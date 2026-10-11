"""
Module: corrected_residual_dca.py
Package: conditional_challengers
Addresses: Section 2 — Correct Residual Training Semantics
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Implements leak-free, deployment-compatible residual DCA training:
1. Restricts DCA fitting strictly to data available at each rolling origin.
2. Fits the identical historical fallback on t in [2002-01-01, T_orig].
3. Predicts genuine out-of-origin historical windows ending <= T_cutoff.
4. Excludes origins with insufficient mature decline history (T_orig < 2003-01-01).
5. Eliminates artificial target padding (no edge padding!). Uses explicit horizon masks.
6. Implements cross-fitting across reservoir cases via GroupKFold to eliminate in-sample bias.
7. Preserves analytical boundary conditions: r(0) = 0 and r(inf) = 0.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from ensemble.data import load_uncertainty, TRAIN_CASES, VAL_CASES, PARAMS
from ensemble.metrics import compute_increment_nrmse, compute_bias
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
    LearnedDCARateForecaster,
    YEAR_DAYS,
)

TABLE_DIR = Path("outputs/research/conditional_challengers/tables")
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def fit_historical_dca_at_origin(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    train_cases: List[int],
    origin: pd.Timestamp,
) -> LearnedDCARateForecaster:
    """
    Fits the identical historical DCA fallback strictly on history ending at origin.
    Restricts decline fitting strictly to data available in [2002-01-01, origin].
    """
    # Create strict subset of case_dict up to origin
    dict_at_orig = {c: case_dict[c][case_dict[c]["date"] <= origin] for c in train_cases}
    
    # In LearnedDCARateForecaster, when horizon_quarters is passed such that no post data exists,
    # it strictly uses pre[pre["date"] >= "2002-01-01"].
    # Setting horizon_quarters=0 forces the identical historical fallback.
    m_dca = LearnedDCARateForecaster(decline_type="exponential")
    m_dca.fit(dict_at_orig, unc_df, train_cases, origin, horizon_quarters=0)
    return m_dca


def extract_state_features(df_case: pd.DataFrame, origin: pd.Timestamp, u_c: pd.Series) -> np.ndarray:
    """Extracts 9-dimensional state feature vector strictly as-of origin."""
    pre = df_case[df_case["date"] <= origin]
    last = pre.iloc[-1]
    qo_0 = float(last["oil_rate_stbd"])
    qw_0 = float(last["water_rate_stbd"])
    t_days = (origin - pre["date"].iloc[0]).days
    field_age = t_days / YEAR_DAYS
    
    # Recent decline over last 4 quarters if available
    qo_hist = pre["oil_rate_stbd"].dropna().values
    if len(qo_hist) >= 5:
        D_rec = -(np.log(max(qo_0, 1.0)) - np.log(max(qo_hist[-5], 1.0)))
    else:
        D_rec = 0.0
    if not np.isfinite(D_rec):
        D_rec = 0.0
        
    return np.array([
        float(u_c["Fault Transmissibility"]),
        float(u_c["Porosity Multiplier"]),
        float(u_c["Permeability Multiplier"]),
        float(u_c["Aquifer Pore Volume"]),
        np.log(max(qo_0, 1.0)),
        np.log(max(qw_0, 1.0) + 1.0),
        field_age,
        D_rec,
        float(last.get("wor", qw_0 / max(qo_0, 1.0))),
    ])


class CorrectedResidualForecaster:
    """
    Deployment-compatible residual DCA forecaster with leak-free training semantics:
    - Genuine out-of-origin residual targets.
    - Explicit step-wise masking (no edge-padding).
    - Cross-fitted residual generation across training reservoirs.
    """
    def __init__(
        self,
        architecture: str = "gp",  # "gp", "linear"
        kappa: float = 1.5,
        gamma: float = 0.25,
        max_log_delta: float = 0.5,
        n_splits_crossfit: int = 5,
        random_state: int = 42,
    ):
        self.architecture = architecture
        self.kappa = kappa
        self.gamma = gamma
        self.max_log_delta = max_log_delta
        self.n_splits_crossfit = n_splits_crossfit
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.models_o = []
        self.models_w = []
        self.max_horizon = 16
        self.fitted = False

    def fit(
        self,
        case_dict: Dict[int, pd.DataFrame],
        unc_df: pd.DataFrame,
        train_cases: List[int],
        cutoff_date: pd.Timestamp,
        max_horizon_quarters: int = 16,
    ):
        """
        Fits strictly using historical rolling origins preceding cutoff_date.
        All predicted target windows end strictly on or before cutoff_date.
        """
        self.max_horizon = max_horizon_quarters
        
        # 1. Select eligible rolling origins with sufficient mature decline history
        # Mature decline began late 2001. Origin must be >= 2003-01-01 (at least 1 yr mature history)
        min_orig = pd.Timestamp("2003-01-01")
        # Must have at least 2 quarters between origin and cutoff to observe an out-of-origin residual
        max_orig = cutoff_date - pd.DateOffset(days=180)
        
        if max_orig < min_orig:
            valid_origins = [min_orig]
        else:
            # Semi-annual rolling origins (Jan 1 and Jul 1) to balance coverage and computation
            valid_origins = [
                d for d in pd.date_range(min_orig, max_orig, freq="6MS")
                if d <= max_orig
            ]
            if len(valid_origins) == 0:
                valid_origins = [min_orig]
                
        # 2. Extract genuine out-of-origin residual targets with cross-fitting
        # GroupKFold across train_cases to avoid in-sample DCA fit bias
        kf = KFold(n_splits=min(self.n_splits_crossfit, len(train_cases)), shuffle=True, random_state=self.random_state)
        case_arr = np.array(train_cases)
        
        # Structure: list of tuples (features, step_k, target_delta_o, target_delta_w)
        samples_per_step_o = {k: [] for k in range(max_horizon_quarters)}
        samples_per_step_w = {k: [] for k in range(max_horizon_quarters)}
        features_per_step = {k: [] for k in range(max_horizon_quarters)}
        
        for orig in valid_origins:
            # Available post-origin dates strictly ending <= cutoff_date
            post_dates_all = pd.date_range(orig + pd.DateOffset(days=1), cutoff_date, freq="QS")
            n_avail = min(len(post_dates_all), max_horizon_quarters)
            if n_avail < 2:
                continue
                
            post_dates = post_dates_all[:n_avail]
            t_years = (post_dates - orig).days.to_numpy(float) / YEAR_DAYS
            env = (1.0 - np.exp(-self.kappa * t_years)) * np.exp(-self.gamma * t_years)
            env_safe = np.maximum(env, 1e-4)
            
            # Cross-fitting loop: fit DCA on fold_train, predict on fold_val
            for tr_idx, val_idx in kf.split(case_arr):
                f_train = case_arr[tr_idx].tolist()
                f_val = case_arr[val_idx].tolist()
                
                # Fit DCA strictly on f_train up to orig (zero post-orig data)
                dca_oof = fit_historical_dca_at_origin(case_dict, unc_df, f_train, orig)
                
                for c in f_val:
                    df_c = case_dict[c]
                    pre = df_c[df_c["date"] <= orig]
                    post = df_c[(df_c["date"] > orig) & (df_c["date"] <= cutoff_date)].iloc[:n_avail]
                    
                    if len(pre) < 4 or len(post) < 2:
                        continue
                        
                    u_c = unc_df.loc[c]
                    last = pre.iloc[-1]
                    init_st = {
                        "qo": float(last["oil_rate_stbd"]),
                        "qw": float(last["water_rate_stbd"]),
                        "qg": float(last["gas_rate_mscfd"]),
                        "Qo": float(last["oil_cum_stb"]),
                        "Qw": float(last["water_cum_stb"]),
                        "Qg": float(last["gas_cum_mscf"]),
                        "wor": float(last.get("wor", float(last["water_rate_stbd"]) / max(float(last["oil_rate_stbd"]), 1.0))),
                        "res_age": (orig - pre["date"].iloc[0]).days / YEAR_DAYS,
                        "origin_date": orig,
                    }
                    
                    # Out-of-origin DCA forecast
                    dca_preds = dca_oof.predict_rates(init_st, u_c, post_dates)
                    qo_dca = np.maximum(dca_preds["oil_rate"], 1.0)
                    qw_dca = np.maximum(dca_preds["water_rate"], 1.0)
                    
                    obs_qo = np.maximum(post["oil_rate_stbd"].values, 1.0)
                    obs_qw = np.maximum(post["water_rate_stbd"].values, 1.0)
                    
                    # Residual: r(t) = ln(obs) - ln(dca)
                    r_o = np.log(obs_qo) - np.log(qo_dca)
                    r_w = np.log(obs_qw) - np.log(qw_dca)
                    
                    # Un-damped residual Delta = r / env
                    delta_o = np.clip(r_o / env_safe[:len(r_o)], -self.max_log_delta, self.max_log_delta)
                    delta_w = np.clip(r_w / env_safe[:len(r_w)], -self.max_log_delta, self.max_log_delta)
                    
                    feats = extract_state_features(df_c, orig, u_c)
                    
                    # Store strictly observed quarters (NO PADDING!)
                    for step_idx in range(len(delta_o)):
                        features_per_step[step_idx].append(feats)
                        samples_per_step_o[step_idx].append(delta_o[step_idx])
                        samples_per_step_w[step_idx].append(delta_w[step_idx])

        # 3. Fit step-wise regressors across valid observed samples
        self.models_o = []
        self.models_w = []
        self.scalers = []
        
        for step_idx in range(max_horizon_quarters):
            if len(features_per_step[step_idx]) < 10:
                # If late quarter has too few historical rolling observations, fallback to earlier step
                self.models_o.append(self.models_o[-1] if self.models_o else None)
                self.models_w.append(self.models_w[-1] if self.models_w else None)
                self.scalers.append(self.scalers[-1] if self.scalers else None)
                continue
                
            X_step = np.array(features_per_step[step_idx])
            Yo_step = np.array(samples_per_step_o[step_idx])
            Yw_step = np.array(samples_per_step_w[step_idx])
            
            sc = StandardScaler()
            Xs_step = sc.fit_transform(X_step)
            self.scalers.append(sc)
            
            if self.architecture == "linear":
                m_o = RidgeCV(alphas=np.logspace(-3, 3, 10)).fit(Xs_step, Yo_step)
                m_w = RidgeCV(alphas=np.logspace(-3, 3, 10)).fit(Xs_step, Yw_step)
            elif self.architecture == "gp":
                # Subsample if large for GP efficiency
                if len(Xs_step) > 200:
                    idx_sub = np.random.choice(len(Xs_step), 200, replace=False)
                    Xs_gp = Xs_step[idx_sub]
                    Yo_gp = Yo_step[idx_sub]
                    Yw_gp = Yw_step[idx_sub]
                else:
                    Xs_gp = Xs_step
                    Yo_gp = Yo_step
                    Yw_gp = Yw_step
                    
                kernel = ConstantKernel(0.1, (1e-2, 10.0)) * Matern(length_scale=np.ones(X_step.shape[1]), nu=2.5) + WhiteKernel(1e-3)
                m_o = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, random_state=self.random_state+step_idx).fit(Xs_gp, Yo_gp)
                m_w = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, random_state=self.random_state+50+step_idx).fit(Xs_gp, Yw_gp)
            else:
                raise ValueError(f"Unknown architecture {self.architecture}")
                
            self.models_o.append(m_o)
            self.models_w.append(m_w)
            
        self.fitted = True
        return self

    def predict_rates(
        self,
        df_target: pd.DataFrame,
        u_c: pd.Series,
        origin: pd.Timestamp,
        forecast_dates: pd.DatetimeIndex,
        base_dca: LearnedDCARateForecaster,
    ) -> Dict[str, np.ndarray]:
        """Generates physically constrained rate predictions with asymptotic damping."""
        assert self.fitted, "Model must be fitted before predict"
        pre = df_target[df_target["date"] <= origin]
        last = pre.iloc[-1]
        
        init_st = {
            "qo": float(last["oil_rate_stbd"]),
            "qw": float(last["water_rate_stbd"]),
            "qg": float(last.get("gas_rate_mscfd", last.get("gas_rate", 0.0))),
            "Qo": float(last["oil_cum_stb"]),
            "Qw": float(last["water_cum_stb"]),
            "Qg": float(last["gas_cum_mscf"]),
            "wor": float(last.get("wor", float(last["water_rate_stbd"]) / max(float(last["oil_rate_stbd"]), 1.0))),
            "res_age": (origin - pre["date"].iloc[0]).days / YEAR_DAYS,
            "origin_date": origin,
        }
        
        # 1. Base DCA rates
        dca_preds = base_dca.predict_rates(init_st, u_c, forecast_dates)
        qo_dca = np.asarray(dca_preds["oil_rate"], dtype=float)
        qw_dca = np.asarray(dca_preds["water_rate"], dtype=float)
        
        n_steps = len(forecast_dates)
        days = (forecast_dates - origin).days.to_numpy(float)
        t_years = days / YEAR_DAYS
        
        # 2. Envelope: r(0) = 0 and r(inf) = 0
        envelope = (1.0 - np.exp(-self.kappa * t_years)) * np.exp(-self.gamma * t_years)
        
        # 3. Predict raw residuals step by step
        feats = extract_state_features(df_target, origin, u_c).reshape(1, -1)
        raw_delta_o = np.zeros(n_steps)
        raw_delta_w = np.zeros(n_steps)
        
        for k in range(n_steps):
            step_m_idx = min(k, len(self.models_o) - 1)
            sc = self.scalers[step_m_idx]
            xs = sc.transform(feats)
            mo = self.models_o[step_m_idx]
            mw = self.models_w[step_m_idx]
            raw_delta_o[k] = float(mo.predict(xs)[0])
            raw_delta_w[k] = float(mw.predict(xs)[0])
            
        eff_res_o = np.clip(raw_delta_o, -self.max_log_delta, self.max_log_delta) * envelope
        eff_res_w = np.clip(raw_delta_w, -self.max_log_delta, self.max_log_delta) * envelope
        
        qo_adj = qo_dca * np.exp(eff_res_o)
        qw_adj = qw_dca * np.exp(eff_res_w)
        
        # 4. Strict physical constraints
        qo_adj = np.maximum(qo_adj, 0.0)
        qw_adj = np.maximum(qw_adj, 0.0)
        
        # Monotonic oil decline post-plateau
        for t in range(1, len(qo_adj)):
            if qo_adj[t] > qo_adj[t-1]:
                qo_adj[t] = qo_adj[t-1]
                
        return {
            "oil_rate": qo_adj,
            "water_rate": qw_adj,
            "gas_rate": qo_adj * 0.3633,
            "envelope": envelope,
            "effective_residual_o": eff_res_o,
            "effective_residual_w": eff_res_w,
            "qo_dca": qo_dca,
            "qw_dca": qw_dca,
        }
