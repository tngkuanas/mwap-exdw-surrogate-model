"""
Module: deployable_residual_dca.py
Package: deployable_challengers
Addresses: Section 3 — Implementation of Deployment-Compatible Residual DCA
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Formulation:
    q(t) = q_DCA(t) * exp(r(t, x))
    r(t, x) = [1 - exp(-kappa * t)] * exp(-gamma * t) * Delta(t, x)

Boundary and Asymptotic Conditions:
1. r(0) = 0 exactly: Forecast starts strictly at observed rate q(0) = q_DCA(0).
2. lim[t->inf] r(t) = 0: Smoothly attenuates to pure physical DCA at late horizons.
3. Post-plateau monotonic decline: Complete oil rate curve q_o(t) is clamped to be
   strictly non-increasing (q_o(t_k) <= q_o(t_{k-1})), preventing unphysical rate spikes.

Three Deployable Challenger Architectures:
- Challenger A: Regularized Linear Residual Model (RidgeCV)
- Challenger B: Gaussian Process Residual Model (GP with Matern 5/2 kernel)
- Challenger C: Low-Dimensional Functional-Basis Residual Model (Chebyshev basis + Ridge)

Training Protocol:
- Strictly constructed from rolling historical origins T_train whose target windows
  end on or before cutoff date T_cutoff. Zero post-cutoff target leakage.
- Fully evaluated at 80 quarters (2008-2028) across all three variants.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

from ensemble.data import load_uncertainty, TRAIN_CASES, VAL_CASES, PARAMS
from ensemble.metrics import compute_increment_nrmse, compute_bias
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
    LearnedDCARateForecaster,
    YEAR_DAYS,
)

TABLE_DIR = Path("outputs/research/deployable_challengers/tables")
FIG_DIR = Path("outputs/research/deployable_challengers/figures")
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


class DeployableResidualForecaster:
    """
    Deployment-compatible residual DCA forecaster with explicit boundary and asymptotic conditions.
    """
    def __init__(
        self,
        architecture: str = "linear",  # "linear", "gp", "functional"
        kappa: float = 1.5,             # Boundary rise rate (1/year): ensures r(0) = 0
        gamma: float = 0.25,            # Asymptotic decay rate (1/year): ensures r(inf) = 0
        max_log_delta: float = 0.5,     # Hard bound on raw residual Delta(t)
        n_functional_bases: int = 3,    # Number of Chebyshev polynomial bases for functional model
        random_state: int = 42,
    ):
        self.architecture = architecture
        self.kappa = kappa
        self.gamma = gamma
        self.max_log_delta = max_log_delta
        self.n_functional_bases = n_functional_bases
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.model_o = None
        self.model_w = None
        self.fitted = False

    def _extract_state_features(self, df_case: pd.DataFrame, origin: pd.Timestamp, u_c: pd.Series) -> np.ndarray:
        pre = df_case[df_case["date"] <= origin]
        last = pre.iloc[-1]
        qo_0 = float(last["oil_rate_stbd"])
        qw_0 = float(last["water_rate_stbd"])
        t_days = (origin - pre["date"].iloc[0]).days
        field_age = t_days / YEAR_DAYS
        
        # Recent decline
        qo_hist = pre["oil_rate_stbd"].values
        D_rec = -(np.log(max(qo_0, 1.0)) - np.log(max(qo_hist[-5], 1.0))) if len(qo_hist) >= 5 else 0.0
        
        feats = np.array([
            float(u_c["Fault Transmissibility"]),
            float(u_c["Porosity Multiplier"]),
            float(u_c["Permeability Multiplier"]),
            float(u_c["Aquifer Pore Volume"]),
            np.log(max(qo_0, 1.0)),
            np.log(max(qw_0, 1.0)),
            field_age,
            D_rec,
            float(last.get("wor", qw_0 / max(qo_0, 1.0))),
        ])
        return feats

    def _evaluate_chebyshev_bases(self, t_years: np.ndarray, max_t: float = 4.0) -> np.ndarray:
        """Evaluates first K Chebyshev polynomial basis functions mapped to [0, max_t]."""
        # Map t to [-1, 1]
        tau = np.clip(2.0 * (t_years / max_t) - 1.0, -1.0, 1.0)
        bases = []
        bases.append(np.ones_like(tau))                  # T_0(tau) = 1
        if self.n_functional_bases > 1:
            bases.append(tau)                           # T_1(tau) = tau
        if self.n_functional_bases > 2:
            bases.append(2.0 * tau**2 - 1.0)            # T_2(tau) = 2tau^2 - 1
        if self.n_functional_bases > 3:
            bases.append(4.0 * tau**3 - 3.0 * tau)      # T_3(tau) = 4tau^3 - 3tau
        return np.column_stack(bases[:self.n_functional_bases])

    def fit(
        self,
        case_dict: Dict[int, pd.DataFrame],
        unc_df: pd.DataFrame,
        train_cases: List[int],
        cutoff_date: pd.Timestamp,
        horizon_quarters: int = 16,
    ):
        """
        Fits using rolling origins strictly preceding cutoff_date.
        All target windows end on or before cutoff_date.
        """
        # Generate rolling origins between 2001-01-01 and cutoff_date - 1 year (at least 4 quarters before cutoff)
        min_date = pd.Timestamp("2001-01-01")
        max_orig = cutoff_date - pd.DateOffset(days=365)
        if max_orig < min_date:
            valid_origins = [min_date]
        else:
            valid_origins = pd.date_range(min_date, max_orig, freq="QS").tolist()

        X_list = []
        Yo_list = []
        Yw_list = []

        for orig in valid_origins:
            # Number of available historical quarters between orig and cutoff_date
            post_dates = pd.date_range(orig + pd.DateOffset(days=1), cutoff_date, freq="QS")
            n_avail = min(len(post_dates), horizon_quarters)
            if n_avail < 4:
                continue
                
            f_dates = post_dates[:n_avail]
            t_years = (f_dates - orig).days.to_numpy(float) / YEAR_DAYS
            
            # Theoretical envelope
            env = (1.0 - np.exp(-self.kappa * t_years)) * np.exp(-self.gamma * t_years)
            env_safe = np.maximum(env, 1e-4)
            
            # Filter case dict strictly up to cutoff_date (guaranteeing no post-cutoff data is touched)
            dict_cutoff = {c: case_dict[c][case_dict[c]["date"] <= cutoff_date] for c in train_cases}
            base_dca = LearnedDCARateForecaster(decline_type="exponential").fit(
                dict_cutoff, unc_df, train_cases, orig, n_avail
            )
            
            for c in train_cases:
                df_c = case_dict[c]
                pre = df_c[df_c["date"] <= orig]
                post = df_c[df_c["date"] > orig].iloc[:horizon_quarters]
                
                if len(pre) < 4 or len(post) < horizon_quarters:
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
                    "res_age": float(last.get("res_age_years", (orig - pre["date"].iloc[0]).days / YEAR_DAYS)),
                    "origin_date": orig,
                }
                
                # Predict base DCA rates
                dca_preds = base_dca.predict_rates(init_st, u_c, f_dates)
                qo_dca = np.maximum(dca_preds["oil_rate"], 1.0)
                qw_dca = np.maximum(dca_preds["water_rate"], 1.0)
                
                # Observed rates
                obs_qo = np.maximum(post["oil_rate_stbd"].values[:n_avail], 1.0)
                obs_qw = np.maximum(post["water_rate_stbd"].values[:n_avail], 1.0)
                
                # True log residuals: r(t) = ln(obs) - ln(dca)
                r_o = np.log(obs_qo) - np.log(qo_dca)
                r_w = np.log(obs_qw) - np.log(qw_dca)
                
                # Invert envelope to get raw residual Delta(t) = r(t) / envelope(t)
                delta_o = np.clip(r_o / env_safe, -self.max_log_delta, self.max_log_delta)
                delta_w = np.clip(r_w / env_safe, -self.max_log_delta, self.max_log_delta)
                
                if len(delta_o) < horizon_quarters:
                    delta_o = np.pad(delta_o, (0, horizon_quarters - len(delta_o)), mode="edge")
                    delta_w = np.pad(delta_w, (0, horizon_quarters - len(delta_w)), mode="edge")
                else:
                    delta_o = delta_o[:horizon_quarters]
                    delta_w = delta_w[:horizon_quarters]
                
                feats = self._extract_state_features(df_c, orig, u_c)
                
                X_list.append(feats)
                Yo_list.append(delta_o)
                Yw_list.append(delta_w)

        assert len(X_list) > 10, f"Too few training samples: {len(X_list)}"
        X = np.array(X_list)
        Yo = np.array(Yo_list)
        Yw = np.array(Yw_list)

        Xs = self.scaler.fit_transform(X)

        if self.architecture == "linear":
            self.model_o = RidgeCV(alphas=np.logspace(-3, 3, 15)).fit(Xs, Yo)
            self.model_w = RidgeCV(alphas=np.logspace(-3, 3, 15)).fit(Xs, Yw)
        elif self.architecture == "gp":
            # Compact GP fitted on mean residual across time steps or per-step
            kernel = ConstantKernel(0.1, (1e-2, 10.0)) * Matern(length_scale=np.ones(X.shape[1]), nu=2.5) + WhiteKernel(1e-3)
            # Multi-output GP using small ensemble across quarters
            self.model_o = [GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, random_state=self.random_state+k).fit(Xs, Yo[:, k]) for k in range(Yo.shape[1])]
            self.model_w = [GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, random_state=self.random_state+50+k).fit(Xs, Yw[:, k]) for k in range(Yw.shape[1])]
        elif self.architecture == "functional":
            # Decompose Yo and Yw into Chebyshev polynomial basis coefficients
            Phi = self._evaluate_chebyshev_bases(np.arange(Yo.shape[1]) * 0.25)
            # Solve for basis weights alpha per training sample: Yo_i = Phi @ alpha_i
            # Alpha: (N, n_bases)
            Alpha_o = np.linalg.lstsq(Phi, Yo.T, rcond=None)[0].T
            Alpha_w = np.linalg.lstsq(Phi, Yw.T, rcond=None)[0].T
            self.model_o = RidgeCV(alphas=np.logspace(-3, 3, 15)).fit(Xs, Alpha_o)
            self.model_w = RidgeCV(alphas=np.logspace(-3, 3, 15)).fit(Xs, Alpha_w)
            self.Phi_train = Phi
        else:
            raise ValueError(f"Unknown architecture {self.architecture}")

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
        assert self.fitted, "Model must be fitted before predict"
        pre = df_target[df_target["date"] <= origin]
        last = pre.iloc[-1]
        
        init_st = {
            "qo": float(last["oil_rate_stbd"]),
            "qw": float(last["water_rate_stbd"]),
            "qg": float(last["gas_rate_mscfd"]),
            "Qo": float(last["oil_cum_stb"]),
            "Qw": float(last["water_cum_stb"]),
            "Qg": float(last["gas_cum_mscf"]),
            "wor": float(last.get("wor", float(last["water_rate_stbd"]) / max(float(last["oil_rate_stbd"]), 1.0))),
            "res_age": float(last.get("res_age_years", (origin - pre["date"].iloc[0]).days / YEAR_DAYS)),
            "origin_date": origin,
        }
        
        # 1. Base analytical DCA rates
        dca_preds = base_dca.predict_rates(init_st, u_c, forecast_dates)
        qo_dca = dca_preds["oil_rate"]
        qw_dca = dca_preds["water_rate"]
        
        n_steps = len(forecast_dates)
        days = (forecast_dates - origin).days.to_numpy(float)
        t_years = days / YEAR_DAYS
        
        # 2. Asymptotic bounded envelope: r(0) = 0 and r(inf) = 0
        envelope = (1.0 - np.exp(-self.kappa * t_years)) * np.exp(-self.gamma * t_years)
        
        feats = self._extract_state_features(df_target, origin, u_c).reshape(1, -1)
        xs = self.scaler.transform(feats)
        
        if self.architecture == "linear":
            raw_delta_o = self.model_o.predict(xs)[0]
            raw_delta_w = self.model_w.predict(xs)[0]
            if len(raw_delta_o) < n_steps:
                delta_o = np.pad(raw_delta_o, (0, n_steps - len(raw_delta_o)), mode="edge")
                delta_w = np.pad(raw_delta_w, (0, n_steps - len(raw_delta_w)), mode="edge")
            else:
                delta_o = raw_delta_o[:n_steps]
                delta_w = raw_delta_w[:n_steps]
        elif self.architecture == "gp":
            raw_delta_o = np.array([gp.predict(xs)[0] for gp in self.model_o])
            raw_delta_w = np.array([gp.predict(xs)[0] for gp in self.model_w])
            if len(raw_delta_o) < n_steps:
                delta_o = np.pad(raw_delta_o, (0, n_steps - len(raw_delta_o)), mode="edge")
                delta_w = np.pad(raw_delta_w, (0, n_steps - len(raw_delta_w)), mode="edge")
            else:
                delta_o = raw_delta_o[:n_steps]
                delta_w = raw_delta_w[:n_steps]
        elif self.architecture == "functional":
            pred_alpha_o = self.model_o.predict(xs)[0]
            pred_alpha_w = self.model_w.predict(xs)[0]
            Phi_fwd = self._evaluate_chebyshev_bases(t_years)
            delta_o = Phi_fwd @ pred_alpha_o
            delta_w = Phi_fwd @ pred_alpha_w
            
        # 3. Apply bounds and envelope
        eff_res_o = np.clip(delta_o, -self.max_log_delta, self.max_log_delta) * envelope
        eff_res_w = np.clip(delta_w, -self.max_log_delta, self.max_log_delta) * envelope
        
        qo_raw = np.asarray(qo_dca, dtype=float) * np.exp(eff_res_o)
        qw_raw = np.asarray(qw_dca, dtype=float) * np.exp(eff_res_w)
        
        # 4. Enforce strict physical constraints:
        # Rate non-negativity
        qo_adj = np.maximum(qo_raw, 0.0).copy()
        qw_adj = np.maximum(qw_raw, 0.0).copy()
        
        # Post-plateau monotonic oil decline: q_o(t_k) <= q_o(t_{k-1})
        for t in range(1, len(qo_adj)):
            if qo_adj[t] > qo_adj[t-1]:
                qo_adj[t] = qo_adj[t-1]
                
        return {
            "oil_rate": qo_adj,
            "water_rate": qw_adj,
            "envelope": envelope,
            "effective_residual_o": eff_res_o,
            "effective_residual_w": eff_res_w,
        }


def run_deployable_residual_experiments(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    origins: List[Tuple[str, int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates:
    1. Challenger A: Linear Damped Residual
    2. Challenger B: GP Damped Residual
    3. Challenger C: Functional Damped Residual
    Across retrospective origins AND 80-quarter (20-year) forward stability audit.
    """
    print("\n" + "=" * 80)
    print("DEPLOYMENT-COMPATIBLE RESIDUAL DCA EXPERIMENTS (Strict As-Of)")
    print("=" * 80)
    
    if origins is None:
        origins = [("2004-01-01", 16), ("2005-01-01", 12)]
        
    dt_days = 91.3125
    records_val = []
    
    for orig_str, hq in origins:
        orig = pd.Timestamp(orig_str)
        f_dates = pd.date_range(orig + pd.DateOffset(days=1), periods=hq, freq="QS")
        print(f"\n>>> Running Origin {orig_str} ({hq}Q) with Strict Cutoff at {orig_str} <<<")
        
        # Base DCA fitted as-of orig
        dict_asof = {c: case_dict[c][case_dict[c]["date"] <= orig] for c in TRAIN_CASES}
        m_dca = LearnedDCARateForecaster(decline_type="exponential").fit(
            dict_asof, unc_df, TRAIN_CASES, orig, hq
        )
        
        # Fit 3 deployable challengers strictly on training samples preceding orig
        m_lin = DeployableResidualForecaster(architecture="linear").fit(case_dict, unc_df, TRAIN_CASES, orig, hq)
        m_gp = DeployableResidualForecaster(architecture="gp").fit(case_dict, unc_df, TRAIN_CASES, orig, hq)
        m_func = DeployableResidualForecaster(architecture="functional").fit(case_dict, unc_df, TRAIN_CASES, orig, hq)
        
        models = {
            "Deployable_Damped_Linear_Residual": m_lin,
            "Deployable_Damped_GP_Residual": m_gp,
            "Deployable_Damped_Functional_Residual": m_func,
        }
        
        for c in VAL_CASES:
            df_c = case_dict[c]
            pre = df_c[df_c["date"] <= orig]
            post = df_c[df_c["date"] > orig].iloc[:hq]
            last = pre.iloc[-1]
            u_c = unc_df.loc[c]
            
            init_Qo = float(last["oil_cum_stb"])
            init_Qw = float(last["water_cum_stb"])
            
            y_true_o = post["oil_cum_stb"].values
            y_true_w = post["water_cum_stb"].values
            inc_o = y_true_o - init_Qo
            inc_w = y_true_w - init_Qw
            
            for m_key, model in models.items():
                preds = model.predict_rates(df_c, u_c, orig, f_dates, m_dca)
                cum_o = init_Qo + np.cumsum(preds["oil_rate"] * dt_days)
                cum_w = init_Qw + np.cumsum(preds["water_rate"] * dt_days)
                
                no = compute_increment_nrmse(cum_o - y_true_o, inc_o)
                nw = compute_increment_nrmse(cum_w - y_true_w, inc_w)
                mac = 0.5 * (no + nw)
                bo = compute_bias(cum_o - y_true_o, inc_o)
                bw = compute_bias(cum_w - y_true_w, inc_w)
                
                records_val.append({
                    "origin": orig_str,
                    "horizon_quarters": hq,
                    "case_id": c,
                    "model_name": m_key,
                    "macro_nrmse": mac,
                    "oil_nrmse": no,
                    "water_nrmse": nw,
                    "oil_bias": bo,
                    "water_bias": bw,
                })

    df_raw = pd.DataFrame(records_val)
    summary_records = []
    print("\n" + "-" * 80)
    print(f"{'Origin':<12} | {'Model':<38} | {'Macro NRMSE':<11} | {'Oil NRMSE':<9} | {'Wat NRMSE':<9}")
    print("-" * 80)
    for (orig_str, m_name), grp in df_raw.groupby(["origin", "model_name"]):
        mac = float(grp["macro_nrmse"].mean())
        no = float(grp["oil_nrmse"].mean())
        nw = float(grp["water_nrmse"].mean())
        summary_records.append({
            "origin": orig_str,
            "model_name": m_name,
            "macro_nrmse": mac,
            "oil_nrmse": no,
            "water_nrmse": nw,
            "oil_bias": float(grp["oil_bias"].mean()),
            "water_bias": float(grp["water_bias"].mean()),
        })
        print(f"{orig_str:<12} | {m_name:<38} | {mac:.4f}      | {no:.4f}    | {nw:.4f}")
    print("-" * 80)
    
    df_val_summary = pd.DataFrame(summary_records)
    out_path = TABLE_DIR / "deployable_residual_dca_comparison.csv"
    df_val_summary.to_csv(out_path, index=False)
    print(f"\nSaved Deployable Residual DCA Benchmark: {out_path}")
    
    # ----------------------------------------------------------------------
    # 20-YEAR (80-QUARTER) FORWARD STABILITY AUDIT (2008-2028)
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("20-YEAR (80-QUARTER) ASYMPTOTIC STABILITY AUDIT (2008-2028)")
    print("=" * 80)
    
    orig_2008 = pd.Timestamp("2008-01-01")
    f_dates_80q = pd.date_range(orig_2008 + pd.DateOffset(days=1), periods=80, freq="QS")
    dict_2008 = {c: case_dict[c][case_dict[c]["date"] <= orig_2008] for c in TRAIN_CASES}
    m_dca_2008 = LearnedDCARateForecaster(decline_type="exponential").fit(
        dict_2008, unc_df, TRAIN_CASES, orig_2008, 16
    )
    
    # Fit all three challengers strictly up to 2008-01-01
    m_lin_2008 = DeployableResidualForecaster(architecture="linear").fit(case_dict, unc_df, TRAIN_CASES, orig_2008, 16)
    m_gp_2008 = DeployableResidualForecaster(architecture="gp").fit(case_dict, unc_df, TRAIN_CASES, orig_2008, 16)
    m_func_2008 = DeployableResidualForecaster(architecture="functional").fit(case_dict, unc_df, TRAIN_CASES, orig_2008, 16)
    
    audit_models = {
        "Base_Exponential_DCA": None,
        "Challenger_A_Linear_Residual": m_lin_2008,
        "Challenger_B_GP_Residual": m_gp_2008,
        "Challenger_C_Functional_Residual": m_func_2008,
    }
    
    stab_records = []
    test_cases = [71, 75, 80, 85]
    
    for c in test_cases:
        df_c = case_dict[c]
        u_c = unc_df.loc[c]
        last = df_c[df_c["date"] <= orig_2008].iloc[-1]
        init_st = {
            "qo": float(last["oil_rate_stbd"]),
            "qw": float(last["water_rate_stbd"]),
            "qg": float(last["gas_rate_mscfd"]),
            "Qo": float(last["oil_cum_stb"]),
            "Qw": float(last["water_cum_stb"]),
            "Qg": float(last["gas_cum_mscf"]),
            "wor": float(last.get("wor", float(last["water_rate_stbd"]) / max(float(last["oil_rate_stbd"]), 1.0))),
            "res_age": float(last.get("res_age_years", 10.0)),
            "origin_date": orig_2008,
        }
        
        # Base DCA forecast
        base_preds = m_dca_2008.predict_rates(init_st, u_c, f_dates_80q)
        
        for m_name, sm in audit_models.items():
            if sm is None:
                r_o = base_preds["oil_rate"]
                r_w = base_preds["water_rate"]
            else:
                preds = sm.predict_rates(df_c, u_c, orig_2008, f_dates_80q, m_dca_2008)
                r_o = preds["oil_rate"]
                r_w = preds["water_rate"]
                
            q_term_o = r_o[-1]
            cum_20yr_o = np.sum(r_o * dt_days)
            cum_20yr_w = np.sum(r_w * dt_days)
            
            # Constraints
            is_monotonic = bool((np.diff(r_o) <= 1e-4).all())
            is_positive = bool((r_o >= 0.0).all() and (r_w >= 0.0).all())
            is_finite = bool(np.isfinite(r_o).all() and np.isfinite(r_w).all())
            
            status = "PHYSICALLY STABLE" if (is_monotonic and is_positive and is_finite) else "VIOLATION DETECTED"
            
            stab_records.append({
                "case_id": c,
                "model_name": m_name,
                "q_init_o": r_o[0],
                "q_term_o": q_term_o,
                "cum_20yr_oil_stb": cum_20yr_o,
                "cum_20yr_wat_stb": cum_20yr_w,
                "is_monotonic": is_monotonic,
                "is_positive": is_positive,
                "is_finite": is_finite,
                "stability_status": status,
            })
            print(f"  Case {c:2d} | {m_name:<32s}: q_init={r_o[0]:.0f} -> q_term={q_term_o:.1f} | 20yr Cum={cum_20yr_o/1e6:.2f} MMstb | {status}")

    df_stab = pd.DataFrame(stab_records)
    out_stab_path = TABLE_DIR / "deployable_residual_20year_stability_audit.csv"
    df_stab.to_csv(out_stab_path, index=False)
    print(f"\nSaved 20-Year Asymptotic Stability Audit: {out_stab_path}")
    
    return df_val_summary, df_stab


if __name__ == "__main__":
    unc = load_uncertainty()
    qdf = load_clean_simulation_quarterly()
    cdict = build_case_quarterly_trajectories(qdf)
    run_deployable_residual_experiments(cdict, unc)
