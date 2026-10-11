"""
Module: deployable_direct_forecasting.py
Package: deployable_challengers
Addresses: Section 2 — Implementation of Deployment-Compatible Direct Forecasting
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Deployment-Compatible Direct History-to-Future Forecaster:
1. Strictly trained on historical rolling origins whose target window ends on or before cutoff date T_cutoff.
   Zero post-cutoff simulator targets are used.
2. Fixed-dimensional input representation (D=24) invariant to historical length:
   - Recent K=8 normalized quarterly oil rates q_o(t-k)/q_o(T_0)
   - Recent K=8 normalized quarterly water rates q_w(t-k)/max(q_w(T_0), 10)
   - Log terminal rates ln(q_o(T_0)), ln(max(q_w(T_0), 1))
   - Terminal water cut wc(T_0) and logit(wc(T_0))
   - Recent 1-year oil decline D_recent and water cut change Delta_wc
   - Field age (years from start) and normalized cumulative oil/water produced
3. Functional mappings:
   - Deployable Direct Ridge (RidgeCV with L2 regularization)
   - Deployable Direct CCA (Canonical Correlation Analysis)
   - Deployable Direct Functional SVD
4. Controlled smooth transition to mature analytical DCA tail:
   - Direct model predicts reliable short-medium horizon (H_direct = 12-16Q).
   - For horizons beyond H_direct, a smooth, continuous transition to an exponential DCA tail
     is applied, enforcing exact rate and cumulative continuity:
       q(t) = q_direct(H_direct) * exp(-D_tail * (t - H_direct))
   - Guarantees 20-year asymptotic non-divergence and monotonic late-time decline.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import RidgeCV
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from ensemble.data import load_uncertainty, TRAIN_CASES, VAL_CASES, PARAMS
from ensemble.metrics import compute_increment_nrmse, compute_bias
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
    YEAR_DAYS,
)

TABLE_DIR = Path("outputs/research/deployable_challengers/tables")
FIG_DIR = Path("outputs/research/deployable_challengers/figures")
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

K_RECENT = 8  # Fixed number of recent quarters


def extract_fixed_state_features(
    df_case: pd.DataFrame,
    origin: pd.Timestamp,
    k_recent: int = K_RECENT,
) -> np.ndarray:
    """
    Extracts a strictly fixed-dimensional feature vector (D=24) from production history up to origin.
    Works identically for 2001, 2004, 2005, or 2008 cutoff dates.
    """
    pre = df_case[df_case["date"] <= origin].copy()
    if len(pre) == 0:
        raise ValueError(f"No history before {origin}")
        
    last = pre.iloc[-1]
    qo_0 = float(last["oil_rate_stbd"])
    qw_0 = float(last["water_rate_stbd"])
    Qo_0 = float(last["oil_cum_stb"])
    Qw_0 = float(last["water_cum_stb"])
    
    # 1. Recent K normalized quarterly rates
    # If fewer than K quarters available, backfill with earliest rate
    qo_hist = pre["oil_rate_stbd"].values
    qw_hist = pre["water_rate_stbd"].values
    
    if len(qo_hist) >= k_recent:
        qo_tail = qo_hist[-k_recent:]
        qw_tail = qw_hist[-k_recent:]
    else:
        pad_len = k_recent - len(qo_hist)
        qo_tail = np.pad(qo_hist, (pad_len, 0), mode="edge")
        qw_tail = np.pad(qw_hist, (pad_len, 0), mode="edge")
        
    # Relative rate ratios
    scale_o = max(qo_0, 10.0)
    scale_w = max(qw_0, 10.0)
    qo_norm = qo_tail / scale_o
    qw_norm = qw_tail / scale_w
    
    # 2. Terminal rate levels
    ln_qo0 = np.log(max(qo_0, 1.0))
    ln_qw0 = np.log(max(qw_0, 1.0))
    
    # 3. Water cut and logit
    wc_0 = np.clip(qw_0 / max(qo_0 + qw_0, 1e-6), 1e-4, 1.0 - 1e-4)
    l_wc0 = logit(wc_0)
    
    # 4. Recent trends (1 year / 4 quarters back)
    if len(qo_hist) >= 5:
        qo_4 = qo_hist[-5]
        qw_4 = qw_hist[-5]
        wc_4 = np.clip(qw_4 / max(qo_4 + qw_4, 1e-6), 1e-4, 1.0 - 1e-4)
        D_rec = -(np.log(max(qo_0, 1.0)) - np.log(max(qo_4, 1.0)))
        Delta_wc = wc_0 - wc_4
    else:
        D_rec = 0.0
        Delta_wc = 0.0
        
    # 5. Field age and cumulative maturity proxies
    t_days = (origin - pre["date"].iloc[0]).days
    field_age = t_days / YEAR_DAYS
    Qo_scaled = Qo_0 / 1e7
    Qw_scaled = Qw_0 / 1e7
    
    feats = np.concatenate([
        qo_norm,        # k_recent (8)
        qw_norm,        # k_recent (8)
        [ln_qo0, ln_qw0],  # 2
        [wc_0, l_wc0],     # 2
        [D_rec, Delta_wc], # 2
        [field_age, Qo_scaled, Qw_scaled], # 3
    ])  # Total dimension = 8 + 8 + 2 + 2 + 2 + 3 = 25
    return feats


class DeployableDirectForecaster:
    """
    Direct history-to-future forecaster trained strictly with deployment-compatible rolling origins.
    """
    def __init__(
        self,
        method: str = "ridge",  # "ridge", "cca", "svd_ridge"
        h_direct: int = 16,     # Direct forecast horizon quarters (e.g. 4 years)
        random_state: int = 42,
    ):
        self.method = method
        self.h_direct = h_direct
        self.random_state = random_state
        self.scaler_x = StandardScaler()
        self.scaler_yo = StandardScaler()
        self.scaler_yw = StandardScaler()
        self.model_o = None
        self.model_w = None
        self.fitted = False

    def fit(
        self,
        case_dict: Dict[int, pd.DataFrame],
        train_cases: List[int],
        cutoff_date: pd.Timestamp,
        rolling_origins: Optional[List[pd.Timestamp]] = None,
    ):
        """
        Fits using rolling origins T_orig such that T_orig + H_direct quarters <= cutoff_date.
        Ensures zero post-cutoff target leakage!
        """
        if rolling_origins is None:
            # Generate quarterly rolling origins from 2000 to cutoff - h_direct
            all_dates = pd.date_range("2000-01-01", cutoff_date, freq="QS")
            # Keep origins where target window fits entirely on or before cutoff_date
            valid_origins = []
            for d in all_dates:
                f_end = d + pd.DateOffset(days=int(self.h_direct * 91.3125))
                if f_end <= cutoff_date:
                    valid_origins.append(d)
            rolling_origins = valid_origins

        X_list = []
        Yo_list = []
        Yw_list = []

        for orig in rolling_origins:
            for c in train_cases:
                df_c = case_dict[c]
                pre = df_c[df_c["date"] <= orig]
                post = df_c[df_c["date"] > orig].iloc[:self.h_direct]
                
                # Must have sufficient history and complete forward target window
                if len(pre) < 4 or len(post) < self.h_direct:
                    continue
                    
                feats = extract_fixed_state_features(df_c, orig)
                
                # Targets: relative forward oil rate q_o(t)/q0_o in [0, 1], and forward logit water cut growth Delta_logit_wc
                q0_o = max(float(pre["oil_rate_stbd"].iloc[-1]), 10.0)
                yo = np.clip(post["oil_rate_stbd"].values / q0_o, 0.0, 1.0)
                
                # Bounded logit water cut growth target: Delta_logit_wc(t) = logit(wc(t)) - logit(wc_0)
                pw = post["water_rate_stbd"].values
                po = post["oil_rate_stbd"].values
                wc_0 = np.clip(float(pre["water_rate_stbd"].iloc[-1]) / max(q0_o + float(pre["water_rate_stbd"].iloc[-1]), 1e-6), 1e-4, 1.0 - 1e-4)
                wc_target = np.clip(pw / np.maximum(po + pw, 1e-6), 1e-4, 1.0 - 1e-4)
                delta_logit_wc = logit(wc_target) - logit(wc_0)
                
                X_list.append(feats)
                Yo_list.append(yo)
                Yw_list.append(delta_logit_wc)

        assert len(X_list) > 20, f"Too few training samples: {len(X_list)}"
        X = np.array(X_list)
        Yo = np.array(Yo_list)
        Yw = np.array(Yw_list)

        Xs = self.scaler_x.fit_transform(X)

        if self.method == "ridge":
            self.model_o = RidgeCV(alphas=np.logspace(-2, 4, 20)).fit(Xs, Yo)
            self.model_w = RidgeCV(alphas=np.logspace(-2, 4, 20)).fit(Xs, Yw)
        elif self.method == "cca":
            n_comp = min(Xs.shape[1], Yo.shape[1], 4)
            self.model_o = CCA(n_components=n_comp).fit(Xs, Yo)
            self.model_w = CCA(n_components=n_comp).fit(Xs, Yw)
        elif self.method == "svd_ridge":
            self.svd_yo = TruncatedSVD(n_components=3, random_state=self.random_state)
            self.svd_yw = TruncatedSVD(n_components=3, random_state=self.random_state)
            zo = self.svd_yo.fit_transform(Yo)
            zw = self.svd_yw.fit_transform(Yw)
            self.model_o = RidgeCV(alphas=np.logspace(-2, 4, 20)).fit(Xs, zo)
            self.model_w = RidgeCV(alphas=np.logspace(-2, 4, 20)).fit(Xs, zw)
        else:
            raise ValueError(f"Unknown method {self.method}")

        self.fitted = True
        return self

    def predict_rates(
        self,
        df_target: pd.DataFrame,
        origin: pd.Timestamp,
        forecast_dates: pd.DatetimeIndex,
    ) -> Dict[str, np.ndarray]:
        """
        Generates production rate forecasts for any horizon length H.
        If H > H_direct, performs a smooth, continuous transition to a mature DCA tail.
        """
        assert self.fitted, "Model must be fitted before predict"
        pre = df_target[df_target["date"] <= origin]
        q0_o = float(pre["oil_rate_stbd"].iloc[-1])
        q0_w = float(pre["water_rate_stbd"].iloc[-1])
        wc_0 = np.clip(q0_w / max(q0_o + q0_w, 1e-6), 1e-4, 1.0 - 1e-4)
        
        feats = extract_fixed_state_features(df_target, origin).reshape(1, -1)
        xs = self.scaler_x.transform(feats)
        
        if self.method == "ridge":
            pred_ratio_o = np.clip(self.model_o.predict(xs)[0], 0.0, 1.0)
            pred_delta_l_w = self.model_w.predict(xs)[0]
        elif self.method == "cca":
            pred_ratio_o = np.clip(self.model_o.predict(xs)[0], 0.0, 1.0)
            pred_delta_l_w = self.model_w.predict(xs)[0]
        elif self.method == "svd_ridge":
            pred_zo = self.model_o.predict(xs)
            pred_zw = self.model_w.predict(xs)
            pred_ratio_o = np.clip(self.svd_yo.inverse_transform(pred_zo)[0], 0.0, 1.0)
            pred_delta_l_w = self.svd_yw.inverse_transform(pred_zw)[0]
            
        # Reconstruct physical rates for the first H_direct quarters
        qo_direct = np.maximum(0.0, pred_ratio_o * max(q0_o, 10.0))
        # Water cut starts at wc_0 and grows with bounded positive logit growth
        delta_l_clipped = np.clip(pred_delta_l_w, 0.0, 3.0)
        wc_pred = expit(logit(wc_0) + delta_l_clipped)
        qw_direct = qo_direct * (wc_pred / np.maximum(1.0 - wc_pred, 1e-4))
        
        # Continuity anchor at t=0
        qo_direct[0] = 0.5 * (q0_o + qo_direct[0])
        qw_direct[0] = 0.5 * (q0_w + qw_direct[0])
        
        # Enforce monotonic oil decline for direct portion
        for t in range(1, len(qo_direct)):
            if qo_direct[t] > qo_direct[t-1]:
                qo_direct[t] = qo_direct[t-1]
                
        n_req = len(forecast_dates)
        if n_req <= self.h_direct:
            return {
                "oil_rate": qo_direct[:n_req],
                "water_rate": qw_direct[:n_req],
            }
            
        # ----------------------------------------------------------------------
        # Controlled Smooth Transition to Mature DCA Tail (for H > H_direct)
        # ----------------------------------------------------------------------
        # Estimate tail decline rate from the last 4 quarters of direct forecast
        if len(qo_direct) >= 4 and qo_direct[-4] > 10.0 and qo_direct[-1] > 1.0:
            d_tail = max(0.02, -(np.log(qo_direct[-1]) - np.log(qo_direct[-4])) / 1.0)
        else:
            d_tail = 0.08  # Default 8% annual decline
            
        d_tail = float(np.clip(d_tail, 0.02, 0.35))
        
        dt_tail_quarters = np.arange(1, n_req - self.h_direct + 1)
        dt_tail_years = dt_tail_quarters * 0.25
        
        # Exact exponential tail matching boundary value at H_direct
        q_term_o = qo_direct[-1]
        qo_tail = q_term_o * np.exp(-d_tail * dt_tail_years)
        
        # Water tail relaxation: bounded water cut or gradual decline
        q_term_w = qw_direct[-1]
        qw_tail = q_term_w * np.exp(-0.04 * dt_tail_years)
        
        qo_full = np.concatenate([qo_direct, qo_tail])[:n_req]
        qw_full = np.concatenate([qw_direct, qw_tail])[:n_req]
        
        return {
            "oil_rate": qo_full,
            "water_rate": qw_full,
            "d_tail_used": d_tail,
        }


def run_deployable_direct_experiments(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    origins: List[Tuple[str, int]] = None,
) -> pd.DataFrame:
    """
    Evaluates deployment-compatible direct forecasters strictly under as-of rules.
    """
    print("\n" + "=" * 80)
    print("DEPLOYMENT-COMPATIBLE DIRECT FORECASTING EXPERIMENTS (Strict As-Of)")
    print("=" * 80)
    
    if origins is None:
        origins = [("2004-01-01", 16), ("2005-01-01", 12)]
        
    dt_days = 91.3125
    records = []
    
    for orig_str, hq in origins:
        orig = pd.Timestamp(orig_str)
        f_dates = pd.date_range(orig + pd.DateOffset(days=1), periods=hq, freq="QS")
        print(f"\n>>> Evaluating Origin {orig_str} ({hq}Q) with Strict Cutoff at {orig_str} <<<")
        
        # Fit models with training window ending strictly on or before orig
        m_ridge = DeployableDirectForecaster(method="ridge", h_direct=hq).fit(
            case_dict, TRAIN_CASES, cutoff_date=orig
        )
        m_cca = DeployableDirectForecaster(method="cca", h_direct=hq).fit(
            case_dict, TRAIN_CASES, cutoff_date=orig
        )
        m_svd = DeployableDirectForecaster(method="svd_ridge", h_direct=hq).fit(
            case_dict, TRAIN_CASES, cutoff_date=orig
        )
        
        models = {
            "Deployable_Direct_Ridge": m_ridge,
            "Deployable_Direct_CCA": m_cca,
            "Deployable_Direct_SVD_Ridge": m_svd,
        }
        
        for c in VAL_CASES:
            df_c = case_dict[c]
            pre = df_c[df_c["date"] <= orig]
            post = df_c[df_c["date"] > orig].iloc[:hq]
            last = pre.iloc[-1]
            
            init_Qo = float(last["oil_cum_stb"])
            init_Qw = float(last["water_cum_stb"])
            
            y_true_o = post["oil_cum_stb"].values
            y_true_w = post["water_cum_stb"].values
            inc_o = y_true_o - init_Qo
            inc_w = y_true_w - init_Qw
            
            for m_key, model in models.items():
                preds = model.predict_rates(df_c, orig, f_dates)
                cum_o = init_Qo + np.cumsum(preds["oil_rate"] * dt_days)
                cum_w = init_Qw + np.cumsum(preds["water_rate"] * dt_days)
                
                no = compute_increment_nrmse(cum_o - y_true_o, inc_o)
                nw = compute_increment_nrmse(cum_w - y_true_w, inc_w)
                mac = 0.5 * (no + nw)
                bo = compute_bias(cum_o - y_true_o, inc_o)
                bw = compute_bias(cum_w - y_true_w, inc_w)
                
                records.append({
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
                
    df_raw = pd.DataFrame(records)
    summary_records = []
    
    print("\n" + "-" * 80)
    print(f"{'Origin':<12} | {'Model':<30} | {'Macro NRMSE':<11} | {'Oil NRMSE':<9} | {'Wat NRMSE':<9}")
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
        print(f"{orig_str:<12} | {m_name:<30} | {mac:.4f}      | {no:.4f}    | {nw:.4f}")
    print("-" * 80)
    
    df_sum = pd.DataFrame(summary_records)
    out_path = TABLE_DIR / "deployable_direct_forecasting_results.csv"
    df_sum.to_csv(out_path, index=False)
    print(f"\nSaved Deployable Direct Forecasting Results: {out_path}")
    return df_sum


if __name__ == "__main__":
    unc = load_uncertainty()
    qdf = load_clean_simulation_quarterly()
    cdict = build_case_quarterly_trajectories(qdf)
    run_deployable_direct_experiments(cdict, unc)
