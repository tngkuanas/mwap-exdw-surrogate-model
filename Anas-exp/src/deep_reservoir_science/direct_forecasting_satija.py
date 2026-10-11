"""
Module: direct_forecasting_satija.py
Addresses: Research 2 — Direct History-to-Future Forecasting (Satija et al. 2017)
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Implements the direct forecasting framework based on:
Satija et al. (2017), "Direct forecasting of reservoir performance using production data without history matching"
Computational Geosciences, 21(2), 331-351. DOI: 10.1007/s10596-017-9614-7

Framework:
1. Functional PCA / SVD dimension reduction on observed historical trajectories (oil & water rates).
2. SVD dimension reduction on future trajectory space (oil & water rates).
3. Mapping in reduced latent space:
   - Method 1: Regularized multivariate Ridge regression (Direct-Ridge)
   - Method 2: Canonical Correlation Analysis (Direct-CCA)
   - Method 3: Gaussian Process conditional prediction in latent space (Direct-GP)
4. Monotonic physical cumulative reconstruction:
   - Rate non-negativity constraint.
   - Anchor continuity constraint: q(T_0+) anchored to q(T_0).
   - Monotonic integration to guarantee strictly increasing cumulative curves.
5. Evaluated under strict reservoir isolation on held-out Validation Cases 71-85.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeCV
from sklearn.cross_decomposition import CCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

from ensemble.data import load_uncertainty, TRAIN_CASES, VAL_CASES
from ensemble.metrics import compute_increment_nrmse, compute_bias
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
    YEAR_DAYS,
)

TABLE_DIR = Path("outputs/research/deep_reservoir_science/tables")
FIG_DIR = Path("outputs/research/deep_reservoir_science/figures")
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


class SatijaDirectForecaster:
    """
    Direct History-to-Future Forecaster (Satija et al. 2017).
    Maps observed historical rate curves directly to future rate trajectories without geological inversion.
    """
    def __init__(
        self,
        n_hist_comp: int = 4,
        n_fut_comp: int = 3,
        method: str = "ridge",  # "ridge", "cca", "gp"
        random_state: int = 42,
    ):
        self.n_hist_comp = n_hist_comp
        self.n_fut_comp = n_fut_comp
        self.method = method
        self.random_state = random_state
        
        self.scaler_hist = StandardScaler()
        self.svd_hist = TruncatedSVD(n_components=n_hist_comp, random_state=random_state)
        self.svd_fut_o = TruncatedSVD(n_components=n_fut_comp, random_state=random_state)
        self.svd_fut_w = TruncatedSVD(n_components=n_fut_comp, random_state=random_state)
        
        self.reg_o = None
        self.reg_w = None
        self.fitted = False

    def _extract_history_vector(self, df: pd.DataFrame, origin: pd.Timestamp) -> np.ndarray:
        pre = df[df["date"] <= origin]
        qo = pre["oil_rate_stbd"].values
        qw = pre["water_rate_stbd"].values
        # Feature vector: concatenated standardized log-rates and rates
        l_qo = np.log(np.maximum(qo, 1.0))
        return np.concatenate([l_qo, qw / 1000.0])

    def fit(
        self,
        case_dict: Dict[int, pd.DataFrame],
        train_cases: List[int],
        origin: pd.Timestamp,
        horizon_quarters: int,
    ):
        H_list = []
        Fo_list = []
        Fw_list = []
        
        for c in train_cases:
            df_c = case_dict[c]
            pre = df_c[df_c["date"] <= origin]
            post = df_c[df_c["date"] > origin].iloc[:horizon_quarters]
            if len(post) < horizon_quarters:
                continue
            h_vec = self._extract_history_vector(df_c, origin)
            fo_vec = post["oil_rate_stbd"].values
            fw_vec = post["water_rate_stbd"].values
            H_list.append(h_vec)
            Fo_list.append(fo_vec)
            Fw_list.append(fw_vec)
            
        H = np.array(H_list)
        Fo = np.array(Fo_list)
        Fw = np.array(Fw_list)
        
        # 1. Dimension reduction
        Hs = self.scaler_hist.fit_transform(H)
        Zh = self.svd_hist.fit_transform(Hs)
        Zfo = self.svd_fut_o.fit_transform(Fo)
        Zfw = self.svd_fut_w.fit_transform(Fw)
        
        # 2. Fit mapping Zh -> Zf
        if self.method == "ridge":
            self.reg_o = RidgeCV(alphas=np.logspace(-3, 3, 10)).fit(Zh, Zfo)
            self.reg_w = RidgeCV(alphas=np.logspace(-3, 3, 10)).fit(Zh, Zfw)
        elif self.method == "cca":
            self.reg_o = CCA(n_components=min(self.n_hist_comp, self.n_fut_comp)).fit(Zh, Zfo)
            self.reg_w = CCA(n_components=min(self.n_hist_comp, self.n_fut_comp)).fit(Zh, Zfw)
        elif self.method == "gp":
            kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(self.n_hist_comp), nu=2.5) + WhiteKernel(1e-3)
            self.reg_o = [GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, random_state=42+k).fit(Zh, Zfo[:, k]) for k in range(self.n_fut_comp)]
            self.reg_w = [GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, random_state=52+k).fit(Zh, Zfw[:, k]) for k in range(self.n_fut_comp)]
        else:
            raise ValueError(f"Unknown method {self.method}")
            
        self.fitted = True
        return self

    def predict(
        self,
        df_target: pd.DataFrame,
        origin: pd.Timestamp,
        init_state: Dict[str, Any],
        forecast_dates: pd.DatetimeIndex,
    ) -> Dict[str, np.ndarray]:
        assert self.fitted, "Model must be fitted before predict"
        h_vec = self._extract_history_vector(df_target, origin).reshape(1, -1)
        hs = self.scaler_hist.transform(h_vec)
        zh = self.svd_hist.transform(hs)
        
        # Latent prediction
        if self.method == "ridge":
            zfo_pred = self.reg_o.predict(zh)
            zfw_pred = self.reg_w.predict(zh)
        elif self.method == "cca":
            # CCA predict returns projection; use linear reconstruction
            zfo_pred = self.reg_o.predict(zh)
            zfw_pred = self.reg_w.predict(zh)
        elif self.method == "gp":
            zfo_pred = np.column_stack([gp.predict(zh) for gp in self.reg_o])
            zfw_pred = np.column_stack([gp.predict(zh) for gp in self.reg_w])
            
        # Inverse transform to rate space
        fo_raw = self.svd_fut_o.inverse_transform(zfo_pred)[0]
        fw_raw = self.svd_fut_w.inverse_transform(zfw_pred)[0]
        
        # Physical boundary constraints
        # 1. Non-negativity
        fo_clamped = np.maximum(0.0, fo_raw)
        fw_clamped = np.maximum(0.0, fw_raw)
        
        # 2. Smooth rate anchoring to init_state
        q0_o = init_state["qo"]
        q0_w = init_state["qw"]
        if fo_clamped[0] > 1e-3:
            scale_o = q0_o / max(fo_clamped[0], 1.0)
            fo_smooth = fo_clamped * np.exp(-0.1 * np.arange(len(fo_clamped))) * scale_o + fo_clamped * (1 - np.exp(-0.1 * np.arange(len(fo_clamped))))
        else:
            fo_smooth = fo_clamped
            
        # 3. Monotonic non-increasing rate decay for oil late-time
        for t in range(1, len(fo_smooth)):
            if fo_smooth[t] > fo_smooth[t-1]:
                fo_smooth[t] = fo_smooth[t-1]  # Enforce monotonic oil decline
                
        return {
            "oil_rate": fo_smooth,
            "water_rate": fw_clamped,
        }


def run_satija_experiments(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    origins: List[Tuple[str, int]] = None,
) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("RESEARCH 2: SATIJA ET AL. (2017) DIRECT FORECASTING EVALUATION")
    print("=" * 80)
    
    if origins is None:
        origins = [("2004-01-01", 16), ("2005-01-01", 12)]
        
    dt_days = 91.3125
    records = []
    
    for orig_str, hq in origins:
        orig = pd.Timestamp(orig_str)
        f_dates = pd.date_range(orig + pd.DateOffset(days=1), periods=hq, freq="QS")
        print(f"\n>>> Running Satija Benchmark at Origin {orig_str} (Horizon: {hq}Q) <<<")
        
        # Fit 3 Satija variants on TRAIN_CASES (1-70)
        satija_ridge = SatijaDirectForecaster(n_hist_comp=4, n_fut_comp=3, method="ridge").fit(case_dict, TRAIN_CASES, orig, hq)
        satija_cca = SatijaDirectForecaster(n_hist_comp=3, n_fut_comp=3, method="cca").fit(case_dict, TRAIN_CASES, orig, hq)
        satija_gp = SatijaDirectForecaster(n_hist_comp=3, n_fut_comp=2, method="gp").fit(case_dict, TRAIN_CASES, orig, hq)
        
        for c in VAL_CASES:
            df_c = case_dict[c]
            pre = df_c[df_c["date"] <= orig]
            post = df_c[df_c["date"] > orig].iloc[:hq]
            last = pre.iloc[-1]
            
            init_st = {
                "qo": float(last["oil_rate_stbd"]),
                "qw": float(last["water_rate_stbd"]),
                "Qo": float(last["oil_cum_stb"]),
                "Qw": float(last["water_cum_stb"]),
                "origin_date": orig,
            }
            y_true_o = post["oil_cum_stb"].values
            y_true_w = post["water_cum_stb"].values
            inc_o = y_true_o - init_st["Qo"]
            inc_w = y_true_w - init_st["Qw"]
            
            # Predict with variants
            preds = {
                "Satija_Direct_Ridge": satija_ridge.predict(df_c, orig, init_st, f_dates),
                "Satija_Direct_CCA": satija_cca.predict(df_c, orig, init_st, f_dates),
                "Satija_Direct_GP": satija_gp.predict(df_c, orig, init_st, f_dates),
            }
            
            for m_key, pr in preds.items():
                cum_o = init_st["Qo"] + np.cumsum(pr["oil_rate"] * dt_days)
                cum_w = init_st["Qw"] + np.cumsum(pr["water_rate"] * dt_days)
                no = compute_increment_nrmse(cum_o - y_true_o, inc_o)
                nw = compute_increment_nrmse(cum_w - y_true_w, inc_w)
                mac = 0.5 * (no + nw)
                bias_o = compute_bias(cum_o - y_true_o, inc_o)
                bias_w = compute_bias(cum_w - y_true_w, inc_w)
                
                records.append({
                    "origin": orig_str,
                    "horizon_quarters": hq,
                    "case_id": c,
                    "model_name": m_key,
                    "macro_nrmse": mac,
                    "oil_nrmse": no,
                    "water_nrmse": nw,
                    "oil_bias": bias_o,
                    "water_bias": bias_w,
                })
                
    df_raw = pd.DataFrame(records)
    summary_records = []
    
    for (orig_str, hq, m_name), grp in df_raw.groupby(["origin", "horizon_quarters", "model_name"]):
        summary_records.append({
            "origin": orig_str,
            "horizon_quarters": hq,
            "model_name": m_name,
            "macro_nrmse": float(grp["macro_nrmse"].mean()),
            "oil_nrmse": float(grp["oil_nrmse"].mean()),
            "water_nrmse": float(grp["water_nrmse"].mean()),
            "oil_bias": float(grp["oil_bias"].mean()),
            "water_bias": float(grp["water_bias"].mean()),
        })
        print(f"  {orig_str} | {m_name:<24s}: Macro = {grp['macro_nrmse'].mean():.4f} (Oil = {grp['oil_nrmse'].mean():.4f}, Wat = {grp['water_nrmse'].mean():.4f})")
        
    df_summary = pd.DataFrame(summary_records)
    out_path = TABLE_DIR / "direct_forecasting_satija_results.csv"
    df_summary.to_csv(out_path, index=False)
    print(f"\nSaved Satija Direct Forecasting Results: {out_path}")
    
    return df_summary


if __name__ == "__main__":
    unc = load_uncertainty()
    qdf = load_clean_simulation_quarterly()
    cdict = build_case_quarterly_trajectories(qdf)
    run_satija_experiments(cdict, unc)
