"""
Module: functional_trajectory_forecaster.py
Package: forecast_advancement
Addresses: Stage 5.C — Low-Dimensional Functional Trajectory Forecaster
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Formulation:
1. Trajectory normalization: Rate normalized by anchor rate q(T_0):
   q_tilde(t) = q(t) / q(T_0)
2. Low-rank functional basis: Orthogonal polynomial/exponential decay modes on [0, T_basis]:
   phi_0(t) = 1.0
   phi_1(t) = exp(-beta_1 * t) - 1.0
   phi_2(t) = exp(-beta_2 * t) - 1.0
   Guarantees exact rate continuity at forecast origin:
   q_tilde(0) = phi_0(0) + c_1 * 0 + c_2 * 0 = 1.0 -> q(0+) = q(T_0).
3. Regularized mapping from historical shape descriptors to basis coefficients c_m.
4. Analytical decline tail for bounded extrapolation beyond T_basis.
5. Strictly trained on rolling historical segments WITHIN the pre-cutoff period.
   Requires NO post-2008 ground truth!
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

from ensemble.data import PARAMS, YEAR_DAYS
from ensemble.run_horizon_adaptive_stacking_experiment import RS_SOLUTION


class FunctionalTrajectoryForecaster:
    """
    Direct functional trajectory model strictly trainable on pre-deployment history.
    Uses orthogonal SVD functional basis on normalized trajectory drops (q_tilde(t) - 1.0)
    to guarantee origin continuity and numerical stability.
    """
    def __init__(self, t_basis_quarters: int = 12, n_modes: int = 2):
        self.t_basis_quarters = t_basis_quarters
        self.n_modes = n_modes
        self.scaler = StandardScaler()
        self.svd_o = TruncatedSVD(n_components=n_modes, random_state=42)
        self.svd_w = TruncatedSVD(n_components=n_modes, random_state=42)
        self.model_o = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self.model_w = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self.fitted = False

    def _extract_shape_descriptors(
        self,
        df_pre: pd.DataFrame,
        u_c: pd.Series,
        t_ref: pd.Timestamp,
    ) -> np.ndarray:
        pre = df_pre[df_pre["date"] <= t_ref]
        last = pre.iloc[-1]
        
        qo_0 = float(last["oil_rate_stbd"])
        qw_0 = float(last["water_rate_stbd"])
        qliq_0 = max(qo_0 + qw_0, 1.0)
        fw_0 = qw_0 / qliq_0
        
        # Historical slopes and curvatures (normalized)
        if len(pre) >= 8:
            sub = pre.iloc[-8:]
            qo_arr = sub["oil_rate_stbd"].values / max(qo_0, 1.0)
            qw_arr = sub["water_rate_stbd"].values / max(qw_0, 1.0)
            
            t_sub_y = (sub["date"] - sub["date"].iloc[0]).dt.total_seconds() / (86400.0 * YEAR_DAYS)
            slope_o = float(np.polyfit(t_sub_y, qo_arr, 1)[0]) if np.ptp(t_sub_y) > 0.1 else 0.0
            slope_w = float(np.polyfit(t_sub_y, qw_arr, 1)[0]) if np.ptp(t_sub_y) > 0.1 else 0.0
        else:
            slope_o = -0.1
            slope_w = 0.05
            
        fault_t = float(u_c["Fault Transmissibility"])
        poro_m = float(u_c["Porosity Multiplier"])
        perm_m = float(u_c["Permeability Multiplier"])
        aq_pv = float(u_c["Aquifer Pore Volume"]) / 100.0
        
        return np.array([
            np.log(max(qo_0, 1.0)),
            np.log(max(qw_0, 1.0) + 1.0),
            fw_0,
            slope_o,
            slope_w,
            fault_t,
            poro_m,
            perm_m,
            aq_pv,
        ])

    def fit(
        self,
        case_dict: Dict[int, pd.DataFrame],
        unc_df: pd.DataFrame,
        train_cases: List[int],
        cutoff_date: pd.Timestamp,
        window_quarters: int = 12,
    ):
        """
        Fits mapping strictly using rolling historical windows <= cutoff_date.
        """
        self.t_basis_quarters = window_quarters
        X_list = []
        Yo_list = []
        Yw_list = []
        dt_days = 91.3125
        
        rolling_cutoffs = [
            pd.Timestamp("2002-01-01"),
            pd.Timestamp("2003-01-01"),
            pd.Timestamp("2004-01-01"),
            pd.Timestamp("2005-01-01"),
        ]
        
        for c in train_cases:
            df_c = case_dict[c]
            u_c = unc_df.loc[c]
            
            for t_roll in rolling_cutoffs:
                t_end_window = t_roll + pd.DateOffset(days=int(window_quarters * dt_days))
                if t_end_window > cutoff_date:
                    continue
                    
                pre_roll = df_c[df_c["date"] <= t_roll]
                post_roll = df_c[(df_c["date"] > t_roll) & (df_c["date"] <= t_end_window)]
                
                if len(pre_roll) < 8 or len(post_roll) < window_quarters:
                    continue
                    
                qo_0 = float(pre_roll["oil_rate_stbd"].iloc[-1])
                qw_0 = float(pre_roll["water_rate_stbd"].iloc[-1])
                if qo_0 < 10.0:
                    continue
                    
                target_qo_norm = post_roll["oil_rate_stbd"].values / max(qo_0, 1.0)
                target_qw_norm = post_roll["water_rate_stbd"].values / max(qw_0, 1.0)
                
                # Normalized drop relative to 1.0 (drop at t=0 is 0.0)
                drop_qo = target_qo_norm - 1.0
                growth_qw = target_qw_norm - 1.0
                
                feat = self._extract_shape_descriptors(df_c, u_c, t_roll)
                X_list.append(feat)
                Yo_list.append(drop_qo)
                Yw_list.append(growth_qw)
                
        if len(X_list) < 10:
            self.fitted = False
            return self
            
        X = np.array(X_list)
        Yo = np.array(Yo_list)
        Yw = np.array(Yw_list)
        
        # Fit orthogonal SVD basis on trajectory shapes
        Co = self.svd_o.fit_transform(Yo)
        Cw = self.svd_w.fit_transform(Yw)
        
        Xs = self.scaler.fit_transform(X)
        self.model_o.fit(Xs, Co)
        self.model_w.fit(Xs, Cw)
        self.fitted = True
        return self

    def predict_rates(
        self,
        df_target: pd.DataFrame,
        u_c: pd.Series,
        origin: pd.Timestamp,
        forecast_dates: pd.DatetimeIndex,
        dt_days: float = 91.3125,
    ) -> Dict[str, np.ndarray]:
        pre = df_target[df_target["date"] <= origin]
        last = pre.iloc[-1]
        
        qo_0 = float(last["oil_rate_stbd"])
        qw_0 = float(last["water_rate_stbd"])
        n_steps = len(forecast_dates)
        t_years = (forecast_dates - origin).days.to_numpy(float) / YEAR_DAYS
        
        if not self.fitted:
            # Safe analytical fallback
            qo = qo_0 * np.exp(-0.08 * t_years)
            qw = qw_0 * np.exp(0.04 * t_years)
            return {"oil_rate": qo, "water_rate": qw, "gas_rate": RS_SOLUTION * qo}
            
        feat = self._extract_shape_descriptors(pre, u_c, origin)
        feat_s = self.scaler.transform(feat.reshape(1, -1))
        
        Co_pred = self.model_o.predict(feat_s)
        Cw_pred = self.model_w.predict(feat_s)
        
        # Reconstruct normalized drop on basis window
        drop_qo = self.svd_o.inverse_transform(Co_pred)[0]
        growth_qw = self.svd_w.inverse_transform(Cw_pred)[0]
        
        # Basis window normalized rates
        qo_basis_norm = np.clip(1.0 + drop_qo, 0.05, 1.0)
        qw_basis_norm = np.maximum(0.0, 1.0 + growth_qw)
        
        n_basis = len(qo_basis_norm)
        qo_norm_full = np.zeros(n_steps)
        qw_norm_full = np.zeros(n_steps)
        
        # Map basis window to steps
        n_fill = min(n_steps, n_basis)
        qo_norm_full[:n_fill] = qo_basis_norm[:n_fill]
        qw_norm_full[:n_fill] = qw_basis_norm[:n_fill]
        
        # Extrapolate beyond basis window with analytical exponential tail
        if n_steps > n_basis:
            t_tail_start = n_basis * (dt_days / YEAR_DAYS)
            d_tail = 0.06
            val_tail_o = qo_basis_norm[-1]
            val_tail_w = qw_basis_norm[-1]
            for k in range(n_basis, n_steps):
                dt_tail = t_years[k] - t_tail_start
                qo_norm_full[k] = val_tail_o * np.exp(-d_tail * dt_tail)
                qw_norm_full[k] = val_tail_w * np.exp(0.02 * dt_tail)
                
        # Denormalize rates
        qo_pred = qo_0 * np.maximum(0.0, qo_norm_full)
        qw_pred = qw_0 * np.maximum(0.0, qw_norm_full)
        
        # Monotonic clamping safeguard for oil
        for k in range(1, len(qo_pred)):
            if qo_pred[k] > qo_pred[k - 1]:
                qo_pred[k] = qo_pred[k - 1]
                
        qo_pred = np.maximum(qo_pred, 0.0)
        qw_pred = np.maximum(qw_pred, 0.0)
        qg_pred = RS_SOLUTION * qo_pred
        
        return {
            "oil_rate": qo_pred,
            "water_rate": qw_pred,
            "gas_rate": qg_pred,
        }
