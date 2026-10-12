"""
Module: constrained_liquid_dynamics.py
Package: forecast_advancement
Addresses: Stage 5.B — Constrained Oil-and-Water Dynamics Challenger
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Physical Formulation:
Decomposition:
    q_liquid(t) = q_o(t) + q_w(t)
    q_o(t) = q_liquid(t) * [1 - f_w(t)]
    q_w(t) = q_liquid(t) * f_w(t)
    q_g(t) = R_s * q_o(t)

Mathematical Invariants:
1. 0 <= f_w(t) <= 1 identically via logistic or WOR fractional flow dynamics.
2. q_o(t) >= 0 and q_w(t) >= 0 everywhere.
3. Mass conservation: q_o(t) + q_w(t) == q_liquid(t) at all timesteps.
4. Strictly non-increasing oil rate dq_o/dt <= 0 guaranteed when liquid declines
   and fractional flow grows monotonically.
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize_scalar
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from ensemble.data import PARAMS, YEAR_DAYS
from ensemble.run_horizon_adaptive_stacking_experiment import RS_SOLUTION


class ConstrainedLiquidDynamicsForecaster:
    """
    Joint liquid production and fractional-flow forecaster.
    """
    def __init__(self, fractional_flow_mode: str = "logistic"):
        # Mode: "logistic" or "wor_exponential"
        self.fractional_flow_mode = fractional_flow_mode
        self.scaler = StandardScaler()
        self.model_D_liq = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self.model_lambda_fw = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self.b_hyperbolic = 0.25
        self.fitted = False

    def _extract_case_descriptors(self, df_pre: pd.DataFrame, u_c: pd.Series, origin: pd.Timestamp) -> np.ndarray:
        pre = df_pre[df_pre["date"] <= origin]
        last = pre.iloc[-1]
        
        qo_0 = float(last["oil_rate_stbd"])
        qw_0 = float(last["water_rate_stbd"])
        qliq_0 = max(qo_0 + qw_0, 1.0)
        fw_0 = np.clip(qw_0 / qliq_0, 1e-4, 1.0 - 1e-4)
        
        # Recent liquid decline rate over last 2 years
        if len(pre) >= 8:
            sub = pre.iloc[-8:]
            qliq_start = float(sub["oil_rate_stbd"].iloc[0]) + float(sub["water_rate_stbd"].iloc[0])
            d_qliq_hist = -np.log(max(qliq_0, 1.0) / max(qliq_start, 1.0)) / 2.0
            
            fw_start = np.clip(
                float(sub["water_rate_stbd"].iloc[0]) / max(qliq_start, 1.0), 1e-4, 1.0 - 1e-4
            )
            d_fw_hist = (fw_0 - fw_start) / 2.0
        else:
            d_qliq_hist = 0.05
            d_fw_hist = 0.02
            
        fault_t = float(u_c["Fault Transmissibility"])
        poro_m = float(u_c["Porosity Multiplier"])
        perm_m = float(u_c["Permeability Multiplier"])
        aq_pv = float(u_c["Aquifer Pore Volume"]) / 100.0
        
        return np.array([
            np.log(qliq_0),
            fw_0,
            float(logit(fw_0)),
            np.clip(d_qliq_hist, -0.2, 1.0),
            np.clip(d_fw_hist, -0.1, 0.5),
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
    ):
        X_list = []
        y_Dliq_list = []
        y_lfw_list = []
        
        for c in train_cases:
            df_c = case_dict[c]
            pre = df_c[df_c["date"] <= cutoff_date]
            if len(pre) < 8:
                continue
                
            u_c = unc_df.loc[c]
            # Use rolling historical windows within pre-origin history
            # to estimate characteristic liquid decline and water-cut breakthrough rates
            sub_decline = pre[pre["date"] >= "2001-01-01"]
            if len(sub_decline) < 4:
                sub_decline = pre.iloc[-4:]
                
            t_y = (sub_decline["date"] - sub_decline["date"].iloc[0]).dt.total_seconds() / (86400.0 * YEAR_DAYS)
            qliq_arr = sub_decline["oil_rate_stbd"].values + sub_decline["water_rate_stbd"].values
            fw_arr = np.clip(sub_decline["water_rate_stbd"].values / np.maximum(qliq_arr, 1.0), 1e-4, 1.0 - 1e-4)
            
            # Liquid exponential decline fit: q(t) = q0 * exp(-D t)
            if t_y.iloc[-1] > 0.5 and qliq_arr[0] > 10.0:
                d_liq_est = float(np.polyfit(t_y, np.log(np.maximum(qliq_arr, 1.0)), 1)[0])
                D_liq = np.clip(-d_liq_est, 0.01, 0.8)
            else:
                D_liq = 0.05
                
            # Water fractional flow advance rate: logit(fw(t)) = logit(fw0) + lambda * t
            l_fw_arr = logit(fw_arr)
            if t_y.iloc[-1] > 0.5:
                lambda_est = float(np.polyfit(t_y, l_fw_arr, 1)[0])
                lambda_fw = np.clip(lambda_est, 0.005, 0.6)
            else:
                lambda_fw = 0.05
                
            feat = self._extract_case_descriptors(pre, u_c, cutoff_date)
            X_list.append(feat)
            y_Dliq_list.append(D_liq)
            y_lfw_list.append(lambda_fw)
            
        X = np.array(X_list)
        Xs = self.scaler.fit_transform(X)
        self.model_D_liq.fit(Xs, np.array(y_Dliq_list))
        self.model_lambda_fw.fit(Xs, np.array(y_lfw_list))
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
        qliq_0 = max(qo_0 + qw_0, 1.0)
        fw_0 = np.clip(qw_0 / qliq_0, 1e-4, 1.0 - 1e-4)
        
        feat = self._extract_case_descriptors(pre, u_c, origin)
        feat_s = self.scaler.transform(feat.reshape(1, -1))
        
        # Predicted liquid decline rate and water advancement rate
        D_liq = max(0.01, float(self.model_D_liq.predict(feat_s)[0]))
        lambda_fw = max(0.005, float(self.model_lambda_fw.predict(feat_s)[0]))
        
        n_steps = len(forecast_dates)
        t_years = (forecast_dates - origin).days.to_numpy(float) / YEAR_DAYS
        
        # 1. Total liquid trajectory (Hyperbolic/Harmonic smooth decline)
        b = self.b_hyperbolic
        q_liquid = qliq_0 / np.power(1.0 + b * D_liq * t_years, 1.0 / b)
        q_liquid = np.maximum(q_liquid, 0.0)
        
        # 2. Fractional flow trajectory
        if self.fractional_flow_mode == "logistic":
            # Logistic S-curve: fw(t) = 1 / (1 + ((1 - fw0)/fw0) * exp(-lambda * t))
            ratio_0 = (1.0 - fw_0) / fw_0
            fw_t = 1.0 / (1.0 + ratio_0 * np.exp(-lambda_fw * t_years))
        else:
            # WOR exponential: WOR(t) = WOR0 * exp(lambda * t)
            wor_0 = fw_0 / (1.0 - fw_0)
            wor_t = wor_0 * np.exp(lambda_fw * t_years)
            fw_t = wor_t / (1.0 + wor_t)
            
        fw_t = np.clip(fw_t, 0.0, 0.999)
        
        # 3. Exact joint decomposition
        qo_pred = q_liquid * (1.0 - fw_t)
        qw_pred = q_liquid * fw_t
        
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
            "liquid_rate": q_liquid,
            "water_cut": fw_t,
        }
