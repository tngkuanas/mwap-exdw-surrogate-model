"""
Module: recovery_state_forecaster.py
Package: forecast_advancement
Addresses: Stage 5.A — Recovery-State Forecasting Challenger
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Physical Formulation:
- Measured variables: q_o, q_w, Q_o, Q_w, f_w = q_w / (q_o + q_w), q_liq = q_o + q_w
- Normalization proxies:
  * Oil recovery factor proxy: R_f = Q_o / (V_base * Porosity_Multiplier)
  * Aquifer water displacement proxy: W_f = Q_w / (Aquifer_PV * 1e6)
  * Transformed fractional flow: logit(f_w)
  * Geological proxies: Fault Transmissibility, Permeability Multiplier
- Direct state-to-decline mapping:
  D_oil(t) = f(R_f(t), W_f(t), f_w(t), q_liq(t), geology)
- Smooth integration guaranteeing non-negative rates and monotonic recovery.
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from ensemble.data import PARAMS, YEAR_DAYS
from ensemble.run_horizon_adaptive_stacking_experiment import RS_SOLUTION


class RecoveryStateForecaster:
    """
    Compact forecasting model based on normalized reservoir recovery state.
    """
    def __init__(self, base_ooip: float = 1.5e8, base_aq_pv: float = 1.0e8):
        self.base_ooip = base_ooip
        self.base_aq_pv = base_aq_pv
        self.scaler = StandardScaler()
        self.model_D_oil = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self.model_D_liq = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self.model_fw_rate = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self.fitted = False

    def _extract_state_features(
        self,
        qo: float,
        qw: float,
        Qo: float,
        Qw: float,
        u_c: pd.Series,
        prev_qliq: Optional[float] = None,
    ) -> np.ndarray:
        poro_m = float(u_c["Porosity Multiplier"])
        perm_m = float(u_c["Permeability Multiplier"])
        fault_t = float(u_c["Fault Transmissibility"])
        aq_pv = float(u_c["Aquifer Pore Volume"])
        
        ooip_proxy = self.base_ooip * poro_m
        aq_proxy = aq_pv * 1e6
        
        rf = Qo / max(ooip_proxy, 1e6)
        wf = Qw / max(aq_proxy, 1e6)
        
        qliq = max(qo + qw, 1.0)
        fw = np.clip(qw / qliq, 1e-4, 1.0 - 1e-4)
        l_fw = float(logit(fw))
        
        d_qliq = (qliq - prev_qliq) / prev_qliq if prev_qliq is not None and prev_qliq > 0 else 0.0
        
        return np.array([
            rf,
            wf,
            fw,
            l_fw,
            np.log(qliq),
            d_qliq,
            fault_t,
            poro_m,
            perm_m,
            aq_pv / 100.0,
        ])

    def fit(
        self,
        case_dict: Dict[int, pd.DataFrame],
        unc_df: pd.DataFrame,
        train_cases: List[int],
        cutoff_date: pd.Timestamp,
    ):
        X_list = []
        y_Do_list = []
        y_Dliq_list = []
        y_dfw_list = []
        
        for c in train_cases:
            df_c = case_dict[c]
            pre = df_c[df_c["date"] <= cutoff_date].sort_values("date").reset_index(drop=True)
            u_c = unc_df.loc[c]
            
            # Start from quarter 4 to allow trend estimation
            for i in range(4, len(pre)):
                prev = pre.iloc[i - 1]
                curr = pre.iloc[i]
                
                qo_p = float(prev["oil_rate_stbd"])
                qw_p = float(prev["water_rate_stbd"])
                Qo_p = float(prev["oil_cum_stb"])
                Qw_p = float(prev["water_cum_stb"])
                
                qo_c = float(curr["oil_rate_stbd"])
                qw_c = float(curr["water_rate_stbd"])
                
                dt_y = 0.25  # 1 quarter = 0.25 years
                
                qliq_p = qo_p + qw_p
                qliq_c = qo_c + qw_c
                
                prev_prev = pre.iloc[i - 2]
                qliq_pp = float(prev_prev["oil_rate_stbd"]) + float(prev_prev["water_rate_stbd"])
                
                # Decline rates (per year)
                Do = -np.log(max(qo_c, 1.0) / max(qo_p, 1.0)) / dt_y
                Dliq = -np.log(max(qliq_c, 1.0) / max(qliq_p, 1.0)) / dt_y
                
                fw_p = np.clip(qw_p / max(qliq_p, 1.0), 1e-4, 1.0 - 1e-4)
                fw_c = np.clip(qw_c / max(qliq_c, 1.0), 1e-4, 1.0 - 1e-4)
                dfw = (fw_c - fw_p) / dt_y
                
                feat = self._extract_state_features(qo_p, qw_p, Qo_p, Qw_p, u_c, qliq_pp)
                
                X_list.append(feat)
                y_Do_list.append(np.clip(Do, -0.5, 2.0))
                y_Dliq_list.append(np.clip(Dliq, -0.5, 2.0))
                y_dfw_list.append(np.clip(dfw, -1.0, 1.0))
                
        X = np.array(X_list)
        Xs = self.scaler.fit_transform(X)
        self.model_D_oil.fit(Xs, np.array(y_Do_list))
        self.model_D_liq.fit(Xs, np.array(y_Dliq_list))
        self.model_fw_rate.fit(Xs, np.array(y_dfw_list))
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
        prev_last = pre.iloc[-2] if len(pre) > 1 else last
        
        curr_qo = float(last["oil_rate_stbd"])
        curr_qw = float(last["water_rate_stbd"])
        curr_Qo = float(last["oil_cum_stb"])
        curr_Qw = float(last["water_cum_stb"])
        prev_qliq = float(prev_last["oil_rate_stbd"]) + float(prev_last["water_rate_stbd"])
        
        n_steps = len(forecast_dates)
        dt_y = dt_days / YEAR_DAYS
        
        qo_preds = np.zeros(n_steps)
        qw_preds = np.zeros(n_steps)
        qg_preds = np.zeros(n_steps)
        
        for k in range(n_steps):
            feat = self._extract_state_features(curr_qo, curr_qw, curr_Qo, curr_Qw, u_c, prev_qliq)
            feat_s = self.scaler.transform(feat.reshape(1, -1))
            
            # Predict state-conditional decline
            Do = float(self.model_D_oil.predict(feat_s)[0])
            Do = max(0.01, Do)  # Minimum physical decline floor
            
            # Predict liquid decline and fractional flow growth
            Dliq = float(self.model_D_liq.predict(feat_s)[0])
            Dliq = max(0.005, Dliq)
            
            curr_qliq = curr_qo + curr_qw
            next_qliq = max(1.0, curr_qliq * np.exp(-Dliq * dt_y))
            
            dfw = float(self.model_fw_rate.predict(feat_s)[0])
            curr_fw = curr_qw / max(curr_qliq, 1.0)
            next_fw = np.clip(curr_fw + dfw * dt_y, 0.0, 0.999)
            
            # Reconcile oil and water via liquid decomposition
            next_qo_liq = next_qliq * (1.0 - next_fw)
            next_qw_liq = next_qliq * next_fw
            
            # Direct oil decline prediction
            next_qo_dca = curr_qo * np.exp(-Do * dt_y)
            
            # Blend state-liquid decomposition with direct oil decline (50/50 balance)
            next_qo = 0.5 * next_qo_liq + 0.5 * next_qo_dca
            next_qw = next_qw_liq
            
            # Monotonic clamping safeguard for oil
            next_qo = min(curr_qo, max(0.0, next_qo))
            next_qw = max(0.0, next_qw)
            
            qo_preds[k] = next_qo
            qw_preds[k] = next_qw
            qg_preds[k] = RS_SOLUTION * next_qo
            
            # State accumulation
            curr_Qo += next_qo * dt_days
            curr_Qw += next_qw * dt_days
            prev_qliq = curr_qliq
            curr_qo = next_qo
            curr_qw = next_qw
            
        return {
            "oil_rate": qo_preds,
            "water_rate": qw_preds,
            "gas_rate": qg_preds,
        }
