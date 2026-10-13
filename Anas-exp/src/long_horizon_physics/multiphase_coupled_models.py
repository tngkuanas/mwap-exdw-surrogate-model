"""
Module: multiphase_coupled_models.py
Package: long_horizon_physics
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

PHASE 2 — Multiphase Coupled Oil and Water Forecasting Models
Implements:
1. KovalFractionalFlowForecaster (Hypothesis A — SPE-173315-MS & Buckley-Leverett).
2. TotalLiquidBoundedWORForecaster (Hypothesis B — Gao et al. 2024 Geofluids).
3. RecoveryStateDependentForecaster (Hypothesis C — Depletion-state decline).
4. BaselinePreservingSelectiveCorrector (Hypothesis D — Selective phase correction).
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from ensemble.data import YEAR_DAYS
from ensemble.run_horizon_adaptive_stacking_experiment import RS_SOLUTION
from long_horizon_physics.literature_foundations import (
    evaluate_exact_multiphase_identities,
    koval_fractional_flow,
    logistic_water_cut_progression,
)


class KovalFractionalFlowForecaster:
    """
    Hypothesis A: Waterflood Fractional-Flow Model (SPE-173315-MS & Buckley-Leverett).
    Forecasts total liquid rate q_L(t) = q_o + q_w and water cut f_w(Q_L) jointly:
      q_L(t) = q_{L,0} * exp(-D_L * t)
      f_w(Q_L) = [f_{w,0} + K_K * xi] / [1 + (K_K - 1) * xi], where xi = (Q_L - Q_{L,0}) / V_D
      q_o(t) = q_L(t) * [1 - f_w(t)]
      q_w(t) = q_L(t) * f_w(t)
      q_g(t) = R_s * q_o(t)
    """
    def __init__(self):
        self.model_DL = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self.model_KK = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self.feat_cols = ["fault", "poro", "perm", "aq", "qL_0", "fw_0", "res_age"]
        self.fitted = False

    def fit(
        self,
        asof_case_dict: Dict[int, pd.DataFrame],
        unc_df: pd.DataFrame,
        train_cases: List[int],
        origin: pd.Timestamp,
    ):
        records = []
        for c in train_cases:
            df_c = asof_case_dict[c]
            pre = df_c[df_c["date"] <= origin]
            if len(pre) < 8:
                continue
                
            # Historical decline window (last 3-5 years)
            tail = pre.tail(min(len(pre), 16)).copy().reset_index(drop=True)
            t_fit = (tail["date"] - tail["date"].iloc[0]).dt.total_seconds() / (86400.0 * YEAR_DAYS)
            
            q_L = tail["oil_rate_stbd"].values + tail["water_rate_stbd"].values
            q_o = tail["oil_rate_stbd"].values
            q_w = tail["water_rate_stbd"].values
            
            # 1. Total liquid decline rate D_L
            log_qL = np.log(np.maximum(q_L, 1.0))
            poly_L = np.polyfit(t_fit.values, log_qL, 1)
            D_L_obs = max(0.005, -poly_L[0])
            
            # 2. Water cut evolution: estimate effective Koval slope K_K
            fw_obs = np.where(q_L > 1.0, q_w / np.maximum(q_L, 1.0), 0.0)
            wor_obs = np.where(q_o > 1.0, q_w / np.maximum(q_o, 1.0), 1e4)
            log_wor = np.log(np.maximum(wor_obs, 1e-4))
            poly_wor = np.polyfit(t_fit.values, log_wor, 1)
            lambda_wor = max(0.01, poly_wor[0])
            
            u_c = unc_df.loc[c]
            qL_0 = float(q_L[-1])
            fw_0 = float(fw_obs[-1])
            res_age = float((origin - pre["date"].iloc[0]).days / YEAR_DAYS)
            
            records.append({
                "case_id": c,
                "fault": float(u_c["Fault Transmissibility"]),
                "poro": float(u_c["Porosity Multiplier"]),
                "perm": float(u_c["Permeability Multiplier"]),
                "aq": float(u_c["Aquifer Pore Volume"]),
                "qL_0": qL_0,
                "fw_0": fw_0,
                "res_age": res_age,
                "D_L": D_L_obs,
                "lambda_wor": lambda_wor,
            })
            
        train_df = pd.DataFrame(records)
        X = train_df[self.feat_cols].values
        self.model_DL.fit(X, train_df["D_L"].values)
        self.model_KK.fit(X, train_df["lambda_wor"].values)
        self.fitted = True
        return self

    def predict_rates(
        self,
        initial_state: Dict[str, float],
        u_c: pd.Series,
        forecast_dates: pd.DatetimeIndex,
    ) -> Dict[str, np.ndarray]:
        origin = pd.Timestamp(initial_state["origin_date"])
        t_years = np.asarray((pd.DatetimeIndex(forecast_dates) - origin).total_seconds() / (86400.0 * YEAR_DAYS), dtype=float)
        
        q0_o = float(initial_state["qo"])
        q0_w = float(initial_state["qw"])
        q0_L = max(q0_o + q0_w, 1.0)
        fw_0 = float(q0_w / q0_L)
        res_age = float(initial_state["res_age"])
        
        x_in = np.array([[
            float(u_c["Fault Transmissibility"]),
            float(u_c["Porosity Multiplier"]),
            float(u_c["Permeability Multiplier"]),
            float(u_c["Aquifer Pore Volume"]),
            q0_L,
            fw_0,
            res_age,
        ]])
        
        D_L = float(np.maximum(0.005, self.model_DL.predict(x_in)[0]))
        lambda_w = float(np.maximum(0.01, self.model_KK.predict(x_in)[0]))
        
        # Total liquid trajectory
        q_L = q0_L * np.exp(-D_L * t_years)
        
        # Bounded Koval / logistic water-cut progression
        fw = logistic_water_cut_progression(t_years, fw_0=fw_0, lambda_w=lambda_w, fw_max=0.995)
        
        # Multiphase rate identities
        qo_preds = np.maximum(0.0, q_L * (1.0 - fw))
        qw_preds = np.maximum(0.0, q_L * fw)
        qg_preds = RS_SOLUTION * qo_preds
        
        return {"oil_rate": qo_preds, "water_rate": qw_preds, "gas_rate": qg_preds}


class TotalLiquidBoundedWORForecaster:
    """
    Hypothesis B: Total-Liquid & Bounded Water-Cut State Model (Gao et al. 2024 Geofluids).
    Forecasts total liquid rate q_L(t) with mild hyperbolic decline:
      q_L(t) = q_{L,0} * (1 + b_L * D_L * t)^{-1 / b_L}
    and bounded water-cut evolution via log-linear WOR progression:
      WOR(t) = WOR_0 * exp(lambda_w * t)
      f_w(t) = WOR(t) / [1 + WOR(t)]
      q_o(t) = q_L(t) / [1 + WOR(t)]
      q_w(t) = q_L(t) * WOR(t) / [1 + WOR(t)]
    """
    def __init__(self, b_liquid: float = 0.20):
        self.b_liquid = b_liquid
        self.model_DL = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self.model_lambda = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self.feat_cols = ["fault", "poro", "perm", "aq", "qo_0", "qw_0", "wor_0", "res_age"]
        self.fitted = False

    def fit(
        self,
        asof_case_dict: Dict[int, pd.DataFrame],
        unc_df: pd.DataFrame,
        train_cases: List[int],
        origin: pd.Timestamp,
    ):
        records = []
        for c in train_cases:
            df_c = asof_case_dict[c]
            pre = df_c[df_c["date"] <= origin]
            if len(pre) < 8:
                continue
                
            tail = pre.tail(min(len(pre), 16)).copy().reset_index(drop=True)
            t_fit = (tail["date"] - tail["date"].iloc[0]).dt.total_seconds() / (86400.0 * YEAR_DAYS)
            
            q_o = tail["oil_rate_stbd"].values
            q_w = tail["water_rate_stbd"].values
            q_L = q_o + q_w
            
            # Fit liquid decline
            log_qL = np.log(np.maximum(q_L, 1.0))
            poly_L = np.polyfit(t_fit.values, log_qL, 1)
            D_L_obs = max(0.005, -poly_L[0])
            
            # Fit WOR exponential growth rate lambda
            wor_obs = np.maximum(q_w / np.maximum(q_o, 1.0), 1e-4)
            log_wor = np.log(wor_obs)
            poly_wor = np.polyfit(t_fit.values, log_wor, 1)
            lambda_obs = np.clip(poly_wor[0], 0.005, 0.60)
            
            u_c = unc_df.loc[c]
            last = pre.iloc[-1]
            res_age = float((origin - pre["date"].iloc[0]).days / YEAR_DAYS)
            
            records.append({
                "case_id": c,
                "fault": float(u_c["Fault Transmissibility"]),
                "poro": float(u_c["Porosity Multiplier"]),
                "perm": float(u_c["Permeability Multiplier"]),
                "aq": float(u_c["Aquifer Pore Volume"]),
                "qo_0": float(last["oil_rate_stbd"]),
                "qw_0": float(last["water_rate_stbd"]),
                "wor_0": float(last["water_rate_stbd"] / max(last["oil_rate_stbd"], 1.0)),
                "res_age": res_age,
                "D_L": D_L_obs,
                "lambda_wor": lambda_obs,
            })
            
        train_df = pd.DataFrame(records)
        X = train_df[self.feat_cols].values
        self.model_DL.fit(X, train_df["D_L"].values)
        self.model_lambda.fit(X, train_df["lambda_wor"].values)
        self.fitted = True
        return self

    def predict_rates(
        self,
        initial_state: Dict[str, float],
        u_c: pd.Series,
        forecast_dates: pd.DatetimeIndex,
    ) -> Dict[str, np.ndarray]:
        origin = pd.Timestamp(initial_state["origin_date"])
        t_years = np.asarray((pd.DatetimeIndex(forecast_dates) - origin).total_seconds() / (86400.0 * YEAR_DAYS), dtype=float)
        
        q0_o = float(initial_state["qo"])
        q0_w = float(initial_state["qw"])
        q0_L = max(q0_o + q0_w, 1.0)
        wor_0 = max(float(initial_state.get("wor", q0_w / max(q0_o, 1.0))), 0.001)
        res_age = float(initial_state["res_age"])
        
        x_in = np.array([[
            float(u_c["Fault Transmissibility"]),
            float(u_c["Porosity Multiplier"]),
            float(u_c["Permeability Multiplier"]),
            float(u_c["Aquifer Pore Volume"]),
            q0_o,
            q0_w,
            wor_0,
            res_age,
        ]])
        
        D_L = float(np.maximum(0.005, self.model_DL.predict(x_in)[0]))
        lambda_w = float(np.clip(self.model_lambda.predict(x_in)[0], 0.005, 0.60))
        
        # Hyperbolic liquid rate decline
        b = self.b_liquid
        q_L = q0_L * ((1.0 + b * D_L * t_years) ** (-1.0 / b))
        
        # Bounded WOR progression
        wor = wor_0 * np.exp(lambda_w * t_years)
        wor = np.minimum(wor, 1000.0)  # max WOR 1000 (99.9% water cut)
        
        qo_preds = np.maximum(0.0, q_L / (1.0 + wor))
        qw_preds = np.maximum(0.0, q_L * wor / (1.0 + wor))
        qg_preds = RS_SOLUTION * qo_preds
        
        return {"oil_rate": qo_preds, "water_rate": qw_preds, "gas_rate": qg_preds}


class RecoveryStateDependentForecaster:
    """
    Hypothesis C: Cumulative Recovery-State Dependent Oil Decline.
    Models oil decline rate as a function of remaining recoverable reserves:
      q_o(t) = q_{o,0} * [1 - (Q_o(t) - Q_{o,0}) / N_rem]_+^alpha
    Coupled with bounded water-cut evolution.
    """
    def __init__(self, alpha: float = 1.2):
        self.alpha = alpha
        self.model_Nrem = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self.model_lambda = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self.feat_cols = ["fault", "poro", "perm", "aq", "qo_0", "qw_0", "Qo_0", "res_age"]
        self.fitted = False

    def fit(
        self,
        asof_case_dict: Dict[int, pd.DataFrame],
        unc_df: pd.DataFrame,
        train_cases: List[int],
        origin: pd.Timestamp,
    ):
        records = []
        for c in train_cases:
            df_c = asof_case_dict[c]
            pre = df_c[df_c["date"] <= origin]
            if len(pre) < 8:
                continue
                
            tail = pre.tail(min(len(pre), 16)).copy().reset_index(drop=True)
            t_fit = (tail["date"] - tail["date"].iloc[0]).dt.total_seconds() / (86400.0 * YEAR_DAYS)
            
            q_o = tail["oil_rate_stbd"].values
            q_w = tail["water_rate_stbd"].values
            
            # Estimate remaining drainable reserves N_rem from initial decline slope
            log_qo = np.log(np.maximum(q_o, 1.0))
            poly_o = np.polyfit(t_fit.values, log_qo, 1)
            D_obs = max(0.01, -poly_o[0])
            # EUR under exponential tail: N_rem ~= q_o / D_obs * YEAR_DAYS
            N_rem_est = (q_o[-1] / D_obs) * YEAR_DAYS
            
            wor_obs = np.maximum(q_w / np.maximum(q_o, 1.0), 1e-4)
            poly_wor = np.polyfit(t_fit.values, np.log(wor_obs), 1)
            lambda_obs = np.clip(poly_wor[0], 0.005, 0.60)
            
            u_c = unc_df.loc[c]
            last = pre.iloc[-1]
            res_age = float((origin - pre["date"].iloc[0]).days / YEAR_DAYS)
            
            records.append({
                "case_id": c,
                "fault": float(u_c["Fault Transmissibility"]),
                "poro": float(u_c["Porosity Multiplier"]),
                "perm": float(u_c["Permeability Multiplier"]),
                "aq": float(u_c["Aquifer Pore Volume"]),
                "qo_0": float(last["oil_rate_stbd"]),
                "qw_0": float(last["water_rate_stbd"]),
                "Qo_0": float(last["oil_cum_stb"]),
                "res_age": res_age,
                "N_rem": N_rem_est,
                "lambda_wor": lambda_obs,
            })
            
        train_df = pd.DataFrame(records)
        X = train_df[self.feat_cols].values
        self.model_Nrem.fit(X, train_df["N_rem"].values)
        self.model_lambda.fit(X, train_df["lambda_wor"].values)
        self.fitted = True
        return self

    def predict_rates(
        self,
        initial_state: Dict[str, float],
        u_c: pd.Series,
        forecast_dates: pd.DatetimeIndex,
        dt_days: float = 91.3125,
    ) -> Dict[str, np.ndarray]:
        n_steps = len(forecast_dates)
        origin = pd.Timestamp(initial_state["origin_date"])
        t_years = np.asarray((pd.DatetimeIndex(forecast_dates) - origin).total_seconds() / (86400.0 * YEAR_DAYS), dtype=float)
        
        q0_o = float(initial_state["qo"])
        q0_w = float(initial_state["qw"])
        Q0_o = float(initial_state["Qo"])
        wor_0 = max(float(initial_state.get("wor", q0_w / max(q0_o, 1.0))), 0.001)
        res_age = float(initial_state["res_age"])
        
        x_in = np.array([[
            float(u_c["Fault Transmissibility"]),
            float(u_c["Porosity Multiplier"]),
            float(u_c["Permeability Multiplier"]),
            float(u_c["Aquifer Pore Volume"]),
            q0_o,
            q0_w,
            Q0_o,
            res_age,
        ]])
        
        N_rem = float(np.maximum(1e5, self.model_Nrem.predict(x_in)[0]))
        lambda_w = float(np.clip(self.model_lambda.predict(x_in)[0], 0.005, 0.60))
        
        qo_preds = np.zeros(n_steps)
        qw_preds = np.zeros(n_steps)
        
        cum_incr_o = 0.0
        for k in range(n_steps):
            frac_depleted = np.clip(cum_incr_o / N_rem, 0.0, 0.99)
            qo_k = q0_o * ((1.0 - frac_depleted) ** self.alpha)
            # Physical decline floor
            qo_k = max(qo_k, 50.0)
            
            wor_k = wor_0 * np.exp(lambda_w * t_years[k])
            wor_k = min(wor_k, 1000.0)
            qw_k = qo_k * wor_k
            
            qo_preds[k] = qo_k
            qw_preds[k] = qw_k
            cum_incr_o += qo_k * dt_days
            
        qg_preds = RS_SOLUTION * qo_preds
        return {"oil_rate": qo_preds, "water_rate": qw_preds, "gas_rate": qg_preds}


class BaselinePreservingSelectiveCorrector:
    """
    Hypothesis D: Baseline-Preserving Selective Phase Corrector.
    Retains the protected frozen ensemble baseline for oil (which achieves excellent oil NRMSE ~0.0322),
    while correcting the water phase via coupled fractional flow / WOR evolution:
      q_o(t) = q_o^{baseline}(t)
      q_w(t) = WOR^{coupled}(t) * q_o^{baseline}(t)
    Also supports the dual mode: retain baseline water, correct oil.
    """
    def __init__(
        self,
        base_models: Dict[str, Any],
        weights: Dict[str, Dict[str, float]],
        correction_mode: str = "correct_water",
    ):
        self.base_models = base_models
        self.weights = weights
        self.correction_mode = correction_mode
        self.coupled_wor_model = TotalLiquidBoundedWORForecaster()

    def fit(
        self,
        asof_case_dict: Dict[int, pd.DataFrame],
        unc_df: pd.DataFrame,
        train_cases: List[int],
        origin: pd.Timestamp,
    ):
        self.coupled_wor_model.fit(asof_case_dict, unc_df, train_cases, origin)
        return self

    def predict_rates(
        self,
        initial_state: Dict[str, float],
        u_c: pd.Series,
        forecast_dates: pd.DatetimeIndex,
    ) -> Dict[str, np.ndarray]:
        # 1. Baseline Ensemble Prediction
        w_oil = self.weights["oil"]
        w_wat = self.weights["water"]
        
        preds_base = {
            m_name: self.base_models[m_name].predict_rates(initial_state, u_c, forecast_dates)
            for m_name in w_oil.keys() if m_name in self.base_models
        }
        
        qo_base = sum(w_oil[m] * preds_base[m]["oil_rate"] for m in w_oil.keys())
        qw_base = sum(w_wat[m] * preds_base[m]["water_rate"] for m in w_wat.keys())
        
        # 2. Coupled Physics Prediction
        coupled_pred = self.coupled_wor_model.predict_rates(initial_state, u_c, forecast_dates)
        qo_coup = coupled_pred["oil_rate"]
        qw_coup = coupled_pred["water_rate"]
        
        if self.correction_mode == "correct_water":
            # Protect baseline oil, replace water with WOR-coupled water
            wor_coup = np.where(qo_coup > 1.0, qw_coup / np.maximum(qo_coup, 1.0), 1.0)
            qw_final = np.maximum(0.0, qo_base * wor_coup)
            qo_final = np.maximum(0.0, qo_base)
        elif self.correction_mode == "correct_oil":
            # Protect baseline water, replace oil with coupled oil
            wor_coup = np.where(qo_coup > 1.0, qw_coup / np.maximum(qo_coup, 1.0), 1.0)
            qo_final = np.maximum(0.0, qw_base / np.maximum(wor_coup, 1e-4))
            qw_final = np.maximum(0.0, qw_base)
        elif self.correction_mode == "fully_coupled":
            qo_final = qo_coup
            qw_final = qw_coup
        else:
            raise ValueError(f"Unknown correction_mode: {self.correction_mode}")
            
        qg_final = RS_SOLUTION * qo_final
        return {"oil_rate": qo_final, "water_rate": qw_final, "gas_rate": qg_final}
