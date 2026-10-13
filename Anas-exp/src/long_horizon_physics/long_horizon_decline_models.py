"""
Module: long_horizon_decline_models.py
Package: long_horizon_physics
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

PHASE 3 — Long-Horizon Oil Decline and Depletion Challengers
Implements:
1. ModifiedArpsDeclineForecaster (Hyperbolic decline transitioning to exponential floor D_lim).
2. StretchedExponentialForecaster (SEPD - Valko 2009, finite asymptotic recovery).
3. TwoCompartmentDepletionForecaster (Parsimonious two-tank near-well + far-field recharge).
4. ConstrainedTailHybridForecaster (Short-horizon ExtraTrees transitioning to physically bounded tail).
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from ensemble.data import YEAR_DAYS
from ensemble.run_horizon_adaptive_stacking_experiment import RS_SOLUTION
from long_horizon_physics.literature_foundations import (
    modified_arps_decline,
    stretched_exponential_decline,
    two_compartment_depletion,
)


class ModifiedArpsDeclineForecaster:
    """
    Modified Arps decline with physically motivated terminal-decline behavior:
      D(t) = Di / (1 + b * Di * t)
      Switch to exponential at D(t) = D_lim (default 0.06/year).
    Eliminates the infinite-EUR paradox of unconstrained hyperbolic extrapolation.
    """
    def __init__(self, b: float = 0.35, D_lim: float = 0.06):
        self.b = b
        self.D_lim = D_lim
        self.model_Di = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self.model_water_growth = RidgeCV(alphas=np.logspace(-3, 3, 10))
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
            
            fit_rates = tail["oil_rate_stbd"].values
            fit_water = tail["water_rate_stbd"].values
            
            log_qo = np.log(np.maximum(fit_rates, 1.0))
            poly_o = np.polyfit(t_fit.values, log_qo, 1)
            D_obs = max(0.01, -poly_o[0])
            
            poly_w = np.polyfit(t_fit.values, fit_water, 1)
            w_growth_obs = poly_w[0]
            
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
                "Di": D_obs,
                "w_growth": w_growth_obs,
            })
            
        train_df = pd.DataFrame(records)
        X = train_df[self.feat_cols].values
        self.model_Di.fit(X, train_df["Di"].values)
        self.model_water_growth.fit(X, train_df["w_growth"].values)
        self.fitted = True
        return self

    def predict_rates(
        self,
        initial_state: Dict[str, float],
        u_c: pd.Series,
        forecast_dates: pd.DatetimeIndex,
        dt_days: float = 91.3125,
    ) -> Dict[str, np.ndarray]:
        origin = pd.Timestamp(initial_state["origin_date"])
        t_years = np.asarray((pd.DatetimeIndex(forecast_dates) - origin).total_seconds() / (86400.0 * YEAR_DAYS), dtype=float)
        
        q0_o = float(initial_state["qo"])
        q0_w = float(initial_state["qw"])
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
        
        Di = float(np.maximum(0.01, self.model_Di.predict(x_in)[0]))
        w_growth = float(self.model_water_growth.predict(x_in)[0])
        
        qo_preds, _ = modified_arps_decline(
            t_years=t_years,
            q0=max(q0_o, 1.0),
            Di=Di,
            b=self.b,
            D_lim=self.D_lim,
            dt_days=dt_days,
        )
        
        # Exponential water relaxation
        Dw = 0.20
        qw_preds = np.maximum(0.0, q0_w + (w_growth / Dw) * (1.0 - np.exp(-Dw * t_years)))
        qg_preds = RS_SOLUTION * qo_preds
        
        return {"oil_rate": qo_preds, "water_rate": qw_preds, "gas_rate": qg_preds}


class StretchedExponentialForecaster:
    """
    Stretched Exponential Production Decline (SEPD - Valko 2009):
      q_o(t) = q_0 * exp(-(t / tau)^n),  0 < n <= 1
    Accommodates early transient curvature while guaranteeing strictly finite EUR.
    """
    def __init__(self, n: float = 0.55):
        self.n = n
        self.model_tau = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self.model_water_growth = RidgeCV(alphas=np.logspace(-3, 3, 10))
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
            
            fit_rates = tail["oil_rate_stbd"].values
            fit_water = tail["water_rate_stbd"].values
            
            # Linearized SEPD: ln(-ln(q / q0)) = n * ln(t) - n * ln(tau)
            log_qo = np.log(np.maximum(fit_rates, 1.0))
            poly_o = np.polyfit(t_fit.values, log_qo, 1)
            D_obs = max(0.01, -poly_o[0])
            # Approximate tau from equivalent initial exponential slope: tau ~= (n / D_obs)
            tau_obs = max(0.5, self.n / D_obs)
            
            poly_w = np.polyfit(t_fit.values, fit_water, 1)
            w_growth_obs = poly_w[0]
            
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
                "tau": tau_obs,
                "w_growth": w_growth_obs,
            })
            
        train_df = pd.DataFrame(records)
        X = train_df[self.feat_cols].values
        self.model_tau.fit(X, train_df["tau"].values)
        self.model_water_growth.fit(X, train_df["w_growth"].values)
        self.fitted = True
        return self

    def predict_rates(
        self,
        initial_state: Dict[str, float],
        u_c: pd.Series,
        forecast_dates: pd.DatetimeIndex,
        dt_days: float = 91.3125,
    ) -> Dict[str, np.ndarray]:
        origin = pd.Timestamp(initial_state["origin_date"])
        t_years = np.asarray((pd.DatetimeIndex(forecast_dates) - origin).total_seconds() / (86400.0 * YEAR_DAYS), dtype=float)
        
        q0_o = float(initial_state["qo"])
        q0_w = float(initial_state["qw"])
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
        
        tau = float(np.maximum(0.5, self.model_tau.predict(x_in)[0]))
        w_growth = float(self.model_water_growth.predict(x_in)[0])
        
        qo_preds, _ = stretched_exponential_decline(
            t_years=t_years,
            q0=max(q0_o, 1.0),
            tau=tau,
            n=self.n,
            dt_days=dt_days,
        )
        
        Dw = 0.20
        qw_preds = np.maximum(0.0, q0_w + (w_growth / Dw) * (1.0 - np.exp(-Dw * t_years)))
        qg_preds = RS_SOLUTION * qo_preds
        
        return {"oil_rate": qo_preds, "water_rate": qw_preds, "gas_rate": qg_preds}


class TwoCompartmentDepletionForecaster:
    """
    Parsimonious Two-Compartment Depletion Forecaster:
      q_o(t) = q_{1,0} * exp(-D1 * t) + q_{2,0} * exp(-D2 * t)
    Decomposes drainage into rapid near-well flush (D1 ~ 0.35/yr) and sustained far-field aquifer support (D2 ~ 0.05/yr).
    """
    def __init__(self, frac_q1: float = 0.60):
        self.frac_q1 = frac_q1
        self.model_D1 = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self.model_D2 = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self.model_water_growth = RidgeCV(alphas=np.logspace(-3, 3, 10))
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
            
            fit_rates = tail["oil_rate_stbd"].values
            fit_water = tail["water_rate_stbd"].values
            
            log_qo = np.log(np.maximum(fit_rates, 1.0))
            poly_o = np.polyfit(t_fit.values, log_qo, 1)
            D_base = max(0.01, -poly_o[0])
            
            D1_est = max(0.15, 1.5 * D_base)
            D2_est = max(0.02, 0.4 * D_base)
            
            poly_w = np.polyfit(t_fit.values, fit_water, 1)
            w_growth_obs = poly_w[0]
            
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
                "D1": D1_est,
                "D2": D2_est,
                "w_growth": w_growth_obs,
            })
            
        train_df = pd.DataFrame(records)
        X = train_df[self.feat_cols].values
        self.model_D1.fit(X, train_df["D1"].values)
        self.model_D2.fit(X, train_df["D2"].values)
        self.model_water_growth.fit(X, train_df["w_growth"].values)
        self.fitted = True
        return self

    def predict_rates(
        self,
        initial_state: Dict[str, float],
        u_c: pd.Series,
        forecast_dates: pd.DatetimeIndex,
        dt_days: float = 91.3125,
    ) -> Dict[str, np.ndarray]:
        origin = pd.Timestamp(initial_state["origin_date"])
        t_years = np.asarray((pd.DatetimeIndex(forecast_dates) - origin).total_seconds() / (86400.0 * YEAR_DAYS), dtype=float)
        
        q0_o = float(initial_state["qo"])
        q0_w = float(initial_state["qw"])
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
        
        D1 = float(np.maximum(0.10, self.model_D1.predict(x_in)[0]))
        D2 = float(np.maximum(0.01, self.model_D2.predict(x_in)[0]))
        w_growth = float(self.model_water_growth.predict(x_in)[0])
        
        q1_0 = max(q0_o, 1.0) * self.frac_q1
        q2_0 = max(q0_o, 1.0) * (1.0 - self.frac_q1)
        
        qo_preds, _ = two_compartment_depletion(
            t_years=t_years,
            q1_0=q1_0,
            D1=D1,
            q2_0=q2_0,
            D2=D2,
            dt_days=dt_days,
        )
        
        Dw = 0.20
        qw_preds = np.maximum(0.0, q0_w + (w_growth / Dw) * (1.0 - np.exp(-Dw * t_years)))
        qg_preds = RS_SOLUTION * qo_preds
        
        return {"oil_rate": qo_preds, "water_rate": qw_preds, "gas_rate": qg_preds}


class ConstrainedTailHybridForecaster:
    """
    Data-driven near-horizon transition model (Recursive ExtraTrees for k <= switch_quarters)
    switching smoothly at handoff to an analytically constrained tail (Modified Arps with D_lim floor or SEPD).
    """
    def __init__(
        self,
        near_model: Any,
        switch_quarters: int = 12,
        tail_type: str = "modified_arps",
        D_lim: float = 0.06,
    ):
        self.near_model = near_model
        self.switch_quarters = switch_quarters
        self.tail_type = tail_type
        self.D_lim = D_lim
        self.tail_model = ModifiedArpsDeclineForecaster(b=0.35, D_lim=D_lim)

    def fit(
        self,
        asof_case_dict: Dict[int, pd.DataFrame],
        unc_df: pd.DataFrame,
        train_cases: List[int],
        origin: pd.Timestamp,
    ):
        self.tail_model.fit(asof_case_dict, unc_df, train_cases, origin)
        return self

    def predict_rates(
        self,
        initial_state: Dict[str, float],
        u_c: pd.Series,
        forecast_dates: pd.DatetimeIndex,
        dt_days: float = 91.3125,
    ) -> Dict[str, np.ndarray]:
        n_steps = len(forecast_dates)
        switch_k = min(self.switch_quarters, n_steps)
        
        # 1. Near dates: predict with data-driven ExtraTrees
        near_pred = self.near_model.predict_rates(initial_state, u_c, forecast_dates[:switch_k])
        
        qo_out = np.zeros(n_steps)
        qw_out = np.zeros(n_steps)
        
        qo_out[:switch_k] = near_pred["oil_rate"]
        qw_out[:switch_k] = near_pred["water_rate"]
        
        if n_steps > switch_k:
            near_dates = forecast_dates[:switch_k]
            handoff_date = near_dates[-1]
            far_dates = forecast_dates[switch_k:]
            
            handoff_qo = float(qo_out[switch_k - 1])
            handoff_qw = float(qw_out[switch_k - 1])
            handoff_wor = handoff_qw / max(handoff_qo, 1.0)
            handoff_age = float(initial_state["res_age"]) + switch_k * 0.25
            
            handoff_state = {
                "qo": handoff_qo,
                "qw": handoff_qw,
                "wor": handoff_wor,
                "res_age": handoff_age,
                "origin_date": handoff_date,
            }
            
            tail_pred = self.tail_model.predict_rates(handoff_state, u_c, far_dates, dt_days)
            qo_out[switch_k:] = tail_pred["oil_rate"]
            qw_out[switch_k:] = tail_pred["water_rate"]
            
        qg_out = RS_SOLUTION * qo_out
        return {"oil_rate": qo_out, "water_rate": qw_out, "gas_rate": qg_out}
