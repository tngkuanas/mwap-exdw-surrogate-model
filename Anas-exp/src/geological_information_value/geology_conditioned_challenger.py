"""
Module: geology_conditioned_challenger.py
Package: geological_information_value
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Addresses: Section 7 — Scientifically Justified Forecasting Challenger
Implements: Geology-Conditioned Physical Decline Forecaster (GCDF)
Derived directly from the empirical diagnosis:
1. The baseline ensemble's DCA model has near-zero coefficients on Perm and AqPV (~1e-4, ~1e-6),
   causing complete insensitivity to aquifer pressure maintenance and permeability contrast.
2. In reservoir physics, active aquifer support sustains reservoir pressure, reducing oil
   decline rate D while imparting hyperbolic curvature (b > 0) and accelerating fractional watercut.
3. GCDF couples dimensionless aquifer support Pi_aq = (AqPV / Poro) * sqrt(Perm) to decline
   curvature b(u), effective decline D(u), and monotonic watercut evolution f_w(t).
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from ensemble.data import YEAR_DAYS, PARAMS
from ensemble.run_horizon_adaptive_stacking_experiment import RS_SOLUTION
from geological_information_value.parameter_bounds import (
    PARAM_NAMES,
    PRIOR_MEANS,
)


class GeologyConditionedPhysicalForecaster:
    """
    Forecaster that explicitly conditions decline curvature and fractional flow
    on subsurface geological parameters (Perm, AqPV, Poro, Ft).
    """
    def __init__(
        self,
        base_b: float = 0.20,
        max_b: float = 0.70,
        gamma_aq: float = 0.35,
        base_lambda_w: float = 0.15,
    ):
        self.base_b = base_b
        self.max_b = max_b
        self.gamma_aq = gamma_aq
        self.base_lambda_w = base_lambda_w
        self.model_D_base = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self.fitted = False

    def _compute_aquifer_support_number(self, u_c: pd.Series) -> float:
        """
        Dimensionless Aquifer Support Number Pi_aq:
        Pi_aq = (AqPV / (Poro * 150)) * sqrt(Perm / 1.0) * Ft^0.25
        """
        ft = max(float(u_c["Fault Transmissibility"]), 0.01)
        poro = max(float(u_c["Porosity Multiplier"]), 0.50)
        perm = max(float(u_c["Permeability Multiplier"]), 0.05)
        aq = max(float(u_c["Aquifer Pore Volume"]), 5.0)
        
        pi_aq = (aq / (poro * 150.0)) * np.sqrt(perm / 1.0) * (ft ** 0.25)
        return float(np.clip(pi_aq, 0.05, 5.0))

    def fit(
        self,
        case_dict: Dict[int, pd.DataFrame],
        unc_df: pd.DataFrame,
        train_cases: List[int],
        cutoff_date: pd.Timestamp,
    ):
        """Fits historical baseline decline rate from pre-cutoff mature history."""
        records = []
        for c in train_cases:
            df_c = case_dict[c]
            pre = df_c[df_c["date"] <= cutoff_date]
            if len(pre) < 8:
                continue
                
            decline_hist = pre[pre["date"] >= "2002-01-01"]
            if len(decline_hist) < 4:
                decline_hist = pre.iloc[-4:]
                
            t_fit_y = (decline_hist["date"] - decline_hist["date"].iloc[0]).dt.total_seconds() / (86400.0 * YEAR_DAYS)
            fit_rates = decline_hist["oil_rate_stbd"].values
            log_qo = np.log(np.maximum(fit_rates, 1.0))
            poly_o = np.polyfit(t_fit_y, log_qo, 1)
            D_obs = max(0.01, -poly_o[0])
            
            u_c = unc_df.loc[c]
            records.append({
                "qo_0": float(pre.iloc[-1]["oil_rate_stbd"]),
                "qw_0": float(pre.iloc[-1]["water_rate_stbd"]),
                "wor_0": float(pre.iloc[-1]["wor"]),
                "poro": float(u_c["Porosity Multiplier"]),
                "D_obs": D_obs,
            })
            
        fit_df = pd.DataFrame(records)
        feat_cols = ["qo_0", "qw_0", "wor_0", "poro"]
        self.feat_cols = feat_cols
        self.model_D_base.fit(fit_df[feat_cols].values, fit_df["D_obs"].values)
        self.fitted = True
        return self

    def predict_rates(
        self,
        initial_state: Dict[str, float],
        u_c: pd.Series,
        forecast_dates: pd.DatetimeIndex,
    ) -> Dict[str, np.ndarray]:
        """
        Predicts oil and water production rates with explicit geological conditioning.
        """
        assert self.fitted, "Model must be fitted before predict_rates."
        t_years = (pd.DatetimeIndex(forecast_dates) - pd.Timestamp(initial_state["origin_date"])).total_seconds() / (86400.0 * YEAR_DAYS)
        
        # 1. Base decline rate from historical state & porosity
        x_in = np.array([[
            initial_state["qo"],
            initial_state["qw"],
            initial_state["wor"],
            float(u_c["Porosity Multiplier"]),
        ]])
        D_unadjusted = max(0.02, float(self.model_D_base.predict(x_in)[0]))
        
        # 2. Physics-grounded geological adjustment via Aquifer Support Number Pi_aq
        pi_aq = self._compute_aquifer_support_number(u_c)
        
        # Aquifer support sustains pressure -> lower effective decline rate D
        D_eff = D_unadjusted / (1.0 + self.gamma_aq * (pi_aq - 1.0))
        D_eff = float(np.clip(D_eff, 0.01, 0.40))
        
        # Aquifer support imparts hyperbolic curvature b > 0
        b_eff = self.base_b + (self.max_b - self.base_b) * (pi_aq / (1.0 + pi_aq))
        b_eff = float(np.clip(b_eff, 0.10, 0.85))
        
        q0_o = max(initial_state["qo"], 1.0)
        q0_w = max(initial_state["qw"], 1.0)
        q0_liq = q0_o + q0_w
        wor_0 = max(initial_state["wor"], 0.01)
        
        # 3. Total Liquid Rate Decline
        # Total liquid declines more stably than oil alone
        q_liq_t = q0_liq * ((1.0 + b_eff * D_eff * t_years) ** (-1.0 / b_eff))
        
        # 4. Geological Fractional Flow Progression
        # Aquifer strength accelerates watercut progression
        lambda_w = self.base_lambda_w * (1.0 + 0.25 * (pi_aq - 1.0))
        wor_t = wor_0 * np.exp(lambda_w * t_years)
        fw_t = wor_t / (1.0 + wor_t)
        
        # Partition rates
        qo_t = q_liq_t * (1.0 - fw_t)
        qw_t = q_liq_t * fw_t
        
        # Enforce monotonicity safeguard on oil rate
        qo_t = np.maximum(qo_t, 0.0)
        qw_t = np.maximum(qw_t, 0.0)
        qg_t = RS_SOLUTION * qo_t
        
        return {
            "oil_rate": qo_t,
            "water_rate": qw_t,
            "gas_rate": qg_t,
            "pi_aq_used": pi_aq,
            "D_eff_used": D_eff,
            "b_eff_used": b_eff,
        }
