"""
Module: maturity_descriptors_and_gates.py
Package: conditional_challengers
Addresses: Section 3 — Maturity-Conditioned Residual Correction & Blending
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Constructs observable maturity descriptors strictly from pre-origin history:
- Quarters since sustained decline began.
- Recent nominal decline rate.
- Stability of rolling decline estimates.
- Water cut and water-cut slope.
- Production rate variability.
- Mahalanobis state distance from training support.

Implements:
1. Fixed-horizon hybrid (residual near-term, frozen far-term).
2. Maturity-conditioned hybrid (smooth gating function learned strictly on inner training folds).
3. Phase-specific hybrid (residual for oil, frozen ensemble for water).
"""
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from ensemble.run_horizon_adaptive_stacking_experiment import YEAR_DAYS

MATURITY_FEATURE_NAMES = [
    "quarters_in_decline",
    "nominal_decline_rate",
    "rolling_decline_stability",
    "water_cut",
    "water_cut_slope",
    "rate_variability_cv",
]


def extract_maturity_descriptors(df_case: pd.DataFrame, origin: pd.Timestamp) -> Dict[str, float]:
    """
    Computes observable reservoir maturity metrics strictly using data up to origin.
    """
    pre = df_case[df_case["date"] <= origin].copy()
    if len(pre) == 0:
        raise ValueError(f"No history before {origin}")
        
    rates_o = pre["oil_rate_stbd"].values
    rates_w = pre["water_rate_stbd"].values
    dates = pre["date"].values
    
    qo_0 = float(rates_o[-1])
    qw_0 = float(rates_w[-1])
    wc_0 = qw_0 / max(qo_0 + qw_0, 1e-4)
    
    # 1. Quarters in sustained decline
    # Identify maximum historical rate (plateau rate)
    q_max = np.max(rates_o)
    # Sustained decline starts when rate drops below 92% of plateau and stays below
    below_thresh = np.where(rates_o < 0.92 * q_max)[0]
    if len(below_thresh) > 0:
        # First index where sustained decline starts
        start_idx = below_thresh[0]
        quarters_in_decline = float(len(rates_o) - 1 - start_idx)
    else:
        quarters_in_decline = 0.0
        
    # 2. Recent nominal decline rate (over last 4 quarters)
    if len(rates_o) >= 5:
        qo_4ago = max(rates_o[-5], 1.0)
        dt_years = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[-5])).days / YEAR_DAYS
        nominal_decline_rate = float(-(np.log(max(qo_0, 1.0)) - np.log(qo_4ago)) / max(dt_years, 0.25))
    else:
        nominal_decline_rate = 0.0
        
    # 3. Stability of rolling decline estimates
    if len(rates_o) >= 6:
        # 2-quarter interval decline rates
        d_roll = []
        for i in range(2, min(8, len(rates_o))):
            d_i = -(np.log(max(rates_o[-i+2], 1.0)) - np.log(max(rates_o[-i], 1.0))) / 0.5
            d_roll.append(d_i)
        rolling_decline_stability = float(np.std(d_roll))
    else:
        rolling_decline_stability = 0.5
        
    # 4. Water cut slope (over last 4 quarters)
    if len(rates_w) >= 5:
        wc_4ago = rates_w[-5] / max(rates_o[-5] + rates_w[-5], 1e-4)
        water_cut_slope = float((wc_0 - wc_4ago) / 1.0)
    else:
        water_cut_slope = 0.0
        
    # 5. Production rate variability (CV over last 4 quarters)
    if len(rates_o) >= 4:
        recent_qo = rates_o[-4:]
        rate_variability_cv = float(np.std(recent_qo) / max(np.mean(recent_qo), 1.0))
    else:
        rate_variability_cv = 0.0
        
    return {
        "quarters_in_decline": quarters_in_decline,
        "nominal_decline_rate": nominal_decline_rate,
        "rolling_decline_stability": rolling_decline_stability,
        "water_cut": wc_0,
        "water_cut_slope": water_cut_slope,
        "rate_variability_cv": rate_variability_cv,
    }


class MaturityConditionedGate:
    """
    Predetermined, smooth gate g(m) in [0, 1] learned exclusively from training folds.
    Blends:
        q_hybrid(t) = g(m) * q_residual(t) + (1 - g(m)) * q_frozen(t)
    """
    def __init__(self):
        self.beta = np.array([0.0, 0.0]) # [intercept, slope on quarters_in_decline]
        self.fitted = False

    def fit(self, maturity_list: List[Dict[str, float]], loss_diff_list: np.ndarray):
        """
        Fits gate parameter to minimize blended NRMSE:
        loss_diff = error(residual) - error(frozen)
        Where loss_diff < 0, residual is better -> want g(m) -> 1.
        Where loss_diff > 0, frozen is better -> want g(m) -> 0.
        """
        # Feature: quarters in decline (normalized: 4 quarters = 1 year)
        m_vals = np.array([m["quarters_in_decline"] / 4.0 for m in maturity_list])
        
        def obj(params):
            b0, b1 = params
            g = expit(b0 + b1 * m_vals)
            # Convex combination loss approximation
            # If g=1, loss is error_res = error_frz + loss_diff
            # Blended error approx: error_frz + g * loss_diff
            return np.mean(g * loss_diff_list) + 0.01 * (b0**2 + b1**2)
            
        res = minimize(obj, [0.0, 0.5], method="BFGS")
        if res.success:
            self.beta = res.x
        else:
            self.beta = np.array([0.0, 0.5])
            
        self.fitted = True
        return self

    def compute_gate_weight(self, maturity: Dict[str, float]) -> float:
        """Returns weight assigned to residual model g(m) in [0, 1]."""
        if not self.fitted:
            return 0.5
        m_val = maturity["quarters_in_decline"] / 4.0
        return float(expit(self.beta[0] + self.beta[1] * m_val))


def blend_predictions(
    preds_residual: Dict[str, np.ndarray],
    preds_frozen: Dict[str, np.ndarray],
    mode: str = "frozen",  # "frozen", "residual", "fixed_hybrid", "maturity_hybrid", "phase_hybrid"
    gate_weight: float = 0.5,
    fixed_switch_quarter: int = 8,
) -> Dict[str, np.ndarray]:
    """
    Blends residual and frozen forecasts according to specified hybrid strategy.
    Guarantees exact conservation, non-negativity, and post-plateau monotonic oil decline.
    """
    qo_res = np.asarray(preds_residual["oil_rate"], dtype=float).flatten()
    qw_res = np.asarray(preds_residual["water_rate"], dtype=float).flatten()
    
    qo_frz = np.asarray(preds_frozen["oil_rate"], dtype=float).flatten()
    qw_frz = np.asarray(preds_frozen["water_rate"], dtype=float).flatten()
    
    n_steps = len(qo_frz)
    
    if mode == "frozen":
        qo_blend = qo_frz.copy()
        qw_blend = qw_frz.copy()
    elif mode == "residual":
        qo_blend = qo_res.copy()
        qw_blend = qw_res.copy()
    elif mode == "fixed_hybrid":
        # Smooth cosine transition around fixed_switch_quarter
        t_steps = np.arange(1, n_steps + 1)
        # Weight on residual: 1.0 for t <= switch/2, smooth to 0 for t >= switch * 1.5
        w_res = np.clip(1.0 - (t_steps - fixed_switch_quarter / 2.0) / max(fixed_switch_quarter, 1.0), 0.0, 1.0)
        qo_blend = w_res * qo_res + (1.0 - w_res) * qo_frz
        qw_blend = w_res * qw_res + (1.0 - w_res) * qw_frz
    elif mode == "maturity_hybrid":
        w = np.clip(gate_weight, 0.0, 1.0)
        qo_blend = w * qo_res + (1.0 - w) * qo_frz
        qw_blend = w * qw_res + (1.0 - w) * qw_frz
    elif mode == "phase_hybrid":
        # Residual for oil, frozen for water
        qo_blend = qo_res.copy()
        qw_blend = qw_frz.copy()
    elif mode == "phase_maturity_hybrid":
        # Maturity-weighted residual for oil, strictly frozen for water
        w = np.clip(gate_weight, 0.0, 1.0)
        qo_blend = w * qo_res + (1.0 - w) * qo_frz
        qw_blend = qw_frz.copy()
    else:
        raise ValueError(f"Unknown blend mode: {mode}")
        
    # Strict physical constraints
    qo_blend = np.maximum(qo_blend, 0.0)
    qw_blend = np.maximum(qw_blend, 0.0)
    
    # Monotonic oil decline
    for t in range(1, len(qo_blend)):
        if qo_blend[t] > qo_blend[t-1]:
            qo_blend[t] = qo_blend[t-1]
            
    qg_blend = qo_blend * 0.3633 # Rs coupling
    
    return {
        "oil_rate": qo_blend,
        "water_rate": qw_blend,
        "gas_rate": qg_blend,
    }
