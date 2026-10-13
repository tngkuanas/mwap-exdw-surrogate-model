"""
Module: phase_recombination.py
Package: phase_specific_research
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Section 2: Phase-Specific Horizon Forecasters.
Implements the 8 controlled candidate models:
  A. Frozen ensemble baseline (unchanged Blend_Phase_Specific_Simplex).
  B. ExtraTrees oil + frozen water.
  C. Frozen oil + ExtraTrees water.
  D. ExtraTrees oil + ExtraTrees water.
  E. Phase-specific weights varying smoothly with forecast horizon.
  F. Phase-specific weights conditioned on observable reservoir state.
  G. Near-term ExtraTrees water forecasts transitioning to frozen water after horizon h*.
  H. Constrained Long-Tail Ensemble (smooth support-tapered ExtraTrees).
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

from ensemble.run_horizon_adaptive_stacking_experiment import RS_SOLUTION

DT_DAYS = 91.3125


def predict_candidate_rates(
    candidate_id: str,
    models: Dict[str, Any],
    locked_weights: Dict[str, Dict[str, float]],
    init_state: Dict[str, Any],
    params: pd.Series,
    forecast_dates: pd.DatetimeIndex,
    h_star_water: int = 8,
    gate_state: Optional[Dict[str, float]] = None,
    base_rates: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    gate_threshold: float = 0.35,
) -> Dict[str, np.ndarray]:
    """
    Generates quarterly oil, water, and gas rates for any candidate architecture
    satisfying strict Protocol A historical anchoring and gas coupling.
    """
    n_steps = len(forecast_dates)
    w_oil = locked_weights["oil"]
    w_wat = locked_weights["water"]
    
    # 1. Base model predictions (use precomputed if provided)
    if base_rates is not None:
        r_et = base_rates["Recursive_ExtraTrees"]
        r_exp = base_rates["Learned_Exponential_DCA"]
        r_hyp = base_rates["Learned_Hyperbolic_DCA"]
        r_hyb = base_rates["Hybrid_Dynamic"]
    else:
        r_et = models["Recursive_ExtraTrees"].predict_rates(init_state, params, forecast_dates)
        r_exp = models["Learned_Exponential_DCA"].predict_rates(init_state, params, forecast_dates)
        r_hyp = models["Learned_Hyperbolic_DCA"].predict_rates(init_state, params, forecast_dates)
        r_hyb = models["Hybrid_Dynamic"].predict_rates(init_state, params, forecast_dates)
    
    # Standard frozen blends
    qo_frozen = (
        w_oil["Recursive_ExtraTrees"] * r_et["oil_rate"]
        + w_oil["Learned_Exponential_DCA"] * r_exp["oil_rate"]
        + w_oil["Learned_Hyperbolic_DCA"] * r_hyp["oil_rate"]
        + w_oil["Hybrid_Dynamic"] * r_hyb["oil_rate"]
    )
    qw_frozen = (
        w_wat["Recursive_ExtraTrees"] * r_et["water_rate"]
        + w_wat["Learned_Exponential_DCA"] * r_exp["water_rate"]
        + w_wat["Learned_Hyperbolic_DCA"] * r_hyp["water_rate"]
        + w_wat["Hybrid_Dynamic"] * r_hyb["water_rate"]
    )
    
    qo_et = r_et["oil_rate"]
    qw_et = r_et["water_rate"]
    
    step_indices = np.arange(1, n_steps + 1)
    
    if candidate_id == "Candidate_A_Frozen_Baseline":
        qo = qo_frozen
        qw = qw_frozen
        
    elif candidate_id == "Candidate_B_ExtraTreesOil_FrozenWater":
        qo = qo_et
        qw = qw_frozen
        
    elif candidate_id == "Candidate_C_FrozenOil_ExtraTreesWater":
        qo = qo_frozen
        qw = qw_et
        
    elif candidate_id == "Candidate_D_ExtraTreesBoth":
        qo = qo_et
        qw = qw_et
        
    elif candidate_id == "Candidate_E_Horizon_Adaptive_Blending":
        # ExtraTrees oil weight smoothly decays with horizon step h
        # tau_h = 10 quarters (~2.5 years): ExtraTrees oil weight halves by quarter 7
        decay_factor = np.exp(-step_indices / 10.0)
        w_et_h = w_oil["Recursive_ExtraTrees"] * decay_factor
        w_exp_h = w_oil["Learned_Exponential_DCA"] + (w_oil["Recursive_ExtraTrees"] * (1.0 - decay_factor))
        
        qo = (
            w_et_h * r_et["oil_rate"]
            + w_exp_h * r_exp["oil_rate"]
            + w_oil["Hybrid_Dynamic"] * r_hyb["oil_rate"]
        )
        # Water maintains ExtraTrees weight in near-mid term
        qw = qw_frozen
        
    elif candidate_id == "Candidate_F_State_Conditioned_Gating":
        # Conditioned on observable water cut and decline rate at origin
        fw_0 = float(init_state["wor"] / (1.0 + init_state["wor"]))
        # High water cut (> gate_threshold) favors ExtraTrees water; low water cut favors Frozen water
        gate_w = 1.0 / (1.0 + np.exp(-10.0 * (fw_0 - gate_threshold)))
        qw = (1.0 - gate_w) * qw_frozen + gate_w * qw_et
        qo = qo_frozen
        
    elif candidate_id == "Candidate_G_Transitioned_ExtraTrees_Water":
        # Near-term ExtraTrees water smoothly transitioning to frozen water after h* quarters
        # Sigmoid transition centered at h_star_water
        sigma_trans = 1.0 / (1.0 + np.exp(-1.5 * (step_indices - h_star_water)))
        qw = (1.0 - sigma_trans) * qw_et + sigma_trans * qw_frozen
        qo = qo_frozen
        
    elif candidate_id == "Candidate_H_Constrained_Long_Tail":
        # Support-tapered ExtraTrees: smoothly reallocates ExtraTrees oil to exponential DCA beyond t = 14 yr
        res_age_0 = float(init_state["res_age"])
        t_years = res_age_0 + step_indices * (DT_DAYS / 365.25)
        eta = 1.0 / (1.0 + np.exp((t_years - 14.0) / 2.0))
        
        w_et_t = w_oil["Recursive_ExtraTrees"] * eta
        w_exp_t = w_oil["Learned_Exponential_DCA"] + w_oil["Recursive_ExtraTrees"] * (1.0 - eta)
        w_hyp = w_oil.get("Learned_Hyperbolic_DCA", 0.0)
        
        qo = (
            w_et_t * r_et["oil_rate"]
            + w_exp_t * r_exp["oil_rate"]
            + w_hyp * r_hyp["oil_rate"]
            + w_oil["Hybrid_Dynamic"] * r_hyb["oil_rate"]
        )
        qw = qw_frozen
        
    else:
        raise ValueError(f"Unknown candidate_id: {candidate_id}")
        
    # Non-negativity
    qo = np.maximum(0.0, qo)
    qw = np.maximum(0.0, qw)
    qg = RS_SOLUTION * qo
    
    # Cumulative integration preserving anchor
    cum_o = np.maximum.accumulate(init_state["Qo"] + np.cumsum(qo * DT_DAYS))
    cum_w = np.maximum.accumulate(init_state["Qw"] + np.cumsum(qw * DT_DAYS))
    cum_g = np.maximum.accumulate(init_state["Qg"] + RS_SOLUTION * (cum_o - init_state["Qo"]))
    
    return {
        "candidate_id": candidate_id,
        "oil_rate": qo,
        "water_rate": qw,
        "gas_rate": qg,
        "cum_oil": cum_o,
        "cum_water": cum_w,
        "cum_gas": cum_g,
    }


ALL_CANDIDATE_IDS = [
    "Candidate_A_Frozen_Baseline",
    "Candidate_B_ExtraTreesOil_FrozenWater",
    "Candidate_C_FrozenOil_ExtraTreesWater",
    "Candidate_D_ExtraTreesBoth",
    "Candidate_E_Horizon_Adaptive_Blending",
    "Candidate_F_State_Conditioned_Gating",
    "Candidate_G_Transitioned_ExtraTrees_Water",
    "Candidate_H_Constrained_Long_Tail",
]

RELEVANT_CANDIDATE_IDS = [
    "Candidate_A_Frozen_Baseline",
    "Candidate_B_ExtraTreesOil_FrozenWater",
    "Candidate_C_FrozenOil_ExtraTreesWater",
    "Candidate_D_ExtraTreesBoth",
    "Candidate_F_State_Conditioned_Gating",
    "Candidate_H_Constrained_Long_Tail",
]

