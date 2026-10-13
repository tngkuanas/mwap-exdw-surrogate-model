"""
Module: phase_recombination.py
Package: phase_specific_research
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Section 2: Phase-Specific Horizon Forecasters & Decisive Selection Candidates.
Implements the full portfolio of benchmark and challenger architectures:
  A. Frozen ensemble baseline (unchanged Blend_Phase_Specific_Simplex).
  B. ExtraTrees oil + frozen water.
  C. Frozen oil + ExtraTrees water.
  D. ExtraTrees oil + ExtraTrees water.
  E. Horizon adaptive blending.
  F. Candidate F continuous sigmoid gate.
  G. Transitioned ExtraTrees water.
  H. Constrained Long-Tail Ensemble (reallocates ExtraTrees oil to Learned Exponential DCA).
  H_Arps. Candidate H Modified Arps (reallocates ExtraTrees oil to Modified Arps with D_lim floor).
  B_Taper_Exp. ExtraTrees oil supported up to 9.75 yr, then tapering to Learned Exponential DCA; Frozen water.
  B_Taper_Arps. ExtraTrees oil supported up to 9.75 yr, then tapering to Modified Arps; Frozen water.
  Hard_WaterCut. Frozen oil + Loss-minimizing hard water cut selector.
  Hard_AgeHorizon. Frozen oil + Loss-minimizing hard age/horizon selector.
  Hard_Combined. Frozen oil + Loss-minimizing combined selector.
  Abstention_Gate. Frozen oil + Asymmetric risk abstention selector.
  Shrinkage_Blend. Frozen oil + Conservative shrinkage blend.
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

from ensemble.run_horizon_adaptive_stacking_experiment import RS_SOLUTION
from phase_specific_research.as_of_gate_selection import (
    compute_modified_arps_rate,
    predict_gated_water_rate,
)

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
    gate_params: Optional[Dict[str, Any]] = None,
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
    res_age_0 = float(init_state["res_age"])
    t_years = res_age_0 + step_indices * (DT_DAYS / 365.25)
    
    # Helper to compute Modified Arps oil rate if requested
    def get_arps_mod_oil_rate():
        x_in = np.array([[
            float(params.get("Fault Transmissibility", params.get("fault", 0.1))),
            float(params.get("Porosity Multiplier", params.get("poro", 1.0))),
            float(params.get("Permeability Multiplier", params.get("perm", 5.0))),
            float(params.get("Aquifer Pore Volume", params.get("aq", 100.0))),
            init_state["qo"],
            init_state["qw"],
            init_state["wor"],
            init_state["res_age"],
        ]])
        if hasattr(models["Learned_Hyperbolic_DCA"], "model_D"):
            D_pred = max(0.01, float(models["Learned_Hyperbolic_DCA"].model_D.predict(x_in)[0]))
        else:
            D_pred = 0.05
        # Time relative to origin
        t_rel_years = step_indices * (DT_DAYS / 365.25)
        return compute_modified_arps_rate(init_state["qo"], D_pred, b=0.35, D_lim=0.05, t_years=t_rel_years)

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
        decay_factor = np.exp(-step_indices / 10.0)
        w_et_h = w_oil["Recursive_ExtraTrees"] * decay_factor
        w_exp_h = w_oil["Learned_Exponential_DCA"] + (w_oil["Recursive_ExtraTrees"] * (1.0 - decay_factor))
        qo = (
            w_et_h * r_et["oil_rate"]
            + w_exp_h * r_exp["oil_rate"]
            + w_oil["Hybrid_Dynamic"] * r_hyb["oil_rate"]
        )
        qw = qw_frozen
        
    elif candidate_id == "Candidate_F_State_Conditioned_Gating" or candidate_id == "Candidate_F_Sigmoid_Gate":
        # Continuous sigmoid gate
        if gate_params is not None:
            qw = predict_gated_water_rate("Candidate_F_Sigmoid", gate_params, init_state, n_steps, qw_frozen, qw_et)
        else:
            fw_0 = float(init_state["wor"] / (1.0 + init_state["wor"]))
            gate_w = 1.0 / (1.0 + np.exp(-10.0 * (fw_0 - gate_threshold)))
            qw = (1.0 - gate_w) * qw_frozen + gate_w * qw_et
        qo = qo_frozen
        
    elif candidate_id == "Candidate_G_Transitioned_ExtraTrees_Water":
        sigma_trans = 1.0 / (1.0 + np.exp(-1.5 * (step_indices - h_star_water)))
        qw = (1.0 - sigma_trans) * qw_et + sigma_trans * qw_frozen
        qo = qo_frozen
        
    elif candidate_id == "Candidate_H_Constrained_Long_Tail":
        # Support-tapered ExtraTrees: smoothly reallocates ExtraTrees oil weight to Learned Exponential DCA beyond t = 14 yr
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
        
    elif candidate_id == "Candidate_H_Modified_Arps":
        # Reallocates ExtraTrees oil weight to Modified Arps decline (with D_lim = 0.05/yr) beyond t = 14 yr
        eta = 1.0 / (1.0 + np.exp((t_years - 14.0) / 2.0))
        r_arps_mod = get_arps_mod_oil_rate()
        w_et_t = w_oil["Recursive_ExtraTrees"] * eta
        w_arps_t = w_oil["Recursive_ExtraTrees"] * (1.0 - eta)
        
        qo = (
            w_et_t * r_et["oil_rate"]
            + w_oil["Learned_Exponential_DCA"] * r_exp["oil_rate"]
            + w_arps_t * r_arps_mod
            + w_oil.get("Learned_Hyperbolic_DCA", 0.0) * r_hyp["oil_rate"]
            + w_oil["Hybrid_Dynamic"] * r_hyb["oil_rate"]
        )
        qw = qw_frozen
        
    elif candidate_id == "Candidate_B_Tapered_Exp":
        # ExtraTrees oil where supported (res_age <= 9.75 yr), smoothly tapering to Learned Exponential DCA; Frozen water
        t_supp = 9.75
        tau = 1.0
        eta_supp = 1.0 / (1.0 + np.exp((t_years - t_supp) / tau))
        qo = eta_supp * r_et["oil_rate"] + (1.0 - eta_supp) * r_exp["oil_rate"]
        qw = qw_frozen
        
    elif candidate_id == "Candidate_B_Tapered_Arps":
        # ExtraTrees oil where supported (res_age <= 9.75 yr), smoothly tapering to Modified Arps; Frozen water
        t_supp = 9.75
        tau = 1.0
        eta_supp = 1.0 / (1.0 + np.exp((t_years - t_supp) / tau))
        r_arps_mod = get_arps_mod_oil_rate()
        qo = eta_supp * r_et["oil_rate"] + (1.0 - eta_supp) * r_arps_mod
        qw = qw_frozen
        
    elif candidate_id == "Candidate_Hard_WaterCut":
        # Frozen oil + Loss-minimizing hard water cut selector
        qw = predict_gated_water_rate("Hard_WaterCut", gate_params or {}, init_state, n_steps, qw_frozen, qw_et)
        qo = qo_frozen
        
    elif candidate_id == "Candidate_Hard_AgeHorizon":
        # Frozen oil + Loss-minimizing hard age/horizon selector
        qw = predict_gated_water_rate("Hard_Age_Horizon", gate_params or {}, init_state, n_steps, qw_frozen, qw_et)
        qo = qo_frozen
        
    elif candidate_id == "Candidate_Hard_Combined":
        # Frozen oil + Loss-minimizing hard combined selector
        qw = predict_gated_water_rate("Hard_Combined", gate_params or {}, init_state, n_steps, qw_frozen, qw_et)
        qo = qo_frozen
        
    elif candidate_id == "Candidate_Abstention_Gate":
        # Frozen oil + Asymmetric risk abstention selector
        qw = predict_gated_water_rate("Abstention_Gate", gate_params or {}, init_state, n_steps, qw_frozen, qw_et)
        qo = qo_frozen
        
    elif candidate_id == "Candidate_Shrinkage_Blend":
        # Frozen oil + Conservative shrinkage blend
        qw = predict_gated_water_rate("Shrinkage_Blend", gate_params or {}, init_state, n_steps, qw_frozen, qw_et)
        qo = qo_frozen
        
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


DECISIVE_CANDIDATE_IDS = [
    "Candidate_A_Frozen_Baseline",
    "Candidate_B_ExtraTreesOil_FrozenWater",
    "Candidate_C_FrozenOil_ExtraTreesWater",
    "Candidate_D_ExtraTreesBoth",
    "Candidate_H_Constrained_Long_Tail",
    "Candidate_H_Modified_Arps",
    "Candidate_B_Tapered_Exp",
    "Candidate_B_Tapered_Arps",
    "Candidate_Hard_WaterCut",
    "Candidate_Hard_AgeHorizon",
    "Candidate_Hard_Combined",
    "Candidate_F_Sigmoid_Gate",
    "Candidate_Abstention_Gate",
    "Candidate_Shrinkage_Blend",
]

ALL_CANDIDATE_IDS = DECISIVE_CANDIDATE_IDS
RELEVANT_CANDIDATE_IDS = DECISIVE_CANDIDATE_IDS
