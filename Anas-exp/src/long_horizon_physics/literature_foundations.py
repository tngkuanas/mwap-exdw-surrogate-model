"""
Module: literature_foundations.py
Package: long_horizon_physics
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

PHASE 1 — Mathematical Derivations, Reservoir Physics & Research Hypotheses
Implements:
1. Exact multiphase volume balance identities (liquid rate, water cut, WOR, solution gas).
2. Fractional flow formulations (Koval 1963, Buckley-Leverett, Gao et al. 2024).
3. Long-horizon decline models (Modified Arps with D_lim floor, Stretched Exponential Valko 2009,
   Two-Compartment Depletion, Bounded Water-Cut Logistic Growth).
4. Parameter identifiability analysis for our challenge dataset.
5. Research hypothesis matrix with predictions, assumptions, and falsification criteria.
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.special import gamma


# -----------------------------------------------------------------------------
# 1. EXACT PHYSICAL IDENTITIES VS PHYSICS-INFORMED APPROXIMATIONS
# -----------------------------------------------------------------------------

def evaluate_exact_multiphase_identities(
    qo: np.ndarray,
    qw: np.ndarray,
    qg: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Computes exact physical multiphase identities at surface standard conditions:
    1. Total liquid rate: q_L(t) = q_o(t) + q_w(t)
    2. Water cut: f_w(t) = q_w(t) / q_L(t)
    3. Water-oil ratio: WOR(t) = q_w(t) / max(q_o(t), eps)
    4. Oil cut: f_o(t) = 1 - f_w(t)
    5. Gas-oil ratio: GOR(t) = q_g(t) / max(q_o(t), eps)
    """
    qo = np.asarray(qo, dtype=float)
    qw = np.asarray(qw, dtype=float)
    q_L = qo + qw
    eps = 1e-12
    
    fw = np.where(q_L > eps, qw / np.maximum(q_L, eps), 0.0)
    fo = 1.0 - fw
    wor = np.where(qo > eps, qw / np.maximum(qo, eps), 1e6)
    
    out = {
        "liquid_rate": q_L,
        "water_cut": fw,
        "oil_cut": fo,
        "wor": wor,
    }
    if qg is not None:
        qg = np.asarray(qg, dtype=float)
        gor = np.where(qo > eps, qg / np.maximum(qo, eps), 0.0)
        out["gor"] = gor
    return out


# -----------------------------------------------------------------------------
# 2. KOVAL FRACTIONAL FLOW MODEL (SPE-173315-MS & BUCKLEY-LEVERETT)
# -----------------------------------------------------------------------------

def koval_fractional_flow(
    Q_L_produced: np.ndarray,
    K_K: float,
    V_D: float,
    Q_L_bt: float = 0.0,
    fw_initial: float = 0.0,
) -> np.ndarray:
    """
    Koval (1963) / SPE-173315-MS fractional flow model parameterized by cumulative liquid:
    f_w(Q_L) = fw_initial for Q_L <= Q_L_bt
    f_w(Q_L) = [fw_initial + K_K * xi] / [1 + (K_K - 1) * xi] for Q_L > Q_L_bt
    where xi = (Q_L - Q_L_bt) / V_D is dimensionless pore volume displaced.
    
    K_K: Koval factor = H * M_eff (effective mobility ratio * heterogeneity factor).
    V_D: Characteristic drainable pore volume (STB).
    Q_L_bt: Cumulative liquid at water breakthrough (STB).
    """
    Q_L = np.asarray(Q_L_produced, dtype=float)
    delta_Q = np.maximum(0.0, Q_L - Q_L_bt)
    xi = delta_Q / max(V_D, 1e-6)
    
    # Koval fractional flow curve
    denom = 1.0 + (K_K - 1.0) * xi
    fw_dynamic = (fw_initial + K_K * xi) / np.maximum(denom, 1e-9)
    fw = np.where(delta_Q <= 0.0, fw_initial, fw_dynamic)
    return np.clip(fw, 0.0, 0.9999)


# -----------------------------------------------------------------------------
# 3. BOUNDED LOGISTIC WATER-CUT / WOR PROGRESSION (GAO ET AL. 2024)
# -----------------------------------------------------------------------------

def logistic_water_cut_progression(
    t_years: np.ndarray,
    fw_0: float,
    lambda_w: float,
    fw_max: float = 0.995,
) -> np.ndarray:
    """
    Bounded logistic water-cut evolution:
    ln(WOR(t)) = ln(WOR_0) + lambda_w * t
    which analytically guarantees fw(t) in [0, fw_max] and asymptotic non-divergence:
    f_w(t) = fw_max / [1 + ((fw_max - fw_0) / max(fw_0, eps)) * exp(-lambda_w * t)]
    """
    t = np.asarray(t_years, dtype=float)
    fw_0 = float(np.clip(fw_0, 0.0001, fw_max - 0.0001))
    logit_ratio = (fw_max - fw_0) / fw_0
    fw = fw_max / (1.0 + logit_ratio * np.exp(-lambda_w * t))
    return np.clip(fw, 0.0, fw_max)


# -----------------------------------------------------------------------------
# 4. LONG-HORIZON DECLINE MODELS
# -----------------------------------------------------------------------------

def modified_arps_decline(
    t_years: np.ndarray,
    q0: float,
    Di: float,
    b: float = 0.35,
    D_lim: float = 0.06,
    Q0: float = 0.0,
    dt_days: float = 91.3125,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Modified Arps decline transitioning from hyperbolic to exponential when
    instantaneous decline rate D(t) drops to D_lim:
    D(t) = Di / (1 + b * Di * t)
    t_switch = (Di / D_lim - 1) / (b * Di) if Di > D_lim else 0.0
    
    Returns:
    - rate: q(t)
    - cum: Q(t) = Q0 + cumulative production
    """
    t = np.asarray(t_years, dtype=float)
    Di = max(Di, 0.001)
    b = max(b, 0.001)
    D_lim = max(D_lim, 0.001)
    
    if Di > D_lim:
        t_switch = (Di / D_lim - 1.0) / (b * Di)
        q_switch = q0 * ((1.0 + b * Di * t_switch) ** (-1.0 / b))
    else:
        t_switch = 0.0
        q_switch = q0
        
    rates = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti <= t_switch:
            rates[i] = q0 * ((1.0 + b * Di * ti) ** (-1.0 / b))
        else:
            rates[i] = q_switch * np.exp(-D_lim * (ti - t_switch))
            
    rates = np.maximum(0.0, rates)
    cums = Q0 + np.cumsum(rates * dt_days)
    return rates, cums


def stretched_exponential_decline(
    t_years: np.ndarray,
    q0: float,
    tau: float,
    n: float = 0.50,
    Q0: float = 0.0,
    dt_days: float = 91.3125,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stretched Exponential Production Decline (SEPD - Valko 2009):
    q(t) = q0 * exp(-(t / tau)^n),  0 < n <= 1
    
    Has finite analytical ultimate recovery as t -> infinity:
    Q_inf = Q0 + q0 * (tau / n) * Gamma(1 / n) * YEAR_DAYS
    
    Returns:
    - rate: q(t)
    - cum: Q(t)
    """
    t = np.asarray(t_years, dtype=float)
    tau = max(tau, 0.1)
    n = np.clip(n, 0.05, 1.0)
    
    rates = q0 * np.exp(-((np.maximum(0.0, t) / tau) ** n))
    rates = np.maximum(0.0, rates)
    cums = Q0 + np.cumsum(rates * dt_days)
    return rates, cums


def two_compartment_depletion(
    t_years: np.ndarray,
    q1_0: float,
    D1: float,
    q2_0: float,
    D2: float,
    Q0: float = 0.0,
    dt_days: float = 91.3125,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parsimonious two-compartment depletion model:
    q(t) = q1_0 * exp(-D1 * t) + q2_0 * exp(-D2 * t)
    
    Compartment 1: Near-well high-permeability flush drainage (fast decline D1).
    Compartment 2: Low-permeability boundary/aquifer support recharge (slow decline D2).
    """
    t = np.asarray(t_years, dtype=float)
    D1 = max(D1, 0.01)
    D2 = max(D2, 0.001)
    
    q1 = q1_0 * np.exp(-D1 * t)
    q2 = q2_0 * np.exp(-D2 * t)
    rates = np.maximum(0.0, q1 + q2)
    cums = Q0 + np.cumsum(rates * dt_days)
    return rates, cums


# -----------------------------------------------------------------------------
# 5. RESEARCH HYPOTHESIS MATRIX
# -----------------------------------------------------------------------------

def build_research_hypothesis_matrix() -> pd.DataFrame:
    """
    Constructs the formal research hypothesis matrix defining candidate models,
    theoretical justifications, mathematical assumptions, and empirical falsification criteria.
    """
    hypotheses = [
        {
            "hypothesis_id": "H1_Multiphase_Coupling",
            "name": "Koval / Buckley-Leverett Fractional Flow Waterflood Model",
            "governing_equations": "q_L(t) = q_o + q_w; f_w(Q_L) = K_K * xi / [1 + (K_K - 1) * xi]; q_o = q_L(1 - f_w); q_w = q_L * f_w",
            "theoretical_justification": "SPE-173315-MS & Buckley-Leverett theory. Models multiphase displacement directly as a function of voidage replacement rather than uncoupled independent regressions.",
            "mathematical_assumptions": "Lumped effective mobility ratio K_K and drainable volume V_D are identifiable from history; fractional flow increases monotonically with cumulative liquid.",
            "unmodeled_physics": "Detailed core-scale relative permeability hysteresis, capillary end effects, well-by-well pattern allocations.",
            "falsification_criteria": "Falsified if macro increment NRMSE exceeds frozen ensemble baseline on leave-reservoir-out pseudo-fields (macro NRMSE > 0.0368).",
        },
        {
            "hypothesis_id": "H2_Liquid_WOR_State",
            "name": "Total-Liquid Decline & Bounded Logistic Water-Cut Model",
            "governing_equations": "q_L(t) = q_{L,0} * exp(-D_L * t); ln(WOR(t)) = ln(WOR_0) + lambda_w * t; f_w(t) = WOR / (1 + WOR)",
            "theoretical_justification": "Gao et al. (2024) Geofluids. Liquid rate decline is physically bounded and smoother than oil; logistic WOR evolution guarantees f_w in [0, 1) and prevents unbounded water growth.",
            "mathematical_assumptions": "WOR increases log-linearly with time or cumulative liquid; liquid decline rate D_L is strictly positive and constant.",
            "unmodeled_physics": "Water shutoff interventions, pump capacity constraints, cyclic waterflood shut-in effects.",
            "falsification_criteria": "Falsified if water NRMSE degrades vs baseline or if oil prediction bias exceeds 5% of true cumulative.",
        },
        {
            "hypothesis_id": "H3_State_Dependent_Decline",
            "name": "Cumulative Recovery-State Dependent Oil Decline",
            "governing_equations": "q_o(t) = q_{o,0} * [1 - (Q_o(t) - Q_0) / N_rem]^alpha; D(t) = f(Q_o, f_w, res_age)",
            "theoretical_justification": "Reservoir drive depletion is governed by remaining recoverable reserves (N_rem) and water cut rather than arbitrary calendar time.",
            "mathematical_assumptions": "N_rem is proportional to Porosity * HCPV and estimable without full volumetric PVT data.",
            "unmodeled_physics": "Solution gas liberation below bubble point, rock compressibility compaction drive.",
            "falsification_criteria": "Falsified if fitted N_rem parameter exhibits equifinality or produces premature economic well death (q_o < 50 STB/d) before 20 years.",
        },
        {
            "hypothesis_id": "H4_Baseline_Preserving_Correction",
            "name": "Selective Phase Correction (Protected Oil Baseline + Coupled Water)",
            "governing_equations": "q_o(t) = q_o^{baseline}(t); q_w(t) = WOR^{coupled}(t) * q_o^{baseline}(t)",
            "theoretical_justification": "The frozen ensemble oil forecast is highly accurate (oil NRMSE ~0.0322), whereas water models exhibit significant uncoupled uncertainty. Correcting only the water phase preserves oil excellence.",
            "mathematical_assumptions": "Baseline oil rate and coupled water-cut evolution are statistically and physically compatible.",
            "unmodeled_physics": "Backpressure coupling between high water-cut lifting and reservoir inflow performance.",
            "falsification_criteria": "Falsified if coupled water prediction yields higher water NRMSE than the frozen baseline water simplex model.",
        },
        {
            "hypothesis_id": "H5_Modified_Arps_Terminal_Floor",
            "name": "Modified Arps Decline with Physical Terminal Floor (D_lim)",
            "governing_equations": "q_o(t) = q_0(1 + b Di t)^{-1/b} for D(t) > D_lim; q_o(t) = q_{switch} exp(-D_lim(t - t_{switch})) for D(t) <= D_lim",
            "theoretical_justification": "Standard hyperbolic decline (b=0.35) under-declines at late times (infinite EUR). Imposing a terminal decline floor D_lim in [0.05, 0.08]/yr enforces finite recovery.",
            "mathematical_assumptions": "Late-life decline rate stabilizes at a constant boundary-dominated exponential rate D_lim.",
            "unmodeled_physics": "Late-stage tertiary recovery or pressure maintenance changes.",
            "falsification_criteria": "Falsified if retrospective validation shows terminal floor causes premature decline on medium horizons (12Q-20Q).",
        },
        {
            "hypothesis_id": "H6_Stretched_Exponential",
            "name": "Stretched Exponential Production Decline (SEPD - Valko 2009)",
            "governing_equations": "q_o(t) = q_0 * exp(-(t / tau)^n); Q_inf = Q_0 + q_0 * (tau / n) * Gamma(1 / n) * dt",
            "theoretical_justification": "SEPD accommodates transient fracture/matrix diffusion early while guaranteeing strictly finite EUR as t -> infinity without ad-hoc threshold switches.",
            "mathematical_assumptions": "Heterogeneous relaxation dynamics follow a Kohlrausch stretched exponential distribution.",
            "unmodeled_physics": "Discontinuous boundary effects or rapid water breakthrough shock waves.",
            "falsification_criteria": "Falsified if SEPD fits poorly on retrospective historical decline or over-declines early.",
        },
    ]
    return pd.DataFrame(hypotheses)
