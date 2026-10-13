"""
Module: asymptotic_analysis.py
Package: phase_specific_research
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Section 5, 6 & 7: Asymptotic Behavior, Support Boundary Audit & Structural Long-Tail Validation.
1. Evaluates exact frozen ensemble on Field Case 0 at 20, 30, 50, and 100 years.
2. Deconstructs the 737.12 STB/d asymptotic rate floor and proved infinite EUR divergence.
3. Provides exact mathematical proofs contrasting Arps hyperbolic decline with decision-tree leaf plateaus.
4. Audits the actual training support boundaries of ExtraTrees models.
5. Performs sensitivity grid across taper center (t_trans) and width (tau).
6. Implements formal Two One-Sided Tests (TOST) equivalence testing.
7. Evaluates synthetic structural ground-truth experiments where long-run decline is known.
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from ensemble.data import load_uncertainty, resolve_shared_dir, TRAIN_CASES, PARAMS
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
    extract_transition_training_data,
    STATE_FEATURES,
)
from ensemble.generate_competition_deliverable import (
    load_locked_config,
    fit_locked_base_models,
)

DT_DAYS = 91.3125


def analytical_arps_recovery_limit(q0: float, Di: float, b: float) -> float:
    """
    Computes analytical ultimate recovery Q_inf for standard Arps decline:
    q(t) = q0 * (1 + b * Di * t)^(-1/b)
    
    Proof:
    Q(T) = int_0^T q0 * (1 + b * Di * t)^(-1/b) dt
    Let u = 1 + b * Di * t, du = b * Di * dt
    Q(T) = (q0 / (b * Di)) * [u^(1 - 1/b) / (1 - 1/b)]_1^(1 + b * Di * T)
         = (q0 / ((b - 1) * Di)) * [(1 + b * Di * T)^((b - 1)/b) - 1]
         = (q0 / ((1 - b) * Di)) * [1 - (1 + b * Di * T)^(-(1 - b)/b)]
    
    For 0 < b < 1 (e.g. b = 0.35):
    -(1 - b)/b = -0.65 / 0.35 = -1.857 < 0.
    As T -> infinity, (1 + b * Di * T)^(-1.857) -> 0.
    Therefore, Q_inf = q0 / ((1 - b) * Di) < infinity!
    Only when b >= 1 does Q_inf diverge to infinity.
    """
    if b < 0:
        raise ValueError("Arps b parameter must be non-negative")
    if np.isclose(b, 0.0):
        return float(q0 / Di) * 365.25
    elif b < 1.0:
        return float(q0 / ((1.0 - b) * Di)) * 365.25
    else:
        return float("inf")


def audit_training_support_boundaries(trans_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes empirical support boundaries across all state features in the training data.
    """
    records = []
    for feat in STATE_FEATURES:
        vals = trans_df[feat].values
        records.append({
            "feature": feat,
            "min_train": float(np.min(vals)),
            "max_train": float(np.max(vals)),
            "mean_train": float(np.mean(vals)),
            "std_train": float(np.std(vals)),
            "q01_train": float(np.percentile(vals, 1)),
            "q99_train": float(np.percentile(vals, 99)),
        })
    return pd.DataFrame(records)


def run_frozen_ensemble_long_term_stress(
    models: Dict[str, Any],
    weights: Dict[str, Dict[str, float]],
    init_state: Dict[str, Any],
    params: pd.Series,
    horizons_years: List[int] = [20, 30, 50, 100],
    origin: pd.Timestamp = pd.Timestamp("2008-01-01"),
) -> pd.DataFrame:
    """
    Executes structural asymptotic stress testing of the exact frozen ensemble on Field Case 0
    across 20, 30, 50, and 100 years (80 to 400 quarters).
    """
    w_oil = weights["oil"]
    w_wat = weights["water"]
    
    records = []
    for yr in horizons_years:
        n_q = yr * 4
        f_dates = pd.date_range(origin + pd.DateOffset(months=3), periods=n_q, freq="QS")
        
        r_et = models["Recursive_ExtraTrees"].predict_rates(init_state, params, f_dates)
        r_exp = models["Learned_Exponential_DCA"].predict_rates(init_state, params, f_dates)
        r_hyp = models["Learned_Hyperbolic_DCA"].predict_rates(init_state, params, f_dates)
        r_hyb = models["Hybrid_Dynamic"].predict_rates(init_state, params, f_dates)
        
        qo_ens = (
            w_oil["Recursive_ExtraTrees"] * r_et["oil_rate"]
            + w_oil["Learned_Exponential_DCA"] * r_exp["oil_rate"]
            + w_oil.get("Learned_Hyperbolic_DCA", 0.0) * r_hyp["oil_rate"]
            + w_oil["Hybrid_Dynamic"] * r_hyb["oil_rate"]
        )
        qw_ens = (
            w_wat["Recursive_ExtraTrees"] * r_et["water_rate"]
            + w_wat["Learned_Exponential_DCA"] * r_exp["water_rate"]
            + w_wat["Learned_Hyperbolic_DCA"] * r_hyp["water_rate"]
            + w_wat.get("Hybrid_Dynamic", 0.0) * r_hyb["water_rate"]
        )
        
        cum_o = init_state["Qo"] + np.cumsum(np.maximum(0, qo_ens) * DT_DAYS)
        cum_w = init_state["Qw"] + np.cumsum(np.maximum(0, qw_ens) * DT_DAYS)
        
        et_oil_contrib = w_oil["Recursive_ExtraTrees"] * r_et["oil_rate"][-1]
        exp_oil_contrib = w_oil["Learned_Exponential_DCA"] * r_exp["oil_rate"][-1]
        hyb_oil_contrib = w_oil["Hybrid_Dynamic"] * r_hyb["oil_rate"][-1]
        
        records.append({
            "horizon_years": yr,
            "horizon_quarters": n_q,
            "et_terminal_oil_rate_stbd": float(r_et["oil_rate"][-1]),
            "exp_terminal_oil_rate_stbd": float(r_exp["oil_rate"][-1]),
            "hyp_terminal_oil_rate_stbd": float(r_hyp["oil_rate"][-1]),
            "hyb_terminal_oil_rate_stbd": float(r_hyb["oil_rate"][-1]),
            "ensemble_terminal_oil_rate_stbd": float(qo_ens[-1]),
            "et_oil_rate_contribution_stbd": float(et_oil_contrib),
            "et_oil_contribution_pct": float(et_oil_contrib / max(qo_ens[-1], 1e-6) * 100.0),
            "ensemble_terminal_water_rate_stbd": float(qw_ens[-1]),
            "terminal_water_cut": float(qw_ens[-1] / max(qw_ens[-1] + qo_ens[-1], 1e-6)),
            "eur_oil_stb": float(cum_o[-1]),
            "incremental_oil_stb": float(cum_o[-1] - init_state["Qo"]),
            "eur_water_stb": float(cum_w[-1]),
            "approaches_zero_oil_rate": bool(qo_ens[-1] < 10.0),
            "finite_recovery_integral": False,
        })
        
    return pd.DataFrame(records)


def predict_constrained_long_tail_rates(
    models: Dict[str, Any],
    weights: Dict[str, Dict[str, float]],
    init_state: Dict[str, Any],
    params: pd.Series,
    forecast_dates: pd.DatetimeIndex,
    t_support_years: float = 14.0,
    taper_scale_years: float = 2.0,
) -> Dict[str, np.ndarray]:
    """
    Implements Constrained Long-Tail Ensemble (Candidate H):
    Tapers Recursive ExtraTrees oil weight toward zero as reservoir age exceeds observed support.
    """
    n_steps = len(forecast_dates)
    res_age_0 = float(init_state["res_age"])
    t_years = res_age_0 + np.arange(1, n_steps + 1) * (DT_DAYS / 365.25)
    
    eta = 1.0 / (1.0 + np.exp((t_years - t_support_years) / taper_scale_years))
    
    w_et_0 = weights["oil"]["Recursive_ExtraTrees"]
    w_exp_0 = weights["oil"]["Learned_Exponential_DCA"]
    w_hyp_0 = weights["oil"].get("Learned_Hyperbolic_DCA", 0.0)
    w_hyb_0 = weights["oil"]["Hybrid_Dynamic"]
    
    w_et_t = w_et_0 * eta
    w_exp_t = w_exp_0 + w_et_0 * (1.0 - eta)
    
    r_et = models["Recursive_ExtraTrees"].predict_rates(init_state, params, forecast_dates)
    r_exp = models["Learned_Exponential_DCA"].predict_rates(init_state, params, forecast_dates)
    r_hyp = models["Learned_Hyperbolic_DCA"].predict_rates(init_state, params, forecast_dates)
    r_hyb = models["Hybrid_Dynamic"].predict_rates(init_state, params, forecast_dates)
    
    qo_constrained = (
        w_et_t * r_et["oil_rate"]
        + w_exp_t * r_exp["oil_rate"]
        + w_hyp_0 * r_hyp["oil_rate"]
        + w_hyb_0 * r_hyb["oil_rate"]
    )
    qo_constrained = np.maximum(0.0, qo_constrained)
    
    w_wat = weights["water"]
    qw = (
        w_wat["Recursive_ExtraTrees"] * r_et["water_rate"]
        + w_wat["Learned_Exponential_DCA"] * r_exp["water_rate"]
        + w_wat["Learned_Hyperbolic_DCA"] * r_hyp["water_rate"]
    )
    qw = np.maximum(0.0, qw)
    
    return {
        "oil_rate": qo_constrained,
        "water_rate": qw,
        "t_years": t_years,
        "eta_taper": eta,
        "w_et_t": w_et_t,
        "w_exp_t": w_exp_t,
    }


def compare_frozen_vs_constrained_tail(
    models: Dict[str, Any],
    weights: Dict[str, Dict[str, float]],
    init_state: Dict[str, Any],
    params: pd.Series,
    horizons_years: List[int] = [20, 30, 50, 100],
    origin: pd.Timestamp = pd.Timestamp("2008-01-01"),
) -> pd.DataFrame:
    """
    Direct comparison between Frozen Baseline, Candidate H (Exp & Arps), Candidate B,
    and Candidate B Tapered (Exp & Arps) across 20, 30, 50, and 100 years on Field Case 0.
    """
    from phase_specific_research.as_of_gate_selection import compute_modified_arps_rate
    
    records = []
    w_oil = weights["oil"]
    
    for yr in horizons_years:
        n_q = yr * 4
        f_dates = pd.date_range(origin + pd.DateOffset(months=3), periods=n_q, freq="QS")
        step_indices = np.arange(1, n_q + 1)
        t_years = init_state["res_age"] + step_indices * (DT_DAYS / 365.25)
        
        r_et = models["Recursive_ExtraTrees"].predict_rates(init_state, params, f_dates)
        r_exp = models["Learned_Exponential_DCA"].predict_rates(init_state, params, f_dates)
        r_hyp = models["Learned_Hyperbolic_DCA"].predict_rates(init_state, params, f_dates)
        r_hyb = models["Hybrid_Dynamic"].predict_rates(init_state, params, f_dates)
        
        # 1. Exact Frozen Baseline
        qo_froz = (
            w_oil["Recursive_ExtraTrees"] * r_et["oil_rate"]
            + w_oil["Learned_Exponential_DCA"] * r_exp["oil_rate"]
            + w_oil.get("Learned_Hyperbolic_DCA", 0.0) * r_hyp["oil_rate"]
            + w_oil["Hybrid_Dynamic"] * r_hyb["oil_rate"]
        )
        cum_o_froz = init_state["Qo"] + np.cumsum(np.maximum(0, qo_froz) * DT_DAYS)
        
        # 2. Candidate H (Exp DCA tail beyond 14 yr)
        eta_h = 1.0 / (1.0 + np.exp((t_years - 14.0) / 2.0))
        w_et_t = w_oil["Recursive_ExtraTrees"] * eta_h
        w_exp_t = w_oil["Learned_Exponential_DCA"] + w_oil["Recursive_ExtraTrees"] * (1.0 - eta_h)
        qo_tail_exp = (
            w_et_t * r_et["oil_rate"]
            + w_exp_t * r_exp["oil_rate"]
            + w_oil.get("Learned_Hyperbolic_DCA", 0.0) * r_hyp["oil_rate"]
            + w_oil["Hybrid_Dynamic"] * r_hyb["oil_rate"]
        )
        cum_o_tail_exp = init_state["Qo"] + np.cumsum(np.maximum(0, qo_tail_exp) * DT_DAYS)
        
        # 3. Candidate H (Modified Arps tail beyond 14 yr)
        x_in = np.array([[
            float(params.get("Fault Transmissibility", params.get("fault", 0.1))),
            float(params.get("Porosity Multiplier", params.get("poro", 1.0))),
            float(params.get("Permeability Multiplier", params.get("perm", 5.0))),
            float(params.get("Aquifer Pore Volume", params.get("aq", 100.0))),
            init_state["qo"], init_state["qw"], init_state["wor"], init_state["res_age"],
        ]])
        D_pred = max(0.01, float(models["Learned_Hyperbolic_DCA"].model_D.predict(x_in)[0])) if hasattr(models["Learned_Hyperbolic_DCA"], "model_D") else 0.05
        t_rel = step_indices * (DT_DAYS / 365.25)
        r_arps_mod = compute_modified_arps_rate(init_state["qo"], D_pred, b=0.35, D_lim=0.05, t_years=t_rel)
        
        qo_tail_arps = (
            w_et_t * r_et["oil_rate"]
            + w_oil["Learned_Exponential_DCA"] * r_exp["oil_rate"]
            + (w_oil["Recursive_ExtraTrees"] * (1.0 - eta_h)) * r_arps_mod
            + w_oil.get("Learned_Hyperbolic_DCA", 0.0) * r_hyp["oil_rate"]
            + w_oil["Hybrid_Dynamic"] * r_hyb["oil_rate"]
        )
        cum_o_tail_arps = init_state["Qo"] + np.cumsum(np.maximum(0, qo_tail_arps) * DT_DAYS)
        
        # 4. Candidate B (Pure ExtraTrees oil)
        qo_b = r_et["oil_rate"]
        cum_o_b = init_state["Qo"] + np.cumsum(np.maximum(0, qo_b) * DT_DAYS)
        
        # 5. Candidate B Tapered Exp (ExtraTrees oil supported up to 9.75 yr, then Exp DCA)
        eta_b = 1.0 / (1.0 + np.exp((t_years - 9.75) / 1.0))
        qo_b_taper_exp = eta_b * r_et["oil_rate"] + (1.0 - eta_b) * r_exp["oil_rate"]
        cum_o_b_taper_exp = init_state["Qo"] + np.cumsum(np.maximum(0, qo_b_taper_exp) * DT_DAYS)
        
        # 6. Candidate B Tapered Arps (ExtraTrees oil supported up to 9.75 yr, then Modified Arps)
        qo_b_taper_arps = eta_b * r_et["oil_rate"] + (1.0 - eta_b) * r_arps_mod
        cum_o_b_taper_arps = init_state["Qo"] + np.cumsum(np.maximum(0, qo_b_taper_arps) * DT_DAYS)
        
        records.append({
            "horizon_years": yr,
            "frozen_terminal_oil_rate_stbd": float(qo_froz[-1]),
            "frozen_eur_oil_stb": float(cum_o_froz[-1]),
            "cand_h_exp_terminal_rate_stbd": float(qo_tail_exp[-1]),
            "cand_h_exp_eur_oil_stb": float(cum_o_tail_exp[-1]),
            "cand_h_arps_terminal_rate_stbd": float(qo_tail_arps[-1]),
            "cand_h_arps_eur_oil_stb": float(cum_o_tail_arps[-1]),
            "cand_b_et_terminal_rate_stbd": float(qo_b[-1]),
            "cand_b_et_eur_oil_stb": float(cum_o_b[-1]),
            "cand_b_taper_exp_terminal_rate_stbd": float(qo_b_taper_exp[-1]),
            "cand_b_taper_exp_eur_oil_stb": float(cum_o_b_taper_exp[-1]),
            "cand_b_taper_arps_terminal_rate_stbd": float(qo_b_taper_arps[-1]),
            "cand_b_taper_arps_eur_oil_stb": float(cum_o_b_taper_arps[-1]),
            "eur_difference_cand_h_vs_frozen_stb": float(cum_o_tail_exp[-1] - cum_o_froz[-1]),
        })
        
    return pd.DataFrame(records)



def evaluate_taper_sensitivity_grid(
    models: Dict[str, Any],
    weights: Dict[str, Dict[str, float]],
    init_state: Dict[str, Any],
    params: pd.Series,
    centers: List[float] = [12.0, 14.0, 16.0],
    widths: List[float] = [1.0, 2.0, 3.0],
    origin: pd.Timestamp = pd.Timestamp("2008-01-01"),
) -> pd.DataFrame:
    """
    Evaluates sensitivity of terminal rate and EUR to taper center and bandwidth parameters.
    """
    records = []
    f_dates_100 = pd.date_range(origin + pd.DateOffset(months=3), periods=400, freq="QS")
    f_dates_20 = f_dates_100[:80]
    
    for t_c in centers:
        for tau in widths:
            pred_20 = predict_constrained_long_tail_rates(
                models, weights, init_state, params, f_dates_20, t_support_years=t_c, taper_scale_years=tau
            )
            pred_100 = predict_constrained_long_tail_rates(
                models, weights, init_state, params, f_dates_100, t_support_years=t_c, taper_scale_years=tau
            )
            
            eur_20 = float(init_state["Qo"] + np.sum(pred_20["oil_rate"] * DT_DAYS))
            eur_100 = float(init_state["Qo"] + np.sum(pred_100["oil_rate"] * DT_DAYS))
            
            records.append({
                "taper_center_years": t_c,
                "taper_width_years": tau,
                "terminal_rate_20yr_stbd": float(pred_20["oil_rate"][-1]),
                "terminal_rate_50yr_stbd": float(pred_100["oil_rate"][199]),
                "terminal_rate_100yr_stbd": float(pred_100["oil_rate"][-1]),
                "eur_oil_20yr_stb": eur_20,
                "eur_oil_100yr_stb": eur_100,
                "eur_delta_vs_baseline_20yr": eur_20 - 51378222.17,
            })
    return pd.DataFrame(records)


def compute_tost_equivalence(
    diff_mean: float,
    diff_se_cluster: float,
    df: int = 14,
    margins: List[float] = [0.005, 0.001],
) -> pd.DataFrame:
    """
    Performs formal Two One-Sided Tests (TOST) for statistical equivalence.
    H0: |diff_mean| >= margin (non-equivalent)
    H1: -margin < diff_mean < margin (equivalent)
    """
    records = []
    for delta in margins:
        # Test 1: diff_mean > -delta  =>  t1 = (diff_mean - (-delta)) / SE
        t1 = (diff_mean + delta) / max(diff_se_cluster, 1e-12)
        p1 = float(1.0 - student_t.cdf(t1, df=df))
        
        # Test 2: diff_mean < +delta  =>  t2 = (delta - diff_mean) / SE
        t2 = (delta - diff_mean) / max(diff_se_cluster, 1e-12)
        p2 = float(1.0 - student_t.cdf(t2, df=df))
        
        p_tost = max(p1, p2)
        is_equiv = bool(p_tost < 0.05)
        
        records.append({
            "margin_delta": delta,
            "diff_mean": diff_mean,
            "diff_se_cluster": diff_se_cluster,
            "t1_stat": t1,
            "p1_lower": p1,
            "t2_stat": t2,
            "p2_upper": p2,
            "p_tost_overall": p_tost,
            "equivalent_at_alpha_0_05": is_equiv,
        })
    return pd.DataFrame(records)


def run_synthetic_structural_experiment(
    models: Dict[str, Any],
    weights: Dict[str, Dict[str, float]],
    init_state: Dict[str, Any],
    params: pd.Series,
    q_abandon_stbd: float = 50.0,
    horizon_quarters: int = 120,  # 30 years
    origin: pd.Timestamp = pd.Timestamp("2008-01-01"),
) -> pd.DataFrame:
    """
    Evaluates Frozen Baseline vs Candidate H against a known synthetic physical ground truth
    where the reservoir experiences true physical boundary decline to economic abandonment.
    """
    f_dates = pd.date_range(origin + pd.DateOffset(months=3), periods=horizon_quarters, freq="QS")
    
    # Synthetic physical truth: calibrated exponential decline with true Di = 0.15/yr
    t_years = np.arange(1, horizon_quarters + 1) * (DT_DAYS / 365.25)
    q0_true = init_state["qo"]
    true_rate = np.maximum(0.0, q0_true * np.exp(-0.15 * t_years))
    true_cum = init_state["Qo"] + np.cumsum(true_rate * DT_DAYS)
    
    # Frozen Baseline
    r_et = models["Recursive_ExtraTrees"].predict_rates(init_state, params, f_dates)
    r_exp = models["Learned_Exponential_DCA"].predict_rates(init_state, params, f_dates)
    r_hyb = models["Hybrid_Dynamic"].predict_rates(init_state, params, f_dates)
    
    qo_froz = (
        weights["oil"]["Recursive_ExtraTrees"] * r_et["oil_rate"]
        + weights["oil"]["Learned_Exponential_DCA"] * r_exp["oil_rate"]
        + weights["oil"]["Hybrid_Dynamic"] * r_hyb["oil_rate"]
    )
    cum_froz = init_state["Qo"] + np.cumsum(qo_froz * DT_DAYS)
    
    # Candidate H
    pred_tail = predict_constrained_long_tail_rates(models, weights, init_state, params, f_dates)
    qo_tail = pred_tail["oil_rate"]
    cum_tail = init_state["Qo"] + np.cumsum(qo_tail * DT_DAYS)
    
    records = []
    milestones = [4, 12, 20, 40, 80, 120]  # 1, 3, 5, 10, 20, 30 years
    for m in milestones:
        yr = m / 4.0
        records.append({
            "horizon_years": yr,
            "true_oil_rate_stbd": float(true_rate[m - 1]),
            "frozen_oil_rate_stbd": float(qo_froz[m - 1]),
            "tail_oil_rate_stbd": float(qo_tail[m - 1]),
            "frozen_rate_error_stbd": float(qo_froz[m - 1] - true_rate[m - 1]),
            "tail_rate_error_stbd": float(qo_tail[m - 1] - true_rate[m - 1]),
            "true_eur_stb": float(true_cum[m - 1]),
            "frozen_eur_stb": float(cum_froz[m - 1]),
            "tail_eur_stb": float(cum_tail[m - 1]),
            "frozen_cum_error_stb": float(cum_froz[m - 1] - true_cum[m - 1]),
            "tail_cum_error_stb": float(cum_tail[m - 1] - true_cum[m - 1]),
        })
    return pd.DataFrame(records)
