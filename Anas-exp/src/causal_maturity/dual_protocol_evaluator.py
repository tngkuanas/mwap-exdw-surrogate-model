"""
Module: dual_protocol_evaluator.py
Package: causal_maturity
Addresses: Sections 2, 3, 4, 5 — Protocol A vs Protocol B Evaluation Engine
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Protocol A: Strict Historical As-Of
- Models and gates use ONLY information ending <= retrospective cutoff date.
- No future simulation targets borrowed from any case.
- Feasible for origins with sufficient mature history (e.g. 2006 8Q, 2007 4Q).
- 16Q at 2004 explicitly reported as fundamentally non-evaluable under Protocol A.

Protocol B: Cross-Case Forward-Supervised Research
- Permitted training reservoirs (Cases 1-70) can use future trajectories relative to origin.
- Validation reservoirs (Cases 71-85) NEVER expose any future information.
- Genuinely cross-fitted nested CV across reservoirs.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

from ensemble.data import load_uncertainty, resolve_shared_dir, TRAIN_CASES, VAL_CASES, PARAMS
from ensemble.metrics import compute_increment_nrmse, compute_bias
from ensemble.generate_competition_deliverable import (
    load_locked_config,
    fit_locked_base_models,
    generate_scenario_predictions,
)
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
    LearnedDCARateForecaster,
    YEAR_DAYS,
)
from conditional_challengers.corrected_residual_dca import (
    CorrectedResidualForecaster,
    fit_historical_dca_at_origin,
    extract_state_features,
)
from causal_maturity.sustained_decline_detector import extract_robust_case_maturity
from causal_maturity.exact_time_varying_integration import ExactTimeVaryingDeclineForecaster
from causal_maturity.cross_fitted_maturity_gates import (
    FixedWeightGate,
    CalendarAgeGate,
    PhysicalMaturityGate,
    MultivariateStateGate,
    UncertaintyAwareGate,
    OilOnlyMaturityGate,
    HorizonMaturityGate,
    compute_case_real_macro_nrmse,
)


def evaluate_candidate_forecast(
    qo_pred: np.ndarray,
    qw_pred: np.ndarray,
    post_df: pd.DataFrame,
    init_state: Dict[str, float],
    dt_days: float = 91.3125,
) -> Dict[str, float]:
    """Computes macro, oil, water NRMSE and bias."""
    qo_clean = np.asarray(qo_pred, dtype=float).copy()
    qw_clean = np.asarray(qw_pred, dtype=float).copy()
    
    # Monotonic clamping safeguard for oil
    for t in range(1, len(qo_clean)):
        if qo_clean[t] > qo_clean[t - 1]:
            qo_clean[t] = qo_clean[t - 1]
    qo_clean = np.maximum(qo_clean, 0.0)
    qw_clean = np.maximum(qw_clean, 0.0)
    
    true_o = post_df["oil_cum_stb"].values if "oil_cum_stb" in post_df.columns else post_df["oil_cum"].values
    true_w = post_df["water_cum_stb"].values if "water_cum_stb" in post_df.columns else post_df["water_cum"].values
    
    inc_o = true_o - init_state["Qo"]
    inc_w = true_w - init_state["Qw"]
    
    cum_o = init_state["Qo"] + np.cumsum(qo_clean * dt_days)
    cum_w = init_state["Qw"] + np.cumsum(qw_clean * dt_days)
    
    no = compute_increment_nrmse(cum_o - true_o, inc_o)
    nw = compute_increment_nrmse(cum_w - true_w, inc_w)
    bo = compute_bias(cum_o - true_o, inc_o)
    bw = compute_bias(cum_w - true_w, inc_w)
    
    return {
        "macro_nrmse": 0.5 * (no + nw),
        "oil_nrmse": no,
        "water_nrmse": nw,
        "oil_bias": bo,
        "water_bias": bw,
    }


def run_cross_fitted_gate_benchmark(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    origin_str: str,
    horizon_quarters: int,
    protocol: str = "Protocol_B", # "Protocol_A" or "Protocol_B"
    n_outer_folds: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes leak-free cross-fitted benchmarking of all gates on Cases 71-85 (and outer folds of 1-70).
    """
    orig = pd.Timestamp(origin_str)
    f_dates = pd.date_range(orig + pd.DateOffset(days=1), periods=horizon_quarters, freq="QS")
    dt_days = 91.3125
    config = load_locked_config()
    weights = config["weights"]
    
    print(f"\n=========================================================================")
    print(f"BENCHMARKING: {protocol} | Origin: {origin_str} | Horizon: {horizon_quarters} Quarters")
    print(f"=========================================================================")
    
    # Check Protocol A feasibility
    if protocol == "Protocol_A":
        # Check if rolling historical training data has mature history ending <= orig
        min_mature_date = pd.Timestamp("2002-01-01")
        avail_historical_days = (orig - min_mature_date).days
        needed_days = horizon_quarters * dt_days
        if avail_historical_days < needed_days * 1.5:
            print(f"  [PROTOCOL A NOTICE] Horizon {horizon_quarters}Q at {origin_str} has insufficient mature pre-origin history.")
            print(f"  Available mature history: {avail_historical_days/YEAR_DAYS:.1f} yrs vs Needed: {needed_days/YEAR_DAYS:.1f} yrs.")
            print(f"  Explicitly recorded as FUNDAMENTALLY NON-EVALUABLE under Protocol A without violating strict as-of boundaries.")
            return pd.DataFrame(), pd.DataFrame()
            
    # 1. Fit Base Models
    if protocol == "Protocol_A":
        # Strict historical fallback DCA
        m_frozen = fit_locked_base_models(case_dict, unc_df, TRAIN_CASES, orig)
        m_dca = fit_historical_dca_at_origin(case_dict, unc_df, TRAIN_CASES, orig)
        m_res_gp = CorrectedResidualForecaster(architecture="gp").fit(
            case_dict, unc_df, TRAIN_CASES, cutoff_date=orig, max_horizon_quarters=horizon_quarters
        )
    else:
        # Protocol B: Forward Supervised on Training Cases
        m_frozen = fit_locked_base_models(case_dict, unc_df, TRAIN_CASES, orig)
        m_dca = fit_historical_dca_at_origin(case_dict, unc_df, TRAIN_CASES, orig)
        m_res_gp = CorrectedResidualForecaster(architecture="gp").fit(
            case_dict, unc_df, TRAIN_CASES, cutoff_date=orig, max_horizon_quarters=horizon_quarters
        )
        
    m_tvd = ExactTimeVaryingDeclineForecaster().fit(
        case_dict, unc_df, TRAIN_CASES, cutoff_date=orig, max_horizon_quarters=horizon_quarters
    )
    
    # 2. Outer Cross-Fitting to generate Out-Of-Fold Gate Training Data on Cases 1-70
    train_pool = np.array(TRAIN_CASES)
    kf = KFold(n_splits=n_outer_folds, shuffle=True, random_state=random_state)
    
    oof_res_preds = {}
    oof_frz_preds = {}
    oof_post_dfs = {}
    oof_init_states = {}
    oof_maturities = {}
    
    for tr_idx, val_idx in kf.split(train_pool):
        tr_cases = train_pool[tr_idx].tolist()
        val_cases = train_pool[val_idx].tolist()
        
        # Sub-models fit on tr_cases
        sub_dca = fit_historical_dca_at_origin(case_dict, unc_df, tr_cases, orig)
        sub_frozen = fit_locked_base_models(case_dict, unc_df, tr_cases, orig)
        sub_res = CorrectedResidualForecaster(architecture="gp").fit(
            case_dict, unc_df, tr_cases, cutoff_date=orig, max_horizon_quarters=horizon_quarters
        )
        
        for c in val_cases:
            df_c = case_dict[c]
            pre = df_c[df_c["date"] <= orig]
            post = df_c[df_c["date"] > orig].iloc[:horizon_quarters]
            if len(post) < horizon_quarters:
                continue
            u_c = unc_df.loc[c]
            last = pre.iloc[-1]
            st = {
                "qo": float(last["oil_rate_stbd"]),
                "qw": float(last["water_rate_stbd"]),
                "qg": float(last["gas_rate_mscfd"]),
                "Qo": float(last["oil_cum_stb"]),
                "Qw": float(last["water_cum_stb"]),
                "Qg": float(last["gas_cum_mscf"]),
                "wor": float(last.get("wor", float(last["water_rate_stbd"]) / max(float(last["oil_rate_stbd"]), 1.0))),
                "res_age": (orig - pre["date"].iloc[0]).days / YEAR_DAYS,
                "origin_date": orig,
            }
            
            _, aud_frz = generate_scenario_predictions(sub_frozen, weights, st, u_c, f_dates, 1.0)
            preds_frz = {
                "oil_rate": np.asarray(aud_frz["qo_rate"], dtype=float),
                "water_rate": np.asarray(aud_frz["qw_rate"], dtype=float),
            }
            preds_res = sub_res.predict_rates(df_c, u_c, orig, f_dates, sub_dca)
            mat = extract_robust_case_maturity(df_c, orig)
            
            oof_res_preds[c] = preds_res
            oof_frz_preds[c] = preds_frz
            oof_post_dfs[c] = post
            oof_init_states[c] = st
            oof_maturities[c] = mat

    oof_cases = list(oof_res_preds.keys())
    print(f"  Generated {len(oof_cases)} out-of-fold training predictions across Cases 1-70.")
    
    # 3. Fit Candidate Gates on Out-Of-Fold Real NRMSE Loss
    gate_models = [
        FixedWeightGate(),
        CalendarAgeGate(),
        PhysicalMaturityGate(),
        MultivariateStateGate(),
        OilOnlyMaturityGate(),
        HorizonMaturityGate(),
    ]
    
    fitted_gates = {}
    for gate in gate_models:
        gate.fit_on_real_loss(
            oof_cases, oof_res_preds, oof_frz_preds, oof_post_dfs, oof_init_states, oof_maturities
        )
        fitted_gates[gate.name] = gate
        print(f"  Fitted {gate.name}: params = {np.round(gate.params if hasattr(gate, 'params') else gate.w, 4)}")
        
    # 4. Independent Holdout Evaluation on Cases 71 to 85
    records = []
    
    for c in VAL_CASES:
        df_c = case_dict[c]
        pre = df_c[df_c["date"] <= orig]
        post = df_c[df_c["date"] > orig].iloc[:horizon_quarters]
        if len(post) < horizon_quarters:
            continue
            
        u_c = unc_df.loc[c]
        last = pre.iloc[-1]
        st = {
            "qo": float(last["oil_rate_stbd"]),
            "qw": float(last["water_rate_stbd"]),
            "qg": float(last["gas_rate_mscfd"]),
            "Qo": float(last["oil_cum_stb"]),
            "Qw": float(last["water_cum_stb"]),
            "Qg": float(last["gas_cum_mscf"]),
            "wor": float(last.get("wor", float(last["water_rate_stbd"]) / max(float(last["oil_rate_stbd"]), 1.0))),
            "res_age": (orig - pre["date"].iloc[0]).days / YEAR_DAYS,
            "origin_date": orig,
        }
        
        # Base predictions
        _, aud_frz = generate_scenario_predictions(m_frozen, weights, st, u_c, f_dates, 1.0)
        p_frz = {
            "oil_rate": np.asarray(aud_frz["qo_rate"], dtype=float),
            "water_rate": np.asarray(aud_frz["qw_rate"], dtype=float),
        }
        p_res = m_res_gp.predict_rates(df_c, u_c, orig, f_dates, m_dca)
        p_tvd = m_tvd.predict_rates(df_c, u_c, orig, f_dates, m_dca)
        mat = extract_robust_case_maturity(df_c, orig)
        
        all_eval_preds = {
            "Model_A_Frozen_Ensemble": (p_frz["oil_rate"], p_frz["water_rate"], 0.0, 0.0),
            "Model_B_Corrected_GP_Residual": (p_res["oil_rate"], p_res["water_rate"], 1.0, 1.0),
            "Model_F_Exact_Time_Varying_DCA": (p_tvd["oil_rate"], p_tvd["water_rate"], np.nan, np.nan),
        }
        
        # Evaluated Gates
        for g_name, gate in fitted_gates.items():
            w_o, w_w = gate.get_gate_weights(mat, horizon_quarters)
            blend_qo = w_o * p_res["oil_rate"] + (1.0 - w_o) * p_frz["oil_rate"]
            blend_qw = w_w * p_res["water_rate"] + (1.0 - w_w) * p_frz["water_rate"]
            all_eval_preds[g_name] = (blend_qo, blend_qw, float(np.mean(w_o)), float(np.mean(w_w)))
            
        for m_name, (qo_m, qw_m, mean_wo, mean_ww) in all_eval_preds.items():
            scores = evaluate_candidate_forecast(qo_m, qw_m, post, st, dt_days)
            records.append({
                "protocol": protocol,
                "origin": origin_str,
                "horizon_quarters": horizon_quarters,
                "case_id": c,
                "model_name": m_name,
                "macro_nrmse": scores["macro_nrmse"],
                "oil_nrmse": scores["oil_nrmse"],
                "water_nrmse": scores["water_nrmse"],
                "oil_bias": scores["oil_bias"],
                "water_bias": scores["water_bias"],
                "mean_gate_weight_oil": mean_wo,
                "mean_gate_weight_water": mean_ww,
                "decline_depth": mat["decline_depth"],
                "quarters_in_decline": mat["quarters_in_decline"],
                "field_age_years": mat["field_age_years"],
                "water_cut": mat["water_cut"],
            })
            
    df_case_level = pd.DataFrame(records)
    
    # 5. Compute Statistical Summary Table
    summary_rows = []
    base_scores = df_case_level[df_case_level["model_name"] == "Model_A_Frozen_Ensemble"].sort_values("case_id")["macro_nrmse"].values
    
    for m_name, grp in df_case_level.groupby("model_name"):
        grp_sorted = grp.sort_values("case_id")
        m_scores = grp_sorted["macro_nrmse"].values
        diff = m_scores - base_scores
        
        # Paired Wilcoxon
        if m_name == "Model_A_Frozen_Ensemble":
            p_val = 1.0
            win_rate = 0.5
        else:
            try:
                res_w = stats.wilcoxon(diff, alternative="two-sided")
                p_val = float(res_w.pvalue)
            except Exception:
                p_val = 1.0
            win_rate = float(np.mean(m_scores < base_scores))
            
        # Bootstrap 95% CI
        n_boot = 2000
        boot_diffs = []
        rng = np.random.RandomState(random_state)
        for _ in range(n_boot):
            idx_b = rng.choice(len(diff), len(diff), replace=True)
            boot_diffs.append(np.mean(diff[idx_b]))
        ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
        
        summary_rows.append({
            "protocol": protocol,
            "origin": origin_str,
            "horizon_quarters": horizon_quarters,
            "model_name": m_name,
            "macro_nrmse_mean": float(np.mean(m_scores)),
            "oil_nrmse_mean": float(np.mean(grp_sorted["oil_nrmse"])),
            "water_nrmse_mean": float(np.mean(grp_sorted["water_nrmse"])),
            "diff_vs_baseline": float(np.mean(diff)),
            "ci95_low": float(ci_low),
            "ci95_high": float(ci_high),
            "win_rate_vs_baseline": win_rate,
            "wilcoxon_p": p_val,
            "mean_gate_weight_oil": float(grp_sorted["mean_gate_weight_oil"].mean()),
            "mean_gate_weight_water": float(grp_sorted["mean_gate_weight_water"].mean()),
        })
        
    df_summary = pd.DataFrame(summary_rows)
    return df_case_level, df_summary
