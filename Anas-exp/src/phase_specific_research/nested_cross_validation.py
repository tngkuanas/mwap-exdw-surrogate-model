"""
Module: nested_cross_validation.py
Package: phase_specific_research
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Strict Leave-One-Reservoir-Out Cross-Validation Engine:
1. Strict outer-fold reservoir isolation: held_out_case is completely excluded from train_pool.
2. Strict as-of gate training:
   Gate parameters and selection rules are trained exclusively on historical forecast windows
   whose outcomes ended on or before the outer evaluation origin (t_end <= t_eval_origin).
3. Direct forecasting loss optimization:
   Gate parameters directly minimize cumulative-increment macro NRMSE on inner hindcasts.
4. Cluster-robust statistical inference (M=15 clusters, df=14) with Holm-Bonferroni corrections.
5. Dual-protocol evaluation (Protocol A vs Protocol B) on secondary confirmation Cases 71-85.
"""
from typing import Dict, List, Tuple, Any, Optional, Callable
import time
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from ensemble.data import (
    load_uncertainty,
    resolve_shared_dir,
    TRAIN_CASES,
    VAL_CASES,
    YEAR_DAYS,
)
from ensemble.generate_competition_deliverable import load_locked_config
from forecast_advancement.protocol_framework import fit_base_models_under_protocol
from geological_information_value.authoritative_scorer import score_cumulative_forecast
from geological_information_value.parameter_bounds import PARAM_NAMES, create_parameter_series
from phase_specific_research.as_of_gate_selection import (
    get_valid_inner_asof_windows,
    precompute_inner_hindcast_database,
    fit_asof_loss_minimizing_selectors,
)
from phase_specific_research.phase_recombination import (
    predict_candidate_rates,
    DECISIVE_CANDIDATE_IDS,
    RELEVANT_CANDIDATE_IDS,
)
from phase_specific_research.selection_rule_analysis import extract_observable_state_features

EVAL_PSEUDO_FIELDS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
DT_DAYS = 91.3125

EVAL_GRID = [
    ("2002-01-01", 4),
    ("2002-01-01", 8),
    ("2002-01-01", 12),
    ("2002-01-01", 16),
    ("2002-01-01", 20),
    ("2003-01-01", 4),
    ("2003-01-01", 8),
    ("2003-01-01", 12),
    ("2003-01-01", 16),
    ("2003-01-01", 20),
    ("2004-01-01", 4),
    ("2004-01-01", 8),
    ("2004-01-01", 12),
    ("2004-01-01", 16),
    ("2005-01-01", 4),
    ("2005-01-01", 8),
    ("2005-01-01", 12),
]


def tune_inner_gate_threshold(
    case_dict: Dict[int, pd.DataFrame],
    train_pool: List[int],
    origins: List[str] = ["2002-01-01", "2003-01-01", "2004-01-01", "2005-01-01"],
    candidate_thresholds: List[float] = [0.25, 0.30, 0.35, 0.40, 0.45],
) -> float:
    """
    Backward-compatible threshold tuner for unit testing.
    """
    train_eval_cases = [c for c in EVAL_PSEUDO_FIELDS if c in train_pool]
    if len(train_eval_cases) == 0:
        return 0.35
    fw_records = []
    for orig_str in origins:
        orig = pd.Timestamp(orig_str)
        is_mature_regime = orig >= pd.Timestamp("2004-01-01")
        for c in train_eval_cases:
            df_c = case_dict[c]
            row = df_c[df_c["date"] == orig]
            if len(row) > 0:
                qo = float(row.iloc[0]["oil_rate_stbd"])
                qw = float(row.iloc[0]["water_rate_stbd"])
                fw = qw / max(qo + qw, 1.0)
                fw_records.append({"case_id": c, "origin": orig_str, "fw": fw, "mature": is_mature_regime})
    fw_df = pd.DataFrame(fw_records)
    if len(fw_df) == 0:
        return 0.35
    best_thresh = 0.35
    best_acc = -1.0
    for th in candidate_thresholds:
        preds = fw_df["fw"] > th
        acc = float(np.mean(preds == fw_df["mature"]))
        if acc > best_acc:
            best_acc = acc
            best_thresh = th
    return float(best_thresh)



def evaluate_single_reservoir_leave_out(
    held_out_case: int,
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    locked_weights: Dict[str, Dict[str, float]],
    eval_grid: List[Tuple[str, int]] = EVAL_GRID,
    candidate_ids: List[str] = DECISIVE_CANDIDATE_IDS,
    tracer_callback: Optional[Callable[[int, List[int]], None]] = None,
) -> List[Dict[str, Any]]:
    r"""
    Evaluates a single outer-fold held-out reservoir strictly trained on train_pool = TRAIN_CASES \ {held_out_case}.
    All base models, scalers, and selection gates are trained under strict Protocol A as-of each origin.
    """
    train_pool = [c for c in TRAIN_CASES if c != held_out_case]
    assert held_out_case not in train_pool, (
        f"DATA LEAKAGE VIOLATION: held_out_case {held_out_case} detected in train_pool!"
    )
    if tracer_callback is not None:
        tracer_callback(held_out_case, train_pool)
        
    df_c = case_dict[held_out_case]
    params = create_parameter_series(unc_df.loc[held_out_case, PARAM_NAMES].to_dict())
    
    unique_origins = sorted(list(set(orig_str for orig_str, _ in eval_grid)))
    records = []
    
    # Inner pseudo-fields for gate training (strictly excluded from held_out_case)
    inner_sample_cases = [c for c in [5, 15, 25, 35, 45, 55, 65] if c != held_out_case]
    
    for orig_str in unique_origins:
        orig = pd.Timestamp(orig_str)
        obs_origin = df_c[df_c["date"] == orig]
        if len(obs_origin) == 0:
            continue
        obs_origin = obs_origin.iloc[0]
        
        # 1. Fit outer base models strictly on train_pool under Protocol A as-of orig
        outer_models = fit_base_models_under_protocol(
            case_dict=case_dict,
            unc_df=unc_df,
            train_cases=train_pool,
            origin=orig,
            protocol="Protocol_A",
            horizon_quarters=20,
        )
        
        # 2. Strict as-of gate training: historical hindcasts ending <= orig
        valid_windows = get_valid_inner_asof_windows(orig)
        unique_inner_origins = sorted(list(set(w[0] for w in valid_windows)))
        
        models_inner = {}
        for in_orig_str in unique_inner_origins:
            models_inner[in_orig_str] = fit_base_models_under_protocol(
                case_dict=case_dict,
                unc_df=unc_df,
                train_cases=train_pool,
                origin=pd.Timestamp(in_orig_str),
                protocol="Protocol_A",
                horizon_quarters=20,
            )
            
        inner_db = precompute_inner_hindcast_database(
            case_dict=case_dict,
            unc_df=unc_df,
            train_pool=train_pool,
            eval_origin=orig,
            models_by_inner_origin=models_inner,
            locked_weights=locked_weights,
            sample_pseudo_fields=inner_sample_cases,
        )
        gate_params = fit_asof_loss_minimizing_selectors(inner_db)
        
        t_days = (orig - df_c["date"].min()).days
        init_state = {
            "qo": float(obs_origin["oil_rate_stbd"]),
            "qw": float(obs_origin["water_rate_stbd"]),
            "qg": float(obs_origin["gas_rate_mscfd"]),
            "Qo": float(obs_origin["oil_cum_stb"]),
            "Qw": float(obs_origin["water_cum_stb"]),
            "Qg": float(obs_origin["gas_cum_mscf"]),
            "wor": float(obs_origin["water_rate_stbd"] / max(obs_origin["oil_rate_stbd"], 1.0)),
            "res_age": float(t_days / YEAR_DAYS),
            "origin_date": orig,
        }
        state_feats = extract_observable_state_features(df_c, orig)
        
        horizons_for_orig = [h for o, h in eval_grid if o == orig_str]
        for h_q in horizons_for_orig:
            f_dates = pd.date_range(orig + pd.DateOffset(months=3), periods=h_q, freq="QS")
            true_fwd = df_c[df_c["date"].isin(f_dates)].sort_values("date")
            if len(true_fwd) < h_q:
                continue
                
            true_cum_o = true_fwd["oil_cum_stb"].values
            true_cum_w = true_fwd["water_cum_stb"].values
            true_rate_o = true_fwd["oil_rate_stbd"].values
            true_rate_w = true_fwd["water_rate_stbd"].values
            true_fw = true_rate_w / np.maximum(true_rate_o + true_rate_w, 1e-6)
            
            # Precompute base model predictions once per (origin, horizon)
            base_rates = {
                "Recursive_ExtraTrees": outer_models["Recursive_ExtraTrees"].predict_rates(init_state, params, f_dates),
                "Learned_Exponential_DCA": outer_models["Learned_Exponential_DCA"].predict_rates(init_state, params, f_dates),
                "Learned_Hyperbolic_DCA": outer_models["Learned_Hyperbolic_DCA"].predict_rates(init_state, params, f_dates),
                "Hybrid_Dynamic": outer_models["Hybrid_Dynamic"].predict_rates(init_state, params, f_dates),
            }
            
            for cand_id in candidate_ids:
                pred = predict_candidate_rates(
                    candidate_id=cand_id,
                    models=outer_models,
                    locked_weights=locked_weights,
                    init_state=init_state,
                    params=params,
                    forecast_dates=f_dates,
                    base_rates=base_rates,
                    gate_params=gate_params,
                )
                
                sc = score_cumulative_forecast(
                    pred_cum_oil=pred["cum_oil"],
                    pred_cum_wat=pred["cum_water"],
                    true_cum_oil=true_cum_o,
                    true_cum_wat=true_cum_w,
                    origin_cum_oil=init_state["Qo"],
                    origin_cum_wat=init_state["Qw"],
                    pred_rate_oil=pred["oil_rate"],
                    pred_rate_wat=pred["water_rate"],
                    true_rate_oil=true_rate_o,
                    true_rate_wat=true_rate_w,
                )
                
                pred_fw = pred["water_rate"] / np.maximum(pred["oil_rate"] + pred["water_rate"], 1e-6)
                fw_rmse = float(np.sqrt(np.mean((pred_fw - true_fw) ** 2)))
                
                neg_rates = int(np.sum(pred["oil_rate"] < -1e-6) + np.sum(pred["water_rate"] < -1e-6))
                non_mono_o = int(np.sum(np.diff(pred["cum_oil"]) < -1e-6))
                non_mono_w = int(np.sum(np.diff(pred["cum_water"]) < -1e-6))
                viol_count = neg_rates + non_mono_o + non_mono_w
                
                rec = {
                    "origin": orig_str,
                    "horizon_quarters": h_q,
                    "case_id": held_out_case,
                    "candidate_id": cand_id,
                    "macro_nrmse": sc["macro_nrmse"],
                    "oil_nrmse": sc["oil_nrmse"],
                    "water_nrmse": sc["water_nrmse"],
                    "oil_bias": sc["oil_bias"],
                    "water_bias": sc["water_bias"],
                    "water_cut_rmse": fw_rmse,
                    "physical_violations": viol_count,
                    "theta_fw_hard_learned": gate_params.get("theta_fw_hard", float("nan")),
                    "theta_abs_high_learned": gate_params.get("theta_abs_high", float("nan")),
                }
                rec.update(state_feats)
                records.append(rec)
                
    return records


def run_full_leave_reservoir_out_cross_validation(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    test_cases: List[int] = EVAL_PSEUDO_FIELDS,
    eval_grid: List[Tuple[str, int]] = EVAL_GRID,
    candidate_ids: List[str] = DECISIVE_CANDIDATE_IDS,
    tracer_callback: Optional[Callable[[int, List[int]], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes true leave-one-reservoir-out cross-validation across all 15 outer pseudo-field folds.
    """
    config = load_locked_config()
    locked_weights = config["weights"]
    
    all_records = []
    print(f"Executing Decisive Leave-One-Reservoir-Out Cross-Validation on {len(test_cases)} outer folds...")
    t_start = time.perf_counter()
    
    for idx, c_held in enumerate(test_cases, 1):
        t0 = time.perf_counter()
        recs = evaluate_single_reservoir_leave_out(
            held_out_case=c_held,
            case_dict=case_dict,
            unc_df=unc_df,
            locked_weights=locked_weights,
            eval_grid=eval_grid,
            candidate_ids=candidate_ids,
            tracer_callback=tracer_callback,
        )
        all_records.extend(recs)
        print(f"  Fold {idx:2d}/{len(test_cases)} (Held-out Case {c_held:2d}): evaluated {len(recs)} forecasts in {time.perf_counter() - t0:.2f}s")
        
    master_case_df = pd.DataFrame(all_records)
    
    # Aggregation with cluster-robust statistics
    base_mask = master_case_df["candidate_id"] == "Candidate_A_Frozen_Baseline"
    base_df = master_case_df[base_mask].set_index(["origin", "horizon_quarters", "case_id"])
    
    summary_records = []
    unique_reservoirs = master_case_df["case_id"].unique()
    M_clusters = len(unique_reservoirs)
    
    for cand_id in candidate_ids:
        cand_df = master_case_df[master_case_df["candidate_id"] == cand_id].set_index(["origin", "horizon_quarters", "case_id"])
        
        macro_mean = float(cand_df["macro_nrmse"].mean())
        oil_mean = float(cand_df["oil_nrmse"].mean())
        wat_mean = float(cand_df["water_nrmse"].mean())
        oil_bias = float(cand_df["oil_bias"].mean())
        wat_bias = float(cand_df["water_bias"].mean())
        fw_rmse_mean = float(cand_df["water_cut_rmse"].mean())
        worst_case_macro = float(cand_df["macro_nrmse"].max())
        total_violations = int(cand_df["physical_violations"].sum())
        
        if cand_id == "Candidate_A_Frozen_Baseline":
            diff_mean = 0.0
            diff_se_cluster = 0.0
            ci_low = 0.0
            ci_high = 0.0
            p_val = 1.0
            win_rate = 0.0
        else:
            diffs = cand_df["macro_nrmse"] - base_df["macro_nrmse"]
            win_rate = float(np.mean(diffs < 0.0))
            
            res_diffs = diffs.groupby("case_id").mean()
            diff_mean = float(res_diffs.mean())
            diff_se_cluster = float(res_diffs.std(ddof=1) / np.sqrt(M_clusters))
            
            t_crit = float(student_t.ppf(0.975, df=M_clusters - 1))
            ci_low = diff_mean - t_crit * diff_se_cluster
            ci_high = diff_mean + t_crit * diff_se_cluster
            
            t_stat = diff_mean / max(diff_se_cluster, 1e-12)
            p_val = float(2.0 * (1.0 - student_t.cdf(abs(t_stat), df=M_clusters - 1)))
            
        summary_records.append({
            "candidate_id": cand_id,
            "macro_nrmse_mean": macro_mean,
            "oil_nrmse_mean": oil_mean,
            "water_nrmse_mean": wat_mean,
            "diff_vs_baseline_macro": diff_mean,
            "diff_se_cluster": diff_se_cluster,
            "ci95_cluster_low": ci_low,
            "ci95_cluster_high": ci_high,
            "p_value_raw": p_val,
            "win_rate_vs_baseline": win_rate,
            "worst_case_macro": worst_case_macro,
            "oil_bias": oil_bias,
            "water_bias": wat_bias,
            "water_cut_rmse": fw_rmse_mean,
            "physical_violations": total_violations,
            "n_evaluations": len(cand_df),
            "n_reservoir_clusters": M_clusters,
        })
        
    summary_df = pd.DataFrame(summary_records).sort_values("macro_nrmse_mean").reset_index(drop=True)
    
    # Calculate Holm-Bonferroni adjusted p-values
    # Non-baseline models sorted by raw p-value
    non_base = summary_df[summary_df["candidate_id"] != "Candidate_A_Frozen_Baseline"].copy()
    m_tests = len(non_base)
    sorted_indices = non_base.sort_values("p_value_raw").index
    
    holm_pvals = {}
    running_max = 0.0
    for rank, idx_row in enumerate(sorted_indices, 1):
        raw_p = summary_df.loc[idx_row, "p_value_raw"]
        adjusted_p = min(1.0, (m_tests - rank + 1) * raw_p)
        running_max = max(running_max, adjusted_p)
        holm_pvals[summary_df.loc[idx_row, "candidate_id"]] = running_max
        
    summary_df["p_value_holm"] = summary_df["candidate_id"].map(lambda c: holm_pvals.get(c, 1.0))
    summary_df["p_value_bonferroni"] = summary_df["candidate_id"].map(
        lambda c: 1.0 if c == "Candidate_A_Frozen_Baseline" else min(1.0, m_tests * summary_df.loc[summary_df["candidate_id"] == c, "p_value_raw"].iloc[0])
    )
    
    print(f"Decisive Leave-One-Reservoir-Out Cross-Validation completed in {time.perf_counter() - t_start:.2f} seconds.")
    return master_case_df, summary_df


def evaluate_cases_71_to_85_dual_protocols(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    locked_weights: Dict[str, Dict[str, float]],
    candidate_ids: List[str] = DECISIVE_CANDIDATE_IDS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates secondary confirmation Cases 71-85 at origin 2005-01-01 (12Q horizon)
    strictly separating Protocol A (strict historical as-of) and Protocol B (forward simulator supervised).
    """
    f_dates_val = pd.date_range("2005-04-01", periods=12, freq="QS")
    eval_origin = pd.Timestamp("2005-01-01")
    
    # 1. Protocol A: Strictly truncated to 2005-01-01
    models_A = fit_base_models_under_protocol(
        case_dict=case_dict,
        unc_df=unc_df,
        train_cases=TRAIN_CASES,
        origin=eval_origin,
        protocol="Protocol_A",
        horizon_quarters=12,
    )
    # Inner gate training under Protocol A for 2005-01-01
    valid_windows_A = get_valid_inner_asof_windows(eval_origin)
    unique_inner_origins = sorted(list(set(w[0] for w in valid_windows_A)))
    models_inner_A = {}
    for in_orig_str in unique_inner_origins:
        models_inner_A[in_orig_str] = fit_base_models_under_protocol(
            case_dict=case_dict,
            unc_df=unc_df,
            train_cases=TRAIN_CASES,
            origin=pd.Timestamp(in_orig_str),
            protocol="Protocol_A",
            horizon_quarters=20,
        )
    inner_db_A = precompute_inner_hindcast_database(
        case_dict=case_dict,
        unc_df=unc_df,
        train_pool=TRAIN_CASES,
        eval_origin=eval_origin,
        models_by_inner_origin=models_inner_A,
        locked_weights=locked_weights,
        sample_pseudo_fields=[5, 15, 25, 35, 45, 55, 65],
    )
    gate_params_A = fit_asof_loss_minimizing_selectors(inner_db_A)
    
    # 2. Protocol B: Forward full-history trajectories
    models_B = fit_base_models_under_protocol(
        case_dict=case_dict,
        unc_df=unc_df,
        train_cases=TRAIN_CASES,
        origin=eval_origin,
        protocol="Protocol_B",
        horizon_quarters=12,
    )
    
    records_A = []
    records_B = []
    
    for c in VAL_CASES:
        df_c = case_dict[c]
        obs_origin = df_c[df_c["date"] == eval_origin].iloc[0]
        true_fwd = df_c[df_c["date"].isin(f_dates_val)].sort_values("date")
        t_days = (eval_origin - df_c["date"].min()).days
        init_state = {
            "qo": float(obs_origin["oil_rate_stbd"]),
            "qw": float(obs_origin["water_rate_stbd"]),
            "qg": float(obs_origin["gas_rate_mscfd"]),
            "Qo": float(obs_origin["oil_cum_stb"]),
            "Qw": float(obs_origin["water_cum_stb"]),
            "Qg": float(obs_origin["gas_cum_mscf"]),
            "wor": float(obs_origin["water_rate_stbd"] / max(obs_origin["oil_rate_stbd"], 1.0)),
            "res_age": float(t_days / YEAR_DAYS),
            "origin_date": eval_origin,
        }
        params = create_parameter_series(unc_df.loc[c, PARAM_NAMES].to_dict())
        
        # Protocol A evaluation
        base_rates_A = {
            "Recursive_ExtraTrees": models_A["Recursive_ExtraTrees"].predict_rates(init_state, params, f_dates_val),
            "Learned_Exponential_DCA": models_A["Learned_Exponential_DCA"].predict_rates(init_state, params, f_dates_val),
            "Learned_Hyperbolic_DCA": models_A["Learned_Hyperbolic_DCA"].predict_rates(init_state, params, f_dates_val),
            "Hybrid_Dynamic": models_A["Hybrid_Dynamic"].predict_rates(init_state, params, f_dates_val),
        }
        for cand_id in candidate_ids:
            pred_A = predict_candidate_rates(
                cand_id, models_A, locked_weights, init_state, params, f_dates_val,
                base_rates=base_rates_A, gate_params=gate_params_A
            )
            sc_A = score_cumulative_forecast(
                pred_A["cum_oil"], pred_A["cum_water"], true_fwd["oil_cum_stb"].values, true_fwd["water_cum_stb"].values,
                init_state["Qo"], init_state["Qw"], pred_A["oil_rate"], pred_A["water_rate"],
                true_fwd["oil_rate_stbd"].values, true_fwd["water_rate_stbd"].values
            )
            records_A.append({
                "case_id": c,
                "protocol": "Protocol_A_Strict_AsOf",
                "candidate_id": cand_id,
                "macro_nrmse": sc_A["macro_nrmse"],
                "oil_nrmse": sc_A["oil_nrmse"],
                "water_nrmse": sc_A["water_nrmse"],
            })
            
        # Protocol B evaluation
        base_rates_B = {
            "Recursive_ExtraTrees": models_B["Recursive_ExtraTrees"].predict_rates(init_state, params, f_dates_val),
            "Learned_Exponential_DCA": models_B["Learned_Exponential_DCA"].predict_rates(init_state, params, f_dates_val),
            "Learned_Hyperbolic_DCA": models_B["Learned_Hyperbolic_DCA"].predict_rates(init_state, params, f_dates_val),
            "Hybrid_Dynamic": models_B["Hybrid_Dynamic"].predict_rates(init_state, params, f_dates_val),
        }
        for cand_id in candidate_ids:
            pred_B = predict_candidate_rates(
                cand_id, models_B, locked_weights, init_state, params, f_dates_val,
                base_rates=base_rates_B, gate_params=gate_params_A
            )
            sc_B = score_cumulative_forecast(
                pred_B["cum_oil"], pred_B["cum_water"], true_fwd["oil_cum_stb"].values, true_fwd["water_cum_stb"].values,
                init_state["Qo"], init_state["Qw"], pred_B["oil_rate"], pred_B["water_rate"],
                true_fwd["oil_rate_stbd"].values, true_fwd["water_rate_stbd"].values
            )
            records_B.append({
                "case_id": c,
                "protocol": "Protocol_B_Forward_Supervised",
                "candidate_id": cand_id,
                "macro_nrmse": sc_B["macro_nrmse"],
                "oil_nrmse": sc_B["oil_nrmse"],
                "water_nrmse": sc_B["water_nrmse"],
            })
            
    return pd.DataFrame(records_A), pd.DataFrame(records_B)
