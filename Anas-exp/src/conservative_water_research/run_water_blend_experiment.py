"""
Master Execution Script: Conservative Water Blending Research.
Project: ExxonMobil DataWorks Challenge 2026
Branch: research/conservative-water-blend-validation

Executes:
1. True Leave-One-Reservoir-Out evaluation across 15 held-out reservoirs x 17 conditions (255 instances).
2. Strict temporal and double-reservoir separation on inner hindcasts.
3. Loss minimization over candidate gamma in {0.0, 0.10, 0.25, 0.50} scoring integrated cumulative trajectories.
4. Evaluation of the exact 3 frozen configurations:
   - Candidate_A_Frozen_Baseline
   - Candidate_Shrinkage_Blend (diagnostic control)
   - Candidate_Corrected_Water_Blend (corrected candidate)
5. Analysis of switching penalties vs gains, worst regressions, and condition-level breakdowns.
6. Evaluation on previously consulted secondary evidence (Cases 71-85).
"""
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from ensemble.generate_competition_deliverable import load_locked_config
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
)
from ensemble.data import (
    load_uncertainty,
    resolve_shared_dir,
    TRAIN_CASES,
    VAL_CASES,
)
from geological_information_value.authoritative_scorer import score_cumulative_forecast
from geological_information_value.parameter_bounds import PARAM_NAMES, create_parameter_series
from conservative_water_research.water_blend_engine import (
    fit_models_protocol_a,
    precompute_isolated_inner_hindcasts,
    calibrate_corrected_water_gamma,
    calibrate_unchanged_shrinkage_blend,
    predict_exact_candidate_forecast,
    CONFIG_BASELINE,
    CONFIG_DIAGNOSTIC_SHRINKAGE,
    CONFIG_CORRECTED_BLEND,
    EXACT_THREE_CANDIDATES,
    DT_DAYS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "outputs" / "research" / "conservative_water_research"
TABLE_DIR = OUTPUT_DIR / "tables"

EVAL_PSEUDO_FIELDS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
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


def run_full_cross_validation(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    locked_weights: Dict[str, Dict[str, float]],
    test_cases: List[int] = EVAL_PSEUDO_FIELDS,
    eval_grid: List[Tuple[str, int]] = EVAL_GRID,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes true Leave-One-Reservoir-Out cross-validation across 15 outer folds x 17 conditions.
    """
    all_records = []
    unique_origins = sorted(list(set(orig_str for orig_str, _ in eval_grid)))
    w_oil = locked_weights["oil"]
    w_wat = locked_weights["water"]
    
    print(f"Executing Conservative Water Blending CV across {len(test_cases)} outer folds x {len(eval_grid)} conditions...")
    t_start = time.perf_counter()
    
    for idx, c_held in enumerate(test_cases, 1):
        t0 = time.perf_counter()
        df_c = case_dict[c_held]
        params = create_parameter_series(unc_df.loc[c_held, PARAM_NAMES].to_dict())
        train_pool = [c for c in TRAIN_CASES if c != c_held]
        
        # Verify outer exclusion
        assert c_held not in train_pool, f"Outer reservoir {c_held} leaked into train_pool!"
        inner_models_cache = {}
        
        for orig_str in unique_origins:
            orig = pd.Timestamp(orig_str)
            obs_origin = df_c[df_c["date"] == orig]
            if len(obs_origin) == 0:
                continue
            obs_origin = obs_origin.iloc[0]
            
            # 1. Fit outer models strictly on train_pool under Protocol A as of orig
            outer_models = fit_models_protocol_a(
                case_dict=case_dict,
                unc_df=unc_df,
                train_cases=train_pool,
                origin=orig,
                horizon_quarters=20,
            )
            
            # 2. Strict temporal and double-reservoir separation: inner hindcasts
            inner_hindcasts = precompute_isolated_inner_hindcasts(
                case_dict=case_dict,
                unc_df=unc_df,
                outer_held_out=c_held,
                eval_origin=orig,
                locked_weights=locked_weights,
                inner_models_cache=inner_models_cache,
            )
            
            # 3. Calibrate corrected gamma (cumulative loss optimization)
            calib_corr = calibrate_corrected_water_gamma(inner_hindcasts)
            gamma_star = calib_corr["selected_gamma"]
            
            # 4. Calibrate diagnostic shrinkage (linear surrogate)
            calib_shrink = calibrate_unchanged_shrinkage_blend(inner_hindcasts)
            
            # Initial state at origin
            qo_0 = float(obs_origin["oil_rate_stbd"])
            qw_0 = float(obs_origin["water_rate_stbd"])
            fw_0 = float(qw_0 / max(qo_0 + qw_0, 1.0))
            t_days = (orig - df_c["date"].min()).days
            res_age_0 = float(t_days / 365.25)
            
            init_state = {
                "qo": qo_0,
                "qw": qw_0,
                "qg": float(obs_origin["gas_rate_mscfd"]),
                "Qo": float(obs_origin["oil_cum_stb"]),
                "Qw": float(obs_origin["water_cum_stb"]),
                "Qg": float(obs_origin["gas_cum_mscf"]),
                "wor": float(qw_0 / max(qo_0, 1.0)),
                "res_age": res_age_0,
                "origin_date": orig,
            }
            
            # Evaluate each horizon for this origin
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
                
                # Base model rates
                r_et = outer_models["Recursive_ExtraTrees"].predict_rates(init_state, params, f_dates)
                r_exp = outer_models["Learned_Exponential_DCA"].predict_rates(init_state, params, f_dates)
                r_hyp = outer_models["Learned_Hyperbolic_DCA"].predict_rates(init_state, params, f_dates)
                r_hyb = outer_models["Hybrid_Dynamic"].predict_rates(init_state, params, f_dates)
                
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
                qw_et = r_et["water_rate"]
                
                # Evaluate the EXACT three configurations
                for cand_id in EXACT_THREE_CANDIDATES:
                    if cand_id == CONFIG_BASELINE:
                        cal_p = {}
                    elif cand_id == CONFIG_DIAGNOSTIC_SHRINKAGE:
                        cal_p = calib_shrink
                    elif cand_id == CONFIG_CORRECTED_BLEND:
                        cal_p = calib_corr
                        
                    pred = predict_exact_candidate_forecast(
                        candidate_id=cand_id,
                        qo_frozen=qo_frozen,
                        qw_frozen=qw_frozen,
                        qw_et=qw_et,
                        init_state=init_state,
                        calib_params=cal_p,
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
                    
                    # Weight applied to ET water
                    if cand_id == CONFIG_BASELINE:
                        w_applied = 0.0
                    elif cand_id == CONFIG_DIAGNOSTIC_SHRINKAGE:
                        w_raw = (fw_0 - calib_shrink["shrink_low"]) / max(calib_shrink["shrink_high"] - calib_shrink["shrink_low"], 1e-6)
                        w_applied = calib_shrink["shrink_gamma"] * float(np.clip(w_raw, 0.0, 1.0))
                    elif cand_id == CONFIG_CORRECTED_BLEND:
                        w_raw = float(np.clip((fw_0 - 0.30) / 0.30, 0.0, 1.0))
                        w_applied = gamma_star * w_raw
                        
                    all_records.append({
                        "case_id": c_held,
                        "origin": orig_str,
                        "horizon_quarters": h_q,
                        "candidate_id": cand_id,
                        "macro_nrmse": sc["macro_nrmse"],
                        "oil_nrmse": sc["oil_nrmse"],
                        "water_nrmse": sc["water_nrmse"],
                        "oil_bias": sc["oil_bias"],
                        "water_bias": sc["water_bias"],
                        "fw_0": fw_0,
                        "res_age": res_age_0,
                        "gamma_selected": gamma_star if cand_id == CONFIG_CORRECTED_BLEND else float("nan"),
                        "w_blend_applied": w_applied,
                        "n_inner_hindcasts": len(inner_hindcasts),
                    })
                    
        print(f"  Fold {idx:2d}/15 (Case {c_held:2d}): evaluated in {time.perf_counter() - t0:.2f}s")
        
    master_df = pd.DataFrame(all_records)
    print(f"Cross-Validation complete in {time.perf_counter() - t_start:.2f}s. Total instances: {len(master_df)}")
    return master_df


def aggregate_benchmark_results(master_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Computes paired cluster-robust statistics, per-reservoir breakdown,
    condition breakdown, and switching penalty analysis.
    """
    base_mask = master_df["candidate_id"] == CONFIG_BASELINE
    base_df = master_df[base_mask].set_index(["case_id", "origin", "horizon_quarters"])
    
    unique_reservoirs = master_df["case_id"].unique()
    M = len(unique_reservoirs)
    df_deg = M - 1
    t_crit = float(student_t.ppf(0.975, df=df_deg))
    
    summary_rows = []
    for cand_id in EXACT_THREE_CANDIDATES:
        cand_df = master_df[master_df["candidate_id"] == cand_id].set_index(["case_id", "origin", "horizon_quarters"])
        
        macro_mean = float(cand_df["macro_nrmse"].mean())
        oil_mean = float(cand_df["oil_nrmse"].mean())
        wat_mean = float(cand_df["water_nrmse"].mean())
        oil_bias = float(cand_df["oil_bias"].mean())
        wat_bias = float(cand_df["water_bias"].mean())
        worst_macro = float(cand_df["macro_nrmse"].max())
        
        if cand_id == CONFIG_BASELINE:
            diff_mean = 0.0
            diff_se = 0.0
            ci_low = 0.0
            ci_high = 0.0
            p_val = 1.0
            win_rate = 0.0
            worst_diff = 0.0
        else:
            diffs = cand_df["macro_nrmse"] - base_df["macro_nrmse"]
            res_diffs = diffs.groupby("case_id").mean()
            diff_mean = float(res_diffs.mean())
            diff_se = float(res_diffs.std(ddof=1) / np.sqrt(M))
            ci_low = diff_mean - t_crit * diff_se
            ci_high = diff_mean + t_crit * diff_se
            t_stat = diff_mean / max(diff_se, 1e-12)
            p_val = float(2.0 * (1.0 - student_t.cdf(abs(t_stat), df=df_deg)))
            win_rate = float(np.mean(diffs < 0.0))
            worst_diff = float(diffs.max())
            
        summary_rows.append({
            "candidate_id": cand_id,
            "macro_nrmse_mean": macro_mean,
            "oil_nrmse_mean": oil_mean,
            "water_nrmse_mean": wat_mean,
            "macro_diff_mean": diff_mean,
            "macro_diff_se": diff_se,
            "macro_ci95_low": ci_low,
            "macro_ci95_high": ci_high,
            "p_value": p_val,
            "win_rate_vs_base": win_rate,
            "worst_case_macro": worst_macro,
            "worst_case_diff": worst_diff,
            "oil_bias_mean": oil_bias,
            "water_bias_mean": wat_bias,
            "n_evaluations": len(cand_df),
            "n_reservoir_clusters": M,
        })
    df_summary = pd.DataFrame(summary_rows)
    
    # Per-reservoir breakdown
    res_rows = []
    piv_macro = master_df.pivot_table(index="case_id", columns="candidate_id", values="macro_nrmse")
    piv_wat = master_df.pivot_table(index="case_id", columns="candidate_id", values="water_nrmse")
    for cid in sorted(unique_reservoirs):
        b_m = float(piv_macro.loc[cid, CONFIG_BASELINE])
        s_m = float(piv_macro.loc[cid, CONFIG_DIAGNOSTIC_SHRINKAGE])
        c_m = float(piv_macro.loc[cid, CONFIG_CORRECTED_BLEND])
        b_w = float(piv_wat.loc[cid, CONFIG_BASELINE])
        c_w = float(piv_wat.loc[cid, CONFIG_CORRECTED_BLEND])
        res_rows.append({
            "case_id": cid,
            "baseline_macro": b_m,
            "diagnostic_shrinkage_macro": s_m,
            "corrected_blend_macro": c_m,
            "corrected_diff_macro": c_m - b_m,
            "baseline_water": b_w,
            "corrected_water": c_w,
            "corrected_diff_water": c_w - b_w,
            "corrected_status": "WIN" if c_m < b_m else "LOSS",
        })
    df_per_res = pd.DataFrame(res_rows)
    
    # Per-condition breakdown (17 conditions)
    cond_piv = master_df.pivot_table(index=["origin", "horizon_quarters"], columns="candidate_id", values=["macro_nrmse", "water_nrmse"])
    cond_gamma = master_df[master_df["candidate_id"] == CONFIG_CORRECTED_BLEND].pivot_table(index=["origin", "horizon_quarters"], values="gamma_selected", aggfunc="mean")
    cond_rows = []
    for (orig, hq), row in cond_piv.iterrows():
        b_m = float(row[("macro_nrmse", CONFIG_BASELINE)])
        s_m = float(row[("macro_nrmse", CONFIG_DIAGNOSTIC_SHRINKAGE)])
        c_m = float(row[("macro_nrmse", CONFIG_CORRECTED_BLEND)])
        b_w = float(row[("water_nrmse", CONFIG_BASELINE)])
        c_w = float(row[("water_nrmse", CONFIG_CORRECTED_BLEND)])
        g_sel = float(cond_gamma.loc[(orig, hq), "gamma_selected"])
        cond_rows.append({
            "origin": orig,
            "horizon_quarters": hq,
            "baseline_macro": b_m,
            "diagnostic_shrinkage_macro": s_m,
            "corrected_blend_macro": c_m,
            "diff_macro": c_m - b_m,
            "diff_water": c_w - b_w,
            "mean_gamma_selected": g_sel,
            "winner": "Corrected_Blend" if c_m < b_m else "Baseline",
        })
    df_per_cond = pd.DataFrame(cond_rows)
    
    # Switching Penalty vs Gain Analysis
    # Compare instances where model diverged from baseline (w_blend > 0)
    switch_rows = []
    for cand_id in [CONFIG_DIAGNOSTIC_SHRINKAGE, CONFIG_CORRECTED_BLEND]:
        c_df = master_df[master_df["candidate_id"] == cand_id].copy()
        b_df = master_df[master_df["candidate_id"] == CONFIG_BASELINE].copy()
        merged = pd.merge(c_df, b_df, on=["case_id", "origin", "horizon_quarters"], suffixes=("_cand", "_base"))
        
        switched = merged[merged["w_blend_applied_cand"] > 1e-6]
        diffs = switched["macro_nrmse_cand"] - switched["macro_nrmse_base"]
        
        gains = diffs[diffs < 0.0]
        penalties = diffs[diffs > 0.0]
        ties = diffs[diffs == 0.0]
        
        mean_gain = float(gains.mean()) if len(gains) > 0 else 0.0
        mean_penalty = float(penalties.mean()) if len(penalties) > 0 else 0.0
        max_penalty = float(penalties.max()) if len(penalties) > 0 else 0.0
        asymmetry_ratio = abs(mean_penalty / mean_gain) if mean_gain != 0 else float("inf")
        
        switch_rows.append({
            "candidate_id": cand_id,
            "total_evaluations": len(merged),
            "switched_evaluations": len(switched),
            "switching_rate": float(len(switched) / len(merged)),
            "n_gains": len(gains),
            "n_penalties": len(penalties),
            "n_neutral": len(ties),
            "mean_gain": mean_gain,
            "mean_penalty": mean_penalty,
            "max_penalty": max_penalty,
            "asymmetry_ratio": asymmetry_ratio,
        })
    df_switch = pd.DataFrame(switch_rows)
    
    return df_summary, df_per_res, df_per_cond, df_switch


def evaluate_secondary_cases_71_to_85(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    locked_weights: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Evaluates previously consulted secondary evidence (Cases 71-85) strictly under Protocol A
    at Origin 2005-01-01 (12Q horizon).
    """
    eval_origin = pd.Timestamp("2005-01-01")
    f_dates_val = pd.date_range("2005-04-01", periods=12, freq="QS")
    w_oil = locked_weights["oil"]
    w_wat = locked_weights["water"]
    
    # 1. Outer models on permitted training cases (Cases 1-70)
    models_A = fit_models_protocol_a(
        case_dict=case_dict,
        unc_df=unc_df,
        train_cases=TRAIN_CASES,
        origin=eval_origin,
        horizon_quarters=12,
    )
    
    # 2. Inner hindcasts for calibration at 2005-01-01
    inner_hindcasts = precompute_isolated_inner_hindcasts(
        case_dict=case_dict,
        unc_df=unc_df,
        outer_held_out=-1,  # none held out from validation set
        eval_origin=eval_origin,
        locked_weights=locked_weights,
        inner_sample_cases=[5, 15, 25, 35, 45, 55, 65],
    )
    calib_corr = calibrate_corrected_water_gamma(inner_hindcasts)
    calib_shrink = calibrate_unchanged_shrinkage_blend(inner_hindcasts)
    
    records = []
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
            "res_age": float(t_days / 365.25),
            "origin_date": eval_origin,
        }
        params = create_parameter_series(unc_df.loc[c, PARAM_NAMES].to_dict())
        
        r_et = models_A["Recursive_ExtraTrees"].predict_rates(init_state, params, f_dates_val)
        r_exp = models_A["Learned_Exponential_DCA"].predict_rates(init_state, params, f_dates_val)
        r_hyp = models_A["Learned_Hyperbolic_DCA"].predict_rates(init_state, params, f_dates_val)
        r_hyb = models_A["Hybrid_Dynamic"].predict_rates(init_state, params, f_dates_val)
        
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
        qw_et = r_et["water_rate"]
        
        true_co = true_fwd["oil_cum_stb"].values
        true_cw = true_fwd["water_cum_stb"].values
        true_ro = true_fwd["oil_rate_stbd"].values
        true_rw = true_fwd["water_rate_stbd"].values
        
        for cand_id in EXACT_THREE_CANDIDATES:
            cal_p = calib_shrink if cand_id == CONFIG_DIAGNOSTIC_SHRINKAGE else (calib_corr if cand_id == CONFIG_CORRECTED_BLEND else {})
            pred = predict_exact_candidate_forecast(
                candidate_id=cand_id,
                qo_frozen=qo_frozen,
                qw_frozen=qw_frozen,
                qw_et=qw_et,
                init_state=init_state,
                calib_params=cal_p,
            )
            sc = score_cumulative_forecast(
                pred["cum_oil"], pred["cum_water"], true_co, true_cw,
                init_state["Qo"], init_state["Qw"], pred["oil_rate"], pred["water_rate"], true_ro, true_rw
            )
            records.append({
                "case_id": c,
                "protocol": "Protocol_A_Strict_AsOf",
                "candidate_id": cand_id,
                "macro_nrmse": sc["macro_nrmse"],
                "oil_nrmse": sc["oil_nrmse"],
                "water_nrmse": sc["water_nrmse"],
                "evidence_type": "Previously_Consulted_Secondary_Evidence",
            })
    return pd.DataFrame(records)


def run_all_experiments():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("CONSERVATIVE WATER BLENDING RESEARCH — STRICT PROTOCOL A EXPERIMENT")
    print("=" * 80)
    
    config = load_locked_config()
    locked_weights = config["weights"]
    
    shared_dir = resolve_shared_dir()
    unc_df = load_uncertainty(shared_dir)
    raw_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(raw_df)
    
    # 1. Run 15 folds x 17 conditions LOO Cross-Validation
    master_df = run_full_cross_validation(case_dict, unc_df, locked_weights)
    master_df.to_csv(TABLE_DIR / "conservative_water_case_evaluations.csv", index=False)
    print(f"Saved case evaluations: {TABLE_DIR / 'conservative_water_case_evaluations.csv'}")
    
    # 2. Aggregations & Statistical Analysis
    df_summary, df_per_res, df_per_cond, df_switch = aggregate_benchmark_results(master_df)
    df_summary.to_csv(TABLE_DIR / "conservative_water_summary_benchmark.csv", index=False)
    df_per_res.to_csv(TABLE_DIR / "conservative_water_per_reservoir_breakdown.csv", index=False)
    df_per_cond.to_csv(TABLE_DIR / "conservative_water_per_condition_breakdown.csv", index=False)
    df_switch.to_csv(TABLE_DIR / "conservative_water_switching_penalties_analysis.csv", index=False)
    
    print("\nSummary Benchmark:")
    print(df_summary[["candidate_id", "macro_nrmse_mean", "macro_diff_mean", "macro_diff_se", "macro_ci95_low", "macro_ci95_high", "p_value", "win_rate_vs_base", "worst_case_diff"]].to_string())
    
    print("\nSwitching Penalty vs Gain Analysis:")
    print(df_switch.to_string())
    
    # 3. Previously Consulted Secondary Evidence (Cases 71-85)
    print("\nEvaluating previously consulted secondary evidence (Cases 71-85)...")
    val_df = evaluate_secondary_cases_71_to_85(case_dict, unc_df, locked_weights)
    val_df.to_csv(TABLE_DIR / "conservative_water_secondary_confirmation_cases_71_to_85.csv", index=False)
    
    val_piv = val_df.pivot_table(index="candidate_id", values=["macro_nrmse", "oil_nrmse", "water_nrmse"])
    print("Secondary Holdout Performance (Cases 71-85):")
    print(val_piv.to_string())
    print("\nAll experiments completed successfully.")


if __name__ == "__main__":
    run_all_experiments()
