"""
Module: iteration_comparison_v3.py
Research Iteration 3 — Phase 7: Quantify Improvements Across Iterations 1, 2, and 3.

Methodology:
Evaluates the 10 selected scenarios from:
  1. Research Iteration 1 (outputs/history_matching/tables/selected_10_cases_specification.csv)
  2. Research Iteration 2 (outputs/research/history_matching_iteration_2/tables/selected_10_cases_specification_v2.csv)
  3. Research Iteration 3 (outputs/research/history_matching_iteration_3/tables/selected_10_cases_specification_v3.csv)
under strictly identical conditions:
  - Exact forward quarterly calendar alignment (Delta t in [90, 92] days)
  - Identical field history observations (field_history.parquet)
  - Identical likelihood functions (Models L1 and L2)
  - Identical metric definitions (Oil Cum NRMSE, Water Cum NRMSE, Oil Rate NRMSE, Water Rate NRMSE, Terminal Errors)
  - Identical frozen forecasting model (Blend_Phase_Specific_Simplex)
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from ensemble.data import load_uncertainty, TRAIN_CASES, PARAMS, resolve_shared_dir
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
)
from history_matching_v2.emulator_revalidation_v2 import HistoricalSVDGPEmulatorV2
from history_matching_v2.history_matching_likelihood_v2 import HistoryMatchingLikelihoodV2
from history_matching_v2.scenario_selection_v2 import (
    load_frozen_forecasting_models,
    forecast_candidate_realization,
)

CANONICAL_PARAMS = [
    "Fault_Transmissibility",
    "Porosity_Multiplier",
    "Permeability_Multiplier",
    "Aquifer_Pore_Volume",
]


def run_iteration_head_to_head_comparison() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("=" * 80)
    print("RESEARCH ITERATION 3 — PHASE 7: HEAD-TO-HEAD ITERATION COMPARISON")
    print("=" * 80)

    base_dir = Path("/Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp")
    out_dir = base_dir / "outputs/research/history_matching_iteration_3/tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    shared_dir = resolve_shared_dir()
    unc_df = load_uncertainty(shared_dir)
    q_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(q_df)
    fh = pd.read_parquet(base_dir / "../data/field_history.parquet")
    dates = pd.to_datetime(case_dict[1]["date"].to_numpy())

    X_train = unc_df.loc[TRAIN_CASES, PARAMS].to_numpy()
    Y_oil_tr = np.array([case_dict[c]["oil_cum_stb"].to_numpy() for c in TRAIN_CASES])
    Y_wat_tr = np.array([case_dict[c]["water_cum_stb"].to_numpy() for c in TRAIN_CASES])

    emu = HistoricalSVDGPEmulatorV2(n_components=4, random_state=42)
    emu.fit(X_train, Y_oil_tr, Y_wat_tr, dates=dates)
    lik_l1 = HistoryMatchingLikelihoodV2(emu, fh, rho_ar1=0.80, obs_noise_rel_oil=0.05, obs_noise_rel_wat=0.05)
    lik_l2 = HistoryMatchingLikelihoodV2(emu, fh, rho_ar1=0.80, obs_noise_rel_oil=0.05, obs_noise_rel_wat=0.05)

    # Load frozen forecasting models
    models, weights = load_frozen_forecasting_models(case_dict, unc_df)
    models["Recursive_ExtraTrees"].model_qo.n_jobs = 1
    models["Recursive_ExtraTrees"].model_qw.n_jobs = 1

    prod = pd.read_parquet(shared_dir / "production_timeseries.parquet")
    c0 = prod[prod["case_num"] == 0]
    c0_2008 = c0[c0["date"] == "2008-01-01"].set_index("metric")["value"]
    field_anchor = {
        "qo": float(c0_2008["oil_rate"]),
        "qw": float(c0_2008["water_rate"]),
        "qg": float(c0_2008["gas_rate"]),
        "Qo": float(c0_2008["oil_cum"]),
        "Qw": float(c0_2008["water_cum"]),
        "Qg": float(c0_2008["gas_cum"]),
        "wor": float(c0_2008["water_rate"] / max(c0_2008["oil_rate"], 1.0)),
        "res_age": 10.0,
        "origin_date": pd.Timestamp("2008-01-01"),
    }

    # Load specifications from all 3 iterations
    spec_v1_path = base_dir / "outputs/history_matching/tables/selected_10_cases_specification.csv"
    spec_v2_path = base_dir / "outputs/research/history_matching_iteration_2/tables/selected_10_cases_specification_v2.csv"
    spec_v3_path = base_dir / "outputs/research/history_matching_iteration_3/tables/selected_10_cases_specification_v3.csv"

    df_v1 = pd.read_csv(spec_v1_path)
    df_v2 = pd.read_csv(spec_v2_path)
    df_v3 = pd.read_csv(spec_v3_path)

    # Standardize column names for parameter extraction
    def extract_thetas(df: pd.DataFrame) -> List[np.ndarray]:
        thetas = []
        for _, r in df.iterrows():
            th = np.array([
                r["Fault_Transmissibility"],
                r["Porosity_Multiplier"],
                r["Permeability_Multiplier"],
                r["Aquifer_Pore_Volume"],
            ])
            thetas.append(th)
        return thetas

    thetas_v1 = extract_thetas(df_v1)
    thetas_v2 = extract_thetas(df_v2)
    thetas_v3 = extract_thetas(df_v3)

    iterations = [
        ("Iteration 1 (Baseline)", thetas_v1, df_v1),
        ("Iteration 2 (Intermediate)", thetas_v2, df_v2),
        ("Iteration 3 (Final Reconciled)", thetas_v3, df_v3),
    ]

    all_case_eval_rows = []
    summary_eval_rows = []

    for iter_name, thetas, df_spec in iterations:
        oil_cum_nrmses = []
        wat_cum_nrmses = []
        oil_rate_nrmses = []
        wat_rate_nrmses = []
        term_oil_errs = []
        loglik_l1_vals = []
        loglik_l2_vals = []
        eur_80q_vals = []

        for i in range(10):
            th = thetas[i]
            comp_l1 = lik_l1.evaluate_log_likelihood(th, model="L1", return_components=True)
            comp_l2 = lik_l2.evaluate_log_likelihood(th, model="L2", return_components=True)
            fc = forecast_candidate_realization(th, models, weights, field_anchor, horizon_qtrs=80)

            oil_cum_nrmses.append(comp_l1["nrmse_Qo"])
            wat_cum_nrmses.append(comp_l1["nrmse_Qw"])
            oil_rate_nrmses.append(comp_l1["nrmse_qo"])
            wat_rate_nrmses.append(comp_l1["nrmse_qw"])
            term_oil_errs.append(comp_l1["terminal_oil_err"])
            loglik_l1_vals.append(comp_l1["log_likelihood"])
            loglik_l2_vals.append(comp_l2["log_likelihood"])
            eur_80q_vals.append(fc["final_oil_cum"])

            all_case_eval_rows.append({
                "Iteration": iter_name,
                "Case_Sheet": f"Case {i + 1}",
                "Fault_Transmissibility": th[0],
                "Porosity_Multiplier": th[1],
                "Permeability_Multiplier": th[2],
                "Aquifer_Pore_Volume": th[3],
                "Log_Likelihood_L1": comp_l1["log_likelihood"],
                "Log_Likelihood_L2": comp_l2["log_likelihood"],
                "Oil_Cum_NRMSE": comp_l1["nrmse_Qo"],
                "Water_Cum_NRMSE": comp_l1["nrmse_Qw"],
                "Oil_Rate_NRMSE": comp_l1["nrmse_qo"],
                "Water_Rate_NRMSE": comp_l1["nrmse_qw"],
                "Terminal_Oil_Rel_Err": comp_l1["terminal_oil_err"],
                "Ranking_80Q_Oil_EUR_STB": fc["final_oil_cum"],
            })

        th_arr = np.array(thetas)
        summary_eval_rows.append({
            "Iteration": iter_name,
            "Mean_Log_Likelihood_L1": float(np.mean(loglik_l1_vals)),
            "Worst_Log_Likelihood_L1": float(np.min(loglik_l1_vals)),
            "Mean_Oil_Cum_NRMSE": float(np.mean(oil_cum_nrmses)),
            "Max_Oil_Cum_NRMSE": float(np.max(oil_cum_nrmses)),
            "Mean_Water_Cum_NRMSE": float(np.mean(wat_cum_nrmses)),
            "Max_Water_Cum_NRMSE": float(np.max(wat_cum_nrmses)),
            "Worst_Case_Water_Error_Case3": float(wat_cum_nrmses[2]),  # Case 3
            "Mean_Oil_Rate_NRMSE": float(np.mean(oil_rate_nrmses)),
            "Mean_Water_Rate_NRMSE": float(np.mean(wat_rate_nrmses)),
            "Mean_Terminal_Oil_Rel_Err": float(np.mean(term_oil_errs)),
            "Min_80Q_Oil_EUR_STB": float(np.min(eur_80q_vals)),
            "Max_80Q_Oil_EUR_STB": float(np.max(eur_80q_vals)),
            "Spread_80Q_Oil_EUR_STB": float(np.max(eur_80q_vals) - np.min(eur_80q_vals)),
            "Porosity_Range": f"[{np.min(th_arr[:, 1]):.3f}, {np.max(th_arr[:, 1]):.3f}]",
            "Permeability_Range": f"[{np.min(th_arr[:, 2]):.3f}, {np.max(th_arr[:, 2]):.3f}]",
        })

    df_all_eval = pd.DataFrame(all_case_eval_rows)
    df_summary_eval = pd.DataFrame(summary_eval_rows)

    df_all_eval.to_csv(out_dir / "all_iterations_standardized_case_evaluation.csv", index=False)
    df_summary_eval.to_csv(out_dir / "head_to_head_iteration_performance_summary.csv", index=False)

    print("\nHead-to-Head Iteration Performance Summary (Strict Forward Scoring):")
    print(df_summary_eval.to_string())

    # Case 3 forensic comparison
    case3_rows = df_all_eval[df_all_eval["Case_Sheet"] == "Case 3"]
    print("\nCase 3 Water Anomaly Resolution Across Iterations:")
    print(case3_rows[["Iteration", "Case_Sheet", "Porosity_Multiplier", "Water_Cum_NRMSE", "Log_Likelihood_L1", "Ranking_80Q_Oil_EUR_STB"]].to_string())

    return df_summary_eval, df_all_eval, case3_rows


if __name__ == "__main__":
    run_iteration_head_to_head_comparison()
