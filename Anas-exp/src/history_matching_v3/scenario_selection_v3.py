"""
Module: scenario_selection_v3.py
Research Iteration 3 — Phase 5: Correct Scenario Quantile Semantics & Selection Objectives.

Key Principles:
1. Petroleum Exceedance vs Ordinary CDF Percentiles:
   - 90th percentile of EUR corresponds to P10 Exceedance (High Recovery: P(EUR >= P10) = 0.10).
   - 50th percentile of EUR corresponds to P50 Exceedance (Median Base Case: P(EUR >= P50) = 0.50).
   - 10th percentile of EUR corresponds to P90 Exceedance (Low Recovery: P(EUR >= P90) = 0.90).
   All intermediate labels (P15, P25, P35, P60, P70, P80, P85) follow the exact same rule:
   Ordinary Percentile p% = (100 - E)%, where E is the exceedance probability.
2. Horizon Separation:
   - All scenario rankings are computed on a consistent 80-quarter research forecast.
   - For Case 1, the deliverable forecast horizon is 40 quarters; its 80-quarter ranking
     metric is tracked separately without horizon mixing.
3. Canonical 4-Parameter Vector:
   [Fault_Transmissibility, Porosity_Multiplier, Permeability_Multiplier, Aquifer_Pore_Volume].
4. Systematic Comparison of 3 Selection Objectives:
   - Strategy A: Ten Best Historical Matches (Deterministic / MAP mode and nearest neighbors).
   - Strategy B: Ten Posterior-Predictive Representative Scenarios (Unconstrained Bayesian EUR quantiles).
   - Strategy C: Ten Diverse Acceptable-History-Match Realizations (Quality-filtered conditional quantiles).
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

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
PARAM_ABBR = ["Ft", "Poro", "Perm", "AqPV"]


def run_scenario_selection_v3_experiment() -> Dict[str, Any]:
    print("=" * 80)
    print("RESEARCH ITERATION 3 — PHASE 5: SCENARIO SELECTION & QUANTILE SEMANTICS")
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

    # Load frozen forecasting models and weights
    models, weights = load_frozen_forecasting_models(case_dict, unc_df)
    models["Recursive_ExtraTrees"].model_qo.n_jobs = 1
    models["Recursive_ExtraTrees"].model_qw.n_jobs = 1

    # Historical field anchor at 2008-01-01
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

    # Load pooled posterior draws from Phase 3 (L1 and L2)
    draws_l1 = np.load(base_dir / "outputs/research/history_matching_iteration_3/bayesian_posterior_draws_l1.npy")
    draws_l2 = np.load(base_dir / "outputs/research/history_matching_iteration_3/bayesian_posterior_draws_l2.npy")

    # Thin draws to 80 candidates for forecast evaluation (80 candidates * ~0.75s = ~60s total)
    thin_idx = np.linspace(0, len(draws_l1) - 1, 80, dtype=int)
    candidates_l1 = draws_l1[thin_idx]

    print(f"Evaluating 80-quarter frozen ensemble on {len(candidates_l1)} posterior candidates...")
    cand_records = []
    for idx, th in enumerate(candidates_l1):
        comp_l1 = lik_l1.evaluate_log_likelihood(th, model="L1", return_components=True)
        comp_l2 = lik_l2.evaluate_log_likelihood(th, model="L2", return_components=True)

        fc_80 = forecast_candidate_realization(th, models, weights, field_anchor, horizon_qtrs=80)
        eur_80q = fc_80["final_oil_cum"]

        pass_oil = comp_l1["nrmse_Qo"] <= 0.015
        pass_wat = comp_l1["nrmse_Qw"] <= 0.080
        pass_term = comp_l1["terminal_oil_err"] <= 0.010
        passes_all = pass_oil and pass_wat and pass_term

        cand_records.append({
            "cand_id": idx,
            "Ft": th[0],
            "Poro": th[1],
            "Perm": th[2],
            "AqPV": th[3],
            "log_lik_l1": comp_l1["log_likelihood"],
            "log_lik_l2": comp_l2["log_likelihood"],
            "oil_cum_nrmse": comp_l1["nrmse_Qo"],
            "water_cum_nrmse": comp_l1["nrmse_Qw"],
            "oil_rate_nrmse": comp_l1["nrmse_qo"],
            "water_rate_nrmse": comp_l1["nrmse_qw"],
            "terminal_oil_err": comp_l1["terminal_oil_err"],
            "frozen_ensemble_80q_eur": eur_80q,
            "passes_quality_filter": passes_all,
        })
    df_all_cands = pd.DataFrame(cand_records)

    # ==========================================================================
    # STRATEGY A: TEN BEST HISTORICAL MATCHES (DETERMINISTIC / MAP / GLS)
    # ==========================================================================
    top10_best = df_all_cands.sort_values("log_lik_l1", ascending=False).head(10).copy()
    top10_best["strategy_name"] = "Strategy A: Ten Best Historical Matches"

    # ==========================================================================
    # STRATEGY B: TEN POSTERIOR-PREDICTIVE REPRESENTATIVE SCENARIOS (UNCONSTRAINED)
    # ==========================================================================
    # Quantile definition:
    # Ordinary CDF percentiles: [90%, 85%, 80%, 75%, 65%, 50%, 40%, 30%, 20%, 10%]
    # Petroleum Exceedance labels: [P10, P15, P20, P25, P35, P50, P60, P70, P80, P90]
    target_quantiles = [
        {"ordinary_cdf_pct": 90.0, "exceedance_label": "P10 High-Recovery", "role": "High-Recovery Upper Bound"},
        {"ordinary_cdf_pct": 85.0, "exceedance_label": "P15", "role": "Upper Quartile Dispersion"},
        {"ordinary_cdf_pct": 80.0, "exceedance_label": "P20", "role": "Upper Quartile Dispersion"},
        {"ordinary_cdf_pct": 75.0, "exceedance_label": "P25", "role": "Moderate High Scenario"},
        {"ordinary_cdf_pct": 65.0, "exceedance_label": "P35", "role": "Moderate High Scenario"},
        {"ordinary_cdf_pct": 50.0, "exceedance_label": "P50 Median Base Case", "role": "Central Expected Forecast"},
        {"ordinary_cdf_pct": 40.0, "exceedance_label": "P60", "role": "Moderate Low Scenario"},
        {"ordinary_cdf_pct": 30.0, "exceedance_label": "P70", "role": "Moderate Low Scenario"},
        {"ordinary_cdf_pct": 20.0, "exceedance_label": "P80", "role": "Lower Quartile Dispersion"},
        {"ordinary_cdf_pct": 10.0, "exceedance_label": "P90 Low-Recovery", "role": "Low-Recovery Lower Bound"},
    ]

    strat_b_indices = []
    for q_spec in target_quantiles:
        p_val = np.percentile(df_all_cands["frozen_ensemble_80q_eur"], q_spec["ordinary_cdf_pct"])
        closest_idx = (df_all_cands["frozen_ensemble_80q_eur"] - p_val).abs().idxmin()
        strat_b_indices.append(closest_idx)
    strat_b_df = df_all_cands.loc[strat_b_indices].copy()
    strat_b_df["strategy_name"] = "Strategy B: Unconstrained Posterior EUR Quantiles"

    # ==========================================================================
    # STRATEGY C: TEN DIVERSE ACCEPTABLE-HISTORY-MATCH REALIZATIONS (FILTERED)
    # ==========================================================================
    df_accepted = df_all_cands[df_all_cands["passes_quality_filter"]].copy()
    strat_c_indices = []
    for q_spec in target_quantiles:
        p_val = np.percentile(df_accepted["frozen_ensemble_80q_eur"], q_spec["ordinary_cdf_pct"])
        closest_idx = (df_accepted["frozen_ensemble_80q_eur"] - p_val).abs().idxmin()
        strat_c_indices.append(closest_idx)
    strat_c_df = df_accepted.loc[strat_c_indices].copy()
    strat_c_df["strategy_name"] = "Strategy C: Quality-Filtered Conditional Quantiles"

    # Comparison summary across the 3 strategies
    comp_rows = []
    for s_name, df_s in [
        ("Strategy A (10 Best Historical Matches)", top10_best),
        ("Strategy B (Unconstrained Posterior Quantiles)", strat_b_df),
        ("Strategy C (Quality-Filtered Conditional Quantiles)", strat_c_df),
    ]:
        comp_rows.append({
            "strategy": s_name,
            "mean_log_lik_l1": float(df_s["log_lik_l1"].mean()),
            "mean_oil_nrmse": float(df_s["oil_cum_nrmse"].mean()),
            "mean_water_nrmse": float(df_s["water_cum_nrmse"].mean()),
            "eur_80q_min_stb": float(df_s["frozen_ensemble_80q_eur"].min()),
            "eur_80q_max_stb": float(df_s["frozen_ensemble_80q_eur"].max()),
            "eur_80q_spread_stb": float(df_s["frozen_ensemble_80q_eur"].max() - df_s["frozen_ensemble_80q_eur"].min()),
            "porosity_min": float(df_s["Poro"].min()),
            "porosity_max": float(df_s["Poro"].max()),
            "permeability_min": float(df_s["Perm"].min()),
            "permeability_max": float(df_s["Perm"].max()),
            "all_cases_pass_engineering_filter": bool(df_s["passes_quality_filter"].all()),
        })
    df_strat_comp = pd.DataFrame(comp_rows)
    df_strat_comp.to_csv(out_dir / "selection_strategies_comparison_v3.csv", index=False)
    print("\nSelection Strategies Comparison:")
    print(df_strat_comp.to_string())

    # ==========================================================================
    # ASSEMBLE DETAILED SPECIFICATION FOR STRATEGY C (RECOMMENDED DELIVERABLE)
    # ==========================================================================
    horizons = [40] + [80] * 9  # Case 1 is 40Q; Cases 2-10 are 80Q
    spec_rows = []
    forecast_rows = []

    for i in range(10):
        row_c = strat_c_df.iloc[i]
        q_info = target_quantiles[i]
        hz = horizons[i]
        th_val = np.array([row_c["Ft"], row_c["Poro"], row_c["Perm"], row_c["AqPV"]])

        # Deliverable forecast at assigned horizon (40Q or 80Q)
        fc_deliv = forecast_candidate_realization(th_val, models, weights, field_anchor, horizon_qtrs=hz)
        # Separate ranking forecast at 80Q (strictly for cross-case comparison)
        fc_80q_rank = forecast_candidate_realization(th_val, models, weights, field_anchor, horizon_qtrs=80)

        case_id = f"Case {i + 1}"
        designation = f"{case_id} ({q_info['exceedance_label']} / {hz}Q)"

        spec_rows.append({
            "Case_Sheet": case_id,
            "Designation": designation,
            "Forecast_Horizon_Qtrs": hz,
            "Ordinary_CDF_Percentile": q_info["ordinary_cdf_pct"],
            "Petroleum_Exceedance_Probability": q_info["exceedance_label"],
            "Selection_Role": q_info["role"],
            "Fault_Transmissibility": row_c["Ft"],
            "Porosity_Multiplier": row_c["Poro"],
            "Permeability_Multiplier": row_c["Perm"],
            "Aquifer_Pore_Volume": row_c["AqPV"],
            "Log_Likelihood_L1": row_c["log_lik_l1"],
            "Log_Likelihood_L2": row_c["log_lik_l2"],
            "Oil_Cum_NRMSE": row_c["oil_cum_nrmse"],
            "Water_Cum_NRMSE": row_c["water_cum_nrmse"],
            "Oil_Rate_NRMSE": row_c["oil_rate_nrmse"],
            "Water_Rate_NRMSE": row_c["water_rate_nrmse"],
            "Terminal_Oil_Rel_Err": row_c["terminal_oil_err"],
            "Ranking_80Q_Oil_EUR_STB": fc_80q_rank["final_oil_cum"],
            "Deliverable_Oil_Cum_Final_STB": fc_deliv["final_oil_cum"],
            "Selection_Objective": "Strategy C: Quality-Filtered Conditional Quantiles",
            "Probabilistic_Interpretation": (
                "Selected from conditional distribution p(theta | y, A) constrained by "
                "engineering quality tolerances (NRMSE_oil <= 0.015, NRMSE_wat <= 0.080, RelErr_term <= 0.010). "
                f"Represents {q_info['ordinary_cdf_pct']}th ordinary percentile of 80-quarter recovery "
                f"({q_info['exceedance_label']})."
            ),
        })

        forecast_rows.append({
            "Case_Sheet": case_id,
            "Designation": designation,
            "Forecast_Horizon_Qtrs": hz,
            "Oil_Cum_Final_STB": fc_deliv["final_oil_cum"],
            "Gas_Cum_Final_MSCF": fc_deliv["final_gas_cum"],
            "Water_Cum_Final_STB": fc_deliv["final_water_cum"],
            "Oil_Incremental_STB": fc_deliv["oil_increment"],
            "Terminal_Oil_Rate_STBD": fc_deliv["terminal_oil_rate"],
            "Terminal_Water_Rate_STBD": fc_deliv["terminal_water_rate"],
            "Ranking_80Q_EUR_STB": fc_80q_rank["final_oil_cum"],
        })

    df_selected_spec = pd.DataFrame(spec_rows)
    df_selected_fc = pd.DataFrame(forecast_rows)

    df_selected_spec.to_csv(out_dir / "selected_10_cases_specification_v3.csv", index=False)
    df_selected_fc.to_csv(out_dir / "selected_10_cases_forecast_summary_v3.csv", index=False)

    print("\nSaved Authoritative Iteration 3 Selected 10 Cases:")
    print(df_selected_spec[["Case_Sheet", "Designation", "Fault_Transmissibility", "Porosity_Multiplier", "Permeability_Multiplier", "Aquifer_Pore_Volume", "Log_Likelihood_L1", "Oil_Cum_NRMSE", "Water_Cum_NRMSE", "Ranking_80Q_Oil_EUR_STB"]].to_string())

    return {
        "spec_df": df_selected_spec,
        "fc_df": df_selected_fc,
        "strat_comp_df": df_strat_comp,
    }


if __name__ == "__main__":
    run_scenario_selection_v3_experiment()
