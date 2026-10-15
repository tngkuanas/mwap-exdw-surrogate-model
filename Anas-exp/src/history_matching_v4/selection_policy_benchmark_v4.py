"""
Module: selection_policy_benchmark_v4.py
Research Iteration 4 — Phase 3: Selection Policy Optimization & Benchmark.

Objectives:
1. Benchmarks 3 distinct selection policies across 600 posterior candidates:
   - Strategy A: Ten Best Historical Matches (Deterministic MAP / GLS fidelity).
   - Strategy B: Unconstrained Posterior EUR Quantiles (Broad probabilistic coverage).
   - Strategy C: Quality-Filtered Conditional Quantiles (Constrained reserves coverage).
2. Evaluates history-match error vs forecast diversity trade-offs.
3. Formulates primary recommendation and fallback based on challenge instruction ("best 10 forecast cases").
4. Exports:
   - outputs/research/history_matching_iteration_4/tables/selection_policy_benchmark_v4.csv
   - outputs/research/history_matching_iteration_4/tables/selected_scenarios_v4.csv
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

CANONICAL_PARAMS = [
    "Fault_Transmissibility",
    "Porosity_Multiplier",
    "Permeability_Multiplier",
    "Aquifer_Pore_Volume",
]

TARGET_QUANTILES = [
    {"ordinary_cdf_pct": 90.0, "exceedance_label": "P10 High-Recovery", "case_id": 1, "role": "P10 High-Recovery Upper Bound"},
    {"ordinary_cdf_pct": 85.0, "exceedance_label": "P15", "case_id": 2, "role": "Upper Quartile Dispersion"},
    {"ordinary_cdf_pct": 80.0, "exceedance_label": "P20", "case_id": 3, "role": "Upper Quartile Dispersion"},
    {"ordinary_cdf_pct": 75.0, "exceedance_label": "P25", "case_id": 4, "role": "Moderate High Scenario"},
    {"ordinary_cdf_pct": 65.0, "exceedance_label": "P35", "case_id": 5, "role": "Moderate High Scenario"},
    {"ordinary_cdf_pct": 50.0, "exceedance_label": "P50 Median Base Case", "case_id": 6, "role": "P50 Central Expected Forecast"},
    {"ordinary_cdf_pct": 40.0, "exceedance_label": "P60", "case_id": 7, "role": "Moderate Low Scenario"},
    {"ordinary_cdf_pct": 30.0, "exceedance_label": "P70", "case_id": 8, "role": "Moderate Low Scenario"},
    {"ordinary_cdf_pct": 20.0, "exceedance_label": "P80", "case_id": 9, "role": "Lower Quartile Dispersion"},
    {"ordinary_cdf_pct": 10.0, "exceedance_label": "P90 Low-Recovery", "case_id": 10, "role": "P90 Low-Recovery Lower Bound"},
]


def run_selection_policy_benchmark() -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("=" * 80)
    print("RESEARCH ITERATION 4 — PHASE 3: SELECTION POLICY BENCHMARK")
    print("=" * 80)

    base_dir = Path("/Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp")
    tables_dir = base_dir / "outputs/research/history_matching_iteration_4/tables"
    cands_path = tables_dir / "candidate_evaluations_600.csv"

    if not cands_path.exists():
        raise FileNotFoundError(f"Missing {cands_path}. Run Phase 2 quantile_stability_v4.py first!")

    df_cands = pd.read_csv(cands_path)
    print(f"Loaded {len(df_cands)} evaluated candidates.")

    # ==========================================================================
    # STRATEGY A: TEN BEST HISTORICAL MATCHES
    # ==========================================================================
    top10_a = df_cands.sort_values("log_lik_l1", ascending=False).head(10).copy().reset_index(drop=True)
    top10_a["strategy"] = "Strategy A: Ten Best Matches"
    top10_a["case_id"] = [f"Case {i+1}" for i in range(10)]
    top10_a["target_role"] = [f"Best Match Rank {i+1}" for i in range(10)]

    # ==========================================================================
    # STRATEGY B: UNCONSTRAINED POSTERIOR EUR QUANTILES
    # ==========================================================================
    strat_b_rows = []
    eur_all = df_cands["frozen_ensemble_80q_eur"].to_numpy()
    for q_spec in TARGET_QUANTILES:
        target_val = float(np.percentile(eur_all, q_spec["ordinary_cdf_pct"]))
        diffs = (df_cands["frozen_ensemble_80q_eur"] - target_val).abs()
        best_cand_idx = diffs.idxmin()
        row = df_cands.loc[best_cand_idx].copy()
        row["strategy"] = "Strategy B: Unconstrained Quantiles"
        row["case_id"] = f"Case {q_spec['case_id']}"
        row["target_role"] = f"{q_spec['exceedance_label']} ({q_spec['role']})"
        strat_b_rows.append(row)
    df_strat_b = pd.DataFrame(strat_b_rows).reset_index(drop=True)

    # ==========================================================================
    # STRATEGY C: QUALITY-FILTERED CONDITIONAL QUANTILES
    # ==========================================================================
    df_accepted = df_cands[df_cands["passes_quality_filter"]].copy()
    print(f"Accepted {len(df_accepted)} / {len(df_cands)} candidates passing quality filter ({len(df_accepted)/len(df_cands)*100:.1f}%).")

    strat_c_rows = []
    eur_acc = df_accepted["frozen_ensemble_80q_eur"].to_numpy()
    for q_spec in TARGET_QUANTILES:
        target_val = float(np.percentile(eur_acc, q_spec["ordinary_cdf_pct"]))
        diffs = (df_accepted["frozen_ensemble_80q_eur"] - target_val).abs()
        best_cand_idx = diffs.idxmin()
        row = df_accepted.loc[best_cand_idx].copy()
        row["strategy"] = "Strategy C: Quality-Filtered Quantiles"
        row["case_id"] = f"Case {q_spec['case_id']}"
        row["target_role"] = f"{q_spec['exceedance_label']} ({q_spec['role']})"
        strat_c_rows.append(row)
    df_strat_c = pd.DataFrame(strat_c_rows).reset_index(drop=True)

    # ==========================================================================
    # COMPARATIVE SUMMARY TABLE
    # ==========================================================================
    summary_rows = []
    for s_name, df_s in [
        ("Strategy A (Ten Best Matches)", top10_a),
        ("Strategy B (Unconstrained Quantiles)", df_strat_b),
        ("Strategy C (Quality-Filtered Quantiles)", df_strat_c),
    ]:
        summary_rows.append({
            "selection_strategy": s_name,
            "mean_log_lik_l1": float(df_s["log_lik_l1"].mean()),
            "mean_log_lik_l2": float(df_s["log_lik_l2"].mean()),
            "mean_oil_cum_nrmse": float(df_s["oil_cum_nrmse"].mean()),
            "mean_water_cum_nrmse": float(df_s["water_cum_nrmse"].mean()),
            "mean_terminal_oil_err": float(df_s["terminal_oil_err"].mean()),
            "eur_80q_p10_stb": float(df_s["frozen_ensemble_80q_eur"].max()),
            "eur_80q_p50_stb": float(df_s["frozen_ensemble_80q_eur"].median()),
            "eur_80q_p90_stb": float(df_s["frozen_ensemble_80q_eur"].min()),
            "eur_80q_spread_stb": float(df_s["frozen_ensemble_80q_eur"].max() - df_s["frozen_ensemble_80q_eur"].min()),
            "poro_range_min": float(df_s["Poro"].min()),
            "poro_range_max": float(df_s["Poro"].max()),
            "perm_range_min": float(df_s["Perm"].min()),
            "perm_range_max": float(df_s["Perm"].max()),
            "aqpv_range_min": float(df_s["AqPV"].min()),
            "aqpv_range_max": float(df_s["AqPV"].max()),
            "quality_pass_count": int(df_s["passes_quality_filter"].sum()),
        })

    df_summary = pd.DataFrame(summary_rows)
    sum_path = tables_dir / "selection_policy_benchmark_v4.csv"
    df_summary.to_csv(sum_path, index=False)
    print(f"\nSaved selection policy benchmark to {sum_path}")

    # Combine all selected scenarios
    df_all_selected = pd.concat([top10_a, df_strat_b, df_strat_c], ignore_index=True)
    scen_path = tables_dir / "selected_scenarios_v4.csv"
    df_all_selected.to_csv(scen_path, index=False)
    print(f"Saved selected scenarios details to {scen_path}")

    return df_summary, df_all_selected


if __name__ == "__main__":
    df_sum, df_scen = run_selection_policy_benchmark()
    print("\nSelection Policy Benchmark Summary:")
    print(df_sum.to_string(index=False))
