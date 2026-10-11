"""
Module: quantile_stability_v4.py
Research Iteration 4 — Phase 2: Quantile Stability & Candidate Pool Sensitivity.

Objectives:
1. Evaluates 80 vs 300 vs 600 posterior candidates under frozen Blend_Phase_Specific_Simplex.
2. Quantifies P10, P50, P90 EUR stability across candidate pool sizes.
3. Computes 1000-sample block-bootstrap 95% confidence intervals and standard errors.
4. Identifies physical drivers of narrow EUR spread (anchor conditioning & reservoir physics).
5. Exports:
   - outputs/research/history_matching_iteration_4/tables/quantile_stability_v4.csv
   - outputs/research/history_matching_iteration_4/tables/candidate_evaluations_600.csv
   - outputs/research/history_matching_iteration_4/candidate_trajectories_600.npz
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr

from ensemble.data import load_uncertainty, TRAIN_CASES, PARAMS, resolve_shared_dir
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
)
from history_matching_v2.emulator_revalidation_v2 import HistoricalSVDGPEmulatorV2
from history_matching_v4.history_matching_likelihood_v4 import (
    HistoryMatchingLikelihoodV4,
    PARAM_BOUNDS_INFERENCE,
)
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


def run_quantile_stability_experiment(
    n_total_candidates: int = 600,
    pool_sizes: Tuple[int, ...] = (80, 300, 600),
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("=" * 80)
    print("RESEARCH ITERATION 4 — PHASE 2: QUANTILE STABILITY & POOL SENSITIVITY")
    print("=" * 80)

    base_dir = Path("/Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp")
    tables_dir = base_dir / "outputs/research/history_matching_iteration_4/tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

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
    lik_l1 = HistoryMatchingLikelihoodV4(emu, fh, rho_ar1=0.80)
    lik_l2 = HistoryMatchingLikelihoodV4(emu, fh, rho_ar1=0.80)

    # Load frozen forecasting models
    print("Loading and preparing frozen forecasting models...")
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

    # Load pooled posterior draws
    draws_l1 = np.load(base_dir / "outputs/research/history_matching_iteration_3/bayesian_posterior_draws_l1.npy")
    print(f"Loaded {len(draws_l1)} MCMC posterior draws.")

    # Select 600 candidates evenly across the chain draws
    step = len(draws_l1) / n_total_candidates
    cand_indices = [int(i * step) for i in range(n_total_candidates)]
    candidates = draws_l1[cand_indices]

    print(f"Evaluating 80-quarter forecasts for {n_total_candidates} candidates using Parallel(n_jobs=4)...")
    t0 = time.time()
    forecast_results = Parallel(n_jobs=4, batch_size=4)(
        delayed(forecast_candidate_realization)(th, models, weights, field_anchor, 80)
        for th in candidates
    )
    t1 = time.time()
    print(f"Completed {n_total_candidates} forecasts in {t1 - t0:.2f}s ({(t1 - t0)/n_total_candidates:.3f}s per candidate).")

    # Evaluate historical match quality metrics for all candidates
    print("Evaluating historical match quality metrics...")
    records = []
    traj_oil_cum = []
    traj_wat_cum = []
    traj_oil_rate = []
    traj_wat_rate = []

    for idx, (th, fc) in enumerate(zip(candidates, forecast_results)):
        comp_l1 = lik_l1.evaluate_log_likelihood(th, model="L1", return_components=True)
        comp_l2 = lik_l2.evaluate_log_likelihood(th, model="L2", return_components=True)

        pass_oil = comp_l1["nrmse_Qo"] <= 0.015
        pass_wat = comp_l1["nrmse_Qw"] <= 0.080
        pass_term = comp_l1["terminal_oil_err"] <= 0.010
        passes_quality = bool(pass_oil and pass_wat and pass_term)

        records.append({
            "cand_id": idx,
            "Ft": float(th[0]),
            "Poro": float(th[1]),
            "Perm": float(th[2]),
            "AqPV": float(th[3]),
            "log_lik_l1": comp_l1["log_likelihood"],
            "log_lik_l2": comp_l2["log_likelihood"],
            "oil_cum_nrmse": comp_l1["nrmse_Qo"],
            "water_cum_nrmse": comp_l1["nrmse_Qw"],
            "oil_rate_nrmse": comp_l1["nrmse_qo"],
            "water_rate_nrmse": comp_l1["nrmse_qw"],
            "terminal_oil_err": comp_l1["terminal_oil_err"],
            "terminal_wat_err": comp_l1["terminal_wat_err"],
            "frozen_ensemble_80q_eur": fc["final_oil_cum"],
            "frozen_ensemble_80q_water": fc["final_water_cum"],
            "frozen_ensemble_80q_gas": fc["final_gas_cum"],
            "oil_increment_80q": fc["oil_increment"],
            "water_increment_80q": fc["water_increment"],
            "terminal_oil_rate_80q": fc["terminal_oil_rate"],
            "terminal_water_rate_80q": fc["terminal_water_rate"],
            "passes_quality_filter": passes_quality,
        })
        traj_oil_cum.append(fc["oil_cum_trajectory"])
        traj_wat_cum.append(fc["water_cum_trajectory"])
        traj_oil_rate.append(fc["oil_rate_trajectory"])
        traj_wat_rate.append(fc["water_rate_trajectory"])

    df_candidates = pd.DataFrame(records)
    cands_path = tables_dir / "candidate_evaluations_600.csv"
    df_candidates.to_csv(cands_path, index=False)
    print(f"Saved all candidate evaluations to {cands_path}")

    # Save trajectories
    npz_path = tables_dir.parent / "candidate_trajectories_600.npz"
    np.savez_compressed(
        npz_path,
        oil_cum=np.array(traj_oil_cum),
        water_cum=np.array(traj_wat_cum),
        oil_rate=np.array(traj_oil_rate),
        water_rate=np.array(traj_wat_rate),
        candidates=candidates,
    )
    print(f"Saved trajectory arrays to {npz_path}")

    # ==========================================================================
    # QUANTILE STABILITY & BOOTSTRAP ANALYSIS
    # ==========================================================================
    rng = np.random.default_rng(seed)
    stability_records = []

    for n_pool in pool_sizes:
        # Subsample evenly from the 600 candidates
        sub_indices = [int(i * (n_total_candidates / n_pool)) for i in range(n_pool)]
        df_sub = df_candidates.iloc[sub_indices]
        eur_vals = df_sub["frozen_ensemble_80q_eur"].to_numpy()

        # Petroleum exceedance quantiles:
        # P10 High Recovery = 90th percentile
        # P50 Median = 50th percentile
        # P90 Low Recovery = 10th percentile
        p10_point = float(np.percentile(eur_vals, 90.0))
        p50_point = float(np.percentile(eur_vals, 50.0))
        p90_point = float(np.percentile(eur_vals, 10.0))
        spread_point = p10_point - p90_point
        mean_point = float(np.mean(eur_vals))
        std_point = float(np.std(eur_vals))

        # Bootstrap resampling
        boot_p10 = []
        boot_p50 = []
        boot_p90 = []
        boot_spread = []
        boot_mean = []

        for _ in range(n_bootstrap):
            boot_sample = rng.choice(eur_vals, size=n_pool, replace=True)
            b_p10 = np.percentile(boot_sample, 90.0)
            b_p50 = np.percentile(boot_sample, 50.0)
            b_p90 = np.percentile(boot_sample, 10.0)
            boot_p10.append(b_p10)
            boot_p50.append(b_p50)
            boot_p90.append(b_p90)
            boot_spread.append(b_p10 - b_p90)
            boot_mean.append(np.mean(boot_sample))

        stability_records.append({
            "candidate_pool_size": n_pool,
            "p10_eur_stb": p10_point,
            "p10_boot_mean_stb": float(np.mean(boot_p10)),
            "p10_boot_se_stb": float(np.std(boot_p10)),
            "p10_ci_lower_95": float(np.percentile(boot_p10, 2.5)),
            "p10_ci_upper_95": float(np.percentile(boot_p10, 97.5)),
            "p50_eur_stb": p50_point,
            "p50_boot_mean_stb": float(np.mean(boot_p50)),
            "p50_boot_se_stb": float(np.std(boot_p50)),
            "p50_ci_lower_95": float(np.percentile(boot_p50, 2.5)),
            "p50_ci_upper_95": float(np.percentile(boot_p50, 97.5)),
            "p90_eur_stb": p90_point,
            "p90_boot_mean_stb": float(np.mean(boot_p90)),
            "p90_boot_se_stb": float(np.std(boot_p90)),
            "p90_ci_lower_95": float(np.percentile(boot_p90, 2.5)),
            "p90_ci_upper_95": float(np.percentile(boot_p90, 97.5)),
            "eur_spread_stb": spread_point,
            "spread_boot_mean_stb": float(np.mean(boot_spread)),
            "spread_boot_se_stb": float(np.std(boot_spread)),
            "spread_ci_lower_95": float(np.percentile(boot_spread, 2.5)),
            "spread_ci_upper_95": float(np.percentile(boot_spread, 97.5)),
            "eur_mean_stb": mean_point,
            "eur_std_stb": std_point,
        })

    df_stability = pd.DataFrame(stability_records)
    stab_path = tables_dir / "quantile_stability_v4.csv"
    df_stability.to_csv(stab_path, index=False)
    print(f"\nSaved quantile stability table to {stab_path}")

    # ==========================================================================
    # PHYSICAL DRIVERS OF NARROW EUR SPREAD
    # ==========================================================================
    print("\n--- Physical Sensitivity Analysis (N=600 candidates) ---")
    eur_600 = df_candidates["frozen_ensemble_80q_eur"]
    corrs = {}
    for p in ["Ft", "Poro", "Perm", "AqPV"]:
        r_p, _ = pearsonr(df_candidates[p], eur_600)
        rho_s, _ = spearmanr(df_candidates[p], eur_600)
        corrs[p] = {"pearson": r_p, "spearman": rho_s}
        print(f"EUR vs {p:6s}: Pearson r={r_p:+.4f}, Spearman rho={rho_s:+.4f}")

    # Relative spread vs total EUR
    base_eur = df_stability.loc[df_stability["candidate_pool_size"] == 600, "p50_eur_stb"].iloc[0]
    spread_600 = df_stability.loc[df_stability["candidate_pool_size"] == 600, "eur_spread_stb"].iloc[0]
    inc_oil_mean = df_candidates["oil_increment_80q"].mean()
    print(f"Total P50 EUR: {base_eur:,.0f} STB")
    print(f"Historical Production at 2008: {field_anchor['Qo']:,.0f} STB ({field_anchor['Qo']/base_eur*100:.1f}% of EUR)")
    print(f"Forecast Incremental Oil: {inc_oil_mean:,.0f} STB ({inc_oil_mean/base_eur*100:.1f}% of EUR)")
    print(f"EUR Spread (P10 - P90): {spread_600:,.0f} STB ({spread_600/base_eur*100:.2f}% of total EUR, {spread_600/inc_oil_mean*100:.2f}% of incremental oil)")

    return df_stability, df_candidates


if __name__ == "__main__":
    df_stab, df_cands = run_quantile_stability_experiment()
    print("\nQuantile Stability Table:")
    print(df_stab.to_string(index=False))
