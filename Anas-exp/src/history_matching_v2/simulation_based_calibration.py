"""
Module: simulation_based_calibration.py
Implements Section 5 of Research Iteration 2:
Simulation-Based Calibration (SBC) and Empirical Credible Interval Coverage Tests.
Methodology:
1. Talts et al. (2018/2020) Simulation-Based Calibration principles:
   - Evaluates rank uniformity of posterior draws conditioned on synthetic data generated from prior.
2. Pseudo-field Bayesian recovery across held-out Cases 71-85:
   - Evaluates empirical coverage of nominal 50%, 80%, and 90% credible intervals.
   - Distinguishes surrogate approximation error from inference algorithm consistency.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy.stats import chisquare

from ensemble.data import load_uncertainty, TRAIN_CASES, VAL_CASES, PARAMS
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
)
from history_matching_v2.emulator_revalidation_v2 import HistoricalSVDGPEmulatorV2
from history_matching_v2.history_matching_likelihood_v2 import HistoryMatchingLikelihoodV2
from history_matching_v2.bayesian_mcmc_v2 import (
    run_adaptive_metropolis_chain,
    PARAM_BOUNDS,
    from_unit_hypercube,
    to_unit_hypercube,
)


def run_sbc_and_coverage_experiments(
    n_sbc_sims: int = 15,
    mcmc_draws_per_sim: int = 500,
    burn_in: int = 150,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes SBC rank uniformity tests and pseudo-field credible interval coverage tests.
    """
    print("=" * 80)
    print("RESEARCH ITERATION 2 — SECTION 5: SIMULATION-BASED CALIBRATION & COVERAGE")
    print("=" * 80)

    unc_df = load_uncertainty()
    q_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(q_df)
    fh = pd.read_parquet("../data/field_history.parquet")
    dates = pd.to_datetime(case_dict[1]["date"].to_numpy())

    X_train = unc_df.loc[TRAIN_CASES, PARAMS].to_numpy()
    Y_oil_tr = np.array([case_dict[c]["oil_cum_stb"].to_numpy() for c in TRAIN_CASES])
    Y_wat_tr = np.array([case_dict[c]["water_cum_stb"].to_numpy() for c in TRAIN_CASES])

    emu = HistoricalSVDGPEmulatorV2(n_components=4, random_state=42)
    emu.fit(X_train, Y_oil_tr, Y_wat_tr, dates=dates)

    # 1. Pseudo-Field Empirical Credible Interval Coverage across Held-Out Cases
    print("\nEvaluating Empirical Credible Interval Coverage on Held-Out Cases 71-85...")
    coverage_cases = [71, 72, 73, 75, 76, 78, 80, 81, 83, 85]  # 10 representative held-out cases
    case_coverage_records = []

    for c in coverage_cases:
        true_th = unc_df.loc[c, PARAMS].to_numpy()
        synth_fh = pd.DataFrame({
            "date": fh["date"],
            "ramp_oil_cum_stb": case_dict[c]["oil_cum_stb"].to_numpy(),
            "ramp_water_cum_stb": case_dict[c]["water_cum_stb"].to_numpy(),
            "ramp_oil_rate_stbd": case_dict[c]["oil_rate_stbd"].to_numpy(),
            "ramp_water_rate_stbd": case_dict[c]["water_rate_stbd"].to_numpy(),
        })
        lik_synth = HistoryMatchingLikelihoodV2(emu, synth_fh, rho_ar1=0.80, obs_noise_rel_oil=0.05, obs_noise_rel_wat=0.05)

        # Run fast MCMC for recovery
        u_init = np.array([0.5, 0.5, 0.5, 0.5])
        res_mcmc = run_adaptive_metropolis_chain(
            lik_synth,
            u_init=u_init,
            n_iterations=mcmc_draws_per_sim,
            burn_in=burn_in,
            model="L1",
            chain_id=1,
            seed=c,
        )
        post_draws = res_mcmc["draws"]  # (N_draws, 4)

        # Check containment in 50%, 80%, 90% credible intervals per parameter
        record = {"case_id": c}
        param_names = ["Ft", "Poro", "Perm", "AqPV"]
        for p_idx, p_name in enumerate(param_names):
            p_draws = post_draws[:, p_idx]
            q05, q10, q25, q75, q90, q95 = np.percentile(p_draws, [5.0, 10.0, 25.0, 75.0, 90.0, 95.0])
            val = true_th[p_idx]
            record[f"{p_name}_in_50"] = int(q25 <= val <= q75)
            record[f"{p_name}_in_80"] = int(q10 <= val <= q90)
            record[f"{p_name}_in_90"] = int(q05 <= val <= q95)
        case_coverage_records.append(record)

    df_coverage = pd.DataFrame(case_coverage_records)

    # Aggregate coverage statistics
    param_names = ["Ft", "Poro", "Perm", "AqPV"]
    agg_rows = []
    for p_name in param_names:
        cov_50 = df_coverage[f"{p_name}_in_50"].mean()
        cov_80 = df_coverage[f"{p_name}_in_80"].mean()
        cov_90 = df_coverage[f"{p_name}_in_90"].mean()
        agg_rows.append({
            "parameter": p_name,
            "nominal_50_empirical_coverage": cov_50,
            "nominal_80_empirical_coverage": cov_80,
            "nominal_90_empirical_coverage": cov_90,
            "n_test_cases": len(coverage_cases),
        })
    df_agg_cov = pd.DataFrame(agg_rows)

    out_dir = Path("outputs/research/history_matching_iteration_2/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_agg_cov.to_csv(out_dir / "pseudo_field_credible_interval_coverage.csv", index=False)
    print(f"\nSaved pseudo-field credible interval coverage summary:")
    print(df_agg_cov.to_string())

    # 2. Simulation-Based Calibration (Talts et al., 2018) on Generative Model
    print(f"\nRunning SBC on Generative Model ({n_sbc_sims} simulations)...")
    sbc_ranks = {p: [] for p in param_names}
    np.random.seed(42)

    for s_idx in range(n_sbc_sims):
        # Draw prior parameter vector theta_true
        u_true = np.random.uniform(0.05, 0.95, size=4)
        th_true = from_unit_hypercube(u_true)

        # Generate synthetic observation from likelihood model
        pred_true = emu.predict(th_true.reshape(1, -1))
        # Add correlated noise
        noise_o = np.random.multivariate_normal(
            np.zeros(40), (0.05 * lik_synth.norm_qo) ** 2 * np.eye(40)
        )
        noise_w = np.random.multivariate_normal(
            np.zeros(40), (0.05 * lik_synth.norm_qw) ** 2 * np.eye(40)
        )
        synth_qo = np.maximum(pred_true["oil_rate_fwd"][0] + noise_o, 0.0)
        synth_qw = np.maximum(pred_true["water_rate_fwd"][0] + noise_w, 0.0)
        synth_Qo = np.concatenate([[0.0], np.cumsum(synth_qo * lik_synth.days_fwd)])
        synth_Qw = np.concatenate([[0.0], np.cumsum(synth_qw * lik_synth.days_fwd)])

        synth_df = pd.DataFrame({
            "date": fh["date"],
            "ramp_oil_cum_stb": synth_Qo,
            "ramp_water_cum_stb": synth_Qw,
            "ramp_oil_rate_stbd": np.append(synth_qo, synth_qo[-1]),
            "ramp_water_rate_stbd": np.append(synth_qw, synth_qw[-1]),
        })
        lik_sbc = HistoryMatchingLikelihoodV2(emu, synth_df, rho_ar1=0.80, obs_noise_rel_oil=0.05, obs_noise_rel_wat=0.05)

        # Run posterior sampling
        u_init = np.random.uniform(0.2, 0.8, size=4)
        res_sbc = run_adaptive_metropolis_chain(
            lik_sbc, u_init=u_init, n_iterations=mcmc_draws_per_sim, burn_in=burn_in, model="L1", seed=200 + s_idx
        )
        draws = res_sbc["draws"]

        # Compute rank of true parameter among posterior draws
        L = len(draws)
        for p_idx, p_name in enumerate(param_names):
            rank_val = int(np.sum(draws[:, p_idx] < th_true[p_idx]))
            sbc_ranks[p_name].append(rank_val / L)

    # Chi-squared test for uniformity across bins [0..0.2, 0.2..0.4, ...]
    sbc_rows = []
    for p_name in param_names:
        r_arr = np.array(sbc_ranks[p_name])
        hist, _ = np.histogram(r_arr, bins=5, range=(0.0, 1.0))
        # Uniform expected frequency
        exp_freq = np.full(5, len(r_arr) / 5.0)
        chi2_stat, p_val = chisquare(hist, exp_freq)
        sbc_rows.append({
            "parameter": p_name,
            "mean_rank_percentile": float(np.mean(r_arr)),
            "std_rank_percentile": float(np.std(r_arr)),
            "chi2_uniformity_stat": float(chi2_stat),
            "p_value_uniformity": float(p_val),
            "is_calibrated": bool(p_val >= 0.05),
        })

    df_sbc = pd.DataFrame(sbc_rows)
    df_sbc.to_csv(out_dir / "simulation_based_calibration_results.csv", index=False)
    print("\nSaved Simulation-Based Calibration (Talts et al., 2018) summary:")
    print(df_sbc.to_string())

    return df_agg_cov, df_sbc


if __name__ == "__main__":
    run_sbc_and_coverage_experiments()
