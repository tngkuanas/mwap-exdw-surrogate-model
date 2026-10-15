"""
Module: simulation_based_calibration_v4.py
Research Iteration 4 — Phase 1: Fixed vs State-Dependent Scale SBC Benchmark.

Methodological Foundation:
Talts, Betancourt, Simpson, Vehtari & Gelman (2018/2020),
'Validating Bayesian Inference Algorithms with Simulation-Based Calibration'.

Key Objectives:
1. Analytical Gaussian Bayesian Toy Model Benchmark (proves code correctness).
2. Experiment 1A: Fixed-Scale SBC (N=100)
   - Generative model and likelihood evaluate the exact same fixed AR(1) covariance.
   - Exact mathematical covariance matching eliminates scale divergence.
3. Experiment 1B: State-Dependent Scale SBC (N=100)
   - Realization-dependent noise scale proportional to rate magnitude.
   - Evaluates empirical heuristic behavior under relative observation noise.
4. Exact Clopper-Pearson 95% Confidence Intervals:
   - Evaluates nominal 50%, 80%, and 90% credible intervals.
   - Tests whether empirical coverage falls within the exact binomial confidence interval.
5. Saves comparative results to outputs/research/history_matching_iteration_4/tables/sbc_coverage_comparison_v4.csv.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy.stats import chisquare, kstest
from scipy.linalg import cholesky

from ensemble.data import load_uncertainty, TRAIN_CASES, PARAMS, resolve_shared_dir
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
)
from history_matching_v2.emulator_revalidation_v2 import HistoricalSVDGPEmulatorV2
from history_matching_v4.history_matching_likelihood_v4 import (
    HistoryMatchingLikelihoodV4,
    build_ar1_covariance,
    PARAM_BOUNDS_INFERENCE,
    from_unit_hypercube_v4,
    to_unit_hypercube_v4,
)
from history_matching_v4.bayesian_mcmc_v4 import (
    run_adaptive_metropolis_chain_v4,
    clopper_pearson_ci,
)

CANONICAL_PARAMS = [
    "Fault_Transmissibility",
    "Porosity_Multiplier",
    "Permeability_Multiplier",
    "Aquifer_Pore_Volume",
]
PARAM_ABBR = ["Ft", "Poro", "Perm", "AqPV"]


# ==============================================================================
# 1. ANALYTICAL GAUSSIAN BAYESIAN TOY MODEL BENCHMARK
# ==============================================================================

def run_analytical_toy_sbc(
    n_sims: int = 500,
    n_obs: int = 5,
    mu_0: float = 0.0,
    sigma_0: float = 1.0,
    sigma_obs: float = 0.5,
    l_draws: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes SBC on an analytically tractable 1D conjugate Gaussian model:
      Prior: theta ~ N(mu_0, sigma_0^2)
      Likelihood: y_i ~ N(theta, sigma_obs^2), i = 1..n_obs
      Exact Posterior: theta | y ~ N(mu_post, sigma_post^2)
    """
    rng = np.random.default_rng(seed)
    ranks_exact = []
    ranks_mcmc = []

    # Analytical posterior variance
    tau_0 = 1.0 / (sigma_0 ** 2)
    tau_obs = n_obs / (sigma_obs ** 2)
    tau_post = tau_0 + tau_obs
    sigma_post = 1.0 / np.sqrt(tau_post)

    for _ in range(n_sims):
        theta_true = rng.normal(mu_0, sigma_0)
        y = rng.normal(theta_true, sigma_obs, size=n_obs)
        y_bar = np.mean(y)

        # Exact posterior
        mu_post = (tau_0 * mu_0 + tau_obs * y_bar) / tau_post
        draws_exact = rng.normal(mu_post, sigma_post, size=l_draws)
        rank_exact = int(np.sum(draws_exact < theta_true))
        ranks_exact.append(rank_exact / l_draws)

        # MCMC sampling
        mcmc_draws = []
        th_curr = mu_0
        for _step in range(l_draws + 200):
            th_prop = th_curr + rng.normal(0.0, sigma_post)
            lp_curr = -0.5 * tau_0 * (th_curr - mu_0) ** 2 - 0.5 * tau_obs * (th_curr - y_bar) ** 2
            lp_prop = -0.5 * tau_0 * (th_prop - mu_0) ** 2 - 0.5 * tau_obs * (th_prop - y_bar) ** 2
            if np.log(rng.uniform(0.0, 1.0)) < (lp_prop - lp_curr):
                th_curr = th_prop
            mcmc_draws.append(th_curr)
        post_mcmc = np.array(mcmc_draws[200:])
        rank_mcmc = int(np.sum(post_mcmc < theta_true))
        ranks_mcmc.append(rank_mcmc / l_draws)

    ranks_exact = np.array(ranks_exact)
    ranks_mcmc = np.array(ranks_mcmc)

    hist_exact, _ = np.histogram(ranks_exact, bins=10, range=(0.0, 1.0))
    hist_mcmc, _ = np.histogram(ranks_mcmc, bins=10, range=(0.0, 1.0))
    exp_count = n_sims / 10.0

    chi2_exact, p_exact = chisquare(hist_exact, np.full(10, exp_count))
    chi2_mcmc, p_mcmc = chisquare(hist_mcmc, np.full(10, exp_count))

    ks_exact = kstest(ranks_exact, "uniform")
    ks_mcmc = kstest(ranks_mcmc, "uniform")

    return {
        "n_sims": n_sims,
        "exact_chi2_p": float(p_exact),
        "exact_ks_p": float(ks_exact.pvalue),
        "mcmc_chi2_p": float(p_mcmc),
        "mcmc_ks_p": float(ks_mcmc.pvalue),
        "is_exact_calibrated": bool(p_exact >= 0.05 and ks_exact.pvalue >= 0.05),
        "is_mcmc_calibrated": bool(p_mcmc >= 0.05 and ks_mcmc.pvalue >= 0.05),
    }


# ==============================================================================
# 2. RESERVOIR SIMULATION-BASED CALIBRATION (EXPERIMENTS 1A & 1B)
# ==============================================================================

def run_reservoir_sbc_v4(
    experiment_type: str = "fixed_scale",  # "fixed_scale" or "state_dependent"
    n_sbc_sims: int = 100,
    mcmc_draws: int = 600,
    burn_in: int = 200,
    rho_ar1: float = 0.80,
    obs_noise_rel: float = 0.05,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes SBC under either fixed-scale or state-dependent noise configuration.
    """
    print(f"\n--- Running Reservoir SBC ({experiment_type.upper()}) with {n_sbc_sims} replications ---")
    base_dir = Path("/Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp")
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

    # Reference likelihood for scale normalizers and day intervals
    lik_ref = HistoryMatchingLikelihoodV4(emu, fh, rho_ar1=rho_ar1)
    norm_qo_field = lik_ref.norm_qo
    norm_qw_field = lik_ref.norm_qw
    fixed_sigma_o = obs_noise_rel * norm_qo_field
    fixed_sigma_w = obs_noise_rel * norm_qw_field
    days_fwd = lik_ref.days_fwd

    # Fixed covariance for 1A
    cov_fixed_o = build_ar1_covariance(40, fixed_sigma_o, rho_ar1)
    cov_fixed_w = build_ar1_covariance(40, fixed_sigma_w, rho_ar1)
    L_fixed_o = cholesky(cov_fixed_o, lower=True)
    L_fixed_w = cholesky(cov_fixed_w, lower=True)

    ranks_dict = {p: [] for p in CANONICAL_PARAMS}
    cov_50_dict = {p: [] for p in CANONICAL_PARAMS}
    cov_80_dict = {p: [] for p in CANONICAL_PARAMS}
    cov_90_dict = {p: [] for p in CANONICAL_PARAMS}

    for s_idx in range(n_sbc_sims):
        sim_seed = seed + 1000 + s_idx
        sim_rng = np.random.default_rng(sim_seed)

        # 1. Draw theta_true from prior Uniform(0, 1)^4
        u_true = sim_rng.uniform(0.0, 1.0, size=4)
        th_true = from_unit_hypercube_v4(u_true)

        # 2. Generative Forward Model
        pred_true = emu.predict(th_true.reshape(1, -1))
        qo_true = pred_true["oil_rate_fwd"][0]
        qw_true = pred_true["water_rate_fwd"][0]

        if experiment_type == "fixed_scale":
            z_o = sim_rng.standard_normal(40)
            z_w = sim_rng.standard_normal(40)
            noise_o = L_fixed_o @ z_o
            noise_w = L_fixed_w @ z_w
            synth_qo = qo_true + noise_o
            synth_qw = qw_true + noise_w
        else:  # state_dependent
            norm_qo_local = float(np.sqrt(np.mean(qo_true ** 2))) + 1e-12
            norm_qw_local = float(np.sqrt(np.mean(qw_true ** 2))) + 1e-12
            cov_local_o = build_ar1_covariance(40, obs_noise_rel * norm_qo_local, rho_ar1)
            cov_local_w = build_ar1_covariance(40, obs_noise_rel * norm_qw_local, rho_ar1)
            L_local_o = cholesky(cov_local_o, lower=True)
            L_local_w = cholesky(cov_local_w, lower=True)
            z_o = sim_rng.standard_normal(40)
            z_w = sim_rng.standard_normal(40)
            noise_o = L_local_o @ z_o
            noise_w = L_local_w @ z_w
            synth_qo = qo_true + noise_o
            synth_qw = qw_true + noise_w

        # Quadrature for cumulative production
        synth_Qo = np.concatenate([[0.0], np.cumsum(synth_qo * days_fwd)])
        synth_Qw = np.concatenate([[0.0], np.cumsum(synth_qw * days_fwd)])

        synth_df = pd.DataFrame({
            "date": fh["date"],
            "ramp_oil_cum_stb": synth_Qo,
            "ramp_water_cum_stb": synth_Qw,
            "ramp_oil_rate_stbd": np.append(synth_qo, synth_qo[-1]),
            "ramp_water_rate_stbd": np.append(synth_qw, synth_qw[-1]),
        })

        # 3. Posterior Inference
        if experiment_type == "fixed_scale":
            lik_synth = HistoryMatchingLikelihoodV4(
                emu,
                synth_df,
                rho_ar1=rho_ar1,
                fixed_sigma_oil=fixed_sigma_o,
                fixed_sigma_wat=fixed_sigma_w,
            )
        else:
            lik_synth = HistoryMatchingLikelihoodV4(
                emu,
                synth_df,
                rho_ar1=rho_ar1,
                obs_noise_rel_oil=obs_noise_rel,
                obs_noise_rel_wat=obs_noise_rel,
            )

        u_init = sim_rng.uniform(0.1, 0.9, size=4)
        res_mcmc = run_adaptive_metropolis_chain_v4(
            lik_synth,
            u_init=u_init,
            n_iterations=mcmc_draws,
            burn_in=burn_in,
            model="L1",
            chain_id=1,
            seed=sim_seed,
        )
        post_draws = res_mcmc["draws"]

        # Evaluate ranks and coverage for each parameter
        for p_idx, p_name in enumerate(CANONICAL_PARAMS):
            true_val = th_true[p_idx]
            samples = post_draws[:, p_idx]
            rank_val = int(np.sum(samples < true_val)) / len(samples)
            ranks_dict[p_name].append(rank_val)

            # Credible intervals
            cov_50_dict[p_name].append(bool(np.percentile(samples, 25.0) <= true_val <= np.percentile(samples, 75.0)))
            cov_80_dict[p_name].append(bool(np.percentile(samples, 10.0) <= true_val <= np.percentile(samples, 90.0)))
            cov_90_dict[p_name].append(bool(np.percentile(samples, 5.0) <= true_val <= np.percentile(samples, 95.0)))

    # Compute statistics and Clopper-Pearson CIs
    records = []
    for p_name in CANONICAL_PARAMS:
        ranks = np.array(ranks_dict[p_name])
        hist, _ = np.histogram(ranks, bins=10, range=(0.0, 1.0))
        chi2_stat, p_chi2 = chisquare(hist, np.full(10, n_sbc_sims / 10.0))
        ks_res = kstest(ranks, "uniform")

        # 50% CI
        k_50 = sum(cov_50_dict[p_name])
        cp_50_low, cp_50_high = clopper_pearson_ci(k_50, n_sbc_sims)
        records.append({
            "experiment": experiment_type,
            "parameter": p_name,
            "nominal_level": 0.50,
            "empirical_coverage": k_50 / n_sbc_sims,
            "cp_ci_lower_95": cp_50_low,
            "cp_ci_upper_95": cp_50_high,
            "nominal_in_ci": bool(cp_50_low <= 0.50 <= cp_50_high),
            "ks_p_value": float(ks_res.pvalue),
            "chi2_p_value": float(p_chi2),
        })

        # 80% CI
        k_80 = sum(cov_80_dict[p_name])
        cp_80_low, cp_80_high = clopper_pearson_ci(k_80, n_sbc_sims)
        records.append({
            "experiment": experiment_type,
            "parameter": p_name,
            "nominal_level": 0.80,
            "empirical_coverage": k_80 / n_sbc_sims,
            "cp_ci_lower_95": cp_80_low,
            "cp_ci_upper_95": cp_80_high,
            "nominal_in_ci": bool(cp_80_low <= 0.80 <= cp_80_high),
            "ks_p_value": float(ks_res.pvalue),
            "chi2_p_value": float(p_chi2),
        })

        # 90% CI
        k_90 = sum(cov_90_dict[p_name])
        cp_90_low, cp_90_high = clopper_pearson_ci(k_90, n_sbc_sims)
        records.append({
            "experiment": experiment_type,
            "parameter": p_name,
            "nominal_level": 0.90,
            "empirical_coverage": k_90 / n_sbc_sims,
            "cp_ci_lower_95": cp_90_low,
            "cp_ci_upper_95": cp_90_high,
            "nominal_in_ci": bool(cp_90_low <= 0.90 <= cp_90_high),
            "ks_p_value": float(ks_res.pvalue),
            "chi2_p_value": float(p_chi2),
        })

    df_out = pd.DataFrame(records)
    return df_out, ranks_dict


def run_sbc_benchmark_suite() -> pd.DataFrame:
    """Executes the full SBC benchmark suite: Toy -> Exp 1A -> Exp 1B."""
    print("=" * 80)
    print("RESEARCH ITERATION 4 — PHASE 1: SBC BENCHMARK SUITE")
    print("=" * 80)

    # 1. Toy benchmark
    toy_res = run_analytical_toy_sbc(n_sims=500, seed=42)
    print(f"Toy Gaussian Model: Exact KS p={toy_res['exact_ks_p']:.4f}, MCMC KS p={toy_res['mcmc_ks_p']:.4f}")
    assert toy_res["is_exact_calibrated"], "Analytical toy model failed exact calibration!"

    # 2. Exp 1A: Fixed scale
    df_1a, ranks_1a = run_reservoir_sbc_v4("fixed_scale", n_sbc_sims=100, seed=42)

    # 3. Exp 1B: State dependent
    df_1b, ranks_1b = run_reservoir_sbc_v4("state_dependent", n_sbc_sims=100, seed=42)

    df_combined = pd.concat([df_1a, df_1b], ignore_index=True)
    out_dir = Path("/Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp/outputs/research/history_matching_iteration_4/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sbc_coverage_comparison_v4.csv"
    df_combined.to_csv(out_path, index=False)
    print(f"\nSaved SBC comparison table to {out_path}")

    # Save ranks for plotting
    np.save(out_dir.parent / "sbc_ranks_fixed_scale.npy", ranks_1a)
    np.save(out_dir.parent / "sbc_ranks_state_dependent.npy", ranks_1b)

    return df_combined


if __name__ == "__main__":
    df = run_sbc_benchmark_suite()
    print("\nSummary of SBC Results:")
    print(df.to_string(index=False))
