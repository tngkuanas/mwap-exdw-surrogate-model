"""
Module: simulation_based_calibration_v3.py
Research Iteration 3 — Phase 2: Repair Simulation-Based Calibration (SBC).

Methodological Foundation:
Talts, Betancourt, Simpson, Vehtari & Gelman (2018/2020),
'Validating Bayesian Inference Algorithms with Simulation-Based Calibration'.

Key Corrections Implemented:
1. Prior Consistency: Prior in generative model is drawn from Uniform(0, 1)^4,
   matching the inference prior exactly.
2. Covariance Consistency: Synthetic observations are generated using the exact
   AR(1) temporal covariance (rho=0.80, sigma_obs=0.05) assumed by the likelihood.
3. Untruncated Gaussian Observations: Synthetic rates are not clipped with maximum(..., 0),
   strictly preserving the Gaussian density family.
4. Scale Normalization: Generative noise scales are anchored to consistent field references.
5. Analytical Toy Benchmark: Analytically tractable conjugate Gaussian model validates
   the calibration testing engine.
6. Adequately Powered SBC: 100 independent simulations with binned chi-square and
   Kolmogorov-Smirnov uniformity tests.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy.stats import chisquare, kstest, uniform
from scipy.linalg import cholesky

from ensemble.data import load_uncertainty, TRAIN_CASES, PARAMS, resolve_shared_dir
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
)
from history_matching_v2.emulator_revalidation_v2 import HistoricalSVDGPEmulatorV2
from history_matching_v2.history_matching_likelihood_v2 import (
    HistoryMatchingLikelihoodV2,
    build_ar1_covariance,
)
from history_matching_v2.bayesian_mcmc_v2 import (
    run_adaptive_metropolis_chain,
    from_unit_hypercube,
    to_unit_hypercube,
    PARAM_BOUNDS,
)


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
        # 1. Draw theta_true from prior
        theta_true = rng.normal(mu_0, sigma_0)

        # 2. Draw observations from likelihood
        y = rng.normal(theta_true, sigma_obs, size=n_obs)
        y_bar = np.mean(y)

        # 3. Exact posterior mean
        mu_post = (tau_0 * mu_0 + tau_obs * y_bar) / tau_post

        # 4. Draw L posterior samples from exact posterior
        draws_exact = rng.normal(mu_post, sigma_post, size=l_draws)
        rank_exact = int(np.sum(draws_exact < theta_true))
        ranks_exact.append(rank_exact / l_draws)

        # 5. Draw L posterior samples via Metropolis-Hastings MCMC
        mcmc_draws = []
        th_curr = mu_0
        for _step in range(l_draws + 200):
            th_prop = th_curr + rng.normal(0.0, sigma_post)
            # log-posterior ratio
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

    # Chi-squared test (10 bins)
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
        "exact_mean_rank": float(np.mean(ranks_exact)),
        "exact_std_rank": float(np.std(ranks_exact)),
        "mcmc_chi2_p": float(p_mcmc),
        "mcmc_ks_p": float(ks_mcmc.pvalue),
        "mcmc_mean_rank": float(np.mean(ranks_mcmc)),
        "mcmc_std_rank": float(np.std(ranks_mcmc)),
        "is_exact_calibrated": bool(p_exact >= 0.05 and ks_exact.pvalue >= 0.05),
        "is_mcmc_calibrated": bool(p_mcmc >= 0.05 and ks_mcmc.pvalue >= 0.05),
    }


# ==============================================================================
# 2. RESERVOIR SIMULATION-BASED CALIBRATION PIPELINE
# ==============================================================================

def run_reservoir_sbc_experiment(
    n_sbc_sims: int = 100,
    mcmc_draws: int = 600,
    burn_in: int = 200,
    rho_ar1: float = 0.80,
    obs_noise_rel: float = 0.05,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Executes a statistically powered SBC experiment on the reservoir likelihood model.
    """
    print(f"\nRunning Reservoir SBC with {n_sbc_sims} independent replications...")
    rng = np.random.default_rng(seed)

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
    lik_ref = HistoryMatchingLikelihoodV2(
        emu, fh, rho_ar1=rho_ar1, obs_noise_rel_oil=obs_noise_rel, obs_noise_rel_wat=obs_noise_rel
    )
    norm_qo = lik_ref.norm_qo
    norm_qw = lik_ref.norm_qw
    days_fwd = lik_ref.days_fwd

    # Exact generative AR(1) covariance matrices
    cov_gen_o = build_ar1_covariance(40, obs_noise_rel * norm_qo, rho_ar1)
    cov_gen_w = build_ar1_covariance(40, obs_noise_rel * norm_qw, rho_ar1)
    L_gen_o = cholesky(cov_gen_o, lower=True)
    L_gen_w = cholesky(cov_gen_w, lower=True)

    param_names = ["Fault_Transmissibility", "Porosity_Multiplier", "Permeability_Multiplier", "Aquifer_Pore_Volume"]
    param_abbr = ["Ft", "Poro", "Perm", "AqPV"]
    sbc_ranks = {p: [] for p in param_names}
    coverage_50 = {p: [] for p in param_names}
    coverage_80 = {p: [] for p in param_names}
    coverage_90 = {p: [] for p in param_names}

    raw_sim_records = []

    for s_idx in range(n_sbc_sims):
        sim_seed = seed + 1000 + s_idx
        sim_rng = np.random.default_rng(sim_seed)

        # 1. Exact Prior Draw: Uniform(0, 1)^4
        u_true = sim_rng.uniform(0.0, 1.0, size=4)
        th_true = from_unit_hypercube(u_true)

        # 2. Generative Forward Model (Untruncated Gaussian with AR(1) Covariance)
        pred_true = emu.predict(th_true.reshape(1, -1))
        z_o = sim_rng.standard_normal(40)
        z_w = sim_rng.standard_normal(40)
        noise_o = L_gen_o @ z_o
        noise_w = L_gen_w @ z_w

        # CRITICAL FIX: UNTRUNCATED GAUSSIAN OBSERVATIONS (NO CLIPPING)
        synth_qo = pred_true["oil_rate_fwd"][0] + noise_o
        synth_qw = pred_true["water_rate_fwd"][0] + noise_w

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

        # 3. Posterior Inference (L1 Gaussian Likelihood + Uniform Prior)
        lik_synth = HistoryMatchingLikelihoodV2(
            emu, synth_df, rho_ar1=rho_ar1, obs_noise_rel_oil=obs_noise_rel, obs_noise_rel_wat=obs_noise_rel
        )

        # Dispersed initial point
        u_init = sim_rng.uniform(0.1, 0.9, size=4)
        res_mcmc = run_adaptive_metropolis_chain(
            lik_synth,
            u_init=u_init,
            n_iterations=mcmc_draws,
            burn_in=burn_in,
            model="L1",
            chain_id=1,
            seed=sim_seed,
        )
        post_draws = res_mcmc["draws"]  # (n_eff, 4)
        n_draws = len(post_draws)

        sim_rec = {"sim_id": s_idx}
        for p_idx, p_name in enumerate(param_names):
            p_draws = post_draws[:, p_idx]
            true_val = th_true[p_idx]
            rank_val = int(np.sum(p_draws < true_val))
            norm_rank = rank_val / n_draws
            sbc_ranks[p_name].append(norm_rank)

            q05, q10, q25, q75, q90, q95 = np.percentile(p_draws, [5.0, 10.0, 25.0, 75.0, 90.0, 95.0])
            c50 = int(q25 <= true_val <= q75)
            c80 = int(q10 <= true_val <= q90)
            c90 = int(q05 <= true_val <= q95)
            coverage_50[p_name].append(c50)
            coverage_80[p_name].append(c80)
            coverage_90[p_name].append(c90)

            sim_rec[f"{param_abbr[p_idx]}_true"] = true_val
            sim_rec[f"{param_abbr[p_idx]}_post_mean"] = float(np.mean(p_draws))
            sim_rec[f"{param_abbr[p_idx]}_rank_percentile"] = norm_rank

        raw_sim_records.append(sim_rec)

    df_raw = pd.DataFrame(raw_sim_records)

    # 4. Statistical Calibration Summary
    n_bins = 10
    exp_freq = np.full(n_bins, n_sbc_sims / n_bins)
    summary_rows = []

    for p_idx, p_name in enumerate(param_names):
        r_arr = np.array(sbc_ranks[p_name])
        hist, _ = np.histogram(r_arr, bins=n_bins, range=(0.0, 1.0))
        chi2_stat, p_chi2 = chisquare(hist, exp_freq)
        ks_res = kstest(r_arr, "uniform")

        mean_rank = np.mean(r_arr)
        std_rank = np.std(r_arr)
        # Theoretical uniform variance: 1/12 = 0.08333, std = 0.28868
        # Standard error of mean: std / sqrt(N)
        mcse_mean = std_rank / np.sqrt(n_sbc_sims)

        cov50 = np.mean(coverage_50[p_name])
        cov80 = np.mean(coverage_80[p_name])
        cov90 = np.mean(coverage_90[p_name])

        is_calibrated = (p_chi2 >= 0.05) and (ks_res.pvalue >= 0.05)

        summary_rows.append({
            "parameter": p_name,
            "abbreviation": param_abbr[p_idx],
            "n_sims": n_sbc_sims,
            "mean_rank": float(mean_rank),
            "mcse_mean_rank": float(mcse_mean),
            "std_rank": float(std_rank),
            "chi2_stat_10bins": float(chi2_stat),
            "chi2_p_value": float(p_chi2),
            "ks_stat": float(ks_res.statistic),
            "ks_p_value": float(ks_res.pvalue),
            "nominal_50_empirical_coverage": float(cov50),
            "nominal_80_empirical_coverage": float(cov80),
            "nominal_90_empirical_coverage": float(cov90),
            "is_statistically_calibrated": bool(is_calibrated),
        })

    df_summary = pd.DataFrame(summary_rows)
    return df_summary, df_raw, {"sbc_ranks": sbc_ranks}


def main():
    print("=" * 80)
    print("RESEARCH ITERATION 3 — PHASE 2: SIMULATION-BASED CALIBRATION (TALTS ET AL. 2018)")
    print("=" * 80)

    out_dir = Path("/Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp/outputs/research/history_matching_iteration_3/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run Analytical Gaussian Benchmark
    print("\n--- 1. Analytical Gaussian Toy SBC Benchmark ---")
    toy_res = run_analytical_toy_sbc(n_sims=500, n_obs=5, seed=42)
    print(f"Analytical Toy SBC Results:")
    print(f"  Exact Posterior:  Chi2 p-value = {toy_res['exact_chi2_p']:.4f}, KS p-value = {toy_res['exact_ks_p']:.4f}, Calibrated = {toy_res['is_exact_calibrated']}")
    print(f"  MCMC Posterior:   Chi2 p-value = {toy_res['mcmc_chi2_p']:.4f}, KS p-value = {toy_res['mcmc_ks_p']:.4f}, Calibrated = {toy_res['is_mcmc_calibrated']}")

    df_toy = pd.DataFrame([toy_res])
    df_toy.to_csv(out_dir / "analytical_toy_sbc_benchmark.csv", index=False)

    # 2. Run Reservoir SBC with N=100
    print("\n--- 2. Adequately Powered Reservoir SBC (100 Replications) ---")
    df_res_summary, df_raw, _ = run_reservoir_sbc_experiment(n_sbc_sims=100, mcmc_draws=600, burn_in=200, seed=42)
    df_res_summary.to_csv(out_dir / "simulation_based_calibration_v3_summary.csv", index=False)
    df_raw.to_csv(out_dir / "simulation_based_calibration_v3_raw_draws.csv", index=False)

    print("\nReservoir Simulation-Based Calibration Summary:")
    print(df_res_summary.to_string())


if __name__ == "__main__":
    main()
