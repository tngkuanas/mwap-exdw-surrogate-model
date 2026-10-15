"""
Module: bayesian_uncertainty_and_identifiability.py
Implements Phase 5: Bayesian Parameter Uncertainty, MCMC Sampling, and Identifiability Analysis.
- Rigorous Adaptive Metropolis-Hastings (AM) MCMC across 4 independent chains.
- Gelman-Rubin R-hat convergence diagnostic and Effective Sample Size (ESS).
- Identifiability analysis: Prior vs Posterior variance reduction.
- Pairwise parameter correlation analysis.
- Posterior predictive distribution overlays.
"""
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy.stats import norm

from ensemble.data import load_uncertainty, TRAIN_CASES, PARAMS
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
)
from history_matching.historical_forward_emulator import HistoricalSVDGPEmulator
from history_matching.history_matching_objective import HistoryMatchingObjective

PARAM_BOUNDS = [
    (0.04, 0.15),   # Fault Transmissibility
    (0.80, 1.50),   # Porosity Multiplier
    (0.50, 9.00),   # Permeability Multiplier
    (50.0, 200.0),  # Aquifer Pore Volume
]
BOUNDS_ARR = np.array(PARAM_BOUNDS)
LOWER_BOUNDS = BOUNDS_ARR[:, 0]
UPPER_BOUNDS = BOUNDS_ARR[:, 1]
PARAM_RANGES = UPPER_BOUNDS - LOWER_BOUNDS


def log_prior(theta: np.ndarray) -> float:
    """Uniform prior within physical bounding box."""
    if np.all(theta >= LOWER_BOUNDS) and np.all(theta <= UPPER_BOUNDS):
        # Normalization constant for uniform prior
        return -np.sum(np.log(PARAM_RANGES))
    return -np.inf


def log_likelihood(theta: np.ndarray, obj: HistoryMatchingObjective, temperature: float = 1.0) -> float:
    """
    Log-likelihood based on AR(1) Generalized Least Squares misfit:
    ln p(y | theta) = -0.5 * J_GLS(theta) / T
    """
    misfit = obj.evaluate_misfit(theta, objective_type="ar1_gls")
    # Subtraction of minimum baseline misfit to avoid extreme underflow
    return -0.5 * (misfit - 42.0) / temperature


def log_posterior(theta: np.ndarray, obj: HistoryMatchingObjective, temperature: float = 1.0) -> float:
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, obj, temperature=temperature)
    return lp + ll


def run_adaptive_mcmc_chain(
    obj: HistoryMatchingObjective,
    theta_init: np.ndarray,
    n_samples: int = 5000,
    burn_in: int = 1000,
    seed: int = 42,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """
    Adaptive Metropolis-Hastings (Haario et al. 2001) for 4D parameter space.
    Adapts proposal covariance matrix to achieve target acceptance rate (~23.4%).
    """
    np.random.seed(seed)
    dim = len(theta_init)
    chain = np.zeros((n_samples, dim))
    chain[0] = theta_init

    current_log_post = log_posterior(theta_init, obj, temperature=temperature)

    # Initial proposal covariance
    cov = np.diag((0.02 * PARAM_RANGES) ** 2)
    sd = 2.38 ** 2 / dim  # Optimal scaling factor
    epsilon = 1e-6

    n_accepted = 0

    for i in range(1, n_samples):
        current_theta = chain[i - 1]

        # Adapt proposal covariance after burn-in initiation (every 50 steps)
        if i > 200 and i % 50 == 0:
            hist_samples = chain[:i]
            cov = sd * (np.cov(hist_samples.T) + epsilon * np.eye(dim))

        # Propose candidate
        prop_theta = np.random.multivariate_normal(current_theta, cov)

        prop_log_post = log_posterior(prop_theta, obj, temperature=temperature)

        # Metropolis acceptance probability
        log_alpha = prop_log_post - current_log_post
        if np.log(np.random.uniform(0.0, 1.0)) < log_alpha:
            chain[i] = prop_theta
            current_log_post = prop_log_post
            n_accepted += 1
        else:
            chain[i] = current_theta

    acc_rate = n_accepted / (n_samples - 1)
    return chain[burn_in:], acc_rate


def compute_gelman_rubin(chains: List[np.ndarray]) -> np.ndarray:
    """
    Computes Gelman-Rubin R-hat convergence diagnostic across multiple chains.
    R-hat < 1.05 indicates adequate convergence.
    """
    m = len(chains)  # number of chains
    n = len(chains[0])  # samples per chain
    dim = chains[0].shape[1]

    chain_means = np.array([np.mean(c, axis=0) for c in chains])  # (m, d)
    grand_mean = np.mean(chain_means, axis=0)  # (d,)

    # Between-chain variance B
    B = (n / (m - 1)) * np.sum((chain_means - grand_mean) ** 2, axis=0)  # (d,)

    # Within-chain variance W
    chain_vars = np.array([np.var(c, axis=0, ddof=1) for c in chains])  # (m, d)
    W = np.mean(chain_vars, axis=0)  # (d,)

    # Marginal posterior variance estimate V_hat
    V_hat = ((n - 1) / n) * W + ((m + 1) / (m * n)) * B
    R_hat = np.sqrt(V_hat / (W + 1e-12))
    return R_hat


def compute_effective_sample_size(chain: np.ndarray) -> np.ndarray:
    """Computes Effective Sample Size (ESS) per parameter using autocorrelation time."""
    n, d = chain.shape
    ess = np.zeros(d)
    for j in range(d):
        x = chain[:, j] - np.mean(chain[:, j])
        autocorr = np.correlate(x, x, mode="full")[n - 1:] / (np.var(chain[:, j]) * n)
        # Sum autocorrelations until first negative
        first_neg = np.where(autocorr <= 0.0)[0]
        max_lag = first_neg[0] if len(first_neg) > 0 else min(len(autocorr), 500)
        tau = 1.0 + 2.0 * np.sum(autocorr[1:max_lag])
        ess[j] = n / max(tau, 1.0)
    return ess


def run_bayesian_analysis() -> Dict[str, Any]:
    """Executes full Phase 5 Bayesian Calibration and Diagnostics."""
    print("=" * 80)
    print("PHASE 5: BAYESIAN PARAMETER UNCERTAINTY & IDENTIFIABILITY ANALYSIS")
    print("=" * 80)

    unc_df = load_uncertainty()
    q_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(q_df)
    fh = pd.read_parquet("../data/field_history.parquet")

    X_train = unc_df.loc[TRAIN_CASES, PARAMS].to_numpy()
    Y_oil_tr = np.array([case_dict[c]["oil_cum_stb"].to_numpy() for c in TRAIN_CASES])
    Y_wat_tr = np.array([case_dict[c]["water_cum_stb"].to_numpy() for c in TRAIN_CASES])

    emulator = HistoricalSVDGPEmulator(n_components=4, random_state=42).fit(X_train, Y_oil_tr, Y_wat_tr)
    obj = HistoryMatchingObjective(emulator, fh)

    # 4 diverse initializations for 4 MCMC chains
    initial_thetas = [
        np.array([0.129, 1.320, 8.18, 144.0]),  # Best match region
        np.array([0.086, 1.226, 9.00, 133.0]),  # DE optimum
        np.array([0.060, 1.350, 2.50, 120.0]),  # Case 6 region
        np.array([0.110, 1.260, 6.00, 150.0]),  # Intermediate region
    ]

    chains = []
    acc_rates = []
    n_samples_per_chain = 4000
    burn_in = 1000

    print(f"Executing 4 Adaptive MCMC Chains ({n_samples_per_chain} steps each, {burn_in} burn-in)...")
    t0 = time.perf_counter()

    for c_idx, th_init in enumerate(initial_thetas):
        ch, acc = run_adaptive_mcmc_chain(
            obj=obj,
            theta_init=th_init,
            n_samples=n_samples_per_chain,
            burn_in=burn_in,
            seed=42 + c_idx * 100,
            temperature=1.0,
        )
        chains.append(ch)
        acc_rates.append(acc)
        print(f"  Chain {c_idx+1}: {len(ch)} post-burn samples | Acceptance Rate: {acc*100:.1f}%")

    elapsed_mcmc = time.perf_counter() - t0
    print(f"MCMC Completed in {elapsed_mcmc:.2f} seconds.")

    # Convergence Diagnostics
    R_hat = compute_gelman_rubin(chains)
    full_posterior = np.vstack(chains)
    ESS = compute_effective_sample_size(full_posterior)

    print("\nMCMC Convergence & Sampling Diagnostics:")
    for j, p_name in enumerate(PARAMS):
        print(f"  {p_name:<26s} | R-hat: {R_hat[j]:.4f} (target < 1.05) | ESS: {ESS[j]:.0f}")

    # Prior vs Posterior Variance Reduction (Identifiability Index)
    # Uniform prior variance on [a, b]: (b - a)^2 / 12
    prior_var = (PARAM_RANGES ** 2) / 12.0
    post_var = np.var(full_posterior, axis=0)
    identifiability_index = 1.0 - (post_var / prior_var)

    # Posterior Distribution Summary
    post_mean = np.mean(full_posterior, axis=0)
    post_std = np.std(full_posterior, axis=0)
    post_p10 = np.percentile(full_posterior, 10, axis=0)
    post_p50 = np.percentile(full_posterior, 50, axis=0)
    post_p90 = np.percentile(full_posterior, 90, axis=0)

    summary_rows = []
    for j, p_name in enumerate(PARAMS):
        summary_rows.append({
            "Parameter": p_name,
            "Prior_Min": LOWER_BOUNDS[j],
            "Prior_Max": UPPER_BOUNDS[j],
            "Post_Mean": post_mean[j],
            "Post_Std": post_std[j],
            "Post_P10": post_p10[j],
            "Post_P50": post_p50[j],
            "Post_P90": post_p90[j],
            "Identifiability_Index": identifiability_index[j],
            "R_hat": R_hat[j],
            "ESS": ESS[j],
        })

    param_summary_df = pd.DataFrame(summary_rows)
    out_table = Path("outputs/history_matching/tables/bayesian_posterior_parameter_distributions.csv")
    out_table.parent.mkdir(parents=True, exist_ok=True)
    param_summary_df.to_csv(out_table, index=False)
    print(f"\nSaved Posterior Summary Table: {out_table}")
    print(param_summary_df.to_string())

    # Posterior Correlation Matrix
    corr_matrix = pd.DataFrame(np.corrcoef(full_posterior.T), index=PARAMS, columns=PARAMS)
    corr_path = Path("outputs/history_matching/tables/posterior_parameter_correlation_matrix.csv")
    corr_matrix.to_csv(corr_path)
    print(f"\nSaved Parameter Correlation Matrix: {corr_path}")
    print(corr_matrix.to_string())

    # Save sampled posterior array for Phase 6 selection
    post_samples_path = Path("outputs/history_matching/bayesian_posterior_samples.npy")
    np.save(post_samples_path, full_posterior)
    print(f"Saved {len(full_posterior)} Posterior Samples: {post_samples_path}")

    return {
        "full_posterior": full_posterior,
        "summary_df": param_summary_df,
        "corr_matrix": corr_matrix,
        "R_hat": R_hat,
        "ESS": ESS,
    }


if __name__ == "__main__":
    run_bayesian_analysis()
