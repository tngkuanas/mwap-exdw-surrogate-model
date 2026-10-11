"""
Module: bayesian_mcmc_v4.py
Research Iteration 4: Enhanced Bayesian MCMC Sampling and Diagnostics.

Provides:
- Vehtari et al. (2021) rank-normalized and folded split-R-hat
- Bulk-ESS and Tail-ESS
- Monte Carlo Standard Errors (MCSE) for mean and quantiles
- Adaptive Metropolis chain sampling supporting HistoryMatchingLikelihoodV4
- Strict separation of inference prior bounds from physical bounds
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm, beta

from history_matching_v4.history_matching_likelihood_v4 import (
    PARAM_BOUNDS_INFERENCE,
    BOUNDS_ARR_INFERENCE,
    LOWER_BOUNDS_INFERENCE,
    UPPER_BOUNDS_INFERENCE,
    to_unit_hypercube_v4,
    from_unit_hypercube_v4,
    HistoryMatchingLikelihoodV4,
)


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Computes exact Clopper-Pearson (1 - alpha) confidence interval for binomial proportion.
    """
    if k <= 0:
        lower = 0.0
    else:
        lower = float(beta.ppf(alpha / 2.0, k, n - k + 1))

    if k >= n:
        upper = 1.0
    else:
        upper = float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))

    return lower, upper


def compute_autocorr_ess(x: np.ndarray) -> float:
    """Computes Geyer's initial positive sequence ESS for a 1D chain."""
    N = len(x)
    if N < 4:
        return float(N)
    x_c = x - np.mean(x)
    var = np.var(x)
    if var <= 1e-12:
        return float(N)

    # Compute autocorrelations via FFT
    n_fft = 2 ** int(np.ceil(np.log2(2 * N)))
    fx = np.fft.fft(x_c, n_fft)
    acov = np.fft.ifft(fx * np.conj(fx)).real[:N] / N
    rho = acov / acov[0]

    # Geyer's initial positive sequence estimator
    K = (N - 1) // 2
    paired_rho = np.empty(K)
    for k in range(K):
        paired_rho[k] = rho[2 * k] + rho[2 * k + 1]

    cutoff = 0
    while cutoff < K and paired_rho[cutoff] > 0:
        cutoff += 1

    if cutoff == 0:
        return float(N)

    gamma = paired_rho[:cutoff]
    for i in range(1, len(gamma)):
        gamma[i] = min(gamma[i], gamma[i - 1])

    tau = -1.0 + 2.0 * np.sum(gamma)
    tau = max(tau, 1.0)
    ess = N / tau
    return float(np.clip(ess, 1.0, float(N)))


def compute_rank_normalized_split_rhat(chains: np.ndarray) -> float:
    """
    chains: (M, N) draws from M chains of length N.
    Splits each chain in half -> 2M chains of length N/2.
    Rank-transforms all draws to normal scores, then computes split R-hat.
    """
    M, N = chains.shape
    if N % 2 != 0:
        chains = chains[:, :-1]
        N -= 1
    half_N = N // 2
    split_chains = np.empty((2 * M, half_N))
    for m in range(M):
        split_chains[2 * m] = chains[m, :half_N]
        split_chains[2 * m + 1] = chains[m, half_N:]

    num_chains = 2 * M
    total_samples = num_chains * half_N

    flat = split_chains.flatten()
    ranks = flat.argsort().argsort() + 1.0
    z = norm.ppf((ranks - 0.375) / (total_samples + 0.25))
    z_chains = z.reshape((num_chains, half_N))

    chain_means = np.mean(z_chains, axis=1)
    overall_mean = np.mean(chain_means)
    B = (half_N / (num_chains - 1)) * np.sum((chain_means - overall_mean) ** 2)
    chain_vars = np.var(z_chains, axis=1, ddof=1)
    W = np.mean(chain_vars)

    if W <= 1e-12:
        return 1.0
    var_plus = ((half_N - 1) / half_N) * W + (1.0 / half_N) * B
    return float(np.sqrt(var_plus / W))


def compute_folded_split_rhat(chains: np.ndarray) -> float:
    """Computes split R-hat on folded draws |x - median(x)| to detect scale/variance differences."""
    med = np.median(chains)
    folded = np.abs(chains - med)
    return compute_rank_normalized_split_rhat(folded)


def compute_bulk_and_tail_ess(chains: np.ndarray) -> Tuple[float, float]:
    """Computes Bulk-ESS (mean/median precision) and Tail-ESS (5% and 95% quantiles)."""
    flat = chains.flatten()
    total_N = len(flat)

    ranks = flat.argsort().argsort() + 1.0
    z = norm.ppf((ranks - 0.375) / (total_N + 0.25))
    ess_bulk = compute_autocorr_ess(z)

    q05 = np.percentile(flat, 5.0)
    q95 = np.percentile(flat, 95.0)
    I_05 = (flat <= q05).astype(float)
    I_95 = (flat >= q95).astype(float)
    ess_05 = compute_autocorr_ess(I_05)
    ess_95 = compute_autocorr_ess(I_95)
    ess_tail = min(ess_05, ess_95)

    return float(ess_bulk), float(ess_tail)


def compute_mcse_mean(draws: np.ndarray, ess_bulk: float) -> float:
    """Computes Monte Carlo Standard Error for the posterior mean."""
    return float(np.std(draws) / np.sqrt(max(ess_bulk, 1.0)))


def compute_mcse_quantile(draws: np.ndarray, p: float, ess_tail: float) -> float:
    """Computes Monte Carlo Standard Error for quantile p using asymptotic normal density."""
    q_val = np.percentile(draws, p * 100.0)
    std = np.std(draws)
    if std <= 1e-12:
        return 0.0
    # Kernel-free normal approximation for density f(q)
    f_q = norm.pdf((q_val - np.mean(draws)) / std) / std
    if f_q <= 1e-12:
        f_q = 1e-6
    return float(np.sqrt(p * (1.0 - p) / (max(ess_tail, 1.0) * (f_q ** 2))))


def run_adaptive_metropolis_chain_v4(
    lik_obj: Any,
    u_init: np.ndarray,
    n_iterations: int = 2000,
    burn_in: int = 500,
    model: str = "L1",
    chain_id: int = 1,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Runs an Adaptive Metropolis MCMC chain in logit-transformed unit hypercube space.
    """
    rng = np.random.default_rng(seed)
    z_curr = logit(np.clip(u_init, 0.01, 0.99))
    th_curr = from_unit_hypercube_v4(expit(z_curr))
    log_post_curr = lik_obj.evaluate_log_likelihood(th_curr, model=model)

    d = len(u_init)
    sd = (2.38 ** 2) / d
    eps_id = 1e-6
    cov_emp = np.eye(d) * 0.1

    z_history = [z_curr.copy()]
    th_history = [th_curr.copy()]
    log_post_history = [log_post_curr]
    n_accepted = 0

    adapt_start = 50
    for it in range(1, n_iterations):
        if it < adapt_start:
            z_prop = rng.multivariate_normal(z_curr, np.eye(d) * 0.05)
        else:
            if rng.uniform(0.0, 1.0) < 0.05:
                z_prop = rng.multivariate_normal(z_curr, np.eye(d) * (0.1 ** 2) / d)
            else:
                prop_cov = sd * cov_emp + eps_id * np.eye(d)
                z_prop = rng.multivariate_normal(z_curr, prop_cov)

        u_prop = expit(z_prop)
        th_prop = from_unit_hypercube_v4(u_prop)

        # Uniform prior in physical domain -> Jacobian in logit space: sum(log(u * (1-u)))
        log_prior_curr = np.sum(np.log(expit(z_curr) * (1.0 - expit(z_curr)) + 1e-12))
        log_prior_prop = np.sum(np.log(u_prop * (1.0 - u_prop) + 1e-12))

        log_lik_prop = lik_obj.evaluate_log_likelihood(th_prop, model=model)
        log_alpha = (log_lik_prop + log_prior_prop) - (log_post_curr + log_prior_curr)

        if np.log(rng.uniform(0.0, 1.0)) < log_alpha:
            z_curr = z_prop
            th_curr = th_prop
            log_post_curr = log_lik_prop
            n_accepted += 1

        z_history.append(z_curr.copy())
        th_history.append(th_curr.copy())
        log_post_history.append(log_post_curr)

        # Online adaptation of empirical covariance matrix
        if it >= adapt_start:
            z_arr = np.array(z_history)
            cov_emp = np.cov(z_arr.T)

    th_history_arr = np.array(th_history)
    post_draws = th_history_arr[burn_in:]
    post_log_liks = np.array(log_post_history[burn_in:])

    return {
        "chain_id": chain_id,
        "draws": post_draws,
        "log_likelihoods": post_log_liks,
        "acceptance_rate": float(n_accepted / n_iterations),
        "th_map": post_draws[np.argmax(post_log_liks)],
        "max_log_lik": float(np.max(post_log_liks)),
    }
