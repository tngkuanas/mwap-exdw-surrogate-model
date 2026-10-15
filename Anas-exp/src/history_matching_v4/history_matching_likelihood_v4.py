"""
Module: history_matching_likelihood_v4.py
Research Iteration 4: Enhanced Likelihood with Fixed-Scale Calibration Support
and Explicit Parameter Bounds Separation.

Key Enhancements:
1. Exact Fixed-Scale Covariance:
   Allows specifying fixed_sigma_oil and fixed_sigma_wat so that synthetic data
   generators and inference likelihoods share the exact same covariance matrix,
   resolving the SBC generative-versus-inference scale mismatch.
2. Observation-Dependent Covariance:
   Retains state-dependent relative noise scale (0.05 * RMS) for historical field inference.
3. Explicit Bounds Separation:
   Separates officially admissible physical parameter ranges from the
   experimental design prior bounds used in Bayesian MCMC inference.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular

from history_matching_v2.emulator_revalidation_v2 import HistoricalSVDGPEmulatorV2

# Admissible physical bounds from challenge specification / reservoir physics
PARAM_BOUNDS_PHYSICAL = {
    "Fault_Transmissibility": (0.0001, 1.0),
    "Porosity_Multiplier": (0.70, 1.50),
    "Permeability_Multiplier": (0.20, 10.0),
    "Aquifer_Pore_Volume": (10.0, 250.0),
}

# Empirical simulation design bounds (used as prior support for MCMC)
PARAM_BOUNDS_INFERENCE = [
    (0.04, 0.15),   # Fault Transmissibility
    (0.80, 1.50),   # Porosity Multiplier
    (0.50, 10.00),  # Permeability Multiplier
    (50.0, 200.0),  # Aquifer Pore Volume
]
BOUNDS_ARR_INFERENCE = np.array(PARAM_BOUNDS_INFERENCE)
LOWER_BOUNDS_INFERENCE = BOUNDS_ARR_INFERENCE[:, 0]
UPPER_BOUNDS_INFERENCE = BOUNDS_ARR_INFERENCE[:, 1]

CANONICAL_PARAMS = [
    "Fault_Transmissibility",
    "Porosity_Multiplier",
    "Permeability_Multiplier",
    "Aquifer_Pore_Volume",
]


def to_unit_hypercube_v4(theta: np.ndarray) -> np.ndarray:
    return (theta - LOWER_BOUNDS_INFERENCE) / (UPPER_BOUNDS_INFERENCE - LOWER_BOUNDS_INFERENCE)


def from_unit_hypercube_v4(u: np.ndarray) -> np.ndarray:
    return LOWER_BOUNDS_INFERENCE + u * (UPPER_BOUNDS_INFERENCE - LOWER_BOUNDS_INFERENCE)


def build_ar1_covariance(n: int, sigma: float, rho: float) -> np.ndarray:
    """Builds an n x n AR(1) covariance matrix: Cov(i, j) = sigma^2 * rho^|i - j|."""
    idx = np.arange(n)
    diff = np.abs(idx[:, None] - idx[None, :])
    return (sigma ** 2) * (rho ** diff)


class HistoryMatchingLikelihoodV4:
    """
    Evaluates calibrated likelihood models for reservoir history matching.
    Supports both fixed observation noise scale (for exact SBC validation)
    and relative observation noise scale (for empirical field calibration).
    """

    def __init__(
        self,
        emulator: HistoricalSVDGPEmulatorV2,
        field_history: pd.DataFrame,
        rho_ar1: float = 0.80,
        obs_noise_rel_oil: float = 0.05,
        obs_noise_rel_wat: float = 0.05,
        fixed_sigma_oil: Optional[float] = None,
        fixed_sigma_wat: Optional[float] = None,
    ):
        self.emulator = emulator
        self.field_history = field_history.sort_values("date").reset_index(drop=True)
        self.rho_ar1 = rho_ar1
        self.obs_noise_rel_oil = obs_noise_rel_oil
        self.obs_noise_rel_wat = obs_noise_rel_wat
        self.fixed_sigma_oil = fixed_sigma_oil
        self.fixed_sigma_wat = fixed_sigma_wat

        self.dates = pd.to_datetime(self.field_history["date"])
        self.n_timestamps = len(self.dates)  # 41
        self.n_intervals = self.n_timestamps - 1  # 40

        # Exact day intervals
        self.days_fwd = (self.dates.diff().dt.total_seconds() / 86400.0).dropna().to_numpy()

        # Cumulative production
        self.obs_Qo = self.field_history["ramp_oil_cum_stb"].to_numpy()
        self.obs_Qw = self.field_history["ramp_water_cum_stb"].to_numpy()

        # True forward quarterly rates
        self.obs_qo_fwd = np.diff(self.obs_Qo) / self.days_fwd
        self.obs_qw_fwd = np.diff(self.obs_Qw) / self.days_fwd

        # Terminal rate at 2008-01-01
        self.obs_qo_term = float(self.field_history["ramp_oil_rate_stbd"].iloc[-1])
        self.obs_qw_term = float(self.field_history["ramp_water_rate_stbd"].iloc[-1])

        # Scale normalizers
        self.norm_qo = float(np.sqrt(np.mean(self.obs_qo_fwd ** 2))) + 1e-12
        self.norm_qw = float(np.sqrt(np.mean(self.obs_qw_fwd ** 2))) + 1e-12
        self.norm_Qo = float(np.sqrt(np.mean(self.obs_Qo ** 2))) + 1e-12
        self.norm_Qw = float(np.sqrt(np.mean(self.obs_Qw ** 2))) + 1e-12

        # Noise scale determination
        if fixed_sigma_oil is not None and fixed_sigma_wat is not None:
            self.sigma_qo = float(fixed_sigma_oil)
            self.sigma_qw = float(fixed_sigma_wat)
            self.is_fixed_scale = True
        else:
            self.sigma_qo = obs_noise_rel_oil * self.norm_qo
            self.sigma_qw = obs_noise_rel_wat * self.norm_qw
            self.is_fixed_scale = False

        # Build AR(1) covariance matrices for rates (40 x 40)
        self.cov_obs_qo = build_ar1_covariance(self.n_intervals, self.sigma_qo, rho_ar1)
        self.cov_obs_qw = build_ar1_covariance(self.n_intervals, self.sigma_qw, rho_ar1)

        # Truncation covariance for L2
        self.sigma_trunc_rate_oil = (0.005 * self.norm_qo) ** 2 * np.eye(self.n_intervals)
        self.sigma_trunc_rate_wat = (0.010 * self.norm_qw) ** 2 * np.eye(self.n_intervals)

        # Cholesky factors for fixed L1 covariance
        self.L_obs_qo = cholesky(self.cov_obs_qo, lower=True)
        self.L_obs_qw = cholesky(self.cov_obs_qw, lower=True)
        self.logdet_obs_qo = 2.0 * np.sum(np.log(np.diag(self.L_obs_qo)))
        self.logdet_obs_qw = 2.0 * np.sum(np.log(np.diag(self.L_obs_qw)))
        self.const_norm = self.n_intervals * np.log(2.0 * np.pi)

    def evaluate_log_likelihood(
        self, theta: np.ndarray, model: str = "L1", return_components: bool = False
    ) -> Any:
        th_2d = theta.reshape(1, -1)
        pred = self.emulator.predict(th_2d, return_cov=(model == "L2"), return_std=True)

        qo_pred = pred["oil_rate_fwd"][0]
        qw_pred = pred["water_rate_fwd"][0]
        Qo_pred = pred["oil_cum"][0]
        Qw_pred = pred["water_cum"][0]

        res_qo = qo_pred - self.obs_qo_fwd
        res_qw = qw_pred - self.obs_qw_fwd

        term_err_o = abs(Qo_pred[-1] - self.obs_Qo[-1]) / (self.obs_Qo[-1] + 1e-12)
        term_err_w = abs(Qw_pred[-1] - self.obs_Qw[-1]) / (self.obs_Qw[-1] + 1e-12)
        term_err = 0.5 * (term_err_o + term_err_w)

        nrmse_Qo = np.sqrt(np.mean((Qo_pred - self.obs_Qo) ** 2)) / self.norm_Qo
        nrmse_Qw = np.sqrt(np.mean((Qw_pred - self.obs_Qw) ** 2)) / self.norm_Qw
        nrmse_qo = np.sqrt(np.mean(res_qo ** 2)) / self.norm_qo
        nrmse_qw = np.sqrt(np.mean(res_qw ** 2)) / self.norm_qw

        if model == "L1":
            z_qo = solve_triangular(self.L_obs_qo, res_qo, lower=True)
            z_qw = solve_triangular(self.L_obs_qw, res_qw, lower=True)
            quad_o = float(np.sum(z_qo ** 2))
            quad_w = float(np.sum(z_qw ** 2))

            ll_o = -0.5 * (quad_o + self.logdet_obs_qo + self.const_norm)
            ll_w = -0.5 * (quad_w + self.logdet_obs_qw + self.const_norm)
            log_lik = float(ll_o + ll_w)
            logdet_total = float(self.logdet_obs_qo + self.logdet_obs_qw)

        elif model == "L2":
            cov_emu_o = pred["oil_rate_cov"][0] + self.sigma_trunc_rate_oil
            cov_emu_w = pred["water_rate_cov"][0] + self.sigma_trunc_rate_wat

            cov_tot_o = self.cov_obs_qo + cov_emu_o
            cov_tot_w = self.cov_obs_qw + cov_emu_w

            L_tot_o = cholesky(cov_tot_o + 1e-6 * np.eye(self.n_intervals), lower=True)
            L_tot_w = cholesky(cov_tot_w + 1e-6 * np.eye(self.n_intervals), lower=True)

            z_qo = solve_triangular(L_tot_o, res_qo, lower=True)
            z_qw = solve_triangular(L_tot_w, res_qw, lower=True)
            quad_o = float(np.sum(z_qo ** 2))
            quad_w = float(np.sum(z_qw ** 2))

            logdet_o = 2.0 * np.sum(np.log(np.diag(L_tot_o)))
            logdet_w = 2.0 * np.sum(np.log(np.diag(L_tot_w)))

            ll_o = -0.5 * (quad_o + logdet_o + self.const_norm)
            ll_w = -0.5 * (quad_w + logdet_w + self.const_norm)
            log_lik = float(ll_o + ll_w)
            logdet_total = float(logdet_o + logdet_w)

        else:
            raise ValueError(f"Unknown model: {model}")

        if return_components:
            return {
                "log_likelihood": float(log_lik),
                "quad_oil": float(quad_o),
                "quad_wat": float(quad_w),
                "logdet_cov": float(logdet_total),
                "nrmse_Qo": float(nrmse_Qo),
                "nrmse_Qw": float(nrmse_Qw),
                "nrmse_qo": float(nrmse_qo),
                "nrmse_qw": float(nrmse_qw),
                "terminal_oil_err": float(term_err_o),
                "terminal_wat_err": float(term_err_w),
            }
        return float(log_lik)
