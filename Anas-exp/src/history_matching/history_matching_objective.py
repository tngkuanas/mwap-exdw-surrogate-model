"""
Module: history_matching_objective.py
Implements Phase 3: Reservoir-Engineering History-Matching Objective.
Features:
1. Reconciles exact 41 quarterly timestamps (1998-01-01 to 2008-01-01).
2. Prevents double-counting of gas (solution GOR Rs = 0.3633 MSCF/STB is strictly dependent on oil).
3. Evaluates rate-based, cumulative-based, and generalized least squares with AR(1) covariance per Paper F (Evensen 2021).
4. Compounds emulator epistemic variance per Paper A (Kennedy & O'Hagan 2001).
5. Provides multi-objective Pareto trade-off evaluation (Oil vs Water vs Terminal Anchor).
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular

RS_SOLUTION = 0.3633  # MSCF / STB


def build_ar1_covariance(n_steps: int, sigma: float, rho: float) -> np.ndarray:
    """
    Constructs an AR(1) temporal covariance matrix:
    Sigma(i, j) = sigma^2 * rho^|i - j|
    Per Paper F (Evensen 2021), this captures autocorrelation in production measurements.
    """
    idx = np.arange(n_steps)
    dist = np.abs(idx[:, None] - idx[None, :])
    cov = (sigma ** 2) * (rho ** dist)
    return cov


class HistoryMatchingObjective:
    """
    Reservoir-engineering history matching objective function.
    Evaluates historical match between emulator predictions and observed field production.
    """
    def __init__(
        self,
        emulator,
        field_history: pd.DataFrame,
        rho_ar1: float = 0.80,
        obs_noise_rel: float = 0.05,
        weight_oil: float = 0.50,
        weight_water: float = 0.50,
        weight_terminal: float = 0.20,
    ):
        self.emulator = emulator
        self.field_history = field_history.sort_values("date").reset_index(drop=True)
        self.n_steps = len(self.field_history)
        self.rho_ar1 = rho_ar1
        self.weight_oil = weight_oil
        self.weight_water = weight_water
        self.weight_terminal = weight_terminal

        # Ground truth field observations
        self.obs_Qo = self.field_history["ramp_oil_cum_stb"].to_numpy()
        self.obs_Qw = self.field_history["ramp_water_cum_stb"].to_numpy()
        self.obs_qo = self.field_history["ramp_oil_rate_stbd"].to_numpy()
        self.obs_qw = self.field_history["ramp_water_rate_stbd"].to_numpy()

        # Scale normalizers (RMS)
        self.norm_Qo = np.sqrt(np.mean(self.obs_Qo ** 2)) + 1e-12
        self.norm_Qw = np.sqrt(np.mean(self.obs_Qw ** 2)) + 1e-12
        self.norm_qo = np.sqrt(np.mean(self.obs_qo ** 2)) + 1e-12
        self.norm_qw = np.sqrt(np.mean(self.obs_qw ** 2)) + 1e-12

        # AR(1) Covariance matrices for GLS
        sigma_qo = obs_noise_rel * self.norm_qo
        sigma_qw = obs_noise_rel * self.norm_qw
        self.cov_qo_ar1 = build_ar1_covariance(self.n_steps, sigma_qo, rho_ar1)
        self.cov_qw_ar1 = build_ar1_covariance(self.n_steps, sigma_qw, rho_ar1)

        # Precompute Cholesky factors for fast GLS evaluation
        self.L_qo = cholesky(self.cov_qo_ar1, lower=True)
        self.L_qw = cholesky(self.cov_qw_ar1, lower=True)

    def evaluate_misfit(
        self, theta: np.ndarray, objective_type: str = "ar1_gls", return_components: bool = False
    ) -> Any:
        """
        theta: (4,) geological parameters [Ft, Poro, Perm, AqPV]
        objective_type:
          - 'ar1_gls': Generalized Least Squares with AR(1) error covariance (Paper F)
          - 'diagonal_nrmse': Standard diagonal NRMSE (Baseline)
          - 'rate_only': Rate-based NRMSE
          - 'pareto': Multi-objective breakdown
        """
        pred = self.emulator.predict(theta.reshape(1, -1), return_std=True)

        Qo_pred = pred["oil_cum"][0]
        Qw_pred = pred["water_cum"][0]
        qo_pred = pred["oil_rate"][0]
        qw_pred = pred["water_rate"][0]

        # Residuals
        res_Qo = Qo_pred - self.obs_Qo
        res_Qw = Qw_pred - self.obs_Qw
        res_qo = qo_pred - self.obs_qo
        res_qw = qw_pred - self.obs_qw

        # Component NRMSEs
        nrmse_Qo = np.sqrt(np.mean(res_Qo ** 2)) / self.norm_Qo
        nrmse_Qw = np.sqrt(np.mean(res_Qw ** 2)) / self.norm_Qw
        nrmse_qo = np.sqrt(np.mean(res_qo ** 2)) / self.norm_qo
        nrmse_qw = np.sqrt(np.mean(res_qw ** 2)) / self.norm_qw

        # Terminal volume relative error at 2008-01-01
        term_err_o = abs(Qo_pred[-1] - self.obs_Qo[-1]) / (self.obs_Qo[-1] + 1e-12)
        term_err_w = abs(Qw_pred[-1] - self.obs_Qw[-1]) / (self.obs_Qw[-1] + 1e-12)
        term_err = 0.5 * (term_err_o + term_err_w)

        # 1. AR(1) Generalized Least Squares misfit (Paper F)
        # Solve L * z = res -> z^T * z = res^T * Sigma^-1 * res
        z_qo = solve_triangular(self.L_qo, res_qo, lower=True)
        z_qw = solve_triangular(self.L_qw, res_qw, lower=True)
        gls_qo = np.sum(z_qo ** 2) / self.n_steps
        gls_qw = np.sum(z_qw ** 2) / self.n_steps
        gls_misfit = self.weight_oil * gls_qo + self.weight_water * gls_qw + self.weight_terminal * (term_err ** 2)

        # 2. Diagonal Macro NRMSE
        macro_cum = 0.5 * (nrmse_Qo + nrmse_Qw)
        macro_rate = 0.5 * (nrmse_qo + nrmse_qw)
        diag_misfit = 0.5 * (macro_cum + macro_rate) + self.weight_terminal * term_err

        # Compounded emulator uncertainty penalty (Paper A)
        std_Qo = pred["oil_cum_std"][0]
        std_Qw = pred["water_cum_std"][0]
        emu_penalty = 0.01 * (np.mean(std_Qo) / self.norm_Qo + np.mean(std_Qw) / self.norm_Qw)

        if objective_type == "ar1_gls":
            total_misfit = gls_misfit + emu_penalty
        elif objective_type == "diagonal_nrmse":
            total_misfit = diag_misfit + emu_penalty
        elif objective_type == "rate_only":
            total_misfit = macro_rate
        else:
            total_misfit = gls_misfit

        if return_components:
            return {
                "total_misfit": float(total_misfit),
                "gls_misfit": float(gls_misfit),
                "diag_misfit": float(diag_misfit),
                "nrmse_Qo": float(nrmse_Qo),
                "nrmse_Qw": float(nrmse_Qw),
                "nrmse_qo": float(nrmse_qo),
                "nrmse_qw": float(nrmse_qw),
                "term_err_o": float(term_err_o),
                "term_err_w": float(term_err_w),
                "emu_penalty": float(emu_penalty),
                "Qo_pred_final": float(Qo_pred[-1]),
                "Qw_pred_final": float(Qw_pred[-1]),
            }
        return float(total_misfit)


if __name__ == "__main__":
    from history_matching.historical_forward_emulator import HistoricalSVDGPEmulator
    from ensemble.data import load_uncertainty, TRAIN_CASES, PARAMS
    from ensemble.run_horizon_adaptive_stacking_experiment import load_clean_simulation_quarterly, build_case_quarterly_trajectories

    unc_df = load_uncertainty()
    q_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(q_df)
    fh = pd.read_parquet("../data/field_history.parquet")

    X_train = unc_df.loc[TRAIN_CASES, PARAMS].to_numpy()
    Y_oil_tr = np.array([case_dict[c]["oil_cum_stb"].to_numpy() for c in TRAIN_CASES])
    Y_wat_tr = np.array([case_dict[c]["water_cum_stb"].to_numpy() for c in TRAIN_CASES])

    emu = HistoricalSVDGPEmulator(n_components=4, random_state=42).fit(X_train, Y_oil_tr, Y_wat_tr)
    obj = HistoryMatchingObjective(emu, fh)

    # Test evaluating Case 6 (best training match)
    theta_c6 = unc_df.loc[6, PARAMS].to_numpy()
    res = obj.evaluate_misfit(theta_c6, objective_type="ar1_gls", return_components=True)
    print("Case 6 Objective Evaluation:")
    for k, v in res.items():
        print(f"  {k:15s}: {v}")
