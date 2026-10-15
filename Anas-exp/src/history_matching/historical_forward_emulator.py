"""
Module: historical_forward_emulator.py
Implements Phase 2: Validated Forward Historical Emulator for Reservoir History Matching.
Maps 4 uncertainty parameters [Ft, Poro, Perm, AqPV] -> 41-quarter historical production trajectories.
Compares Model H0 (Nearest Neighbor Baseline), Model H1 (SVD-GP K=3, K=4), and Model H2 (Kernel Ridge RBF).
"""
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
    RS_SOLUTION,
)
from ensemble.data import load_uncertainty, TRAIN_CASES, VAL_CASES, PARAMS


class HistoricalSVDGPEmulator:
    """
    Forward historical trajectory emulator using Truncated SVD basis decomposition
    and Gaussian Process Regression with Matern-5/2 kernels.
    """
    def __init__(self, n_components: int = 4, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.scaler_X = StandardScaler()
        self.svd_oil = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.svd_wat = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.gps_oil = []
        self.gps_wat = []
        self.fitted = False

    def fit(self, X: np.ndarray, Y_oil: np.ndarray, Y_wat: np.ndarray):
        """
        X: (N, 4) geological parameters
        Y_oil: (N, 41) historical cumulative oil trajectories
        Y_wat: (N, 41) historical cumulative water trajectories
        """
        X_sc = self.scaler_X.fit_transform(X)
        C_oil = self.svd_oil.fit_transform(Y_oil)
        C_wat = self.svd_wat.fit_transform(Y_wat)

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e4))
            * Matern(length_scale=np.ones(4), length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(1e-4, (1e-6, 1e-1))
        )

        self.gps_oil = [
            GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=5, random_state=self.random_state + k
            ).fit(X_sc, C_oil[:, k])
            for k in range(self.n_components)
        ]
        self.gps_wat = [
            GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=5, random_state=self.random_state + 10 + k
            ).fit(X_sc, C_wat[:, k])
            for k in range(self.n_components)
        ]
        self.fitted = True
        return self

    def predict(self, X: np.ndarray, return_std: bool = False) -> Dict[str, Any]:
        """
        Predicts trajectories for query parameters.
        Returns oil_cum, water_cum, gas_cum, oil_rate, water_rate, gas_rate.
        """
        assert self.fitted, "Model must be fitted before predict"
        X_sc = self.scaler_X.transform(X)

        C_pred_oil = []
        C_std_oil = []
        for gp in self.gps_oil:
            if return_std:
                m, s = gp.predict(X_sc, return_std=True)
                C_pred_oil.append(m)
                C_std_oil.append(s)
            else:
                C_pred_oil.append(gp.predict(X_sc))

        C_pred_wat = []
        C_std_wat = []
        for gp in self.gps_wat:
            if return_std:
                m, s = gp.predict(X_sc, return_std=True)
                C_pred_wat.append(m)
                C_std_wat.append(s)
            else:
                C_pred_wat.append(gp.predict(X_sc))

        C_mat_oil = np.column_stack(C_pred_oil)
        C_mat_wat = np.column_stack(C_pred_wat)

        Y_pred_oil = self.svd_oil.inverse_transform(C_mat_oil)
        Y_pred_wat = self.svd_wat.inverse_transform(C_mat_wat)

        # Non-negativity and monotonic enforcement
        Y_pred_oil = np.maximum(Y_pred_oil, 0.0)
        Y_pred_wat = np.maximum(Y_pred_wat, 0.0)
        for i in range(len(Y_pred_oil)):
            Y_pred_oil[i] = np.maximum.accumulate(Y_pred_oil[i])
            Y_pred_wat[i] = np.maximum.accumulate(Y_pred_wat[i])

        # Gas strictly coupled via Rs = 0.3633
        Y_pred_gas = RS_SOLUTION * Y_pred_oil

        # Approximate rate by discrete differentiation (dt = 91.3125 days)
        dt_days = 91.3125
        rates_oil = np.zeros_like(Y_pred_oil)
        rates_wat = np.zeros_like(Y_pred_wat)
        rates_oil[:, 0] = Y_pred_oil[:, 0] / dt_days
        rates_wat[:, 0] = Y_pred_wat[:, 0] / dt_days
        rates_oil[:, 1:] = np.diff(Y_pred_oil, axis=1) / dt_days
        rates_wat[:, 1:] = np.diff(Y_pred_wat, axis=1) / dt_days
        rates_gas = RS_SOLUTION * rates_oil

        res = {
            "oil_cum": Y_pred_oil,
            "water_cum": Y_pred_wat,
            "gas_cum": Y_pred_gas,
            "oil_rate": rates_oil,
            "water_rate": rates_wat,
            "gas_rate": rates_gas,
        }
        if return_std:
            # Trajectory standard deviations via linear propagation through SVD components
            std_oil = np.sqrt(np.dot(np.square(np.column_stack(C_std_oil)), np.square(self.svd_oil.components_)))
            std_wat = np.sqrt(np.dot(np.square(np.column_stack(C_std_wat)), np.square(self.svd_wat.components_)))
            res["oil_cum_std"] = std_oil
            res["water_cum_std"] = std_wat
            res["gas_cum_std"] = RS_SOLUTION * std_oil
        return res


class HistoricalKernelRidgeEmulator:
    """Alternative multi-output forward emulator using Kernel Ridge Regression with RBF kernel."""
    def __init__(self, alpha: float = 1e-3, gamma: float = 0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.scaler_X = StandardScaler()
        self.krr_oil = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
        self.krr_wat = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
        self.fitted = False

    def fit(self, X: np.ndarray, Y_oil: np.ndarray, Y_wat: np.ndarray):
        X_sc = self.scaler_X.fit_transform(X)
        self.krr_oil.fit(X_sc, Y_oil)
        self.krr_wat.fit(X_sc, Y_wat)
        self.fitted = True
        return self

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        assert self.fitted
        X_sc = self.scaler_X.transform(X)
        Y_oil = np.maximum(self.krr_oil.predict(X_sc), 0.0)
        Y_wat = np.maximum(self.krr_wat.predict(X_sc), 0.0)
        for i in range(len(Y_oil)):
            Y_oil[i] = np.maximum.accumulate(Y_oil[i])
            Y_wat[i] = np.maximum.accumulate(Y_wat[i])
        Y_gas = RS_SOLUTION * Y_oil

        dt_days = 91.3125
        rates_oil = np.zeros_like(Y_oil)
        rates_wat = np.zeros_like(Y_wat)
        rates_oil[:, 0] = Y_oil[:, 0] / dt_days
        rates_wat[:, 0] = Y_wat[:, 0] / dt_days
        rates_oil[:, 1:] = np.diff(Y_oil, axis=1) / dt_days
        rates_wat[:, 1:] = np.diff(Y_wat, axis=1) / dt_days
        rates_gas = RS_SOLUTION * rates_oil

        return {
            "oil_cum": Y_oil,
            "water_cum": Y_wat,
            "gas_cum": Y_gas,
            "oil_rate": rates_oil,
            "water_rate": rates_wat,
            "gas_rate": rates_gas,
        }


def compute_trajectory_metrics(
    Y_pred: np.ndarray, Y_true: np.ndarray, R_pred: np.ndarray, R_true: np.ndarray
) -> Dict[str, float]:
    """Computes trajectory NRMSE, rate NRMSE, terminal error, and worst case error."""
    denom_cum = np.sqrt(np.mean(Y_true ** 2, axis=1, keepdims=True)) + 1e-12
    nrmse_per_case_cum = np.sqrt(np.mean((Y_pred - Y_true) ** 2, axis=1, keepdims=True)) / denom_cum
    cum_nrmse = float(np.mean(nrmse_per_case_cum))

    denom_rate = np.sqrt(np.mean(R_true ** 2, axis=1, keepdims=True)) + 1e-12
    nrmse_per_case_rate = np.sqrt(np.mean((R_pred - R_true) ** 2, axis=1, keepdims=True)) / denom_rate
    rate_nrmse = float(np.mean(nrmse_per_case_rate))

    # Terminal error at 2008-01-01
    terminal_rel_err = float(np.mean(np.abs(Y_pred[:, -1] - Y_true[:, -1]) / (Y_true[:, -1] + 1e-12)))
    worst_case_nrmse = float(np.max(nrmse_per_case_cum))

    return {
        "cum_nrmse": cum_nrmse,
        "rate_nrmse": rate_nrmse,
        "terminal_rel_err": terminal_rel_err,
        "worst_case_nrmse": worst_case_nrmse,
    }


def evaluate_forward_emulators() -> pd.DataFrame:
    """
    Executes Phase 2 Benchmark on Held-Out Cases 71-85:
    - Model H0: Nearest Neighbor (Existing Training Case)
    - Model H1_K3: SVD-GP (K=3)
    - Model H1_K4: SVD-GP (K=4)
    - Model H2: Kernel Ridge Regression (RBF)
    """
    unc_df = load_uncertainty()
    q_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(q_df)

    X_train = unc_df.loc[TRAIN_CASES, PARAMS].to_numpy()
    X_val = unc_df.loc[VAL_CASES, PARAMS].to_numpy()

    Y_oil_tr = np.array([case_dict[c]["oil_cum_stb"].to_numpy() for c in TRAIN_CASES])
    Y_wat_tr = np.array([case_dict[c]["water_cum_stb"].to_numpy() for c in TRAIN_CASES])
    Y_gas_tr = RS_SOLUTION * Y_oil_tr

    R_oil_tr = np.array([case_dict[c]["oil_rate_stbd"].to_numpy() for c in TRAIN_CASES])
    R_wat_tr = np.array([case_dict[c]["water_rate_stbd"].to_numpy() for c in TRAIN_CASES])
    R_gas_tr = RS_SOLUTION * R_oil_tr

    Y_oil_val = np.array([case_dict[c]["oil_cum_stb"].to_numpy() for c in VAL_CASES])
    Y_wat_val = np.array([case_dict[c]["water_cum_stb"].to_numpy() for c in VAL_CASES])
    Y_gas_val = RS_SOLUTION * Y_oil_val

    R_oil_val = np.array([case_dict[c]["oil_rate_stbd"].to_numpy() for c in VAL_CASES])
    R_wat_val = np.array([case_dict[c]["water_rate_stbd"].to_numpy() for c in VAL_CASES])
    R_gas_val = RS_SOLUTION * R_oil_val

    results = []

    # 1. Model H0: Nearest-Neighbor Baseline (picks closest training case in parameter space)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(X_train)
    _, nn_idx = nn.kneighbors(X_val)
    nn_idx = nn_idx.flatten()

    Y_nn_oil = Y_oil_tr[nn_idx]
    Y_nn_wat = Y_wat_tr[nn_idx]
    R_nn_oil = R_oil_tr[nn_idx]
    R_nn_wat = R_wat_tr[nn_idx]

    m_oil_h0 = compute_trajectory_metrics(Y_nn_oil, Y_oil_val, R_nn_oil, R_oil_val)
    m_wat_h0 = compute_trajectory_metrics(Y_nn_wat, Y_wat_val, R_nn_wat, R_wat_val)

    results.append({
        "model_id": "Model_H0_Nearest_Neighbor",
        "oil_cum_nrmse": m_oil_h0["cum_nrmse"],
        "water_cum_nrmse": m_wat_h0["cum_nrmse"],
        "gas_cum_nrmse": m_oil_h0["cum_nrmse"],
        "macro_cum_nrmse": 0.5 * (m_oil_h0["cum_nrmse"] + m_wat_h0["cum_nrmse"]),
        "oil_rate_nrmse": m_oil_h0["rate_nrmse"],
        "water_rate_nrmse": m_wat_h0["rate_nrmse"],
        "terminal_oil_rel_err": m_oil_h0["terminal_rel_err"],
        "terminal_wat_rel_err": m_wat_h0["terminal_rel_err"],
        "worst_case_oil_nrmse": m_oil_h0["worst_case_nrmse"],
        "coverage_90_pct": np.nan,  # Deterministic nearest neighbor
    })

    # 2. Model H1: SVD-GP (K=3)
    emu_k3 = HistoricalSVDGPEmulator(n_components=3, random_state=42).fit(X_train, Y_oil_tr, Y_wat_tr)
    pred_k3 = emu_k3.predict(X_val, return_std=True)
    m_oil_k3 = compute_trajectory_metrics(pred_k3["oil_cum"], Y_oil_val, pred_k3["oil_rate"], R_oil_val)
    m_wat_k3 = compute_trajectory_metrics(pred_k3["water_cum"], Y_wat_val, pred_k3["water_rate"], R_wat_val)

    # Coverage 90%
    z_90 = 1.645
    cov_oil_k3 = np.mean(
        (Y_oil_val >= pred_k3["oil_cum"] - z_90 * pred_k3["oil_cum_std"])
        & (Y_oil_val <= pred_k3["oil_cum"] + z_90 * pred_k3["oil_cum_std"])
    )

    results.append({
        "model_id": "Model_H1_SVD_GP_K3",
        "oil_cum_nrmse": m_oil_k3["cum_nrmse"],
        "water_cum_nrmse": m_wat_k3["cum_nrmse"],
        "gas_cum_nrmse": m_oil_k3["cum_nrmse"],
        "macro_cum_nrmse": 0.5 * (m_oil_k3["cum_nrmse"] + m_wat_k3["cum_nrmse"]),
        "oil_rate_nrmse": m_oil_k3["rate_nrmse"],
        "water_rate_nrmse": m_wat_k3["rate_nrmse"],
        "terminal_oil_rel_err": m_oil_k3["terminal_rel_err"],
        "terminal_wat_rel_err": m_wat_k3["terminal_rel_err"],
        "worst_case_oil_nrmse": m_oil_k3["worst_case_nrmse"],
        "coverage_90_pct": float(cov_oil_k3),
    })

    # 3. Model H1: SVD-GP (K=4)
    emu_k4 = HistoricalSVDGPEmulator(n_components=4, random_state=42).fit(X_train, Y_oil_tr, Y_wat_tr)
    pred_k4 = emu_k4.predict(X_val, return_std=True)
    m_oil_k4 = compute_trajectory_metrics(pred_k4["oil_cum"], Y_oil_val, pred_k4["oil_rate"], R_oil_val)
    m_wat_k4 = compute_trajectory_metrics(pred_k4["water_cum"], Y_wat_val, pred_k4["water_rate"], R_wat_val)
    cov_oil_k4 = np.mean(
        (Y_oil_val >= pred_k4["oil_cum"] - z_90 * pred_k4["oil_cum_std"])
        & (Y_oil_val <= pred_k4["oil_cum"] + z_90 * pred_k4["oil_cum_std"])
    )

    results.append({
        "model_id": "Model_H1_SVD_GP_K4",
        "oil_cum_nrmse": m_oil_k4["cum_nrmse"],
        "water_cum_nrmse": m_wat_k4["cum_nrmse"],
        "gas_cum_nrmse": m_oil_k4["cum_nrmse"],
        "macro_cum_nrmse": 0.5 * (m_oil_k4["cum_nrmse"] + m_wat_k4["cum_nrmse"]),
        "oil_rate_nrmse": m_oil_k4["rate_nrmse"],
        "water_rate_nrmse": m_wat_k4["rate_nrmse"],
        "terminal_oil_rel_err": m_oil_k4["terminal_rel_err"],
        "terminal_wat_rel_err": m_wat_k4["terminal_rel_err"],
        "worst_case_oil_nrmse": m_oil_k4["worst_case_nrmse"],
        "coverage_90_pct": float(cov_oil_k4),
    })

    # 4. Model H2: Kernel Ridge Regression (RBF)
    emu_krr = HistoricalKernelRidgeEmulator(alpha=1e-3, gamma=0.25).fit(X_train, Y_oil_tr, Y_wat_tr)
    pred_krr = emu_krr.predict(X_val)
    m_oil_krr = compute_trajectory_metrics(pred_krr["oil_cum"], Y_oil_val, pred_krr["oil_rate"], R_oil_val)
    m_wat_krr = compute_trajectory_metrics(pred_krr["water_cum"], Y_wat_val, pred_krr["water_rate"], R_wat_val)

    results.append({
        "model_id": "Model_H2_Kernel_Ridge_RBF",
        "oil_cum_nrmse": m_oil_krr["cum_nrmse"],
        "water_cum_nrmse": m_wat_krr["cum_nrmse"],
        "gas_cum_nrmse": m_oil_krr["cum_nrmse"],
        "macro_cum_nrmse": 0.5 * (m_oil_krr["cum_nrmse"] + m_wat_krr["cum_nrmse"]),
        "oil_rate_nrmse": m_oil_krr["rate_nrmse"],
        "water_rate_nrmse": m_wat_krr["rate_nrmse"],
        "terminal_oil_rel_err": m_oil_krr["terminal_rel_err"],
        "terminal_wat_rel_err": m_wat_krr["terminal_rel_err"],
        "worst_case_oil_nrmse": m_oil_krr["worst_case_nrmse"],
        "coverage_90_pct": np.nan,
    })

    res_df = pd.DataFrame(results)
    out_path = Path("outputs/history_matching/tables/forward_emulator_benchmark.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_path, index=False)
    print(f"Saved Forward Emulator Benchmark: {out_path}")
    print(res_df.to_string())
    return res_df


if __name__ == "__main__":
    evaluate_forward_emulators()
