"""
Module: emulator_revalidation_v2.py
Implements Section 3 of Research Iteration 2:
Rigorous revalidation of the Historical Forward SVD-GP (K=4) emulator.
Key features:
1. Exact time-aligned rate differentiation using quarterly calendar day intervals (dt in [90, 91, 92] days).
2. Distinguishes forward production rates (aligned with field history) from backward rates.
3. Computes cumulative trajectories, incremental cumulative volumes, quarterly rates, terminal states,
   water breakthrough timing, empirical GP uncertainty coverage, and parameter interpolation support.
4. Performs 5-fold cross-validation on Cases 1-70 and secondary held-out validation on Cases 71-85.
"""
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay
from sklearn.decomposition import TruncatedSVD
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
    RS_SOLUTION,
)
from ensemble.data import load_uncertainty, TRAIN_CASES, VAL_CASES, PARAMS


def compute_quarterly_days(dates: pd.Series) -> np.ndarray:
    """Computes exact calendar days in each quarterly interval."""
    dt_series = pd.to_datetime(dates)
    days = dt_series.diff().dt.days.to_numpy()
    # First entry is NaN; subsequent entries are days from prev to current
    return days


class HistoricalSVDGPEmulatorV2:
    """
    Forward historical trajectory emulator using Truncated SVD basis decomposition
    and Gaussian Process Regression with Matérn-5/2 kernels, with rigorous time alignment.
    """
    def __init__(self, n_components: int = 4, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.scaler_X = StandardScaler()
        self.svd_oil = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.svd_wat = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.gps_oil: List[GaussianProcessRegressor] = []
        self.gps_wat: List[GaussianProcessRegressor] = []
        self.dates: Optional[pd.Series] = None
        self.days_fwd: Optional[np.ndarray] = None  # (40,) days between t_i and t_{i+1}
        self.diff_matrix_fwd: Optional[np.ndarray] = None  # (40, 41) differencing operator
        self.train_X_norm: Optional[np.ndarray] = None
        self.fitted = False

    def fit(self, X: np.ndarray, Y_oil: np.ndarray, Y_wat: np.ndarray, dates: Optional[pd.Series] = None):
        """
        X: (N, 4) geological parameters
        Y_oil: (N, 41) historical cumulative oil trajectories
        Y_wat: (N, 41) historical cumulative water trajectories
        dates: (41,) quarterly timestamps
        """
        self.dates = pd.to_datetime(dates) if dates is not None else pd.date_range("1998-01-01", "2008-01-01", freq="QS")
        dt_days = np.diff(self.dates).astype("timedelta64[D]").astype(float)
        self.days_fwd = dt_days  # length 40

        # Construct forward differencing operator D: (40, 41)
        # (D @ Y)[i] = (Y[i+1] - Y[i]) / days_fwd[i]
        D = np.zeros((len(self.days_fwd), len(self.dates)))
        for i in range(len(self.days_fwd)):
            D[i, i] = -1.0 / self.days_fwd[i]
            D[i, i + 1] = 1.0 / self.days_fwd[i]
        self.diff_matrix_fwd = D

        X_sc = self.scaler_X.fit_transform(X)
        self.train_X_norm = X_sc
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

    def predict(self, X: np.ndarray, return_cov: bool = False, return_std: bool = False) -> Dict[str, Any]:
        """
        Predicts forward trajectories and uncertainties.
        Returns:
          - oil_cum, water_cum, gas_cum: (N, 41)
          - oil_cum_inc, water_cum_inc: (N, 40)
          - oil_rate_fwd, water_rate_fwd: (N, 40)
          - terminal_oil_rate, terminal_water_rate: (N,)
          - oil_cum_std, water_cum_std: (N, 41) (if return_std or return_cov)
          - oil_rate_cov, water_rate_cov: (N, 40, 40) (if return_cov)
        """
        assert self.fitted, "Emulator must be fitted before predict"
        X_sc = self.scaler_X.transform(X)
        N_query = len(X)

        C_pred_oil = []
        C_std_oil = []
        for gp in self.gps_oil:
            if return_std or return_cov:
                m, s = gp.predict(X_sc, return_std=True)
                C_pred_oil.append(m)
                C_std_oil.append(s)
            else:
                C_pred_oil.append(gp.predict(X_sc))

        C_pred_wat = []
        C_std_wat = []
        for gp in self.gps_wat:
            if return_std or return_cov:
                m, s = gp.predict(X_sc, return_std=True)
                C_pred_wat.append(m)
                C_std_wat.append(s)
            else:
                C_pred_wat.append(gp.predict(X_sc))

        C_mat_oil = np.column_stack(C_pred_oil)
        C_mat_wat = np.column_stack(C_pred_wat)

        Y_pred_oil = self.svd_oil.inverse_transform(C_mat_oil)
        Y_pred_wat = self.svd_wat.inverse_transform(C_mat_wat)

        # Monotonicity and non-negativity enforcement
        Y_pred_oil = np.maximum(Y_pred_oil, 0.0)
        Y_pred_wat = np.maximum(Y_pred_wat, 0.0)
        for i in range(N_query):
            Y_pred_oil[i] = np.maximum.accumulate(Y_pred_oil[i])
            Y_pred_wat[i] = np.maximum.accumulate(Y_pred_wat[i])

        Y_pred_gas = RS_SOLUTION * Y_pred_oil

        # Incremental cumulative volumes: (N, 40)
        inc_oil = np.diff(Y_pred_oil, axis=1)
        inc_wat = np.diff(Y_pred_wat, axis=1)

        # Forward rates: (N, 40)
        D = self.diff_matrix_fwd
        rates_oil_fwd = (D @ Y_pred_oil.T).T
        rates_wat_fwd = (D @ Y_pred_wat.T).T

        # Terminal rates at 2008-01-01 (extrapolated from the final quarter rate trend)
        # Using linear trend of the last 2 quarters
        q_term_oil = rates_oil_fwd[:, -1] + (rates_oil_fwd[:, -1] - rates_oil_fwd[:, -2]) * 0.5
        q_term_wat = rates_wat_fwd[:, -1] + (rates_wat_fwd[:, -1] - rates_wat_fwd[:, -2]) * 0.5
        q_term_oil = np.maximum(q_term_oil, 100.0)
        q_term_wat = np.maximum(q_term_wat, 100.0)

        res = {
            "oil_cum": Y_pred_oil,
            "water_cum": Y_pred_wat,
            "gas_cum": Y_pred_gas,
            "oil_cum_inc": inc_oil,
            "water_cum_inc": inc_wat,
            "oil_rate_fwd": rates_oil_fwd,
            "water_rate_fwd": rates_wat_fwd,
            "terminal_oil_rate": q_term_oil,
            "terminal_water_rate": q_term_wat,
        }

        if return_std or return_cov:
            C_std_oil_arr = np.column_stack(C_std_oil)  # (N, K)
            C_std_wat_arr = np.column_stack(C_std_wat)  # (N, K)

            # Cumulative standard deviations (N, 41)
            std_oil_cum = np.sqrt(np.dot(np.square(C_std_oil_arr), np.square(self.svd_oil.components_)))
            std_wat_cum = np.sqrt(np.dot(np.square(C_std_wat_arr), np.square(self.svd_wat.components_)))
            res["oil_cum_std"] = std_oil_cum
            res["water_cum_std"] = std_wat_cum

            if return_cov:
                n_int = len(self.days_fwd)
                cov_rate_oil = np.zeros((N_query, n_int, n_int))
                cov_rate_wat = np.zeros((N_query, n_int, n_int))
                V_o = self.svd_oil.components_.T  # (41, K)
                V_w = self.svd_wat.components_.T  # (41, K)
                DV_o = D @ V_o  # (40, K)
                DV_w = D @ V_w  # (40, K)

                for i in range(N_query):
                    S_o = np.diag(np.square(C_std_oil_arr[i]))
                    S_w = np.diag(np.square(C_std_wat_arr[i]))
                    cov_rate_oil[i] = DV_o @ S_o @ DV_o.T
                    cov_rate_wat[i] = DV_w @ S_w @ DV_w.T

                res["oil_rate_cov"] = cov_rate_oil
                res["water_rate_cov"] = cov_rate_wat

        return res

    def compute_interpolation_distance(self, X: np.ndarray) -> np.ndarray:
        """Computes Euclidean distance in normalized parameter space to nearest training point."""
        X_sc = self.scaler_X.transform(X)
        dists = np.zeros(len(X))
        for i, pt in enumerate(X_sc):
            dists[i] = np.min(np.linalg.norm(self.train_X_norm - pt, axis=1))
        return dists


def compute_water_breakthrough_quarter(water_cum: np.ndarray, threshold_stb: float = 5000.0) -> np.ndarray:
    """Returns the integer quarter index (0..40) when water cumulative exceeds threshold."""
    quarters = np.zeros(len(water_cum), dtype=int)
    for i in range(len(water_cum)):
        idx = np.where(water_cum[i] >= threshold_stb)[0]
        quarters[i] = idx[0] if len(idx) > 0 else len(water_cum[i]) - 1
    return quarters


def evaluate_comprehensive_emulator_metrics(
    pred_dict: Dict[str, Any],
    true_cum_oil: np.ndarray,
    true_cum_wat: np.ndarray,
    true_rates_oil: np.ndarray,
    true_rates_wat: np.ndarray,
) -> Dict[str, float]:
    """Computes full suite of cumulative, rate, terminal, and coverage metrics."""
    # 1. Raw Cumulative NRMSE
    denom_oil_cum = np.sqrt(np.mean(true_cum_oil ** 2, axis=1, keepdims=True)) + 1e-12
    denom_wat_cum = np.sqrt(np.mean(true_cum_wat ** 2, axis=1, keepdims=True)) + 1e-12
    nrmse_oil_cum = np.mean(np.sqrt(np.mean((pred_dict["oil_cum"] - true_cum_oil) ** 2, axis=1, keepdims=True)) / denom_oil_cum)
    nrmse_wat_cum = np.mean(np.sqrt(np.mean((pred_dict["water_cum"] - true_cum_wat) ** 2, axis=1, keepdims=True)) / denom_wat_cum)
    macro_cum = 0.5 * (nrmse_oil_cum + nrmse_wat_cum)

    # 2. Incremental Cumulative NRMSE (40 intervals)
    true_inc_oil = np.diff(true_cum_oil, axis=1)
    true_inc_wat = np.diff(true_cum_wat, axis=1)
    denom_inc_oil = np.sqrt(np.mean(true_inc_oil ** 2, axis=1, keepdims=True)) + 1e-12
    denom_inc_wat = np.sqrt(np.mean(true_inc_wat ** 2, axis=1, keepdims=True)) + 1e-12
    nrmse_inc_oil = np.mean(np.sqrt(np.mean((pred_dict["oil_cum_inc"] - true_inc_oil) ** 2, axis=1, keepdims=True)) / denom_inc_oil)
    nrmse_inc_wat = np.mean(np.sqrt(np.mean((pred_dict["water_cum_inc"] - true_inc_wat) ** 2, axis=1, keepdims=True)) / denom_inc_wat)
    macro_inc = 0.5 * (nrmse_inc_oil + nrmse_inc_wat)

    # 3. Quarterly Forward Rate NRMSE (40 intervals)
    denom_oil_rate = np.sqrt(np.mean(true_rates_oil ** 2, axis=1, keepdims=True)) + 1e-12
    denom_wat_rate = np.sqrt(np.mean(true_rates_wat ** 2, axis=1, keepdims=True)) + 1e-12
    nrmse_oil_rate = np.mean(np.sqrt(np.mean((pred_dict["oil_rate_fwd"] - true_rates_oil) ** 2, axis=1, keepdims=True)) / denom_oil_rate)
    nrmse_wat_rate = np.mean(np.sqrt(np.mean((pred_dict["water_rate_fwd"] - true_rates_wat) ** 2, axis=1, keepdims=True)) / denom_wat_rate)
    macro_rate = 0.5 * (nrmse_oil_rate + nrmse_wat_rate)

    # 4. Terminal Errors at 2008-01-01
    term_oil_err = np.mean(np.abs(pred_dict["oil_cum"][:, -1] - true_cum_oil[:, -1]) / (true_cum_oil[:, -1] + 1e-12))
    term_wat_err = np.mean(np.abs(pred_dict["water_cum"][:, -1] - true_cum_wat[:, -1]) / (true_cum_wat[:, -1] + 1e-12))

    # 5. Water Breakthrough Timing Mismatch (in quarters)
    wb_pred = compute_water_breakthrough_quarter(pred_dict["water_cum"])
    wb_true = compute_water_breakthrough_quarter(true_cum_wat)
    wb_mae_quarters = float(np.mean(np.abs(wb_pred - wb_true)))

    # 6. Empirical GP Uncertainty Coverage (Nominal 95% = +/- 1.96 std)
    if "oil_cum_std" in pred_dict:
        oil_std = pred_dict["oil_cum_std"]
        wat_std = pred_dict["water_cum_std"]
        in_95_oil = np.mean((true_cum_oil >= pred_dict["oil_cum"] - 1.96 * oil_std) &
                            (true_cum_oil <= pred_dict["oil_cum"] + 1.96 * oil_std))
        in_95_wat = np.mean((true_cum_wat >= pred_dict["water_cum"] - 1.96 * wat_std) &
                            (true_cum_wat <= pred_dict["water_cum"] + 1.96 * wat_std))
        cov_95 = 0.5 * (float(in_95_oil) + float(in_95_wat))
    else:
        cov_95 = np.nan

    # 7. Worst-case Case-Level Macro Cumulative NRMSE
    case_macro_cum = 0.5 * (
        (np.sqrt(np.mean((pred_dict["oil_cum"] - true_cum_oil) ** 2, axis=1)) / denom_oil_cum.squeeze()) +
        (np.sqrt(np.mean((pred_dict["water_cum"] - true_cum_wat) ** 2, axis=1)) / denom_wat_cum.squeeze())
    )
    worst_case_macro = float(np.max(case_macro_cum))

    return {
        "oil_cum_nrmse": float(nrmse_oil_cum),
        "water_cum_nrmse": float(nrmse_wat_cum),
        "macro_cum_nrmse": float(macro_cum),
        "oil_inc_nrmse": float(nrmse_inc_oil),
        "water_inc_nrmse": float(nrmse_inc_wat),
        "macro_inc_nrmse": float(macro_inc),
        "oil_rate_nrmse": float(nrmse_oil_rate),
        "water_rate_nrmse": float(nrmse_wat_rate),
        "macro_rate_nrmse": float(macro_rate),
        "terminal_oil_rel_err": float(term_oil_err),
        "terminal_wat_rel_err": float(term_wat_err),
        "water_breakthrough_mae_qtrs": float(wb_mae_quarters),
        "gp_95_coverage": float(cov_95),
        "worst_case_macro_nrmse": float(worst_case_macro),
    }


def run_emulator_revalidation_experiment() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes Section 3 Emulator Revalidation:
    1. 5-Fold Cross Validation on Training Cases 1-70.
    2. Evaluation on Held-Out Validation Cases 71-85.
    Saves tables to outputs/research/history_matching_iteration_2/tables/.
    """
    print("=" * 80)
    print("RESEARCH ITERATION 2 — SECTION 3: FORWARD EMULATOR REVALIDATION")
    print("=" * 80)

    unc_df = load_uncertainty()
    q_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(q_df)
    dates = pd.to_datetime(case_dict[1]["date"].to_numpy())
    dt_days = np.diff(dates).astype("timedelta64[D]").astype(float)

    X_train = unc_df.loc[TRAIN_CASES, PARAMS].to_numpy()
    Y_oil_tr = np.array([case_dict[c]["oil_cum_stb"].to_numpy() for c in TRAIN_CASES])
    Y_wat_tr = np.array([case_dict[c]["water_cum_stb"].to_numpy() for c in TRAIN_CASES])
    # Forward rates for training cases: (70, 40)
    R_oil_fwd_tr = np.diff(Y_oil_tr, axis=1) / dt_days
    R_wat_fwd_tr = np.diff(Y_wat_tr, axis=1) / dt_days

    X_val = unc_df.loc[VAL_CASES, PARAMS].to_numpy()
    Y_oil_val = np.array([case_dict[c]["oil_cum_stb"].to_numpy() for c in VAL_CASES])
    Y_wat_val = np.array([case_dict[c]["water_cum_stb"].to_numpy() for c in VAL_CASES])
    R_oil_fwd_val = np.diff(Y_oil_val, axis=1) / dt_days
    R_wat_fwd_val = np.diff(Y_wat_val, axis=1) / dt_days

    # 1. 5-Fold Cross Validation on Cases 1-70
    print("\nRunning 5-Fold Cross Validation on Cases 1-70...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_oof_oil_cum = np.zeros_like(Y_oil_tr)
    cv_oof_wat_cum = np.zeros_like(Y_wat_tr)
    cv_oof_oil_inc = np.zeros((len(TRAIN_CASES), 40))
    cv_oof_wat_inc = np.zeros((len(TRAIN_CASES), 40))
    cv_oof_oil_rate = np.zeros((len(TRAIN_CASES), 40))
    cv_oof_wat_rate = np.zeros((len(TRAIN_CASES), 40))
    cv_oof_oil_std = np.zeros_like(Y_oil_tr)
    cv_oof_wat_std = np.zeros_like(Y_wat_tr)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        emu_fold = HistoricalSVDGPEmulatorV2(n_components=4, random_state=42 + fold)
        emu_fold.fit(X_train[tr_idx], Y_oil_tr[tr_idx], Y_wat_tr[tr_idx], dates=dates)
        pred_fold = emu_fold.predict(X_train[val_idx], return_std=True)

        cv_oof_oil_cum[val_idx] = pred_fold["oil_cum"]
        cv_oof_wat_cum[val_idx] = pred_fold["water_cum"]
        cv_oof_oil_inc[val_idx] = pred_fold["oil_cum_inc"]
        cv_oof_wat_inc[val_idx] = pred_fold["water_cum_inc"]
        cv_oof_oil_rate[val_idx] = pred_fold["oil_rate_fwd"]
        cv_oof_wat_rate[val_idx] = pred_fold["water_rate_fwd"]
        cv_oof_oil_std[val_idx] = pred_fold["oil_cum_std"]
        cv_oof_wat_std[val_idx] = pred_fold["water_cum_std"]

    cv_pred_dict = {
        "oil_cum": cv_oof_oil_cum,
        "water_cum": cv_oof_wat_cum,
        "oil_cum_inc": cv_oof_oil_inc,
        "water_cum_inc": cv_oof_wat_inc,
        "oil_rate_fwd": cv_oof_oil_rate,
        "water_rate_fwd": cv_oof_wat_rate,
        "oil_cum_std": cv_oof_oil_std,
        "water_cum_std": cv_oof_wat_std,
    }
    cv_metrics = evaluate_comprehensive_emulator_metrics(
        cv_pred_dict, Y_oil_tr, Y_wat_tr, R_oil_fwd_tr, R_wat_fwd_tr
    )
    cv_metrics["dataset_scope"] = "Cases 1-70 (5-Fold CV OOF)"

    # 2. Secondary Held-Out Validation on Cases 71-85
    print("Evaluating Held-Out Validation on Cases 71-85...")
    emu_full = HistoricalSVDGPEmulatorV2(n_components=4, random_state=42)
    emu_full.fit(X_train, Y_oil_tr, Y_wat_tr, dates=dates)
    val_pred = emu_full.predict(X_val, return_std=True)

    val_metrics = evaluate_comprehensive_emulator_metrics(
        val_pred, Y_oil_val, Y_wat_val, R_oil_fwd_val, R_wat_fwd_val
    )
    val_metrics["dataset_scope"] = "Cases 71-85 (Held-Out Test)"

    # Summary Table
    df_metrics = pd.DataFrame([cv_metrics, val_metrics])
    cols_order = [
        "dataset_scope", "macro_cum_nrmse", "oil_cum_nrmse", "water_cum_nrmse",
        "macro_inc_nrmse", "oil_inc_nrmse", "water_inc_nrmse",
        "macro_rate_nrmse", "oil_rate_nrmse", "water_rate_nrmse",
        "terminal_oil_rel_err", "terminal_wat_rel_err",
        "water_breakthrough_mae_qtrs", "gp_95_coverage", "worst_case_macro_nrmse"
    ]
    df_metrics = df_metrics[cols_order]

    # Save to table
    out_dir = Path("outputs/research/history_matching_iteration_2/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "forward_emulator_revalidation_metrics.csv"
    df_metrics.to_csv(csv_path, index=False)
    print(f"\nSaved revalidation metrics to {csv_path}:")
    print(df_metrics.to_string())

    # Case-level breakdown for held-out cases
    case_rows = []
    dists = emu_full.compute_interpolation_distance(X_val)
    for i, c in enumerate(VAL_CASES):
        denom_o = np.sqrt(np.mean(Y_oil_val[i] ** 2)) + 1e-12
        denom_w = np.sqrt(np.mean(Y_wat_val[i] ** 2)) + 1e-12
        nrmse_o = np.sqrt(np.mean((val_pred["oil_cum"][i] - Y_oil_val[i]) ** 2)) / denom_o
        nrmse_w = np.sqrt(np.mean((val_pred["water_cum"][i] - Y_wat_val[i]) ** 2)) / denom_w
        term_o = abs(val_pred["oil_cum"][i, -1] - Y_oil_val[i, -1]) / Y_oil_val[i, -1]
        term_w = abs(val_pred["water_cum"][i, -1] - Y_wat_val[i, -1]) / Y_wat_val[i, -1]
        case_rows.append({
            "case_id": c,
            "macro_cum_nrmse": 0.5 * (nrmse_o + nrmse_w),
            "oil_cum_nrmse": nrmse_o,
            "water_cum_nrmse": nrmse_w,
            "terminal_oil_err": term_o,
            "terminal_wat_err": term_w,
            "param_dist_to_train": dists[i],
        })
    df_cases = pd.DataFrame(case_rows)
    case_csv_path = out_dir / "held_out_cases_revalidation_breakdown.csv"
    df_cases.to_csv(case_csv_path, index=False)
    print(f"Saved held-out case breakdown to {case_csv_path}")

    return df_metrics, df_cases


if __name__ == "__main__":
    run_emulator_revalidation_experiment()
