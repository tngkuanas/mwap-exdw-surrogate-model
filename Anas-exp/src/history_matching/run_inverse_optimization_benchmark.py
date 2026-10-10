"""
Module: run_inverse_optimization_benchmark.py
Implements Phase 4: Benchmark Competing Inverse Optimization Methods for History Matching.
Compares:
- Optimizer A: Direct Best-Match Baseline (Model H0)
- Optimizer B: Multi-Start Local Optimization (L-BFGS-B, 25 starts)
- Optimizer C: Differential Evolution (DE/best/1/bin)
- Optimizer D: Bayesian Optimization (GP-assisted Expected Improvement)
- Optimizer E: ES-MDA (Ensemble Smoother with Multiple Data Assimilation, 5 steps)
Evaluates on both actual field observations and pseudo-field held-out cases.
"""
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.special import expit, logit

from ensemble.data import load_uncertainty, TRAIN_CASES, VAL_CASES, PARAMS
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


def to_unit_hypercube(theta: np.ndarray) -> np.ndarray:
    return (theta - LOWER_BOUNDS) / (UPPER_BOUNDS - LOWER_BOUNDS)


def from_unit_hypercube(u: np.ndarray) -> np.ndarray:
    return LOWER_BOUNDS + u * (UPPER_BOUNDS - LOWER_BOUNDS)


# ---------------------------------------------------------------------------
# OPTIMIZERS IMPLEMENTATION
# ---------------------------------------------------------------------------

def run_optimizer_a_baseline(obj: HistoryMatchingObjective, train_params: np.ndarray) -> Dict[str, Any]:
    """Optimizer A: Direct Best-Match from existing training simulator runs."""
    t0 = time.perf_counter()
    best_misfit = np.inf
    best_theta = None
    best_comp = None

    for i, theta in enumerate(train_params):
        comp = obj.evaluate_misfit(theta, objective_type="ar1_gls", return_components=True)
        if comp["total_misfit"] < best_misfit:
            best_misfit = comp["total_misfit"]
            best_theta = theta
            best_comp = comp

    elapsed = time.perf_counter() - t0
    return {
        "optimizer": "Optimizer_A_Direct_Baseline",
        "best_misfit": best_misfit,
        "best_theta": best_theta,
        "n_evals": len(train_params),
        "runtime_sec": elapsed,
        "components": best_comp,
    }


def run_optimizer_b_multistart_lbfgsb(
    obj: HistoryMatchingObjective, n_starts: int = 25, max_evals: int = 500, seed: int = 42
) -> Dict[str, Any]:
    """Optimizer B: Multi-start bounded L-BFGS-B local optimization."""
    t0 = time.perf_counter()
    np.random.seed(seed)
    eval_count = [0]

    def wrapper(theta):
        eval_count[0] += 1
        return obj.evaluate_misfit(theta, objective_type="ar1_gls")

    best_misfit = np.inf
    best_theta = None
    starts = np.random.uniform(LOWER_BOUNDS, UPPER_BOUNDS, size=(n_starts, 4))

    for start_theta in starts:
        if eval_count[0] >= max_evals:
            break
        res = minimize(
            wrapper,
            start_theta,
            method="L-BFGS-B",
            bounds=PARAM_BOUNDS,
            options={"maxfun": 25, "ftol": 1e-6},
        )
        if res.fun < best_misfit:
            best_misfit = res.fun
            best_theta = res.x

    elapsed = time.perf_counter() - t0
    best_comp = obj.evaluate_misfit(best_theta, objective_type="ar1_gls", return_components=True)
    return {
        "optimizer": "Optimizer_B_MultiStart_LBFGSB",
        "best_misfit": best_misfit,
        "best_theta": best_theta,
        "n_evals": eval_count[0],
        "runtime_sec": elapsed,
        "components": best_comp,
    }


def run_optimizer_c_differential_evolution(
    obj: HistoryMatchingObjective, max_evals: int = 500, seed: int = 42
) -> Dict[str, Any]:
    """Optimizer C: Population-based Differential Evolution."""
    t0 = time.perf_counter()
    eval_count = [0]

    def wrapper(theta):
        eval_count[0] += 1
        return obj.evaluate_misfit(theta, objective_type="ar1_gls")

    popsize = 15
    maxiter = int(max_evals / (popsize * 4))

    res = differential_evolution(
        wrapper,
        bounds=PARAM_BOUNDS,
        strategy="best1bin",
        popsize=popsize,
        mutation=(0.5, 0.9),
        recombination=0.8,
        seed=seed,
        maxiter=maxiter,
    )

    elapsed = time.perf_counter() - t0
    best_comp = obj.evaluate_misfit(res.x, objective_type="ar1_gls", return_components=True)
    return {
        "optimizer": "Optimizer_C_Differential_Evolution",
        "best_misfit": res.fun,
        "best_theta": res.x,
        "n_evals": eval_count[0],
        "runtime_sec": elapsed,
        "components": best_comp,
    }


def run_optimizer_d_bayesian_optimization(
    obj: HistoryMatchingObjective, n_init: int = 20, n_iter: int = 80, seed: int = 42
) -> Dict[str, Any]:
    """Optimizer D: Bayesian Optimization using GP surrogate and Expected Improvement."""
    t0 = time.perf_counter()
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    from scipy.stats import norm

    np.random.seed(seed)
    # Initial sample in unit hypercube
    u_samples = np.random.uniform(0.0, 1.0, size=(n_init, 4))
    y_vals = np.array([obj.evaluate_misfit(from_unit_hypercube(u)) for u in u_samples])

    for it in range(n_iter):
        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5) + WhiteKernel(1e-5),
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=seed + it,
        ).fit(u_samples, y_vals)

        # Maximize Expected Improvement over candidate grid
        candidates = np.random.uniform(0.0, 1.0, size=(500, 4))
        mu, sigma = gp.predict(candidates, return_std=True)
        sigma = np.maximum(sigma, 1e-9)

        f_best = np.min(y_vals)
        improvement = f_best - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)

        best_cand = candidates[np.argmax(ei)]
        new_y = obj.evaluate_misfit(from_unit_hypercube(best_cand))

        u_samples = np.vstack([u_samples, best_cand])
        y_vals = np.append(y_vals, new_y)

    best_idx = np.argmin(y_vals)
    best_theta = from_unit_hypercube(u_samples[best_idx])
    elapsed = time.perf_counter() - t0
    best_comp = obj.evaluate_misfit(best_theta, objective_type="ar1_gls", return_components=True)

    return {
        "optimizer": "Optimizer_D_Bayesian_Optimization",
        "best_misfit": float(y_vals[best_idx]),
        "best_theta": best_theta,
        "n_evals": len(y_vals),
        "runtime_sec": elapsed,
        "components": best_comp,
    }


def run_optimizer_e_esmda(
    obj: HistoryMatchingObjective, n_ensemble: int = 100, n_assimilations: int = 4, seed: int = 42
) -> Dict[str, Any]:
    """
    Optimizer E: ES-MDA (Ensemble Smoother with Multiple Data Assimilation) per Paper B (Emerick & Reynolds 2013).
    Operates on logit-transformed bounded parameters to ensure physical bounds.
    """
    t0 = time.perf_counter()
    np.random.seed(seed)
    alphas = [float(n_assimilations)] * n_assimilations  # sum(1/alpha) = 1.0

    # Initial ensemble drawn from prior (unit hypercube -> logit space)
    u_ens = np.random.uniform(0.01, 0.99, size=(n_ensemble, 4))
    z_ens = logit(u_ens)

    # Observation vector: observed oil and water rates (2 * 41 = 82 data points)
    d_obs = np.concatenate([obj.obs_qo, obj.obs_qw])
    n_d = len(d_obs)

    # Observation noise covariance
    sigma_d = 0.05 * np.maximum(d_obs, 100.0)
    C_D = np.diag(sigma_d ** 2)

    eval_count = 0

    for l_idx, alpha_l in enumerate(alphas):
        # Forecast step
        u_curr = expit(z_ens)
        theta_curr = from_unit_hypercube(u_curr)

        Y_preds = []
        for j in range(n_ensemble):
            pred_j = obj.emulator.predict(theta_curr[j].reshape(1, -1))
            d_j = np.concatenate([pred_j["oil_rate"][0], pred_j["water_rate"][0]])
            Y_preds.append(d_j)
            eval_count += 1

        Y_ens = np.array(Y_preds)  # (Ne, Nd)

        # Compute ensemble covariances
        z_mean = np.mean(z_ens, axis=0)
        Y_mean = np.mean(Y_ens, axis=0)
        Z_dev = z_ens - z_mean
        Y_dev = Y_ens - Y_mean

        C_zd = (Z_dev.T @ Y_dev) / (n_ensemble - 1)  # (4, Nd)
        C_dd = (Y_dev.T @ Y_dev) / (n_ensemble - 1)  # (Nd, Nd)

        # Kalman gain with inflated covariance alpha_l * C_D
        C_total = C_dd + alpha_l * C_D
        # Inversion via pseudo-inverse / regularized solve
        K_l = C_zd @ np.linalg.pinv(C_total)

        # Perturb observations
        noise = np.random.normal(0.0, 1.0, size=(n_ensemble, n_d))
        d_pert = d_obs[None, :] + np.sqrt(alpha_l) * (noise * sigma_d[None, :])

        # Assimilation update
        z_ens = z_ens + (K_l @ (d_pert - Y_ens).T).T

    # Final ensemble
    u_final = expit(z_ens)
    theta_final = from_unit_hypercube(u_final)

    # Evaluate misfits of final ensemble
    misfits = [obj.evaluate_misfit(th, objective_type="ar1_gls") for th in theta_final]
    best_idx = np.argmin(misfits)
    best_theta = theta_final[best_idx]
    elapsed = time.perf_counter() - t0
    best_comp = obj.evaluate_misfit(best_theta, objective_type="ar1_gls", return_components=True)

    return {
        "optimizer": "Optimizer_E_ES_MDA",
        "best_misfit": float(misfits[best_idx]),
        "best_theta": best_theta,
        "n_evals": eval_count,
        "runtime_sec": elapsed,
        "ensemble_thetas": theta_final,
        "ensemble_misfits": np.array(misfits),
        "components": best_comp,
    }


def run_comprehensive_benchmark() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs all 5 optimizers on field history and benchmarks performance."""
    print("=" * 80)
    print("PHASE 4: INVERSE OPTIMIZATION BENCHMARK ON OBSERVED FIELD HISTORY")
    print("=" * 80)

    unc_df = load_uncertainty()
    q_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(q_df)
    fh = pd.read_parquet("../data/field_history.parquet")

    X_train = unc_df.loc[TRAIN_CASES, PARAMS].to_numpy()
    Y_oil_tr = np.array([case_dict[c]["oil_cum_stb"].to_numpy() for c in TRAIN_CASES])
    Y_wat_tr = np.array([case_dict[c]["water_cum_stb"].to_numpy() for c in TRAIN_CASES])

    # Fit forward emulator
    emulator = HistoricalSVDGPEmulator(n_components=4, random_state=42).fit(X_train, Y_oil_tr, Y_wat_tr)
    obj = HistoryMatchingObjective(emulator, fh)

    opt_results = []
    opt_runs = {}

    # 1. Optimizer A
    print("Running Optimizer A (Direct Training Baseline)...")
    res_a = run_optimizer_a_baseline(obj, X_train)
    opt_results.append(res_a)
    opt_runs["Optimizer_A"] = res_a

    # 2. Optimizer B
    print("Running Optimizer B (Multi-Start L-BFGS-B)...")
    res_b = run_optimizer_b_multistart_lbfgsb(obj, n_starts=25, max_evals=500, seed=42)
    opt_results.append(res_b)
    opt_runs["Optimizer_B"] = res_b

    # 3. Optimizer C
    print("Running Optimizer C (Differential Evolution)...")
    res_c = run_optimizer_c_differential_evolution(obj, max_evals=500, seed=42)
    opt_results.append(res_c)
    opt_runs["Optimizer_C"] = res_c

    # 4. Optimizer D
    print("Running Optimizer D (Bayesian Optimization)...")
    res_d = run_optimizer_d_bayesian_optimization(obj, n_init=20, n_iter=80, seed=42)
    opt_results.append(res_d)
    opt_runs["Optimizer_D"] = res_d

    # 5. Optimizer E
    print("Running Optimizer E (ES-MDA)...")
    res_e = run_optimizer_e_esmda(obj, n_ensemble=100, n_assimilations=4, seed=42)
    opt_results.append(res_e)
    opt_runs["Optimizer_E"] = res_e

    # Build comparison DataFrame
    table_rows = []
    for r in opt_results:
        th = r["best_theta"]
        comp = r["components"]
        table_rows.append({
            "Optimizer": r["optimizer"],
            "GLS_Misfit": r["best_misfit"],
            "Diag_NRMSE": comp["diag_misfit"],
            "Oil_Cum_NRMSE": comp["nrmse_Qo"],
            "Water_Cum_NRMSE": comp["nrmse_Qw"],
            "Terminal_Oil_Err": comp["term_err_o"],
            "Terminal_Wat_Err": comp["term_err_w"],
            "Ft": th[0],
            "Porosity": th[1],
            "Permeability": th[2],
            "Aquifer_PV": th[3],
            "Evaluations": r["n_evals"],
            "Runtime_s": r["runtime_sec"],
        })

    summary_df = pd.DataFrame(table_rows).sort_values("GLS_Misfit").reset_index(drop=True)
    out_path = Path("outputs/history_matching/tables/optimizer_benchmark_summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved Optimizer Benchmark: {out_path}")
    print(summary_df.to_string())

    # -----------------------------------------------------------------------
    # PSEUDO-FIELD RECOVERY EXPERIMENT ON HELDOUT CASES
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PSEUDO-FIELD INVERSE RECOVERY EVALUATION (CASES 71, 73, 76, 81)")
    print("=" * 80)
    pseudo_cases = [71, 73, 76, 81]
    pseudo_records = []

    for c in pseudo_cases:
        df_c = case_dict[c]
        true_theta = unc_df.loc[c, PARAMS].to_numpy()
        # Build pseudo field history DataFrame
        pseudo_fh = pd.DataFrame({
            "date": df_c["date"],
            "ramp_oil_cum_stb": df_c["oil_cum_stb"],
            "ramp_water_cum_stb": df_c["water_cum_stb"],
            "ramp_gas_cum_mscf": df_c["gas_cum_mscf"],
            "ramp_oil_rate_stbd": df_c["oil_rate_stbd"],
            "ramp_water_rate_stbd": df_c["water_rate_stbd"],
            "ramp_gas_rate_mscfd": df_c["gas_rate_mscfd"],
        })
        pseudo_obj = HistoryMatchingObjective(emulator, pseudo_fh)

        # Test Differential Evolution on pseudo-field
        res_rec = run_optimizer_c_differential_evolution(pseudo_obj, max_evals=400, seed=42)
        rec_theta = res_rec["best_theta"]
        rel_param_err = np.abs(rec_theta - true_theta) / true_theta

        pseudo_records.append({
            "pseudo_case": c,
            "true_Ft": true_theta[0], "inferred_Ft": rec_theta[0], "err_Ft": rel_param_err[0],
            "true_Poro": true_theta[1], "inferred_Poro": rec_theta[1], "err_Poro": rel_param_err[1],
            "true_Perm": true_theta[2], "inferred_Perm": rec_theta[2], "err_Perm": rel_param_err[2],
            "true_AqPV": true_theta[3], "inferred_AqPV": rec_theta[3], "err_AqPV": rel_param_err[3],
            "recovery_misfit": res_rec["best_misfit"],
            "traj_oil_nrmse": res_rec["components"]["nrmse_Qo"],
            "traj_wat_nrmse": res_rec["components"]["nrmse_Qw"],
        })

    pseudo_df = pd.DataFrame(pseudo_records)
    pseudo_out = Path("outputs/history_matching/tables/pseudo_field_recovery_benchmark.csv")
    pseudo_df.to_csv(pseudo_out, index=False)
    print(f"Saved Pseudo-Field Recovery Benchmark: {pseudo_out}")
    print(pseudo_df.to_string())

    return summary_df, opt_runs


if __name__ == "__main__":
    run_comprehensive_benchmark()
