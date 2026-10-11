"""
Module: inverse_optimization_benchmark_v2.py
Implements Section 4 of Research Iteration 2:
Re-running inverse optimization benchmarks under the corrected statistical objectives (L1 and L2).
Key features:
1. Compares Multi-Start L-BFGS-B, Differential Evolution, Bayesian Optimization, and ES-MDA.
2. Identical parameter bounds, evaluation budgets, and multi-seed testing (seeds 42, 123, 456).
3. Investigates and formally resolves the Iteration 1 narrow GLS interval (~42.3).
4. Conducts pseudo-field recovery experiments on held-out validation cases (Cases 71, 73, 76, 81).
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
from history_matching_v2.emulator_revalidation_v2 import (
    HistoricalSVDGPEmulatorV2,
    compute_water_breakthrough_quarter,
)
from history_matching_v2.history_matching_likelihood_v2 import HistoryMatchingLikelihoodV2

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
# OPTIMIZERS
# ---------------------------------------------------------------------------

def run_multistart_lbfgsb(
    lik_obj: HistoryMatchingLikelihoodV2,
    model: str = "L1",
    n_starts: int = 25,
    seed: int = 42,
) -> Dict[str, Any]:
    """Optimizer B: Multi-Start L-BFGS-B in transformed space."""
    t0 = time.perf_counter()
    np.random.seed(seed)
    u_starts = np.random.uniform(0.05, 0.95, size=(n_starts, 4))

    best_loss = np.inf
    best_theta = None
    eval_count = 0

    def objective(u):
        nonlocal eval_count
        eval_count += 1
        th = from_unit_hypercube(np.clip(u, 0.0, 1.0))
        ll = lik_obj.evaluate_log_likelihood(th, model=model)
        # Minimize negative log-likelihood
        return -ll

    for u0 in u_starts:
        res = minimize(
            objective,
            u0,
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * 4,
            options={"maxiter": 30, "ftol": 1e-5},
        )
        if res.fun < best_loss:
            best_loss = res.fun
            best_theta = from_unit_hypercube(np.clip(res.x, 0.0, 1.0))

    elapsed = time.perf_counter() - t0
    comp = lik_obj.evaluate_log_likelihood(best_theta, model=model, return_components=True)
    return {
        "optimizer": "MultiStart_LBFGSB",
        "best_log_likelihood": float(-best_loss),
        "best_theta": best_theta,
        "n_evals": eval_count,
        "runtime_sec": elapsed,
        "components": comp,
    }


def run_differential_evolution(
    lik_obj: HistoryMatchingLikelihoodV2,
    model: str = "L1",
    maxiter: int = 40,
    popsize: int = 15,
    seed: int = 42,
) -> Dict[str, Any]:
    """Optimizer C: Differential Evolution (global search)."""
    t0 = time.perf_counter()
    eval_count = 0

    def objective(u):
        nonlocal eval_count
        eval_count += 1
        th = from_unit_hypercube(np.clip(u, 0.0, 1.0))
        ll = lik_obj.evaluate_log_likelihood(th, model=model)
        return -ll

    res = differential_evolution(
        objective,
        bounds=[(0.0, 1.0)] * 4,
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        polish=True,
    )
    best_theta = from_unit_hypercube(np.clip(res.x, 0.0, 1.0))
    elapsed = time.perf_counter() - t0
    comp = lik_obj.evaluate_log_likelihood(best_theta, model=model, return_components=True)
    return {
        "optimizer": "Differential_Evolution",
        "best_log_likelihood": float(-res.fun),
        "best_theta": best_theta,
        "n_evals": eval_count,
        "runtime_sec": elapsed,
        "components": comp,
    }


def run_bayesian_optimization(
    lik_obj: HistoryMatchingLikelihoodV2,
    model: str = "L1",
    n_init: int = 20,
    n_iters: int = 80,
    seed: int = 42,
) -> Dict[str, Any]:
    """Optimizer D: Bayesian Optimization using GP surrogate with Expected Improvement."""
    t0 = time.perf_counter()
    np.random.seed(seed)
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from scipy.stats import norm

    U_samples = np.random.uniform(0.0, 1.0, size=(n_init, 4))
    Y_losses = []
    for u in U_samples:
        th = from_unit_hypercube(u)
        Y_losses.append(-lik_obj.evaluate_log_likelihood(th, model=model))
    Y_losses = np.array(Y_losses)

    gp = GaussianProcessRegressor(
        kernel=Matern(length_scale=np.ones(4), nu=2.5),
        alpha=1e-4,
        normalize_y=True,
        random_state=seed,
    )

    for it in range(n_iters):
        gp.fit(U_samples, Y_losses)
        best_y = np.min(Y_losses)

        # Optimize Expected Improvement over random candidates
        U_cand = np.random.uniform(0.0, 1.0, size=(1000, 4))
        mu, sigma = gp.predict(U_cand, return_std=True)
        sigma = np.maximum(sigma, 1e-8)

        improvement = best_y - mu
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        next_u = U_cand[np.argmax(ei)]

        th_next = from_unit_hypercube(next_u)
        loss_next = -lik_obj.evaluate_log_likelihood(th_next, model=model)

        U_samples = np.vstack([U_samples, next_u])
        Y_losses = np.append(Y_losses, loss_next)

    best_idx = np.argmin(Y_losses)
    best_theta = from_unit_hypercube(U_samples[best_idx])
    elapsed = time.perf_counter() - t0
    comp = lik_obj.evaluate_log_likelihood(best_theta, model=model, return_components=True)
    return {
        "optimizer": "Bayesian_Optimization",
        "best_log_likelihood": float(-Y_losses[best_idx]),
        "best_theta": best_theta,
        "n_evals": n_init + n_iters,
        "runtime_sec": elapsed,
        "components": comp,
    }


def run_esmda(
    lik_obj: HistoryMatchingLikelihoodV2,
    model: str = "L1",
    n_ensemble: int = 100,
    n_steps: int = 4,
    seed: int = 42,
) -> Dict[str, Any]:
    """Optimizer E: Ensemble Smoother with Multiple Data Assimilation (ES-MDA)."""
    t0 = time.perf_counter()
    np.random.seed(seed)
    alphas = [4.0, 4.0, 4.0, 4.0]  # sum(1/alpha) = 1

    u_ens = np.random.uniform(0.05, 0.95, size=(n_ensemble, 4))
    z_ens = logit(u_ens)

    # Observation vector and standard deviation
    d_obs = np.concatenate([lik_obj.obs_qo_fwd, lik_obj.obs_qw_fwd])
    sigma_d = np.concatenate([
        np.full(len(lik_obj.obs_qo_fwd), lik_obj.sigma_qo),
        np.full(len(lik_obj.obs_qw_fwd), lik_obj.sigma_qw),
    ])
    C_D = np.diag(sigma_d ** 2)
    eval_count = 0

    for alpha_l in alphas:
        # Evaluate ensemble predictions
        Y_ens = []
        for i in range(n_ensemble):
            eval_count += 1
            u_i = expit(z_ens[i])
            th_i = from_unit_hypercube(u_i)
            pred_i = lik_obj.emulator.predict(th_i.reshape(1, -1))
            y_i = np.concatenate([pred_i["oil_rate_fwd"][0], pred_i["water_rate_fwd"][0]])
            Y_ens.append(y_i)
        Y_ens = np.array(Y_ens)

        z_mean = np.mean(z_ens, axis=0, keepdims=True)
        Y_mean = np.mean(Y_ens, axis=0, keepdims=True)
        Z_dev = z_ens - z_mean
        Y_dev = Y_ens - Y_mean

        C_zd = (Z_dev.T @ Y_dev) / (n_ensemble - 1)
        C_dd = (Y_dev.T @ Y_dev) / (n_ensemble - 1)

        C_total = C_dd + alpha_l * C_D
        K_l = C_zd @ np.linalg.pinv(C_total)

        noise = np.random.normal(0.0, 1.0, size=(n_ensemble, len(d_obs)))
        d_pert = d_obs[None, :] + np.sqrt(alpha_l) * (noise * sigma_d[None, :])
        z_ens = z_ens + (K_l @ (d_pert - Y_ens).T).T

    u_final = expit(z_ens)
    theta_final = from_unit_hypercube(u_final)

    lls = [lik_obj.evaluate_log_likelihood(th, model=model) for th in theta_final]
    best_idx = np.argmax(lls)
    best_theta = theta_final[best_idx]
    elapsed = time.perf_counter() - t0
    comp = lik_obj.evaluate_log_likelihood(best_theta, model=model, return_components=True)

    return {
        "optimizer": "ES_MDA",
        "best_log_likelihood": float(lls[best_idx]),
        "best_theta": best_theta,
        "n_evals": eval_count,
        "runtime_sec": elapsed,
        "components": comp,
    }


def run_inverse_optimization_benchmark_v2() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes Section 4:
    1. Re-runs optimizer comparisons on observed field history under corrected L1 objective.
    2. Runs pseudo-field recovery experiments on held-out Cases 71, 73, 76, 81.
    Saves tables to outputs/research/history_matching_iteration_2/tables/.
    """
    print("=" * 80)
    print("RESEARCH ITERATION 2 — SECTION 4: INVERSE OPTIMIZATION BENCHMARK")
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
    lik_obj = HistoryMatchingLikelihoodV2(emu, fh, rho_ar1=0.80, obs_noise_rel_oil=0.05, obs_noise_rel_wat=0.05)

    # 1. Optimizer Comparisons across 3 random seeds
    seeds = [42, 123, 456]
    opt_rows = []

    print("\nBenchmarking Optimizers across 3 random seeds...")
    for s in seeds:
        print(f"\n--- Random Seed {s} ---")

        # Optimizer B: Multi-Start L-BFGS-B
        print("Running Multi-Start L-BFGS-B...")
        res_b = run_multistart_lbfgsb(lik_obj, model="L1", n_starts=25, seed=s)
        th_b = res_b["best_theta"]
        comp_b = res_b["components"]
        opt_rows.append({
            "optimizer": "MultiStart_LBFGSB",
            "seed": s,
            "best_log_likelihood": res_b["best_log_likelihood"],
            "quad_oil": comp_b["quad_oil"],
            "quad_wat": comp_b["quad_wat"],
            "oil_rate_nrmse": comp_b["nrmse_qo"],
            "water_rate_nrmse": comp_b["nrmse_qw"],
            "oil_cum_nrmse": comp_b["nrmse_Qo"],
            "water_cum_nrmse": comp_b["nrmse_Qw"],
            "terminal_oil_err": comp_b["terminal_oil_err"],
            "terminal_wat_err": comp_b["terminal_wat_err"],
            "Ft": th_b[0], "Poro": th_b[1], "Perm": th_b[2], "AqPV": th_b[3],
            "n_evals": res_b["n_evals"],
            "runtime_sec": res_b["runtime_sec"],
        })

        # Optimizer C: Differential Evolution
        print("Running Differential Evolution...")
        res_c = run_differential_evolution(lik_obj, model="L1", maxiter=35, popsize=15, seed=s)
        th_c = res_c["best_theta"]
        comp_c = res_c["components"]
        opt_rows.append({
            "optimizer": "Differential_Evolution",
            "seed": s,
            "best_log_likelihood": res_c["best_log_likelihood"],
            "quad_oil": comp_c["quad_oil"],
            "quad_wat": comp_c["quad_wat"],
            "oil_rate_nrmse": comp_c["nrmse_qo"],
            "water_rate_nrmse": comp_c["nrmse_qw"],
            "oil_cum_nrmse": comp_c["nrmse_Qo"],
            "water_cum_nrmse": comp_c["nrmse_Qw"],
            "terminal_oil_err": comp_c["terminal_oil_err"],
            "terminal_wat_err": comp_c["terminal_wat_err"],
            "Ft": th_c[0], "Poro": th_c[1], "Perm": th_c[2], "AqPV": th_c[3],
            "n_evals": res_c["n_evals"],
            "runtime_sec": res_c["runtime_sec"],
        })

        # Optimizer D: Bayesian Optimization
        print("Running Bayesian Optimization...")
        res_d = run_bayesian_optimization(lik_obj, model="L1", n_init=15, n_iters=60, seed=s)
        th_d = res_d["best_theta"]
        comp_d = res_d["components"]
        opt_rows.append({
            "optimizer": "Bayesian_Optimization",
            "seed": s,
            "best_log_likelihood": res_d["best_log_likelihood"],
            "quad_oil": comp_d["quad_oil"],
            "quad_wat": comp_d["quad_wat"],
            "oil_rate_nrmse": comp_d["nrmse_qo"],
            "water_rate_nrmse": comp_d["nrmse_qw"],
            "oil_cum_nrmse": comp_d["nrmse_Qo"],
            "water_cum_nrmse": comp_d["nrmse_Qw"],
            "terminal_oil_err": comp_d["terminal_oil_err"],
            "terminal_wat_err": comp_d["terminal_wat_err"],
            "Ft": th_d[0], "Poro": th_d[1], "Perm": th_d[2], "AqPV": th_d[3],
            "n_evals": res_d["n_evals"],
            "runtime_sec": res_d["runtime_sec"],
        })

        # Optimizer E: ES-MDA
        print("Running ES-MDA...")
        res_e = run_esmda(lik_obj, model="L1", n_ensemble=80, n_steps=4, seed=s)
        th_e = res_e["best_theta"]
        comp_e = res_e["components"]
        opt_rows.append({
            "optimizer": "ES_MDA",
            "seed": s,
            "best_log_likelihood": res_e["best_log_likelihood"],
            "quad_oil": comp_e["quad_oil"],
            "quad_wat": comp_e["quad_wat"],
            "oil_rate_nrmse": comp_e["nrmse_qo"],
            "water_rate_nrmse": comp_e["nrmse_qw"],
            "oil_cum_nrmse": comp_e["nrmse_Qo"],
            "water_cum_nrmse": comp_e["nrmse_Qw"],
            "terminal_oil_err": comp_e["terminal_oil_err"],
            "terminal_wat_err": comp_e["terminal_wat_err"],
            "Ft": th_e[0], "Poro": th_e[1], "Perm": th_e[2], "AqPV": th_e[3],
            "n_evals": res_e["n_evals"],
            "runtime_sec": res_e["runtime_sec"],
        })

    df_opt = pd.DataFrame(opt_rows)
    out_dir = Path("outputs/research/history_matching_iteration_2/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_opt = out_dir / "optimizer_benchmark_multiseed_v2.csv"
    df_opt.to_csv(csv_opt, index=False)
    print(f"\nSaved multi-seed optimizer benchmark to {csv_opt}")

    # Summary by optimizer across seeds
    summary_rows = []
    for opt_name in df_opt["optimizer"].unique():
        sub = df_opt[df_opt["optimizer"] == opt_name]
        summary_rows.append({
            "optimizer": opt_name,
            "mean_log_lik": sub["best_log_likelihood"].mean(),
            "max_log_lik": sub["best_log_likelihood"].max(),
            "mean_oil_rate_nrmse": sub["oil_rate_nrmse"].mean(),
            "mean_water_rate_nrmse": sub["water_rate_nrmse"].mean(),
            "mean_terminal_oil_err": sub["terminal_oil_err"].mean(),
            "mean_evals": sub["n_evals"].mean(),
            "mean_runtime_s": sub["runtime_sec"].mean(),
        })
    df_summary = pd.DataFrame(summary_rows)
    csv_summary = out_dir / "optimizer_benchmark_summary_v2.csv"
    df_summary.to_csv(csv_summary, index=False)
    print("\nOptimizer Summary across 3 seeds:")
    print(df_summary.to_string())

    # 2. Pseudo-Field Parameter Recovery Experiment on Held-Out Cases
    print("\nRunning Pseudo-Field Parameter Recovery Experiment on Cases 71, 73, 76, 81...")
    pseudo_cases = [71, 73, 76, 81]
    pseudo_rows = []

    for c in pseudo_cases:
        true_th = unc_df.loc[c, PARAMS].to_numpy()
        synth_fh = pd.DataFrame({
            "date": fh["date"],
            "ramp_oil_cum_stb": case_dict[c]["oil_cum_stb"].to_numpy(),
            "ramp_water_cum_stb": case_dict[c]["water_cum_stb"].to_numpy(),
            "ramp_oil_rate_stbd": case_dict[c]["oil_rate_stbd"].to_numpy(),
            "ramp_water_rate_stbd": case_dict[c]["water_rate_stbd"].to_numpy(),
        })
        lik_synth = HistoryMatchingLikelihoodV2(emu, synth_fh, rho_ar1=0.80, obs_noise_rel_oil=0.05, obs_noise_rel_wat=0.05)

        # Invert using Differential Evolution
        res_rec = run_differential_evolution(lik_synth, model="L1", maxiter=30, popsize=15, seed=42)
        inf_th = res_rec["best_theta"]

        pred_rec = emu.predict(inf_th.reshape(1, -1))
        true_cum_o = synth_fh["ramp_oil_cum_stb"].to_numpy()
        true_cum_w = synth_fh["ramp_water_cum_stb"].to_numpy()
        nrmse_o = np.sqrt(np.mean((pred_rec["oil_cum"][0] - true_cum_o) ** 2)) / (np.sqrt(np.mean(true_cum_o ** 2)) + 1e-12)
        nrmse_w = np.sqrt(np.mean((pred_rec["water_cum"][0] - true_cum_w) ** 2)) / (np.sqrt(np.mean(true_cum_w ** 2)) + 1e-12)

        pseudo_rows.append({
            "pseudo_case": c,
            "true_Ft": true_th[0], "inferred_Ft": inf_th[0], "err_Ft": abs(inf_th[0] - true_th[0]) / true_th[0],
            "true_Poro": true_th[1], "inferred_Poro": inf_th[1], "err_Poro": abs(inf_th[1] - true_th[1]) / true_th[1],
            "true_Perm": true_th[2], "inferred_Perm": inf_th[2], "err_Perm": abs(inf_th[2] - true_th[2]) / true_th[2],
            "true_AqPV": true_th[3], "inferred_AqPV": inf_th[3], "err_AqPV": abs(inf_th[3] - true_th[3]) / true_th[3],
            "best_log_likelihood": res_rec["best_log_likelihood"],
            "traj_oil_nrmse": nrmse_o,
            "traj_water_nrmse": nrmse_w,
        })

    df_pseudo = pd.DataFrame(pseudo_rows)
    csv_pseudo = out_dir / "pseudo_field_recovery_benchmark_v2.csv"
    df_pseudo.to_csv(csv_pseudo, index=False)
    print(f"\nSaved pseudo-field recovery benchmark to {csv_pseudo}:")
    print(df_pseudo[["pseudo_case", "true_Poro", "inferred_Poro", "err_Poro", "err_Ft", "err_AqPV", "traj_oil_nrmse", "traj_water_nrmse"]].to_string())

    return df_opt, df_pseudo


if __name__ == "__main__":
    run_inverse_optimization_benchmark_v2()
