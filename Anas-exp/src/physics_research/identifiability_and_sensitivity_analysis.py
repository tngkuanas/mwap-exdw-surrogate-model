"""
Module: identifiability_and_sensitivity_analysis.py
Investigates Priority Hypothesis 2: Parameter Identifiability & EUR Sensitivity Compression.
Mathematical and empirical analysis of:
1. Historical Sensitivity Matrix J_hist = dh/dtheta
2. Fisher Information Matrix F = J^T Sigma^-1 J
3. Spectral Eigendecomposition of F (principal axes of posterior information)
4. Forecast EUR Gradient grad(g) = d Q_oil(2028) / dtheta
5. Directional projection of forecast gradient onto identifiable vs unidentifiable subspaces.
6. Theoretical vs empirical EUR sensitivity spread quantification.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg

from ensemble.data import (
    load_uncertainty,
    resolve_shared_dir,
    TRAIN_CASES,
    VAL_CASES,
    PARAMS,
)
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
    RecursiveTransitionForecaster,
    LearnedDCARateForecaster,
    HybridDynamicForecaster,
    YEAR_DAYS,
    RS_SOLUTION,
)
from history_matching_v2.emulator_revalidation_v2 import HistoricalSVDGPEmulatorV2

OUTPUT_TABLE_DIR = Path("outputs/research/physics_first/tables")
OUTPUT_FIG_DIR = Path("outputs/research/physics_first/figures")
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

PARAM_NAMES = [
    "Fault Transmissibility",
    "Porosity Multiplier",
    "Permeability Multiplier",
    "Aquifer Pore Volume",
]
PARAM_ABBR = ["Ft", "Poro", "Perm", "AqPV"]

# Prior bounds
PARAM_RANGES = {
    "Fault Transmissibility": (0.01, 1.00),
    "Porosity Multiplier": (0.80, 1.50),
    "Permeability Multiplier": (0.10, 10.00),
    "Aquifer Pore Volume": (10.00, 300.00),
}


def fit_historical_emulator(case_dict: Dict[int, pd.DataFrame], unc_df: pd.DataFrame) -> HistoricalSVDGPEmulatorV2:
    """Fits the SVD-GP forward emulator on historical training cases (1-70)."""
    X_list = []
    Yo_list = []
    Yw_list = []
    
    dates_ref = None
    for c in TRAIN_CASES:
        df_c = case_dict[c]
        u_c = unc_df.loc[c]
        X_list.append([float(u_c[p]) for p in PARAM_NAMES])
        Yo_list.append(df_c["oil_cum_stb"].values[:41])  # 41 points: 1998-01-01 to 2008-01-01
        Yw_list.append(df_c["water_cum_stb"].values[:41])
        if dates_ref is None:
            dates_ref = df_c["date"].iloc[:41]
            
    X_arr = np.array(X_list)
    Yo_arr = np.array(Yo_list)
    Yw_arr = np.array(Yw_list)
    
    emu = HistoricalSVDGPEmulatorV2(n_components=4, random_state=42)
    emu.fit(X_arr, Yo_arr, Yw_arr, dates=dates_ref)
    return emu


def fit_forecast_models(case_dict: Dict[int, pd.DataFrame], unc_df: pd.DataFrame):
    """Fits frozen base forecasters for 20-year predictions."""
    from ensemble.generate_competition_deliverable import fit_locked_base_models
    return fit_locked_base_models(case_dict, unc_df, TRAIN_CASES, pd.Timestamp("2008-01-01"))


def evaluate_forecast_eur(models, weights, field_anchor, params: pd.Series, horizon_quarters: int = 80) -> float:
    """Evaluates 20-year cumulative oil forecast for a given parameter vector."""
    from ensemble.generate_competition_deliverable import generate_scenario_predictions
    forecast_dates = pd.date_range("2008-04-01", periods=horizon_quarters, freq="QS")
    df_pred, _ = generate_scenario_predictions(
        models=models,
        weights=weights,
        init_state=field_anchor,
        params=params,
        forecast_dates=forecast_dates,
        decline_sensitivity=1.0,
    )
    return float(df_pred["cum_oil_stb"].iloc[-1])


def analyze_identifiability_and_sensitivity():
    print("=" * 80)
    print("HYPOTHESIS 2: PARAMETER IDENTIFIABILITY & EUR SENSITIVITY COMPRESSION")
    print("=" * 80)
    
    unc_df = load_uncertainty()
    q_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(q_df)
    
    shared_dir = resolve_shared_dir()
    prod = pd.read_parquet(shared_dir / "production_timeseries.parquet")
    c0_2008 = prod[(prod["case_num"] == 0) & (prod["date"] == "2008-01-01")].set_index("metric")["value"]
    
    field_anchor = {
        "qo": float(c0_2008["oil_rate"]),
        "qw": float(c0_2008["water_rate"]),
        "qg": float(c0_2008["gas_rate"]),
        "Qo": float(c0_2008["oil_cum"]),
        "Qw": float(c0_2008["water_cum"]),
        "Qg": float(c0_2008["gas_cum"]),
        "wor": float(c0_2008["water_rate"] / max(c0_2008["oil_rate"], 1.0)),
        "res_age": 10.0,
        "origin_date": pd.Timestamp("2008-01-01"),
    }
    
    # 1. Fit historical forward emulator
    print("Fitting historical forward SVD-GP emulator...")
    emu = fit_historical_emulator(case_dict, unc_df)
    
    # Nominal posterior center (mean of top matches)
    theta_0 = np.array([0.086, 1.336, 1.88, 155.0])
    
    # Numerical Jacobian computation in normalized coordinates
    # Normalized coordinates u_i in [0, 1]
    bounds = np.array([PARAM_RANGES[p] for p in PARAM_NAMES])
    ranges = bounds[:, 1] - bounds[:, 0]
    
    def denorm(u):
        return bounds[:, 0] + u * ranges
        
    def norm(theta):
        return (theta - bounds[:, 0]) / ranges
        
    u_0 = norm(theta_0)
    
    # Step size for numerical differentiation in unit hypercube
    eps = 1e-4
    
    # Function evaluating historical trajectory vector h(u):
    # Combines 40 quarterly oil rates and 40 quarterly water rates
    def eval_h(u):
        th = denorm(u)
        pred = emu.predict(th.reshape(1, -1))
        # Rate vector: (40 oil rates, 40 water rates)
        rates_o = pred["oil_rate_fwd"][0]
        rates_w = pred["water_rate_fwd"][0]
        return np.concatenate([rates_o, rates_w])
        
    h_0 = eval_h(u_0)
    m = len(h_0)  # 80 observations
    
    # Compute Jacobian J = dh / du (in normalized coordinates)
    J_norm = np.zeros((m, 4))
    for j in range(4):
        u_plus = u_0.copy()
        u_plus[j] += eps
        u_minus = u_0.copy()
        u_minus[j] -= eps
        J_norm[:, j] = (eval_h(u_plus) - eval_h(u_minus)) / (2.0 * eps)
        
    # Observation noise covariance Sigma
    # Field measurement standard error: 150 STB/d for oil, 200 STB/d for water
    sigma_o = 150.0
    sigma_w = 200.0
    sigma_vec = np.concatenate([np.full(40, sigma_o), np.full(40, sigma_w)])
    inv_sigma_vec = 1.0 / (sigma_vec ** 2)
    
    # Fisher Information Matrix in normalized coordinates: F = J^T Sigma^-1 J
    F_norm = J_norm.T @ (inv_sigma_vec[:, None] * J_norm)
    
    # Eigendecomposition of F_norm
    eigvals, eigvecs = linalg.eigh(F_norm)
    # Sort descending
    sort_idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs[:, sort_idx]
    
    condition_number = eigvals[0] / max(1e-12, eigvals[-1])
    variance_shares = eigvals / np.sum(eigvals)
    
    print("\n[Fisher Information Matrix Spectral Decomposition (Normalized Space)]")
    print(f"  Condition Number kappa(F): {condition_number:.2e}")
    for i in range(4):
        v = eigvecs[:, i]
        print(f"  Mode {i+1}: Eigenval = {eigvals[i]:>12.2e} ({variance_shares[i]*100:>5.1f}% Info) | "
              f"v = [{v[0]:+.3f} Ft, {v[1]:+.3f} Poro, {v[2]:+.3f} Perm, {v[3]:+.3f} AqPV]")
              
    # 2. Compute 20-Year Forecast EUR Sensitivity Gradient
    print("\nFitting frozen forecast models and computing 20-year forecast gradient...")
    models = fit_forecast_models(case_dict, unc_df)
    
    # Frozen ensemble weights
    weights = {
        "oil": {
            "Recursive_ExtraTrees": 0.240466,
            "Learned_Exponential_DCA": 0.617902,
            "Learned_Hyperbolic_DCA": 0.0,
            "Hybrid_Dynamic": 0.141631,
        },
        "water": {
            "Recursive_ExtraTrees": 0.497170,
            "Learned_Exponential_DCA": 0.251415,
            "Learned_Hyperbolic_DCA": 0.251415,
            "Hybrid_Dynamic": 0.0,
        }
    }
    
    def eval_eur(u):
        th = denorm(u)
        p_series = pd.Series(th, index=PARAM_NAMES)
        return evaluate_forecast_eur(models, weights, field_anchor, p_series, horizon_quarters=80)
        
    eur_0 = eval_eur(u_0)
    print(f"  Baseline 20-Year EUR at theta_0: {eur_0:,.0f} STB")
    
    grad_eur_norm = np.zeros(4)
    for j in range(4):
        u_plus = u_0.copy()
        u_plus[j] += eps
        u_minus = u_0.copy()
        u_minus[j] -= eps
        grad_eur_norm[j] = (eval_eur(u_plus) - eval_eur(u_minus)) / (2.0 * eps)
        
    grad_eur_physical = grad_eur_norm / ranges
    grad_norm_mag = linalg.norm(grad_eur_norm)
    
    print("\n[20-Year Forecast Sensitivity Gradient dQ_oil(2028)/dtheta]")
    for j in range(4):
        print(f"  {PARAM_NAMES[j]:<25s}: Normalized Grad = {grad_eur_norm[j]:>+10.1f} STB | "
              f"Physical Grad = {grad_eur_physical[j]:>+12.1f} STB/unit")
              
    # Projections onto Fisher Information Modes
    projections = grad_eur_norm @ eigvecs
    proj_shares = (projections ** 2) / max(1e-12, grad_norm_mag ** 2)
    
    print("\n[Forecast Gradient Projections onto Fisher Information Modes]")
    for i in range(4):
        print(f"  Mode {i+1} Projection: {projections[i]:>+10.1f} STB ({proj_shares[i]*100:>5.1f}% of gradient energy)")
        
    # Projected Forecast Uncertainty (Laplace / Cramér-Rao Posterior Variance)
    # Var(g) approx grad^T F^-1 grad
    inv_F = linalg.pinv(F_norm)
    post_eur_variance = float(grad_eur_norm.T @ inv_F @ grad_eur_norm)
    post_eur_std = np.sqrt(max(0.0, post_eur_variance))
    
    # Compare with Unconstrained Prior Variance (assuming uniform prior across full range)
    # Var_prior(u_i) = 1/12 for uniform in [0, 1]
    prior_eur_variance = float(np.sum((grad_eur_norm ** 2) / 12.0))
    prior_eur_std = np.sqrt(prior_eur_variance)
    
    compression_ratio = prior_eur_std / max(1.0, post_eur_std)
    
    print(f"\n[Theoretical EUR Uncertainty Compression]")
    print(f"  Unconstrained Prior 20-Yr EUR Std:    {prior_eur_std:>12,.0f} STB")
    print(f"  Conditioned Posterior 20-Yr EUR Std:  {post_eur_std:>12,.0f} STB")
    print(f"  Uncertainty Compression Factor:       {compression_ratio:>12.1f}x")
    
    # Save Tables
    fisher_df = pd.DataFrame(F_norm, index=PARAM_ABBR, columns=PARAM_ABBR)
    fisher_df.to_csv(OUTPUT_TABLE_DIR / "fisher_information_matrix.csv")
    
    modes_df = pd.DataFrame({
        "mode": [f"Mode {i+1}" for i in range(4)],
        "eigenvalue": eigvals,
        "info_share_pct": variance_shares * 100.0,
        "v_Ft": eigvecs[0, :],
        "v_Poro": eigvecs[1, :],
        "v_Perm": eigvecs[2, :],
        "v_AqPV": eigvecs[3, :],
        "grad_projection_stb": projections,
        "energy_share_pct": proj_shares * 100.0,
    })
    modes_df.to_csv(OUTPUT_TABLE_DIR / "fisher_modes_and_gradient_projections.csv", index=False)
    
    # Diagnostic Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Subplot 1: Fisher Information Eigenvalues & Energy Share
    ax = axes[0]
    modes = [f"Mode {i+1}" for i in range(4)]
    x = np.arange(len(modes))
    width = 0.35
    
    ax.bar(x - width/2, variance_shares * 100.0, width, label="History Info Share (%)", color="#3498db", edgecolor="black")
    ax.bar(x + width/2, proj_shares * 100.0, width, label="Forecast Energy Share (%)", color="#e67e22", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Mode 1\n(Poro dominant)", f"Mode 2\n(Ft/Perm mixed)", f"Mode 3\n(AqPV mixed)", f"Mode 4\n(Ill-conditioned)"], fontweight="bold")
    ax.set_ylabel("Share of Variance / Energy (%)", fontweight="bold")
    ax.set_title(f"Fisher Information vs Forecast Gradient Alignment\n(Condition Number kappa = {condition_number:.1e})", fontsize=11, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Subplot 2: Parameter Sensitivities (Physical Units)
    ax2 = axes[1]
    norm_grads = np.abs(grad_eur_norm)
    bars = ax2.bar(PARAM_ABBR, norm_grads, color=["#1abc9c", "#9b59b6", "#34495e", "#e74c3c"], edgecolor="black", width=0.5)
    ax2.set_ylabel("Absolute EUR Sensitivity |dQ/du| [STB]", fontweight="bold")
    ax2.set_title("20-Year EUR Sensitivity by Normalized Dimension\n(Dominance of Porosity vs Fault/AqPV)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:,.0f} STB", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
                     
    plt.tight_layout()
    fig_path = OUTPUT_FIG_DIR / "fisher_eigensystem_and_forecast_gradient.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved diagnostic figure: {fig_path}")
    
    results = {
        "condition_number": float(condition_number),
        "eigenvalues": eigvals.tolist(),
        "info_shares_pct": (variance_shares * 100.0).tolist(),
        "grad_norm_stb": grad_eur_norm.tolist(),
        "grad_projections_stb": projections.tolist(),
        "prior_eur_std_stb": float(prior_eur_std),
        "post_eur_std_stb": float(post_eur_std),
        "compression_ratio": float(compression_ratio),
    }
    with open(OUTPUT_TABLE_DIR / "identifiability_analysis_summary.json", "w") as f:
        json.dump(results, f, indent=2)
        
    return results


if __name__ == "__main__":
    analyze_identifiability_and_sensitivity()
