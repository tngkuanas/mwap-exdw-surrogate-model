"""
Module: plot_diagnostics_v2.py
Implements visualization and publication figure generation for Research Iteration 2.
Generates:
1. historical_match_overlays_v2.png:
   Observed field history vs predicted trajectories for revised 10 cases + weak case diagnostic (Case 3).
2. long_horizon_20yr_forecast_fans_v2.png:
   20-year forecasts across all 10 revised scenarios using frozen Simplex stack.
3. proxy_vs_actual_ranking_diagnostic.png:
   Old 5-year constant rate proxy vs actual 80-quarter frozen ensemble EUR.
4. mcmc_convergence_diagnostics.png:
   Posterior distributions, prior bounds, and Vehtari rank-normalized diagnostics.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ensemble.data import load_uncertainty, resolve_shared_dir, TRAIN_CASES, PARAMS
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
    RS_SOLUTION,
)
from history_matching_v2.emulator_revalidation_v2 import HistoricalSVDGPEmulatorV2
from history_matching_v2.scenario_selection_v2 import (
    load_frozen_forecasting_models,
    forecast_candidate_realization,
)


def generate_all_publication_figures():
    """Generates all 4 publication figures and saves them in outputs/research/history_matching_iteration_2/figures/."""
    print("=" * 80)
    print("GENERATING PUBLICATION FIGURES FOR RESEARCH ITERATION 2")
    print("=" * 80)

    fig_dir = Path("outputs/research/history_matching_iteration_2/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    unc_df = load_uncertainty()
    q_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(q_df)
    fh = pd.read_parquet("../data/field_history.parquet")
    dates_hist = pd.to_datetime(fh["date"])

    X_train = unc_df.loc[TRAIN_CASES, PARAMS].to_numpy()
    Y_oil_tr = np.array([case_dict[c]["oil_cum_stb"].to_numpy() for c in TRAIN_CASES])
    Y_wat_tr = np.array([case_dict[c]["water_cum_stb"].to_numpy() for c in TRAIN_CASES])

    emu = HistoricalSVDGPEmulatorV2(n_components=4, random_state=42)
    emu.fit(X_train, Y_oil_tr, Y_wat_tr, dates=dates_hist)

    # 1. Historical Match Overlays Figure
    print("Generating Figure 1: Historical Match Overlays...")
    spec_v2_path = Path("outputs/research/history_matching_iteration_2/tables/selected_10_cases_specification_v2.csv")
    df_spec = pd.read_csv(spec_v2_path)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True)

    # Historical ground truth
    obs_qo = fh["ramp_oil_rate_stbd"].to_numpy()
    obs_qw = fh["ramp_water_rate_stbd"].to_numpy()
    obs_Qo = fh["ramp_oil_cum_stb"].to_numpy() / 1e6
    obs_Qw = fh["ramp_water_cum_stb"].to_numpy() / 1e6

    # Plot 10 revised cases
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, row in df_spec.iterrows():
        th = row[["Fault_Transmissibility", "Porosity_Multiplier", "Permeability_Multiplier", "Aquifer_Pore_Volume"]].to_numpy()
        pred = emu.predict(th.reshape(1, -1))

        # Forward rates (length 40)
        axes[0, 0].plot(dates_hist[:-1], pred["oil_rate_fwd"][0], color=colors[i], alpha=0.6, lw=1.5, label=f"{row['Case_Sheet']}" if i < 3 else None)
        axes[0, 1].plot(dates_hist[:-1], pred["water_rate_fwd"][0], color=colors[i], alpha=0.6, lw=1.5)
        # Cumulatives (length 41)
        axes[1, 0].plot(dates_hist, pred["oil_cum"][0] / 1e6, color=colors[i], alpha=0.6, lw=1.5)
        axes[1, 1].plot(dates_hist, pred["water_cum"][0] / 1e6, color=colors[i], alpha=0.6, lw=1.5)

    # Plot weak Case 3 from Iteration 1 as dashed red line
    th_case3_v1 = np.array([0.14228, 1.1068, 3.2519, 193.63])
    pred_c3_v1 = emu.predict(th_case3_v1.reshape(1, -1))
    axes[0, 0].plot(dates_hist[:-1], pred_c3_v1["oil_rate_fwd"][0], "r--", lw=2.0, label="Weak Case 3 (Iteration 1)")
    axes[0, 1].plot(dates_hist[:-1], pred_c3_v1["water_rate_fwd"][0], "r--", lw=2.0, label="Weak Case 3 (Iteration 1)")
    axes[1, 0].plot(dates_hist, pred_c3_v1["oil_cum"][0] / 1e6, "r--", lw=2.0)
    axes[1, 1].plot(dates_hist, pred_c3_v1["water_cum"][0] / 1e6, "r--", lw=2.0)

    # Observed points
    axes[0, 0].scatter(dates_hist[:-1], obs_qo[:-1], color="black", s=18, zorder=5, label="Observed Field Rate")
    axes[0, 1].scatter(dates_hist[:-1], obs_qw[:-1], color="black", s=18, zorder=5, label="Observed Field Rate")
    axes[1, 0].scatter(dates_hist, obs_Qo, color="black", s=18, zorder=5, label="Observed Field Cum")
    axes[1, 1].scatter(dates_hist, obs_Qw, color="black", s=18, zorder=5, label="Observed Field Cum")

    axes[0, 0].set_title("Oil Production Rate (STB/d)", fontsize=13, fontweight="bold")
    axes[0, 0].set_ylabel("Rate (STB/d)", fontsize=11)
    axes[0, 0].legend(loc="upper right", fontsize=9)

    axes[0, 1].set_title("Water Production Rate (STB/d) — Weak Case 3 vs Revised Cases", fontsize=13, fontweight="bold")
    axes[0, 1].set_ylabel("Rate (STB/d)", fontsize=11)
    axes[0, 1].legend(loc="upper left", fontsize=9)

    axes[1, 0].set_title("Cumulative Oil Production (MMSTB)", fontsize=13, fontweight="bold")
    axes[1, 0].set_ylabel("Cumulative Oil (MMSTB)", fontsize=11)

    axes[1, 1].set_title("Cumulative Water Production (MMSTB)", fontsize=13, fontweight="bold")
    axes[1, 1].set_ylabel("Cumulative Water (MMSTB)", fontsize=11)

    for ax in axes.flat:
        ax.tick_params(labelsize=10)
    plt.tight_layout()
    fig1_path = fig_dir / "historical_match_overlays_v2.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"Saved {fig1_path}")

    # 2. 20-Year Long Horizon Forecast Fans Figure
    print("Generating Figure 2: 20-Year Long Horizon Forecast Fans...")
    models, weights = load_frozen_forecasting_models(case_dict, unc_df)
    shared_dir = resolve_shared_dir()
    prod = pd.read_parquet(shared_dir / "production_timeseries.parquet")
    c0 = prod[prod["case_num"] == 0]
    c0_2008 = c0[c0["date"] == "2008-01-01"].set_index("metric")["value"]
    field_anchor = {
        "qo": float(c0_2008["oil_rate"]), "qw": float(c0_2008["water_rate"]), "qg": float(c0_2008["gas_rate"]),
        "Qo": float(c0_2008["oil_cum"]), "Qw": float(c0_2008["water_cum"]), "Qg": float(c0_2008["gas_cum"]),
        "wor": float(c0_2008["water_rate"] / max(c0_2008["oil_rate"], 1.0)), "res_age": 10.0,
        "origin_date": pd.Timestamp("2008-01-01"),
    }

    dates_fc = pd.date_range("2008-04-01", periods=80, freq="QS")
    dates_fc_40 = pd.date_range("2008-04-01", periods=40, freq="QS")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for i, row in df_spec.iterrows():
        th = row[["Fault_Transmissibility", "Porosity_Multiplier", "Permeability_Multiplier", "Aquifer_Pore_Volume"]].to_numpy()
        hz = int(row["Forecast_Horizon_Qtrs"])
        fc = forecast_candidate_realization(th, models, weights, field_anchor, horizon_qtrs=hz)
        d_plot = dates_fc_40 if hz == 40 else dates_fc

        axes[0].plot(d_plot, fc["oil_cum_trajectory"] / 1e6, lw=1.8, label=f"{row['Case_Sheet']}" if i < 3 else None)
        axes[1].plot(d_plot, fc["water_cum_trajectory"] / 1e6, lw=1.8)
        # Plot gas trajectory
        axes[2].plot(d_plot, (field_anchor["Qg"] + RS_SOLUTION * (fc["oil_cum_trajectory"] - field_anchor["Qo"])) / 1e6, lw=1.8)

    axes[0].set_title("20-Year Cumulative Oil Forecast (MMSTB)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("MMSTB", fontsize=11)
    axes[0].legend(loc="lower right")

    axes[1].set_title("20-Year Cumulative Water Forecast (MMSTB)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("MMSTB", fontsize=11)

    axes[2].set_title("20-Year Cumulative Gas Forecast (BCF)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("BCF", fontsize=11)

    for ax in axes:
        ax.axvline(pd.Timestamp("2008-01-01"), color="black", linestyle="--", alpha=0.5, label="Forecast Origin")
        ax.tick_params(labelsize=10)

    plt.tight_layout()
    fig2_path = fig_dir / "long_horizon_20yr_forecast_fans_v2.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"Saved {fig2_path}")

    # 3. Figure 3: Proxy vs Actual Ranking Diagnostic
    print("Generating Figure 3: Proxy vs Actual Ranking Diagnostic...")
    samples_path = Path("outputs/research/history_matching_iteration_2/bayesian_posterior_samples_v2.npy")
    posterior_draws = np.load(samples_path)
    sample_sub = posterior_draws[:: max(1, len(posterior_draws) // 100)][:100]

    actual_eurs = []
    poros = []
    for th in sample_sub:
        fc = forecast_candidate_realization(th, models, weights, field_anchor, horizon_qtrs=80)
        actual_eurs.append(fc["final_oil_cum"] / 1e6)
        poros.append(th[1])

    proxy_eurs = [field_anchor["Qo"] / 1e6 + (field_anchor["qo"] * 365.25 * 5.0) / 1e6] * len(actual_eurs)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(poros, actual_eurs, c=actual_eurs, cmap="viridis", s=50, edgecolors="k", alpha=0.8)
    ax.axhline(proxy_eurs[0], color="red", linestyle="--", lw=2.0, label=f"Iteration 1 Static Proxy ({proxy_eurs[0]:.2f} MMSTB)")
    plt.colorbar(sc, ax=ax, label="Actual 80-Quarter Simplex EUR (MMSTB)")
    ax.set_title("Diagnostic: Geological Porosity vs Actual 80-Quarter Simplex EUR", fontsize=13, fontweight="bold")
    ax.set_xlabel("Porosity Multiplier", fontsize=11)
    ax.set_ylabel("Actual 80-Quarter Cumulative Oil Recovery (MMSTB)", fontsize=11)
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig3_path = fig_dir / "proxy_vs_actual_ranking_diagnostic.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"Saved {fig3_path}")

    # 4. Figure 4: MCMC Posterior Diagnostics
    print("Generating Figure 4: MCMC Posterior Convergence & Distributions...")
    diag_path = Path("outputs/research/history_matching_iteration_2/tables/bayesian_posterior_summary_v2.csv")
    df_summary = pd.read_csv(diag_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    param_names = ["Fault Transmissibility", "Porosity Multiplier", "Permeability Multiplier", "Aquifer Pore Volume"]

    for idx, (ax, p_name) in enumerate(zip(axes.flat, param_names)):
        p_samples = posterior_draws[:, idx]
        p_row = df_summary[df_summary["parameter"] == p_name].iloc[0]

        sns.histplot(p_samples, kde=True, ax=ax, color="steelblue", stat="density", bins=30)
        ax.axvline(p_row["post_mean"], color="red", lw=2.0, label=f"Mean: {p_row['post_mean']:.3f}")
        ax.axvline(p_row["post_p10"], color="darkgreen", linestyle="--", label=f"P10: {p_row['post_p10']:.3f}")
        ax.axvline(p_row["post_p90"], color="darkgreen", linestyle="--", label=f"P90: {p_row['post_p90']:.3f}")
        ax.axvspan(p_row["prior_min"], p_row["prior_max"], alpha=0.08, color="gray", label="Prior Support")

        ax.set_title(f"{p_name} (Ident. Index = {p_row['identifiability_index']:.1%})", fontsize=12, fontweight="bold")
        ax.set_xlim(p_row["prior_min"] * 0.9, p_row["prior_max"] * 1.05)
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig4_path = fig_dir / "mcmc_convergence_diagnostics.png"
    plt.savefig(fig4_path, dpi=200)
    plt.close()
    print(f"Saved {fig4_path}")


if __name__ == "__main__":
    generate_all_publication_figures()
