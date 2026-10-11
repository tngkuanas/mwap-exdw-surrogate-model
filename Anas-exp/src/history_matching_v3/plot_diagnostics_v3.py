"""
Module: plot_diagnostics_v3.py
Research Iteration 3: Generates publication-grade diagnostic figures.

Figures generated:
1. figures/sbc_rank_histograms_v3.png: Rank histograms from 100-sim SBC with 95% binomial intervals.
2. figures/l1_vs_l2_posterior_densities.png: Overlaid L1 vs L2 marginal posteriors.
3. figures/scenario_forecast_fans_v3.png: 80-quarter forecasts with corrected petroleum exceedance labels.
4. figures/historical_match_overlays_v3.png: Production history matches for oil and water.
5. figures/anova_variance_shares_v3.png: Two-way ANOVA variance shares and spread comparison.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from ensemble.data import load_uncertainty, TRAIN_CASES, PARAMS, resolve_shared_dir
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
)
from history_matching_v2.emulator_revalidation_v2 import HistoricalSVDGPEmulatorV2
from history_matching_v2.scenario_selection_v2 import (
    load_frozen_forecasting_models,
    forecast_candidate_realization,
)

CANONICAL_PARAMS = [
    "Fault_Transmissibility",
    "Porosity_Multiplier",
    "Permeability_Multiplier",
    "Aquifer_Pore_Volume",
]
PARAM_TITLES = [
    "Fault Transmissibility ($F_t$)",
    "Porosity Multiplier ($\\Phi$)",
    "Permeability Multiplier ($K_{\\text{mult}}$)",
    "Aquifer Pore Volume ($V_{p,\\text{aq}}$)",
]


def generate_publication_figures():
    print("=" * 80)
    print("GENERATING RESEARCH ITERATION 3 PUBLICATION FIGURES")
    print("=" * 80)

    base_dir = Path("/Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp")
    fig_dir = base_dir / "outputs/research/history_matching_iteration_3/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = base_dir / "outputs/research/history_matching_iteration_3/tables"

    # Artifact directory for chat presentation
    artifact_dir = Path("/Users/tengkuanas/.gemini/antigravity-cli/brain/4b7e1d00-9795-42a6-b5bb-c629e9cf25cb")

    # --------------------------------------------------------------------------
    # 1. SBC RANK HISTOGRAMS (TALTS ET AL. 2018)
    # --------------------------------------------------------------------------
    df_sbc_raw = pd.read_csv(tables_dir / "simulation_based_calibration_v3_raw_draws.csv")
    df_sbc_summ = pd.read_csv(tables_dir / "simulation_based_calibration_v3_summary.csv").set_index("parameter")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    n_sims = len(df_sbc_raw)
    n_bins = 10
    exp_count = n_sims / n_bins
    # 95% binomial confidence bounds
    ci_lower = exp_count - 1.96 * np.sqrt(exp_count * (1.0 - 1.0 / n_bins))
    ci_upper = exp_count + 1.96 * np.sqrt(exp_count * (1.0 - 1.0 / n_bins))

    abbrs = ["Ft", "Poro", "Perm", "AqPV"]
    for i, p_name in enumerate(CANONICAL_PARAMS):
        ax = axes[i]
        ranks = df_sbc_raw[f"{abbrs[i]}_rank_percentile"].to_numpy()
        counts, bin_edges, _ = ax.hist(ranks, bins=n_bins, range=(0.0, 1.0), color="#2b5c8f", edgecolor="white", alpha=0.85)

        ax.axhline(exp_count, color="#d95f02", linestyle="--", linewidth=2, label=f"Uniform Expected ({exp_count:.0f})")
        ax.fill_between([0.0, 1.0], ci_lower, ci_upper, color="#d95f02", alpha=0.15, label="95% Binomial Conf. Band")

        chi2_p = df_sbc_summ.loc[p_name, "chi2_p_value"]
        ks_p = df_sbc_summ.loc[p_name, "ks_p_value"]
        ax.set_title(f"{PARAM_TITLES[i]}\n($\\chi^2$ p={chi2_p:.3f}, KS p={ks_p:.3f})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Normalized Posterior Rank Percentile", fontsize=10)
        ax.set_ylabel("Simulation Frequency", fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, linestyle=":", alpha=0.6)
        if i == 0:
            ax.legend(loc="upper right", fontsize=9)

    plt.suptitle("Simulation-Based Calibration (Talts et al. 2018): 100 Replications on Reservoir Forward Model", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    sbc_fig_path = fig_dir / "sbc_rank_histograms_v3.png"
    plt.savefig(sbc_fig_path, dpi=300)
    plt.savefig(artifact_dir / "sbc_rank_histograms_v3.png", dpi=300)
    plt.close()
    print(f"Saved: {sbc_fig_path}")

    # --------------------------------------------------------------------------
    # 2. L1 VS L2 POSTERIOR DENSITIES (KENNEDY & O'HAGAN 2001)
    # --------------------------------------------------------------------------
    draws_l1 = np.load(base_dir / "outputs/research/history_matching_iteration_3/bayesian_posterior_draws_l1.npy")
    draws_l2 = np.load(base_dir / "outputs/research/history_matching_iteration_3/bayesian_posterior_draws_l2.npy")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for i, p_name in enumerate(CANONICAL_PARAMS):
        ax = axes[i]
        d1 = draws_l1[:, i]
        d2 = draws_l2[:, i]

        x_grid = np.linspace(min(np.min(d1), np.min(d2)), max(np.max(d1), np.max(d2)), 300)
        kde1 = gaussian_kde(d1)(x_grid)
        kde2 = gaussian_kde(d2)(x_grid)

        ax.plot(x_grid, kde1, color="#1b9e77", linewidth=2.5, label="Model L1 (Fixed AR(1) Obs)")
        ax.plot(x_grid, kde2, color="#7570b3", linewidth=2.5, linestyle="--", label="Model L2 (Epistemic Cov + LogDet)")
        ax.fill_between(x_grid, 0, kde1, color="#1b9e77", alpha=0.15)
        ax.fill_between(x_grid, 0, kde2, color="#7570b3", alpha=0.10)

        ax.set_title(PARAM_TITLES[i], fontsize=12, fontweight="bold")
        ax.set_ylabel("Posterior Probability Density", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        if i == 0:
            ax.legend(loc="upper right", fontsize=9)

    plt.suptitle("Posterior Density Comparison: Model L1 (Observation Noise) vs Model L2 (Epistemic Covariance)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    l1_l2_fig_path = fig_dir / "l1_vs_l2_posterior_densities.png"
    plt.savefig(l1_l2_fig_path, dpi=300)
    plt.savefig(artifact_dir / "l1_vs_l2_posterior_densities.png", dpi=300)
    plt.close()
    print(f"Saved: {l1_l2_fig_path}")

    # --------------------------------------------------------------------------
    # 3. 20-YEAR FORECAST FANS WITH CORRECTED PETROLEUM LABELS
    # --------------------------------------------------------------------------
    shared_dir = resolve_shared_dir()
    unc_df = load_uncertainty(shared_dir)
    q_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(q_df)
    models, weights = load_frozen_forecasting_models(case_dict, unc_df)
    models["Recursive_ExtraTrees"].model_qo.n_jobs = 1
    models["Recursive_ExtraTrees"].model_qw.n_jobs = 1

    prod = pd.read_parquet(shared_dir / "production_timeseries.parquet")
    c0 = prod[prod["case_num"] == 0]
    c0_2008 = c0[c0["date"] == "2008-01-01"].set_index("metric")["value"]
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

    df_spec_v3 = pd.read_csv(tables_dir / "selected_10_cases_specification_v3.csv")
    f_dates = pd.date_range("2008-04-01", periods=80, freq="QS")

    fig, (ax_oil, ax_wat) = plt.subplots(1, 2, figsize=(15, 6))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, 10))
    for i, r in df_spec_v3.iterrows():
        th = r[["Fault_Transmissibility", "Porosity_Multiplier", "Permeability_Multiplier", "Aquifer_Pore_Volume"]].to_numpy()
        fc = forecast_candidate_realization(th, models, weights, field_anchor, horizon_qtrs=80)
        label = f"{r['Case_Sheet']}: {r['Petroleum_Exceedance_Probability']}"
        ax_oil.plot(f_dates, fc["oil_cum_trajectory"] / 1e6, label=label, color=colors[i], linewidth=2.0)
        ax_wat.plot(f_dates, fc["water_cum_trajectory"] / 1e6, label=label, color=colors[i], linewidth=2.0)

    ax_oil.set_title("Cumulative Oil Production Forecast (80 Quarters / 20 Years)", fontsize=12, fontweight="bold")
    ax_oil.set_xlabel("Forecast Quarter", fontsize=11)
    ax_oil.set_ylabel("Cumulative Oil (MMSTB)", fontsize=11)
    ax_oil.grid(True, linestyle=":", alpha=0.6)
    ax_oil.legend(loc="lower right", fontsize=8)

    ax_wat.set_title("Cumulative Water Production Forecast (80 Quarters / 20 Years)", fontsize=12, fontweight="bold")
    ax_wat.set_xlabel("Forecast Quarter", fontsize=11)
    ax_wat.set_ylabel("Cumulative Water (MMSTB)", fontsize=11)
    ax_wat.grid(True, linestyle=":", alpha=0.6)
    ax_wat.legend(loc="lower right", fontsize=8)

    plt.suptitle("Iteration 3 Deliverable Scenarios: 20-Year Production Forecast Fans (Monotonic Exceedance Ordering)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fans_fig_path = fig_dir / "scenario_forecast_fans_v3.png"
    plt.savefig(fans_fig_path, dpi=300)
    plt.savefig(artifact_dir / "scenario_forecast_fans_v3.png", dpi=300)
    plt.close()
    print(f"Saved: {fans_fig_path}")

    # --------------------------------------------------------------------------
    # 4. HISTORICAL PRODUCTION MATCH OVERLAYS
    # --------------------------------------------------------------------------
    fh = pd.read_parquet(base_dir / "../data/field_history.parquet")
    h_dates = pd.to_datetime(fh["date"])

    dates_emu = pd.to_datetime(case_dict[1]["date"].to_numpy())
    X_train = unc_df.loc[TRAIN_CASES, PARAMS].to_numpy()
    Y_oil_tr = np.array([case_dict[c]["oil_cum_stb"].to_numpy() for c in TRAIN_CASES])
    Y_wat_tr = np.array([case_dict[c]["water_cum_stb"].to_numpy() for c in TRAIN_CASES])
    emu = HistoricalSVDGPEmulatorV2(n_components=4, random_state=42)
    emu.fit(X_train, Y_oil_tr, Y_wat_tr, dates=dates_emu)

    fig, (ax_h_oil, ax_h_wat) = plt.subplots(1, 2, figsize=(15, 6))

    for i, r in df_spec_v3.iterrows():
        th = r[["Fault_Transmissibility", "Porosity_Multiplier", "Permeability_Multiplier", "Aquifer_Pore_Volume"]].to_numpy().reshape(1, -1)
        pred = emu.predict(th)
        ax_h_oil.plot(h_dates, pred["oil_cum"][0] / 1e6, color=colors[i], alpha=0.7, linewidth=1.5)
        ax_h_wat.plot(h_dates, pred["water_cum"][0] / 1e6, color=colors[i], alpha=0.7, linewidth=1.5)

    # Actual field history
    ax_h_oil.plot(h_dates, fh["ramp_oil_cum_stb"] / 1e6, "k--", linewidth=3.0, label="Observed Field History")
    ax_h_wat.plot(h_dates, fh["ramp_water_cum_stb"] / 1e6, "k--", linewidth=3.0, label="Observed Field History")

    ax_h_oil.set_title("Historical Oil Cumulative Match (1998–2008)", fontsize=12, fontweight="bold")
    ax_h_oil.set_xlabel("Historical Date", fontsize=11)
    ax_h_oil.set_ylabel("Cumulative Oil (MMSTB)", fontsize=11)
    ax_h_oil.grid(True, linestyle=":", alpha=0.6)
    ax_h_oil.legend(loc="upper left", fontsize=10)

    ax_h_wat.set_title("Historical Water Cumulative Match (1998–2008)", fontsize=12, fontweight="bold")
    ax_h_wat.set_xlabel("Historical Date", fontsize=11)
    ax_h_wat.set_ylabel("Cumulative Water (MMSTB)", fontsize=11)
    ax_h_wat.grid(True, linestyle=":", alpha=0.6)
    ax_h_wat.legend(loc="upper left", fontsize=10)

    plt.suptitle("Historical Production Matches Across All 10 Revised Deliverable Scenarios", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    hist_fig_path = fig_dir / "historical_match_overlays_v3.png"
    plt.savefig(hist_fig_path, dpi=300)
    plt.savefig(artifact_dir / "historical_match_overlays_v3.png", dpi=300)
    plt.close()
    print(f"Saved: {hist_fig_path}")

    # --------------------------------------------------------------------------
    # 5. ANOVA VARIANCE DECOMPOSITION
    # --------------------------------------------------------------------------
    df_anova = pd.read_csv(tables_dir / "formal_uncertainty_decomposition_anova_v3.csv")
    df_spread = pd.read_csv(tables_dir / "per_model_forecast_uncertainty_spread_v3.csv")

    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart of variance shares (Configuration 1)
    row_anova = df_anova.iloc[0]
    labels = ["Model Architecture\n(99.99%)", "Geological Parameters\n(0.0006%)", "Interaction / Resid\n(0.0049%)"]
    sizes = [row_anova["Model_Architecture_Share_Pct"], row_anova["Geological_Parameter_Share_Pct"] * 500, row_anova["Interaction_Residual_Share_Pct"] * 50]
    ax_pie.bar(["Model Architecture", "Geol. Params (x1000)", "Interaction (x100)"],
               [row_anova["Model_Architecture_Share_Pct"], row_anova["Geological_Parameter_Share_Pct"] * 1000, row_anova["Interaction_Residual_Share_Pct"] * 100],
               color=["#d95f02", "#1b9e77", "#7570b3"])
    ax_pie.set_title("Two-Way ANOVA Variance Decomposition (%)", fontsize=12, fontweight="bold")
    ax_pie.set_ylabel("Variance Share / Magnified Scale (%)", fontsize=11)
    ax_pie.grid(True, linestyle=":", alpha=0.6)

    # Model spread comparison
    model_names_short = ["Exp DCA", "Hyp DCA", "ExtraTrees", "Hybrid", "Simplex Stack"]
    means = df_spread["Mean_80Q_EUR_STB"] / 1e6
    spreads = df_spread["Geological_Param_Spread_STB"] / 1e3
    ax_bar.bar(model_names_short, means, yerr=df_spread["Std_80Q_EUR_STB"] / 1e6, capsize=5, color="#2b5c8f", alpha=0.85)
    ax_bar.set_title("Mean 20-Year EUR STB Across Models (Error Bars: 1-Sigma)", fontsize=12, fontweight="bold")
    ax_bar.set_ylabel("20-Year Oil EUR (MMSTB)", fontsize=11)
    ax_bar.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Formal Uncertainty Decomposition: Dominance of Model Architecture Selection over Geological Parameters", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    anova_fig_path = fig_dir / "anova_variance_shares_v3.png"
    plt.savefig(anova_fig_path, dpi=300)
    plt.savefig(artifact_dir / "anova_variance_shares_v3.png", dpi=300)
    plt.close()
    print(f"Saved: {anova_fig_path}")

    print("\nAll publication figures successfully generated and saved to artifact directory.")


if __name__ == "__main__":
    generate_publication_figures()
