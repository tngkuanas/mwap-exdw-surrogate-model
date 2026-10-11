"""
Module: plot_diagnostics_v4.py
Research Iteration 4: High-Quality Diagnostic Plots for Final Scientific Acceptance Gate.

Generates:
1. sbc_coverage_and_rank_histograms_v4.png:
   - Rank histograms for 4 parameters under Fixed-Scale vs State-Dependent Scale.
   - Nominal vs empirical coverage comparison with Clopper-Pearson 95% CIs.
2. quantile_stability_and_bootstrap_v4.png:
   - EUR quantiles (P10, P50, P90) across candidate pool sizes (80, 300, 600).
   - Bootstrap standard error convergence and 95% bootstrap confidence intervals.
3. scenario_comparison_and_profiles_v4.png:
   - Strategy A vs B vs C forecast profiles and parameter distributions.
4. reservoir_stress_test_diagnostics_v4.png:
   - Water cut evolution, oil decline monotonicity, recovery factor bounds, and tail sensitivity.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11


def plot_all_diagnostics_v4():
    base_dir = Path("/Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp")
    fig_dir = base_dir / "outputs/research/history_matching_iteration_4/figures"
    tables_dir = base_dir / "outputs/research/history_matching_iteration_4/tables"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # 1. SBC COVERAGE & RANK HISTOGRAMS
    # ==========================================================================
    print("Generating Figure 1: SBC Coverage and Rank Histograms...")
    sbc_df = pd.read_csv(tables_dir / "sbc_coverage_comparison_v4.csv")
    ranks_fixed = np.load(tables_dir.parent / "sbc_ranks_fixed_scale.npy", allow_pickle=True).item()
    ranks_state = np.load(tables_dir.parent / "sbc_ranks_state_dependent.npy", allow_pickle=True).item()

    param_names = ["Fault_Transmissibility", "Porosity_Multiplier", "Permeability_Multiplier", "Aquifer_Pore_Volume"]
    param_titles = [r"Fault Transmissibility ($F_t$)", r"Porosity Multiplier ($\Phi$)", r"Permeability Multiplier ($K_{mult}$)", r"Aquifer Pore Volume ($V_{p,aq}$)"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=300)

    # Top row: Fixed scale rank histograms
    for idx, (p, title) in enumerate(zip(param_names, param_titles)):
        ax = axes[0, idx]
        ax.hist(ranks_fixed[p], bins=10, range=(0, 1), color="#2b5c8f", edgecolor="white", alpha=0.85)
        ax.axhline(10.0, color="#d95f02", linestyle="--", linewidth=1.5, label="Uniform Expectation")
        ks_p = sbc_df[(sbc_df["experiment"] == "fixed_scale") & (sbc_df["parameter"] == p)]["ks_p_value"].iloc[0]
        ax.set_title(f"Fixed Scale: {title}\n(KS p = {ks_p:.4f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Normalized Rank")
        ax.set_ylabel("Count (N=100)")
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    # Bottom row: State-dependent scale rank histograms
    for idx, (p, title) in enumerate(zip(param_names, param_titles)):
        ax = axes[1, idx]
        ax.hist(ranks_state[p], bins=10, range=(0, 1), color="#238b45", edgecolor="white", alpha=0.85)
        ax.axhline(10.0, color="#d95f02", linestyle="--", linewidth=1.5, label="Uniform Expectation")
        ks_p = sbc_df[(sbc_df["experiment"] == "state_dependent") & (sbc_df["parameter"] == p)]["ks_p_value"].iloc[0]
        ax.set_title(f"State-Dependent: {title}\n(KS p = {ks_p:.4f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Normalized Rank")
        ax.set_ylabel("Count (N=100)")
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    plt.suptitle("Simulation-Based Calibration (SBC): Fixed vs State-Dependent Scale Rank Histograms", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    fig1_path = fig_dir / "sbc_coverage_and_rank_histograms_v4.png"
    plt.savefig(fig1_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig1_path}")

    # ==========================================================================
    # 2. QUANTILE STABILITY & BOOTSTRAP UNCERTAINTY
    # ==========================================================================
    print("Generating Figure 2: Quantile Stability and Bootstrap Uncertainty...")
    stab_df = pd.read_csv(tables_dir / "quantile_stability_v4.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Left: P10, P50, P90 EUR vs pool size with 95% bootstrap CIs
    ax = axes[0]
    sizes = stab_df["candidate_pool_size"]

    # P10
    ax.errorbar(
        sizes, stab_df["p10_eur_stb"] / 1e6,
        yerr=[(stab_df["p10_eur_stb"] - stab_df["p10_ci_lower_95"]) / 1e6,
              (stab_df["p10_ci_upper_95"] - stab_df["p10_eur_stb"]) / 1e6],
        fmt="o-", color="#e41a1c", linewidth=2, capsize=5, label="P10 High Recovery (90% Exceedance)"
    )
    # P50
    ax.errorbar(
        sizes, stab_df["p50_eur_stb"] / 1e6,
        yerr=[(stab_df["p50_eur_stb"] - stab_df["p50_ci_lower_95"]) / 1e6,
              (stab_df["p50_ci_upper_95"] - stab_df["p50_eur_stb"]) / 1e6],
        fmt="s-", color="#377eb8", linewidth=2, capsize=5, label="P50 Median Base Case"
    )
    # P90
    ax.errorbar(
        sizes, stab_df["p90_eur_stb"] / 1e6,
        yerr=[(stab_df["p90_eur_stb"] - stab_df["p90_ci_lower_95"]) / 1e6,
              (stab_df["p90_ci_upper_95"] - stab_df["p90_eur_stb"]) / 1e6],
        fmt="^-", color="#4daf4a", linewidth=2, capsize=5, label="P90 Low Recovery (10% Exceedance)"
    )

    ax.set_title("20-Year EUR Quantiles vs Posterior Candidate Pool Size", fontweight="bold")
    ax.set_xlabel("Candidate Pool Size ($N$)")
    ax.set_ylabel("Estimated EUR (MMSTB)")
    ax.set_xticks(sizes)
    ax.legend(loc="center right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Right: Bootstrap Standard Error Reduction
    ax = axes[1]
    ax.plot(sizes, stab_df["spread_boot_se_stb"], "o-", color="#984ea3", linewidth=2.5, markersize=8, label="EUR Spread SE ($P10 - P90$)")
    ax.plot(sizes, stab_df["p10_boot_se_stb"], "s--", color="#e41a1c", linewidth=1.5, label="P10 SE")
    ax.plot(sizes, stab_df["p50_boot_se_stb"], "d--", color="#377eb8", linewidth=1.5, label="P50 SE")
    ax.plot(sizes, stab_df["p90_boot_se_stb"], "^--", color="#4daf4a", linewidth=1.5, label="P90 SE")

    ax.set_title("Bootstrap Monte Carlo Standard Error Convergence ($B=1000$)", fontweight="bold")
    ax.set_xlabel("Candidate Pool Size ($N$)")
    ax.set_ylabel("Bootstrap Standard Error (STB)")
    ax.set_xticks(sizes)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Quantile Stability and Monte Carlo Precision under Frozen Ensemble Forecasting", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig2_path = fig_dir / "quantile_stability_and_bootstrap_v4.png"
    plt.savefig(fig2_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig2_path}")

    # ==========================================================================
    # 3. SELECTION POLICY SCENARIO PROFILES
    # ==========================================================================
    print("Generating Figure 3: Scenario Comparison and Profiles...")
    scen_df = pd.read_csv(tables_dir / "selected_scenarios_v4.csv")
    npz_data = np.load(tables_dir.parent / "candidate_trajectories_600.npz")
    traj_oil_cum = npz_data["oil_cum"]  # (600, 80)
    traj_oil_rate = npz_data["oil_rate"]  # (600, 80)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    strategies = [
        ("Strategy A: Ten Best Matches", axes[0], "#2b5c8f"),
        ("Strategy B: Unconstrained Quantiles", axes[1], "#d95f02"),
        ("Strategy C: Quality-Filtered Quantiles", axes[2], "#238b45"),
    ]

    dates_fc = pd.date_range("2008-04-01", periods=80, freq="QS")
    time_years = np.arange(1, 81) * 0.25  # 0.25 to 20 years

    for s_name, ax, color in strategies:
        df_strat = scen_df[scen_df["strategy"] == s_name]
        cand_ids = df_strat["cand_id"].tolist()

        for cid in cand_ids:
            ax.plot(time_years, traj_oil_cum[cid] / 1e6, color=color, alpha=0.7, linewidth=1.5)

        p50_row = df_strat[df_strat["target_role"].str.contains("P50|Rank 1", na=False)]
        if len(p50_row) > 0:
            cid_p50 = p50_row["cand_id"].iloc[0]
            ax.plot(time_years, traj_oil_cum[cid_p50] / 1e6, color="black", linewidth=2.5, linestyle="--", label="P50 / Central Case")

        spread = df_strat["frozen_ensemble_80q_eur"].max() - df_strat["frozen_ensemble_80q_eur"].min()
        ax.set_title(f"{s_name}\n(EUR Spread: {spread:,.0f} STB)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Forecast Elapsed Time (Years)")
        ax.set_ylabel("Cumulative Oil (MMSTB)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", fontsize=8)

    plt.suptitle("20-Year Cumulative Production Fan Charts Across Selection Policies (2008–2028)", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    fig3_path = fig_dir / "scenario_comparison_and_profiles_v4.png"
    plt.savefig(fig3_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig3_path}")

    # ==========================================================================
    # 4. RESERVOIR STRESS TEST DIAGNOSTICS
    # ==========================================================================
    print("Generating Figure 4: Reservoir Stress Test Diagnostics...")
    traj_wat_rate = npz_data["water_rate"]
    fw_all = traj_wat_rate / np.maximum(traj_oil_rate + traj_wat_rate, 1e-6)
    eur_all = traj_oil_cum[:, -1]
    rf_all = eur_all / 153.51e6 * 100.0  # %

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # Top-Left: Water Cut Evolution
    ax = axes[0, 0]
    for i in range(0, 600, 20):
        ax.plot(time_years, fw_all[i], color="#3182bd", alpha=0.3, linewidth=1)
    ax.axhline(0.98, color="red", linestyle="--", linewidth=1.5, label="Physical Cap ($f_w=0.98$)")
    ax.axhline(np.mean(fw_all[:, -1]), color="black", linestyle=":", linewidth=2, label=f"Mean Terminal $f_w$: {np.mean(fw_all[:, -1]):.3f}")
    ax.set_title(r"Test 1: Water Cut Evolution ($f_w(t) \in [0, 1]$)", fontweight="bold")
    ax.set_xlabel("Forecast Elapsed Time (Years)")
    ax.set_ylabel(r"Water Cut ($f_w$)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Top-Right: Oil Rate Decline Monotonicity
    ax = axes[0, 1]
    for i in range(0, 600, 20):
        ax.plot(time_years, traj_oil_rate[i], color="#31a354", alpha=0.3, linewidth=1)
    ax.set_title(r"Test 2: Oil Rate Monotonic Decline ($dq_o/dt \leq 0$)", fontweight="bold")
    ax.set_xlabel("Forecast Elapsed Time (Years)")
    ax.set_ylabel("Oil Production Rate (STB/d)")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Bottom-Left: Volumetric Recovery Factor Distribution
    ax = axes[1, 0]
    ax.hist(rf_all, bins=25, color="#756bb1", edgecolor="white", alpha=0.85)
    ax.axvline(25.0, color="red", linestyle="--", linewidth=1.5, label="Lower Bound (25%)")
    ax.axvline(35.0, color="red", linestyle="--", linewidth=1.5, label="Upper Bound (35%)")
    ax.axvline(np.mean(rf_all), color="black", linestyle=":", linewidth=2, label=f"Mean RF: {np.mean(rf_all):.2f}%")
    ax.set_title("Test 5: 30-Year Recovery Factor vs OOIP ($153.5$ MMSTB)", fontweight="bold")
    ax.set_xlabel("Recovery Factor (%)")
    ax.set_ylabel("Candidate Count (N=600)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Bottom-Right: DCA Hyperbolic Decline Tail Sensitivity
    ax = axes[1, 1]
    stress_df = pd.read_csv(tables_dir / "reservoir_stress_test_v4.csv")
    test_names = [f"T{row['test_number']}: {row['test_name'][:18]}..." for _, row in stress_df.iterrows()]
    pass_rates = stress_df["pass_rate_pct"].tolist()
    colors = ["#31a354" if p >= 95.0 else "#e6550d" for p in pass_rates]
    bars = ax.barh(test_names, pass_rates, color=colors, edgecolor="black", alpha=0.85)
    ax.axvline(95.0, color="red", linestyle="--", linewidth=1.5, label="Acceptance Gate (95%)")
    ax.set_xlim(0, 115)
    ax.set_title("Summary of All 6 Physical Stress Tests", fontweight="bold")
    ax.set_xlabel("Candidate Pass Rate (%)")
    for bar, p in zip(bars, pass_rates):
        ax.text(p + 1.5, bar.get_y() + bar.get_height() / 2, f"{p:.1f}%", va="center", fontsize=9, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("20-Year Long-Horizon Reservoir-Engineering Physical Verification", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    fig4_path = fig_dir / "reservoir_stress_test_diagnostics_v4.png"
    plt.savefig(fig4_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig4_path}")


if __name__ == "__main__":
    plot_all_diagnostics_v4()
