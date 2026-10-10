"""
Phase 2: Rigorous Controlled Exploratory Data Analysis (EDA) on Cases 1-70.
Computes input-space characteristics, target trajectory SVD properties,
cross-validation basis stability, and generates publication-grade diagnostic plots.
Strict boundary: Cases 86-100 are NEVER loaded or analyzed.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import mahalanobis
from scipy.linalg import svd, inv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from ensemble.data import (
    load_uncertainty,
    load_curves,
    load_forecast_truth,
    get_case_folds,
    PARAMS,
    PHASES,
    EXPERIMENTS,
    TRAIN_CASES,
    VAL_CASES,
)
from ensemble.metrics import compute_increment_nrmse
from ensemble.gp_models import GaussianProcessCurveModel

def run_phase2_eda():
    output_dir = Path("outputs/eda_discovery")
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("RUNNING PHASE 2: RIGOROUS CONTROLLED EDA (CASES 1-70 ONLY)")
    print("=" * 80)
    
    # 1. Load data
    unc_df = load_uncertainty()
    curves = load_curves()
    truth_df = load_forecast_truth()
    folds_df = get_case_folds(unc_df, n_splits=5, seed=42)
    
    train_unc = unc_df.loc[TRAIN_CASES, PARAMS].copy()
    val_unc = unc_df.loc[VAL_CASES, PARAMS].copy()
    
    # -------------------------------------------------------------
    # 2.1 INPUT-SPACE CHARACTERISTICS (Cases 1-70)
    # -------------------------------------------------------------
    print("\n--- 2.1 Input-Space Characteristics (Cases 1-70) ---")
    input_stats = {}
    for col in PARAMS:
        vals = train_unc[col].to_numpy(float)
        q25, q75 = np.percentile(vals, [25, 75])
        input_stats[col] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "iqr": float(q75 - q25),
            "skewness": float(stats.skew(vals)),
            "kurtosis": float(stats.kurtosis(vals)),
        }
    
    pearson_corr = train_unc.corr(method="pearson").round(4).to_dict()
    spearman_corr = train_unc.corr(method="spearman").round(4).to_dict()
    
    # PCA on 4 standardized parameters
    X_train_params = train_unc.to_numpy(float)
    X_train_std = (X_train_params - X_train_params.mean(axis=0)) / X_train_params.std(axis=0, ddof=1)
    cov_params = np.cov(X_train_std, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov_params)
    eigvals = np.sort(eigvals)[::-1]
    explained_var_ratio = eigvals / np.sum(eigvals)
    cum_var_ratio = np.cumsum(explained_var_ratio)
    
    pca_params_stats = {
        "singular_values": [float(v) for v in np.sqrt(eigvals * (len(TRAIN_CASES) - 1))],
        "explained_variance_ratio": [float(r) for r in explained_var_ratio],
        "cumulative_variance_ratio": [float(r) for r in cum_var_ratio],
    }
    
    # Mahalanobis Distance Analysis
    cov_mat = np.cov(X_train_params, rowvar=False)
    inv_cov = inv(cov_mat)
    train_mean = X_train_params.mean(axis=0)
    
    train_mahalanobis = [
        float(mahalanobis(X_train_params[i], train_mean, inv_cov))
        for i in range(len(TRAIN_CASES))
    ]
    X_val_params = val_unc.to_numpy(float)
    val_mahalanobis = {
        int(c): float(mahalanobis(X_val_params[i], train_mean, inv_cov))
        for i, c in enumerate(VAL_CASES)
    }
    val_mahal_vals = list(val_mahalanobis.values())
    
    mahalanobis_analysis = {
        "train_mean": float(np.mean(train_mahalanobis)),
        "train_std": float(np.std(train_mahalanobis)),
        "train_p90": float(np.percentile(train_mahalanobis, 90)),
        "train_max": float(np.max(train_mahalanobis)),
        "val_mean": float(np.mean(val_mahal_vals)),
        "val_std": float(np.std(val_mahal_vals)),
        "val_min": float(np.min(val_mahal_vals)),
        "val_max": float(np.max(val_mahal_vals)),
        "val_cases_above_train_p90": [c for c, d in val_mahalanobis.items() if d > np.percentile(train_mahalanobis, 90)],
        "val_cases_above_train_max": [c for c, d in val_mahalanobis.items() if d > np.max(train_mahalanobis)],
        "val_case_distances": val_mahalanobis,
    }
    
    # -------------------------------------------------------------
    # 2.2 TARGET TRAJECTORY BEHAVIOR & SVD ANALYSIS (Cases 1-70)
    # -------------------------------------------------------------
    print("\n--- 2.2 Target Trajectory Behavior & SVD Decomposition ---")
    svd_experiment_stats = {}
    
    for origin_str, horizon in EXPERIMENTS:
        origin = pd.Timestamp(origin_str)
        forecast_dates = pd.date_range(
            origin + pd.DateOffset(months=3),
            origin + pd.DateOffset(years=horizon),
            freq="QS",
        )
        T_steps = len(forecast_dates)
        
        for phase in PHASES:
            exp_key = f"{origin_str}_{horizon}y_{phase}"
            
            # Form increment matrix for Cases 1-70
            inc_matrix = []
            for c in TRAIN_CASES:
                series = curves[(c, phase)]
                anchor = float(series.loc[:origin].iloc[-1])
                future_vals = np.interp(
                    forecast_dates.asi8.astype(float),
                    series.index.asi8.astype(float),
                    series.to_numpy(float),
                )
                inc_matrix.append(future_vals - anchor)
            inc_matrix = np.array(inc_matrix)  # (70, T)
            
            mean_inc = inc_matrix.mean(axis=0, keepdims=True)
            centered = inc_matrix - mean_inc
            U, S, Vt = svd(centered, full_matrices=False)
            
            var_explained = (S ** 2) / np.sum(S ** 2)
            cum_var = np.cumsum(var_explained)
            
            # Reconstruction NRMSE without regression for K=1..5
            recon_nrmse = {}
            for k in range(1, min(6, len(S) + 1)):
                basis_k = Vt[:k, :]
                recon_inc = mean_inc + (centered @ basis_k.T) @ basis_k
                err = recon_inc - inc_matrix
                nrmse_k = float(np.sqrt(np.mean(err ** 2)) / np.sqrt(np.mean(inc_matrix ** 2)))
                recon_nrmse[k] = nrmse_k
                
            # Fold stability of basis vectors (5-fold CV)
            fold_cosine_sims = {1: [], 2: [], 3: []}
            for fold in range(5):
                fold_train = folds_df[folds_df["fold"] != fold]["case_num"].tolist()
                fold_inc = []
                for c in fold_train:
                    series = curves[(c, phase)]
                    anchor = float(series.loc[:origin].iloc[-1])
                    future_vals = np.interp(
                        forecast_dates.asi8.astype(float),
                        series.index.asi8.astype(float),
                        series.to_numpy(float),
                    )
                    fold_inc.append(future_vals - anchor)
                fold_inc = np.array(fold_inc)
                _, _, fold_Vt = svd(fold_inc - fold_inc.mean(axis=0, keepdims=True), full_matrices=False)
                for comp_idx in range(min(3, len(S))):
                    sim = float(np.abs(np.dot(fold_Vt[comp_idx, :], Vt[comp_idx, :])))
                    fold_cosine_sims[comp_idx + 1].append(sim)
                    
            svd_experiment_stats[exp_key] = {
                "singular_values": [float(s) for s in S[:5]],
                "explained_variance_ratio": [float(v) for v in var_explained[:5]],
                "cumulative_explained_variance": [float(v) for v in cum_var[:5]],
                "svd_reconstruction_nrmse": recon_nrmse,
                "fold_basis_stability_cosine_sim": {
                    f"comp_{k}": {
                        "mean": float(np.mean(sims)),
                        "min": float(np.min(sims)),
                    } for k, sims in fold_cosine_sims.items() if len(sims) > 0
                },
            }
            
    # -------------------------------------------------------------
    # 2.3 INPUT-TARGET CORRELATIONS (1-yr and 3-yr increments)
    # -------------------------------------------------------------
    print("\n--- 2.3 Input-Target Correlation & Milestone Analysis ---")
    correlation_rows = []
    for origin_str, horizon in [("2003-01-01", 3), ("2005-01-01", 3)]:
        origin = pd.Timestamp(origin_str)
        t_1y = origin + pd.DateOffset(years=1)
        t_3y = origin + pd.DateOffset(years=3)
        
        for phase in PHASES:
            for c in TRAIN_CASES:
                series = curves[(c, phase)]
                anchor = float(series.loc[:origin].iloc[-1])
                val_1y = float(np.interp(t_1y.timestamp(), series.index.asi8 / 1e9, series.to_numpy(float)))
                val_3y = float(np.interp(t_3y.timestamp(), series.index.asi8 / 1e9, series.to_numpy(float)))
                
                row = {
                    "case_num": c,
                    "origin": origin_str,
                    "phase": phase,
                    "inc_1y": val_1y - anchor,
                    "inc_3y": val_3y - anchor,
                    "anchor": anchor,
                }
                for p_col in PARAMS:
                    row[p_col] = float(unc_df.loc[c, p_col])
                correlation_rows.append(row)
                
    corr_df = pd.DataFrame(correlation_rows)
    
    milestone_correlations = {}
    for (origin_str, phase), grp in corr_df.groupby(["origin", "phase"]):
        key = f"{origin_str}_{phase}"
        milestone_correlations[key] = {}
        for feature in PARAMS + ["anchor"]:
            r_1y, p_1y = stats.pearsonr(grp[feature], grp["inc_1y"])
            rho_1y, _ = stats.spearmanr(grp[feature], grp["inc_1y"])
            r_3y, p_3y = stats.pearsonr(grp[feature], grp["inc_3y"])
            rho_3y, _ = stats.spearmanr(grp[feature], grp["inc_3y"])
            milestone_correlations[key][feature] = {
                "pearson_1y": float(r_1y),
                "spearman_1y": float(rho_1y),
                "pearson_3y": float(r_3y),
                "spearman_3y": float(rho_3y),
            }
            
    # -------------------------------------------------------------
    # 2.4 GENERATE PUBLICATION DIAGNOSTIC PLOTS
    # -------------------------------------------------------------
    print("\n--- 2.4 Generating Diagnostic Figures ---")
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # FIG 1: input_distributions.png
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()
    for idx, col in enumerate(PARAMS):
        ax = axes[idx]
        sns.histplot(train_unc[col], kde=True, ax=ax, color="#1f77b4", bins=15, stat="density", label="Train (Cases 1-70)")
        sns.rugplot(train_unc[col], ax=ax, color="#1f77b4", height=0.1)
        sns.rugplot(val_unc[col], ax=ax, color="#d62728", height=0.2, lw=2, label="Validation (Cases 71-85)")
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("Parameter Value")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    fig1_path = fig_dir / "input_distributions.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")
    
    # FIG 2: input_target_correlations.png
    # Heatmap of Pearson and Spearman correlations with 3-year increments
    heatmap_data_pearson = []
    heatmap_data_spearman = []
    y_labels = []
    for (origin_str, phase), grp in corr_df.groupby(["origin", "phase"]):
        y_labels.append(f"{origin_str} | {phase[:3].upper()}")
        p_row = [stats.pearsonr(grp[feat], grp["inc_3y"])[0] for feat in PARAMS + ["anchor"]]
        s_row = [stats.spearmanr(grp[feat], grp["inc_3y"])[0] for feat in PARAMS + ["anchor"]]
        heatmap_data_pearson.append(p_row)
        heatmap_data_spearman.append(s_row)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    x_labels = [p.replace(" ", "\n") for p in PARAMS] + ["Anchor\nVal"]
    
    sns.heatmap(
        np.array(heatmap_data_pearson),
        annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
        xticklabels=x_labels, yticklabels=y_labels, ax=ax1, cbar_kws={"label": "Pearson r"}
    )
    ax1.set_title("Pearson Linear Correlation (3-yr Increment)", fontweight="bold")
    
    sns.heatmap(
        np.array(heatmap_data_spearman),
        annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
        xticklabels=x_labels, yticklabels=y_labels, ax=ax2, cbar_kws={"label": "Spearman rho"}
    )
    ax2.set_title("Spearman Rank Correlation (3-yr Increment)", fontweight="bold")
    plt.tight_layout()
    fig2_path = fig_dir / "input_target_correlations.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")
    
    # FIG 3: representative_trajectories.png
    # Spaghetti plots of 10 representative training cases across all 3 phases
    rep_cases = [1, 7, 14, 21, 28, 35, 42, 49, 56, 65]
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    phase_titles = {"oil_cum": "Oil Cumulative (MSTB)", "gas_cum": "Gas Cumulative (MMSCF)", "water_cum": "Water Cumulative (MSTB)"}
    
    for idx, phase in enumerate(PHASES):
        ax = axes[idx]
        for c in rep_cases:
            s = curves[(c, phase)]
            ax.plot(s.index, s.values, alpha=0.7, lw=1.5, label=f"Case {c}" if idx == 0 else None)
        # Add cutoff lines
        for cut_str, _ in EXPERIMENTS:
            ax.axvline(pd.Timestamp(cut_str), color="gray", ls="--", alpha=0.6, lw=1)
        ax.set_title(phase_titles[phase], fontweight="bold")
        ax.set_ylabel("Production")
    axes[0].legend(ncol=5, loc="upper left", fontsize=9)
    axes[2].set_xlabel("Calendar Date")
    plt.tight_layout()
    fig3_path = fig_dir / "representative_trajectories.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")
    
    # FIG 4: svd_reconstruction_error.png
    # Reconstruction NRMSE vs components (1..5) by phase and experiment
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    axes = axes.flatten()
    for exp_idx, (origin_str, horizon) in enumerate(EXPERIMENTS):
        ax = axes[exp_idx]
        for phase, color in zip(PHASES, ["#2ca02c", "#ff7f0e", "#1f77b4"]):
            key = f"{origin_str}_{horizon}y_{phase}"
            nrmse_dict = svd_experiment_stats[key]["svd_reconstruction_nrmse"]
            comps = sorted(nrmse_dict.keys())
            vals = [nrmse_dict[k] for k in comps]
            ax.plot(comps, vals, marker="o", lw=2, color=color, label=phase)
        ax.set_title(f"Cutoff {origin_str} ({horizon}y horizon)", fontweight="bold")
        ax.set_xlabel("Number of SVD Components (K)")
        ax.set_ylabel("Increment NRMSE (Reconstruction Only)")
        ax.set_xticks(range(1, 6))
        ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    fig4_path = fig_dir / "svd_reconstruction_error.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved: {fig4_path}")
    
    # FIG 5: residuals_by_phase_quarter.png
    # Fit baseline GP on full Cases 1-70, generate predictions on Cases 1-70 OOF or Val
    # Let's run a 5-fold CV of the baseline GP to get genuine OOF residuals per quarter!
    print("Computing 5-fold OOF predictions for residual diagnostics plot...")
    gp_base = GaussianProcessCurveModel(n_components=1, nu=2.5, noise_level=1e-2)
    oof_rows = []
    for fold in range(5):
        val_cases_f = folds_df[folds_df["fold"] == fold]["case_num"].tolist()
        train_cases_f = folds_df[folds_df["fold"] != fold]["case_num"].tolist()
        for origin_str, horizon in EXPERIMENTS:
            origin = pd.Timestamp(origin_str)
            dates = pd.date_range(origin + pd.DateOffset(months=3), origin + pd.DateOffset(years=horizon), freq="QS")
            gp_base.fit(train_cases_f, origin, horizon, dates, curves, unc_df)
            p = gp_base.predict_cases(val_cases_f, origin, horizon, dates, curves, unc_df)
            oof_rows.append(p)
            
    oof_df = pd.concat(oof_rows, ignore_index=True)
    # Join with truth to calculate residuals per quarter
    t_copy = truth_df[truth_df["case_num"].isin(TRAIN_CASES)].copy()
    m_res = pd.merge(
        oof_df, t_copy,
        on=["case_num", "cutoff", "horizon_years", "date", "phase"],
        suffixes=("", "_truth"),
    )
    m_res["residual"] = m_res["prediction"] - m_res["truth"]
    m_res["quarter_step"] = m_res.groupby(["case_num", "cutoff", "horizon_years", "phase"]).cumcount() + 1
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for idx, phase in enumerate(PHASES):
        ax = axes[idx]
        p_data = m_res[m_res["phase"] == phase]
        sns.boxplot(x="quarter_step", y="residual", data=p_data, ax=ax, color="#4c72b0", fliersize=2)
        ax.axhline(0, color="red", ls="--", lw=1.2, alpha=0.8)
        ax.set_title(f"Residual Distribution by Forecast Quarter: {phase}", fontweight="bold")
        ax.set_ylabel("Error (Pred - Truth)")
    axes[2].set_xlabel("Forecast Quarter (t_0 + 3 months, 6 months, ...)")
    plt.tight_layout()
    fig5_path = fig_dir / "residuals_by_phase_quarter.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"Saved: {fig5_path}")
    
    # Save overall summary JSON
    summary = {
        "dataset_scope": {
            "training_cases": len(TRAIN_CASES),
            "validation_cases": len(VAL_CASES),
            "blind_test_cases_untouched": 15,
        },
        "input_statistics": input_stats,
        "pca_uncertainty_parameters": pca_params_stats,
        "mahalanobis_analysis": mahalanobis_analysis,
        "svd_trajectory_decomposition": svd_experiment_stats,
        "milestone_correlations": milestone_correlations,
    }
    
    summary_path = output_dir / "eda_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nPhase 2 EDA completed successfully! Summary written to: {summary_path}")
    return summary

if __name__ == "__main__":
    run_phase2_eda()
