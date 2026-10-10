"""
Follow-up Audit: SVD Rank Selection, Generalization Stability, and Component Predictability.
Evaluates K=1, 2, 3, 4, 5 under identical leakage-free outer cross-validation on Cases 1-70.
Conducts component-level predictability analysis, paired statistical tests,
and stability checks across repeated outer partitions.
Strict Boundary: Cases 86-100 are completely sealed.
"""
from pathlib import Path
import json
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

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
from ensemble.metrics import compute_increment_nrmse, score_predictions_df

def build_dataset_slice(cases, origin, forecast_dates, phase, curves, unc_df, train_anchor_mean=None):
    inc_matrix = []
    anchors = []
    for c in sorted(cases):
        series = curves[(c, phase)]
        anchor = float(series.loc[:origin].iloc[-1])
        anchors.append(anchor)
        future_vals = np.interp(
            forecast_dates.asi8.astype(float),
            series.index.asi8.astype(float),
            series.to_numpy(float),
        )
        inc_matrix.append(future_vals - anchor)
        
    inc_matrix = np.array(inc_matrix)
    anchors = np.array(anchors)[:, None]
    if train_anchor_mean is None:
        anchor_mean = float(anchors.mean())
    else:
        anchor_mean = float(train_anchor_mean)
        
    anchor_scaled = anchors / (anchor_mean + 1e-6)
    X_unc = unc_df.loc[sorted(cases), PARAMS].to_numpy(float)
    X = np.hstack([X_unc, anchor_scaled])
    return X, inc_matrix, anchors, anchor_mean

def evaluate_rank_cv(ranks, seed=42, n_splits=5):
    """
    Evaluates ranks K in ranks on Cases 1-70 using n_splits outer CV with specified seed.
    Returns:
        oof_predictions: dict mapping K -> list of prediction dicts
        metrics_by_rank: dict mapping K -> summary dict
        case_scores_by_rank: dict mapping K -> dict mapping case_num -> macro NRMSE
        fold_scores_by_rank: dict mapping K -> dict mapping fold_idx -> macro NRMSE
        component_diagnostics: list of dicts with coefficient predictability details
        runtime_and_warnings: dict mapping K -> stats
    """
    unc_df = load_uncertainty()
    curves = load_curves()
    truth_df = load_forecast_truth()
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_cases_arr = np.array(TRAIN_CASES)
    folds = list(kf.split(train_cases_arr))
    
    oof_predictions = {K: [] for K in ranks}
    component_diag_list = []
    runtime_and_warnings = {K: {"fit_time_sec": 0.0, "convergence_warnings": 0, "n_gps_fitted": 0} for K in ranks}
    
    print(f"\n--- Running Rank Sweep (K={ranks}) with Seed {seed} ({n_splits} folds) ---")
    
    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        train_c = sorted(train_cases_arr[tr_idx].tolist())
        val_c = sorted(train_cases_arr[val_idx].tolist())
        print(f"  Fold {fold_idx}: {len(train_c)} train cases, {len(val_c)} val cases...")
        
        for origin_str, horizon in EXPERIMENTS:
            origin = pd.Timestamp(origin_str)
            forecast_dates = pd.date_range(
                origin + pd.DateOffset(months=3),
                origin + pd.DateOffset(years=horizon),
                freq="QS",
            )
            
            for phase in PHASES:
                X_tr, inc_tr, anchors_tr, anchor_mean = build_dataset_slice(
                    train_c, origin, forecast_dates, phase, curves, unc_df
                )
                X_val, inc_val, anchors_val, _ = build_dataset_slice(
                    val_c, origin, forecast_dates, phase, curves, unc_df, train_anchor_mean=anchor_mean
                )
                
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_val_s = scaler.transform(X_val)
                
                mean_inc = inc_tr.mean(axis=0, keepdims=True)
                centered_tr = inc_tr - mean_inc
                U, S, Vt = svd(centered_tr, full_matrices=False)
                
                max_K = max(ranks)
                max_basis = Vt[:max_K, :]
                coeffs_tr = centered_tr @ max_basis.T  # (N_train, max_K)
                
                # Fit separate GP for each component up to max_K
                gp_models = []
                pred_coeffs_all = np.zeros((len(val_c), max_K))
                
                for k_idx in range(max_K):
                    n_features = X_tr_s.shape[1]
                    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                        length_scale=np.ones(n_features),
                        length_scale_bounds=(1e-2, 1e2),
                        nu=2.5,
                    ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
                    
                    gp = GaussianProcessRegressor(
                        kernel=kernel,
                        alpha=1e-6,
                        n_restarts_optimizer=2,
                        random_state=seed + k_idx * 17,
                        normalize_y=True,
                    )
                    
                    t0 = time.time()
                    with warnings.catch_warnings(record=True) as w_list:
                        warnings.simplefilter("always")
                        gp.fit(X_tr_s, coeffs_tr[:, k_idx])
                    fit_dur = time.time() - t0
                    
                    n_warn = len(w_list)
                    gp_models.append(gp)
                    pred_c_k = gp.predict(X_val_s)
                    pred_coeffs_all[:, k_idx] = pred_c_k
                    
                    # Record runtime & warning stats for ranks that include this component
                    for K in ranks:
                        if k_idx < K:
                            runtime_and_warnings[K]["fit_time_sec"] += fit_dur
                            runtime_and_warnings[K]["convergence_warnings"] += n_warn
                            runtime_and_warnings[K]["n_gps_fitted"] += 1
                            
                    # Record component-level predictability diagnostics (for fold 0 or aggregate)
                    if fold_idx == 0:
                        # True validation coefficients
                        true_val_c = (inc_val - mean_inc) @ max_basis[k_idx:k_idx+1, :].T
                        rmse_c = float(np.sqrt(np.mean((pred_c_k - true_val_c.flatten()) ** 2)))
                        var_ratio = float((S[k_idx] ** 2) / np.sum(S ** 2))
                        
                        component_diag_list.append({
                            "seed": seed,
                            "origin": origin_str,
                            "horizon": horizon,
                            "phase": phase,
                            "component_idx": k_idx + 1,
                            "singular_value": float(S[k_idx]),
                            "variance_explained_ratio": var_ratio,
                            "coeff_std_train": float(np.std(coeffs_tr[:, k_idx])),
                            "coeff_min_train": float(np.min(coeffs_tr[:, k_idx])),
                            "coeff_max_train": float(np.max(coeffs_tr[:, k_idx])),
                            "coeff_iqr_train": float(np.percentile(coeffs_tr[:, k_idx], 75) - np.percentile(coeffs_tr[:, k_idx], 25)),
                            "val_coeff_rmse": rmse_c,
                            "norm_rmse_ratio": float(rmse_c / (np.std(coeffs_tr[:, k_idx]) + 1e-6)),
                        })
                        
                # Form trajectory predictions for each rank K
                for K in ranks:
                    basis_K = max_basis[:K, :]
                    pred_coeffs_K = pred_coeffs_all[:, :K]
                    pred_inc_K = mean_inc + pred_coeffs_K @ basis_K
                    
                    for i, c in enumerate(val_c):
                        cum_pred = anchors_val[i, 0] + pred_inc_K[i, :]
                        for t_idx, d in enumerate(forecast_dates):
                            oof_predictions[K].append({
                                "model_id": f"GP_K{K}",
                                "case_num": c,
                                "fold": fold_idx,
                                "cutoff": origin,
                                "horizon_years": horizon,
                                "date": d,
                                "phase": phase,
                                "prediction": float(cum_pred[t_idx]),
                                "split": "train",
                            })
                            
    # Score predictions for all ranks
    summary_by_rank = {}
    case_scores_by_rank = {}
    fold_scores_by_rank = {}
    
    for K in ranks:
        oof_df = pd.DataFrame(oof_predictions[K])
        p_scores, m_scores, c_scores = score_predictions_df(oof_df, truth_df)
        
        macro_nrmse = float(m_scores[m_scores["split"] == "train"]["mean_dev_NRMSE"].iloc[0])
        worst_exp = float(m_scores[m_scores["split"] == "train"]["worst_exp_NRMSE"].iloc[0])
        
        by_exp = p_scores[p_scores["split"] == "train"].groupby(["cutoff", "horizon_years"])["increment_NRMSE"].mean().to_dict()
        by_phase = p_scores[p_scores["split"] == "train"].groupby("phase")["increment_NRMSE"].mean().to_dict()
        
        # Case macro scores (average over 4 exps and 3 phases)
        case_macro = c_scores.groupby("case_num")["case_NRMSE"].mean().to_dict()
        case_scores_by_rank[K] = case_macro
        
        # Fold macro scores
        fold_cases = {f_idx: sorted(train_cases_arr[val_indices].tolist()) for f_idx, (_, val_indices) in enumerate(folds)}
        fold_scores = {}
        for f_idx, c_list in fold_cases.items():
            f_nrmse = np.mean([case_macro[c] for c in c_list])
            fold_scores[f_idx] = float(f_nrmse)
        fold_scores_by_rank[K] = fold_scores
        
        case_vals = list(case_macro.values())
        summary_by_rank[K] = {
            "seed": seed,
            "macro_NRMSE": macro_nrmse,
            "worst_exp_NRMSE": worst_exp,
            "oil_NRMSE": by_phase.get("oil_cum", np.nan),
            "gas_NRMSE": by_phase.get("gas_cum", np.nan),
            "water_NRMSE": by_phase.get("water_cum", np.nan),
            "exp_2003_3y": by_exp.get(("2003-01-01", 3), np.nan),
            "exp_2003_5y": by_exp.get(("2003-01-01", 5), np.nan),
            "exp_2004_3y": by_exp.get(("2004-01-01", 3), np.nan),
            "exp_2005_3y": by_exp.get(("2005-01-01", 3), np.nan),
            "case_mean": float(np.mean(case_vals)),
            "case_std": float(np.std(case_vals, ddof=1)),
            "case_median": float(np.median(case_vals)),
            "case_iqr": float(np.percentile(case_vals, 75) - np.percentile(case_vals, 25)),
            "case_max": float(np.max(case_vals)),
            "fold_scores": fold_scores,
            "runtime_sec": runtime_and_warnings[K]["fit_time_sec"],
            "convergence_warnings": runtime_and_warnings[K]["convergence_warnings"],
            "n_gps_fitted": runtime_and_warnings[K]["n_gps_fitted"],
        }
        
    return summary_by_rank, case_scores_by_rank, fold_scores_by_rank, component_diag_list

def evaluate_validation_ranks(ranks):
    """
    Evaluates ranks K in ranks on holdout validation Cases 71-85 (trained on full Cases 1-70).
    """
    unc_df = load_uncertainty()
    curves = load_curves()
    truth_df = load_forecast_truth()
    
    val_predictions = {K: [] for K in ranks}
    train_c = sorted(TRAIN_CASES)
    val_c = sorted(VAL_CASES)
    
    print(f"\n--- Running Secondary Validation Confirmation on Cases 71-85 (K={ranks}) ---")
    
    for origin_str, horizon in EXPERIMENTS:
        origin = pd.Timestamp(origin_str)
        forecast_dates = pd.date_range(
            origin + pd.DateOffset(months=3),
            origin + pd.DateOffset(years=horizon),
            freq="QS",
        )
        
        for phase in PHASES:
            X_tr, inc_tr, anchors_tr, anchor_mean = build_dataset_slice(
                train_c, origin, forecast_dates, phase, curves, unc_df
            )
            X_val, inc_val, anchors_val, _ = build_dataset_slice(
                val_c, origin, forecast_dates, phase, curves, unc_df, train_anchor_mean=anchor_mean
            )
            
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)
            
            mean_inc = inc_tr.mean(axis=0, keepdims=True)
            centered_tr = inc_tr - mean_inc
            U, S, Vt = svd(centered_tr, full_matrices=False)
            
            max_K = max(ranks)
            max_basis = Vt[:max_K, :]
            coeffs_tr = centered_tr @ max_basis.T
            
            pred_coeffs_all = np.zeros((len(val_c), max_K))
            for k_idx in range(max_K):
                n_features = X_tr_s.shape[1]
                kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                    length_scale=np.ones(n_features),
                    length_scale_bounds=(1e-2, 1e2),
                    nu=2.5,
                ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
                
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-6,
                    n_restarts_optimizer=2,
                    random_state=42 + k_idx * 17,
                    normalize_y=True,
                )
                gp.fit(X_tr_s, coeffs_tr[:, k_idx])
                pred_coeffs_all[:, k_idx] = gp.predict(X_val_s)
                
            for K in ranks:
                basis_K = max_basis[:K, :]
                pred_coeffs_K = pred_coeffs_all[:, :K]
                pred_inc_K = mean_inc + pred_coeffs_K @ basis_K
                
                for i, c in enumerate(val_c):
                    cum_pred = anchors_val[i, 0] + pred_inc_K[i, :]
                    for t_idx, d in enumerate(forecast_dates):
                        val_predictions[K].append({
                            "model_id": f"GP_K{K}",
                            "case_num": c,
                            "cutoff": origin,
                            "horizon_years": horizon,
                            "date": d,
                            "phase": phase,
                            "prediction": float(cum_pred[t_idx]),
                            "split": "validation",
                        })
                        
    val_summary = {}
    val_case_scores = {}
    for K in ranks:
        val_df = pd.DataFrame(val_predictions[K])
        p_scores, m_scores, c_scores = score_predictions_df(val_df, truth_df)
        macro_nrmse = float(m_scores[m_scores["split"] == "validation"]["mean_dev_NRMSE"].iloc[0])
        worst_exp = float(m_scores[m_scores["split"] == "validation"]["worst_exp_NRMSE"].iloc[0])
        
        by_exp = p_scores[p_scores["split"] == "validation"].groupby(["cutoff", "horizon_years"])["increment_NRMSE"].mean().to_dict()
        by_phase = p_scores[p_scores["split"] == "validation"].groupby("phase")["increment_NRMSE"].mean().to_dict()
        case_macro = c_scores.groupby("case_num")["case_NRMSE"].mean().to_dict()
        val_case_scores[K] = case_macro
        
        case_vals = list(case_macro.values())
        val_summary[K] = {
            "macro_NRMSE": macro_nrmse,
            "worst_exp_NRMSE": worst_exp,
            "oil_NRMSE": by_phase.get("oil_cum", np.nan),
            "gas_NRMSE": by_phase.get("gas_cum", np.nan),
            "water_NRMSE": by_phase.get("water_cum", np.nan),
            "exp_2003_3y": by_exp.get(("2003-01-01", 3), np.nan),
            "exp_2003_5y": by_exp.get(("2003-01-01", 5), np.nan),
            "exp_2004_3y": by_exp.get(("2004-01-01", 3), np.nan),
            "exp_2005_3y": by_exp.get(("2005-01-01", 3), np.nan),
            "case_mean": float(np.mean(case_vals)),
            "case_std": float(np.std(case_vals, ddof=1)),
            "case_median": float(np.median(case_vals)),
            "case_iqr": float(np.percentile(case_vals, 75) - np.percentile(case_vals, 25)),
            "case_max": float(np.max(case_vals)),
        }
        
    return val_summary, val_case_scores

def compute_component_predictability_full():
    """
    Audits the predictable signal vs noise for components 1..5 on Cases 1-70 across all 12 experiment/phase combinations.
    """
    unc_df = load_uncertainty()
    curves = load_curves()
    folds_df = get_case_folds(unc_df, n_splits=5, seed=42)
    train_c = sorted(TRAIN_CASES)
    
    rows = []
    
    for origin_str, horizon in EXPERIMENTS:
        origin = pd.Timestamp(origin_str)
        forecast_dates = pd.date_range(
            origin + pd.DateOffset(months=3),
            origin + pd.DateOffset(years=horizon),
            freq="QS",
        )
        
        for phase in PHASES:
            X_tr, inc_tr, anchors_tr, anchor_mean = build_dataset_slice(
                train_c, origin, forecast_dates, phase, curves, unc_df
            )
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            
            mean_inc = inc_tr.mean(axis=0, keepdims=True)
            centered_tr = inc_tr - mean_inc
            U, S, Vt = svd(centered_tr, full_matrices=False)
            
            var_explained = (S ** 2) / np.sum(S ** 2)
            cum_var = np.cumsum(var_explained)
            
            coeffs_5 = centered_tr @ Vt[:5, :].T  # (70, 5)
            
            # SVD pure reconstruction floor for K=1..5
            recon_floors = {}
            for k in range(1, 6):
                basis_k = Vt[:k, :]
                recon_inc = mean_inc + (centered_tr @ basis_k.T) @ basis_k
                err = recon_inc - inc_tr
                recon_floors[k] = float(np.sqrt(np.mean(err ** 2)) / np.sqrt(np.mean(inc_tr ** 2)))
                
            for k in range(5):
                comp_idx = k + 1
                c_k = coeffs_5[:, k]
                
                # Check correlations with 5 features
                corr_dict = {}
                feature_names = PARAMS + ["anchor"]
                for f_idx, f_name in enumerate(feature_names):
                    r_val, _ = stats.pearsonr(X_tr[:, f_idx], c_k)
                    corr_dict[f_name] = float(r_val)
                    
                # 5-fold CV predictability of this coefficient alone
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                pred_c_oof = np.zeros(len(train_c))
                for tr_i, val_i in kf.split(train_c):
                    gp = GaussianProcessRegressor(
                        kernel=ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(X_tr_s.shape[1]), nu=2.5) + WhiteKernel(1e-2, (1e-5, 1e1)),
                        alpha=1e-6, n_restarts_optimizer=1, random_state=42 + k, normalize_y=True
                    )
                    gp.fit(X_tr_s[tr_i], c_k[tr_i])
                    pred_c_oof[val_i] = gp.predict(X_tr_s[val_i])
                    
                c_rmse = float(np.sqrt(np.mean((pred_c_oof - c_k) ** 2)))
                c_std = float(np.std(c_k, ddof=1))
                norm_c_rmse = float(c_rmse / (c_std + 1e-6))
                r2_c = float(r2_score(c_k, pred_c_oof))
                
                # Fold basis vector cosine stability
                cos_sims = []
                for tr_i, _ in kf.split(train_c):
                    f_inc = inc_tr[tr_i]
                    _, _, f_Vt = svd(f_inc - f_inc.mean(axis=0, keepdims=True), full_matrices=False)
                    cos_sims.append(float(np.abs(np.dot(f_Vt[k], Vt[k]))))
                    
                rows.append({
                    "origin": origin_str,
                    "horizon": horizon,
                    "phase": phase,
                    "component": comp_idx,
                    "singular_value": float(S[k]),
                    "variance_pct": float(var_explained[k] * 100),
                    "cum_variance_pct": float(cum_var[k] * 100),
                    "coeff_mean": float(np.mean(c_k)),
                    "coeff_std": c_std,
                    "coeff_min": float(np.min(c_k)),
                    "coeff_max": float(np.max(c_k)),
                    "coeff_iqr": float(np.percentile(c_k, 75) - np.percentile(c_k, 25)),
                    "oof_coeff_rmse": c_rmse,
                    "oof_norm_rmse": norm_c_rmse,
                    "oof_coeff_r2": r2_c,
                    "basis_cosine_sim_mean": float(np.mean(cos_sims)),
                    "svd_recon_floor": recon_floors[comp_idx],
                    "corr_fault": corr_dict["Fault Transmissibility"],
                    "corr_poro": corr_dict["Porosity Multiplier"],
                    "corr_perm": corr_dict["Permeability Multiplier"],
                    "corr_aquifer": corr_dict["Aquifer Pore Volume"],
                    "corr_anchor": corr_dict["anchor"],
                })
                
    return pd.DataFrame(rows)

def run_paired_comparisons_cv(case_scores_primary):
    """
    Computes rigorous paired statistical tests on identical 5-fold CV partitions (Cases 1-70).
    Pairs: K=3 vs K=2, K=4 vs K=3, K=5 vs K=4, K=2 vs K=1.
    """
    pairs = [(2, 1), (3, 2), (4, 3), (5, 4)]
    paired_results = []
    
    rng = np.random.default_rng(42)
    n_resamples = 10000
    
    for k_high, k_low in pairs:
        scores_high = np.array([case_scores_primary[k_high][c] for c in sorted(TRAIN_CASES)])
        scores_low = np.array([case_scores_primary[k_low][c] for c in sorted(TRAIN_CASES)])
        diffs = scores_high - scores_low  # negative means k_high wins (lower error)
        
        mean_diff = float(np.mean(diffs))
        median_diff = float(np.median(diffs))
        std_diff = float(np.std(diffs, ddof=1))
        cohen_d = float(mean_diff / (std_diff + 1e-12))
        
        # Bootstrap 95% CI on mean difference
        boot_means = np.array([
            rng.choice(diffs, size=len(diffs), replace=True).mean()
            for _ in range(n_resamples)
        ])
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
        
        wins = int(np.sum(diffs < -1e-6))
        losses = int(np.sum(diffs > 1e-6))
        ties = int(len(diffs) - wins - losses)
        
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(scores_high, scores_low)
        except Exception:
            wilcoxon_stat, wilcoxon_p = np.nan, np.nan
        ttest_stat, ttest_p = stats.ttest_rel(scores_high, scores_low)
        
        paired_results.append({
            "comparison": f"K={k_high} vs K={k_low}",
            "k_challenger": k_high,
            "k_baseline": k_low,
            "mean_case_diff": mean_diff,
            "median_case_diff": median_diff,
            "std_case_diff": std_diff,
            "cohens_d": cohen_d,
            "bootstrap_ci_95_lower": ci_lower,
            "bootstrap_ci_95_upper": ci_upper,
            "challenger_wins": wins,
            "challenger_losses": losses,
            "ties": ties,
            "wilcoxon_p_value": float(wilcoxon_p),
            "ttest_p_value": float(ttest_p),
        })
        
    return pd.DataFrame(paired_results)

def main():
    out_dir = Path("outputs/rank_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("STARTING SVD RANK AUDIT AND GENERALIZATION STABILITY STUDY")
    print("=" * 80)
    
    ranks = [1, 2, 3, 4, 5]
    
    # 1. Primary 5-Fold CV on Cases 1-70 (Seed 42)
    summary_s42, case_s42, fold_s42, comp_diag = evaluate_rank_cv(ranks, seed=42, n_splits=5)
    
    # 2. Repeated 5-Fold CV on Cases 1-70 (Seed 123) to audit partition stability
    summary_s123, case_s123, fold_s123, _ = evaluate_rank_cv(ranks, seed=123, n_splits=5)
    
    # 3. Secondary Validation Confirmation on Cases 71-85
    val_summary, val_cases = evaluate_validation_ranks(ranks)
    
    # 4. Component-Level Predictability & Signal-to-Noise Analysis
    print("\n--- Running Component Predictability & Signal-to-Noise Audit ---")
    comp_pred_df = compute_component_predictability_full()
    comp_pred_path = out_dir / "component_predictability.csv"
    comp_pred_df.to_csv(comp_pred_path, index=False)
    print(f"Saved component predictability to: {comp_pred_path}")
    
    # 5. Paired Statistical Comparisons on Cases 1-70
    print("\n--- Running Paired Statistical Comparisons (Cases 1-70) ---")
    paired_cv_df = run_paired_comparisons_cv(case_s42)
    paired_cv_path = out_dir / "paired_rank_comparisons.csv"
    paired_cv_df.to_csv(paired_cv_path, index=False)
    print(f"Saved paired comparisons to: {paired_cv_path}")
    
    # 6. Compile Consolidated Rank Comparison Table
    rank_table_rows = []
    for K in ranks:
        s42 = summary_s42[K]
        s123 = summary_s123[K]
        val = val_summary[K]
        
        # Win / loss vs K=2 on Cases 1-70 (Seed 42)
        diff_vs_k2 = np.array([case_s42[K][c] - case_s42[2][c] for c in sorted(TRAIN_CASES)])
        wins_vs_k2 = int(np.sum(diff_vs_k2 < -1e-6))
        losses_vs_k2 = int(np.sum(diff_vs_k2 > 1e-6))
        
        rank_table_rows.append({
            "rank_K": K,
            "oof_macro_seed42": s42["macro_NRMSE"],
            "oof_macro_seed123": s123["macro_NRMSE"],
            "oof_worst_exp": s42["worst_exp_NRMSE"],
            "oof_oil": s42["oil_NRMSE"],
            "oof_gas": s42["gas_NRMSE"],
            "oof_water": s42["water_NRMSE"],
            "oof_case_mean": s42["case_mean"],
            "oof_case_median": s42["case_median"],
            "oof_case_std": s42["case_std"],
            "oof_case_iqr": s42["case_iqr"],
            "wins_vs_K2_oof": wins_vs_k2,
            "losses_vs_K2_oof": losses_vs_k2,
            "val_macro_NRMSE": val["macro_NRMSE"],
            "val_worst_exp": val["worst_exp_NRMSE"],
            "val_oil": val["oil_NRMSE"],
            "val_gas": val["gas_NRMSE"],
            "val_water": val["water_NRMSE"],
            "val_case_median": val["case_median"],
            "val_case_iqr": val["case_iqr"],
            "fit_time_sec": s42["runtime_sec"],
            "convergence_warnings": s42["convergence_warnings"],
            "n_gps_fitted": s42["n_gps_fitted"],
        })
        
    rank_table_df = pd.DataFrame(rank_table_rows)
    rank_table_path = out_dir / "rank_comparison_table.csv"
    rank_table_df.to_csv(rank_table_path, index=False)
    print(f"Saved rank comparison table to: {rank_table_path}")
    
    # Save detailed JSON summary
    full_audit_results = {
        "summary_seed42": summary_s42,
        "summary_seed123": summary_s123,
        "validation_summary": val_summary,
    }
    json_path = out_dir / "rank_audit_summary.json"
    with open(json_path, "w") as f:
        json.dump(full_audit_results, f, indent=2)
    print(f"Saved full JSON summary to: {json_path}")
    
    print("\nRank audit script execution complete!")

if __name__ == "__main__":
    main()
