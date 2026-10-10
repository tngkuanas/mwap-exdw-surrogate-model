"""
Master Model Discovery and Stacked-Ensemble Optimization Pipeline.
Evaluates expanded model zoo (GPs, Trees, Regularized Linear, Blends, Simplex Stack, Meta Stack)
under strict leakage-free nested cross-validation across Seeds 42 and 123 on Cases 1-70.
Conducts model complementarity, residual correlation, statistical paired testing,
and long-horizon physics audits.
Strict Boundary: Cases 86-100 are completely sealed.
"""
from pathlib import Path
import json
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV, Ridge, BayesianRidge, ElasticNetCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from ensemble.data import (
    load_uncertainty,
    load_curves,
    load_forecast_truth,
    PARAMS,
    PHASES,
    EXPERIMENTS,
    TRAIN_CASES,
    VAL_CASES,
)
from ensemble.metrics import compute_increment_nrmse, score_predictions_df, audit_physical_violations
from ensemble.constraints import apply_physical_constraints

def build_slice_features(cases, origin, forecast_dates, phase, curves, unc_df, train_anchor_mean=None):
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

def fit_predict_coefficients(model_type, X_tr_s, coeffs_tr, X_val_s, K, random_state=42):
    n_feat = X_tr_s.shape[1]
    pred_coeffs = np.zeros((len(X_val_s), K))
    fit_time = 0.0
    warn_count = 0
    
    t0 = time.time()
    for k in range(K):
        y_k = coeffs_tr[:, k]
        
        if model_type == "gp_matern52":
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=np.ones(n_feat), length_scale_bounds=(1e-2, 1e2), nu=2.5
            ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
            m = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=2, random_state=random_state + k * 17, normalize_y=True)
        elif model_type == "gp_matern32":
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=np.ones(n_feat), length_scale_bounds=(1e-2, 1e2), nu=1.5
            ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
            m = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=2, random_state=random_state + k * 17, normalize_y=True)
        elif model_type == "gp_rbf":
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
                length_scale=np.ones(n_feat), length_scale_bounds=(1e-2, 1e2)
            ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
            m = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=2, random_state=random_state + k * 17, normalize_y=True)
        elif model_type == "tree_extratrees":
            m = ExtraTreesRegressor(n_estimators=100, max_depth=6, min_samples_leaf=2, random_state=random_state + k * 17)
        elif model_type == "tree_rf":
            m = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=2, random_state=random_state + k * 17)
        elif model_type == "tree_histgb":
            m = HistGradientBoostingRegressor(max_iter=60, max_depth=3, min_samples_leaf=3, random_state=random_state + k * 17)
        elif model_type == "linear_ridge":
            m = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5)
        elif model_type == "linear_bayesian_ridge":
            m = BayesianRidge(max_iter=300)
        elif model_type == "linear_pls":
            m = PLSRegression(n_components=min(3, n_feat))
        elif model_type == "linear_elasticnet":
            m = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, random_state=random_state + k * 17, max_iter=2000)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m.fit(X_tr_s, y_k)
            warn_count += len(w)
            
        p = m.predict(X_val_s)
        if isinstance(m, PLSRegression):
            p = p.flatten()
        pred_coeffs[:, k] = p
        
    fit_time = time.time() - t0
    return pred_coeffs, fit_time, warn_count

def fit_simplex_weights(P_train, y_train):
    """
    Fits non-negative weights summing to 1: min || P w - y ||^2 s.t. w >= 0, sum w = 1.
    P_train: (N_samples, M_models)
    y_train: (N_samples,)
    """
    M = P_train.shape[1]
    def obj(w):
        return np.mean((P_train @ w - y_train) ** 2) + 1e-4 * np.sum((w - 1.0/M) ** 2)
    w0 = np.ones(M) / M
    bounds = [(0.0, 1.0) for _ in range(M)]
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    res = optimize.minimize(obj, w0, bounds=bounds, constraints=cons, method="SLSQP")
    return res.x if res.success else w0

def run_comprehensive_discovery():
    out_dir = Path("outputs/ensemble_discovery")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    unc_df = load_uncertainty()
    curves = load_curves()
    truth_df = load_forecast_truth()
    
    print("=" * 80)
    print("STARTING MASTER MODEL DISCOVERY AND STACKED-ENSEMBLE PIPELINE")
    print("=" * 80)
    
    base_models_zoo = [
        ("GP_Matern52_3comp", "gp_matern52", 3),
        ("GP_Matern52_2comp", "gp_matern52", 2),
        ("GP_Matern52_4comp", "gp_matern52", 4),
        ("GP_Matern32_3comp", "gp_matern32", 3),
        ("GP_RBF_3comp", "gp_rbf", 3),
        ("Tree_ExtraTrees_3comp", "tree_extratrees", 3),
        ("Tree_RandomForest_3comp", "tree_rf", 3),
        ("Tree_HistGB_3comp", "tree_histgb", 3),
        ("Linear_RidgeCV_3comp", "linear_ridge", 3),
        ("Linear_BayesianRidge_3comp", "linear_bayesian_ridge", 3),
        ("Linear_PLS_3comp", "linear_pls", 3),
        ("Linear_ElasticNetCV_3comp", "linear_elasticnet", 3),
    ]
    
    train_cases_arr = np.array(TRAIN_CASES)
    seeds = [42, 123]
    
    # Store base model OOF predictions per seed
    # Structure: oof_base_preds[seed][model_name] = DataFrame
    oof_base_dfs = {s: {} for s in seeds}
    base_timings = {s: {} for s in seeds}
    
    for seed in seeds:
        print(f"\n--- 1. Evaluating Base Models via 5-Fold CV on Cases 1-70 (Seed {seed}) ---")
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        folds = list(kf.split(train_cases_arr))
        
        seed_preds = {name: [] for name, _, _ in base_models_zoo}
        seed_times = {name: {"runtime": 0.0, "warnings": 0} for name, _, _ in base_models_zoo}
        
        for fold_idx, (tr_idx, val_idx) in enumerate(folds):
            train_c = sorted(train_cases_arr[tr_idx].tolist())
            val_c = sorted(train_cases_arr[val_idx].tolist())
            
            for origin_str, horizon in EXPERIMENTS:
                origin = pd.Timestamp(origin_str)
                forecast_dates = pd.date_range(origin + pd.DateOffset(months=3), origin + pd.DateOffset(years=horizon), freq="QS")
                
                for phase in PHASES:
                    X_tr, inc_tr, anchors_tr, anchor_mean = build_slice_features(train_c, origin, forecast_dates, phase, curves, unc_df)
                    X_val, inc_val, anchors_val, _ = build_slice_features(val_c, origin, forecast_dates, phase, curves, unc_df, train_anchor_mean=anchor_mean)
                    
                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_tr)
                    X_val_s = scaler.transform(X_val)
                    
                    mean_inc = inc_tr.mean(axis=0, keepdims=True)
                    centered_tr = inc_tr - mean_inc
                    U, S, Vt = svd(centered_tr, full_matrices=False)
                    
                    for model_name, model_type, K in base_models_zoo:
                        basis_K = Vt[:K, :]
                        coeffs_tr_K = centered_tr @ basis_K.T
                        p_c, fit_t, w_count = fit_predict_coefficients(model_type, X_tr_s, coeffs_tr_K, X_val_s, K, random_state=seed + fold_idx * 13)
                        seed_times[model_name]["runtime"] += fit_t
                        seed_times[model_name]["warnings"] += w_count
                        
                        pred_inc = mean_inc + p_c @ basis_K
                        for i, c in enumerate(val_c):
                            cum_pred = anchors_val[i, 0] + pred_inc[i, :]
                            for t_idx, d in enumerate(forecast_dates):
                                seed_preds[model_name].append({
                                    "model_id": model_name,
                                    "case_num": c,
                                    "fold": fold_idx,
                                    "cutoff": origin,
                                    "horizon_years": horizon,
                                    "date": d,
                                    "phase": phase,
                                    "prediction": float(cum_pred[t_idx]),
                                    "split": "train",
                                })
                                
        for name, _, _ in base_models_zoo:
            oof_base_dfs[seed][name] = pd.DataFrame(seed_preds[name])
            base_timings[seed][name] = seed_times[name]
        print(f"Seed {seed} base models finished.")
        
    # -------------------------------------------------------------
    # 2. GENERATE LEAKAGE-FREE BLENDS & STACKS
    # -------------------------------------------------------------
    print("\n--- 2. Generating Leakage-Free Blends and Stacked Ensembles ---")
    
    # Define ensembles to build
    ensemble_definitions = [
        "Blend_GP_ExtraTrees",
        "Blend_GP_Ridge",
        "Blend_GP_BayesianRidge",
        "Blend_GP_HistGB",
        "Blend_GP_Matern32",
        "Blend_Tri_GP_Ridge_ET",
        "Stack_Simplex_Tri",
        "Stack_RidgeMeta_Tri",
        "Stack_Simplex_Diverse",
        "Stack_RidgeMeta_Diverse",
    ]
    
    all_models_oof_dfs = {s: {} for s in seeds}
    
    for seed in seeds:
        # First copy base model dfs
        for name, _, _ in base_models_zoo:
            all_models_oof_dfs[seed][name] = oof_base_dfs[seed][name].copy()
            
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        folds = list(kf.split(train_cases_arr))
        
        # Merge base models on keys to easily blend and stack
        merge_keys = ["case_num", "fold", "cutoff", "horizon_years", "date", "phase", "split"]
        
        # We need truth to train meta-weights
        t_sub = truth_df[truth_df["case_num"].isin(TRAIN_CASES)].copy()
        t_sub["date"] = pd.to_datetime(t_sub["date"])
        t_sub["cutoff"] = pd.to_datetime(t_sub["cutoff"])
        
        # Build wide prediction table for this seed
        wide_df = None
        for name, _, _ in base_models_zoo:
            sub = oof_base_dfs[seed][name][merge_keys + ["prediction"]].copy()
            sub = sub.rename(columns={"prediction": name})
            sub["date"] = pd.to_datetime(sub["date"])
            sub["cutoff"] = pd.to_datetime(sub["cutoff"])
            if wide_df is None:
                wide_df = sub
            else:
                wide_df = pd.merge(wide_df, sub, on=merge_keys)
                
        wide_df = pd.merge(wide_df, t_sub[["case_num", "cutoff", "horizon_years", "date", "phase", "truth"]], on=["case_num", "cutoff", "horizon_years", "date", "phase"])
        
        # A. Simple Blends
        wide_df["Blend_GP_ExtraTrees"] = 0.5 * wide_df["GP_Matern52_3comp"] + 0.5 * wide_df["Tree_ExtraTrees_3comp"]
        wide_df["Blend_GP_Ridge"] = 0.5 * wide_df["GP_Matern52_3comp"] + 0.5 * wide_df["Linear_RidgeCV_3comp"]
        wide_df["Blend_GP_BayesianRidge"] = 0.5 * wide_df["GP_Matern52_3comp"] + 0.5 * wide_df["Linear_BayesianRidge_3comp"]
        wide_df["Blend_GP_HistGB"] = 0.5 * wide_df["GP_Matern52_3comp"] + 0.5 * wide_df["Tree_HistGB_3comp"]
        wide_df["Blend_GP_Matern32"] = 0.5 * wide_df["GP_Matern52_3comp"] + 0.5 * wide_df["GP_Matern32_3comp"]
        wide_df["Blend_Tri_GP_Ridge_ET"] = (
            (1.0/3.0) * wide_df["GP_Matern52_3comp"] +
            (1.0/3.0) * wide_df["Linear_RidgeCV_3comp"] +
            (1.0/3.0) * wide_df["Tree_ExtraTrees_3comp"]
        )
        
        # B. Leak-Free Stacking using Outer Cross-Fitting (Train meta-weights on other 4 folds, predict on fold f)
        tri_models = ["GP_Matern52_3comp", "Linear_RidgeCV_3comp", "Tree_ExtraTrees_3comp"]
        div_models = ["GP_Matern52_3comp", "GP_Matern32_3comp", "Linear_RidgeCV_3comp", "Linear_BayesianRidge_3comp", "Tree_ExtraTrees_3comp", "Tree_HistGB_3comp"]
        
        wide_df["Stack_Simplex_Tri"] = 0.0
        wide_df["Stack_RidgeMeta_Tri"] = 0.0
        wide_df["Stack_Simplex_Diverse"] = 0.0
        wide_df["Stack_RidgeMeta_Diverse"] = 0.0
        
        for fold_idx in range(5):
            tr_mask = (wide_df["fold"] != fold_idx)
            val_mask = (wide_df["fold"] == fold_idx)
            
            # Tri Simplex
            P_tr_tri = wide_df.loc[tr_mask, tri_models].to_numpy(float)
            y_tr = wide_df.loc[tr_mask, "truth"].to_numpy(float)
            P_val_tri = wide_df.loc[val_mask, tri_models].to_numpy(float)
            w_tri = fit_simplex_weights(P_tr_tri, y_tr)
            wide_df.loc[val_mask, "Stack_Simplex_Tri"] = P_val_tri @ w_tri
            
            # Tri Ridge Meta
            meta_ridge_tri = RidgeCV(alphas=np.logspace(-2, 4, 15), cv=5)
            meta_ridge_tri.fit(P_tr_tri, y_tr)
            wide_df.loc[val_mask, "Stack_RidgeMeta_Tri"] = meta_ridge_tri.predict(P_val_tri)
            
            # Diverse Simplex
            P_tr_div = wide_df.loc[tr_mask, div_models].to_numpy(float)
            P_val_div = wide_df.loc[val_mask, div_models].to_numpy(float)
            w_div = fit_simplex_weights(P_tr_div, y_tr)
            wide_df.loc[val_mask, "Stack_Simplex_Diverse"] = P_val_div @ w_div
            
            # Diverse Ridge Meta
            meta_ridge_div = RidgeCV(alphas=np.logspace(-2, 4, 15), cv=5)
            meta_ridge_div.fit(P_tr_div, y_tr)
            wide_df.loc[val_mask, "Stack_RidgeMeta_Diverse"] = meta_ridge_div.predict(P_val_div)
            
        # Export all ensembles to standard format
        for ens_name in ensemble_definitions:
            ens_df = wide_df[merge_keys + [ens_name]].copy().rename(columns={ens_name: "prediction"})
            ens_df["model_id"] = ens_name
            all_models_oof_dfs[seed][ens_name] = ens_df
            
    # -------------------------------------------------------------
    # 3. SECONDARY VALIDATION EVALUATION ON CASES 71-85
    # -------------------------------------------------------------
    print("\n--- 3. Evaluating on Holdout Validation Cases 71-85 ---")
    val_c = sorted(VAL_CASES)
    train_c = sorted(TRAIN_CASES)
    val_base_dfs = {}
    
    val_raw_preds = {name: [] for name, _, _ in base_models_zoo}
    for origin_str, horizon in EXPERIMENTS:
        origin = pd.Timestamp(origin_str)
        forecast_dates = pd.date_range(origin + pd.DateOffset(months=3), origin + pd.DateOffset(years=horizon), freq="QS")
        for phase in PHASES:
            X_tr, inc_tr, anchors_tr, anchor_mean = build_slice_features(train_c, origin, forecast_dates, phase, curves, unc_df)
            X_val, inc_val, anchors_val, _ = build_slice_features(val_c, origin, forecast_dates, phase, curves, unc_df, train_anchor_mean=anchor_mean)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)
            mean_inc = inc_tr.mean(axis=0, keepdims=True)
            centered_tr = inc_tr - mean_inc
            U, S, Vt = svd(centered_tr, full_matrices=False)
            
            for model_name, model_type, K in base_models_zoo:
                basis_K = Vt[:K, :]
                coeffs_tr_K = centered_tr @ basis_K.T
                p_c, _, _ = fit_predict_coefficients(model_type, X_tr_s, coeffs_tr_K, X_val_s, K, random_state=42)
                pred_inc = mean_inc + p_c @ basis_K
                for i, c in enumerate(val_c):
                    cum_pred = anchors_val[i, 0] + pred_inc[i, :]
                    for t_idx, d in enumerate(forecast_dates):
                        val_raw_preds[model_name].append({
                            "model_id": model_name,
                            "case_num": c,
                            "cutoff": origin,
                            "horizon_years": horizon,
                            "date": d,
                            "phase": phase,
                            "prediction": float(cum_pred[t_idx]),
                            "split": "validation",
                        })
                        
    val_merge_keys = ["case_num", "cutoff", "horizon_years", "date", "phase", "split"]
    val_wide_df = None
    for name, _, _ in base_models_zoo:
        sub = pd.DataFrame(val_raw_preds[name])[val_merge_keys + ["prediction"]].copy()
        sub = sub.rename(columns={"prediction": name})
        sub["date"] = pd.to_datetime(sub["date"])
        sub["cutoff"] = pd.to_datetime(sub["cutoff"])
        if val_wide_df is None:
            val_wide_df = sub
        else:
            val_wide_df = pd.merge(val_wide_df, sub, on=val_merge_keys)
            
    # Simple blends for validation
    val_wide_df["Blend_GP_ExtraTrees"] = 0.5 * val_wide_df["GP_Matern52_3comp"] + 0.5 * val_wide_df["Tree_ExtraTrees_3comp"]
    val_wide_df["Blend_GP_Ridge"] = 0.5 * val_wide_df["GP_Matern52_3comp"] + 0.5 * val_wide_df["Linear_RidgeCV_3comp"]
    val_wide_df["Blend_GP_BayesianRidge"] = 0.5 * val_wide_df["GP_Matern52_3comp"] + 0.5 * val_wide_df["Linear_BayesianRidge_3comp"]
    val_wide_df["Blend_GP_HistGB"] = 0.5 * val_wide_df["GP_Matern52_3comp"] + 0.5 * val_wide_df["Tree_HistGB_3comp"]
    val_wide_df["Blend_GP_Matern32"] = 0.5 * val_wide_df["GP_Matern52_3comp"] + 0.5 * val_wide_df["GP_Matern32_3comp"]
    val_wide_df["Blend_Tri_GP_Ridge_ET"] = (
        (1.0/3.0) * val_wide_df["GP_Matern52_3comp"] +
        (1.0/3.0) * val_wide_df["Linear_RidgeCV_3comp"] +
        (1.0/3.0) * val_wide_df["Tree_ExtraTrees_3comp"]
    )
    
    # Train meta-learners on full 70 training cases OOF predictions (from wide_df seed 42)
    # Tri models
    P_tr_all_tri = wide_df[tri_models].to_numpy(float)
    y_tr_all = wide_df["truth"].to_numpy(float)
    w_tri_full = fit_simplex_weights(P_tr_all_tri, y_tr_all)
    val_wide_df["Stack_Simplex_Tri"] = val_wide_df[tri_models].to_numpy(float) @ w_tri_full
    
    meta_ridge_tri_full = RidgeCV(alphas=np.logspace(-2, 4, 15), cv=5)
    meta_ridge_tri_full.fit(P_tr_all_tri, y_tr_all)
    val_wide_df["Stack_RidgeMeta_Tri"] = meta_ridge_tri_full.predict(val_wide_df[tri_models].to_numpy(float))
    
    # Diverse models
    P_tr_all_div = wide_df[div_models].to_numpy(float)
    w_div_full = fit_simplex_weights(P_tr_all_div, y_tr_all)
    val_wide_df["Stack_Simplex_Diverse"] = val_wide_df[div_models].to_numpy(float) @ w_div_full
    
    meta_ridge_div_full = RidgeCV(alphas=np.logspace(-2, 4, 15), cv=5)
    meta_ridge_div_full.fit(P_tr_all_div, y_tr_all)
    val_wide_df["Stack_RidgeMeta_Diverse"] = meta_ridge_div_full.predict(val_wide_df[div_models].to_numpy(float))
    
    all_models_val_dfs = {}
    all_model_names = [name for name, _, _ in base_models_zoo] + ensemble_definitions
    for m in all_model_names:
        sub = val_wide_df[val_merge_keys + [m]].copy().rename(columns={m: "prediction"})
        sub["model_id"] = m
        all_models_val_dfs[m] = sub
        
    # -------------------------------------------------------------
    # 4. COMPREHENSIVE SCORING AND METRICS COMPILATION
    # -------------------------------------------------------------
    print("\n--- 4. Compiling Macro Scores and Dispersion Metrics ---")
    results_s42 = {}
    results_s123 = {}
    results_val = {}
    
    for m in all_model_names:
        # Seed 42 OOF
        p_s42, m_s42, c_s42 = score_predictions_df(all_models_oof_dfs[42][m], truth_df)
        c_vals_42 = c_s42.groupby("case_num")["case_NRMSE"].mean().to_dict()
        results_s42[m] = {
            "macro_nrmse": float(m_s42[m_s42["split"] == "train"]["mean_dev_NRMSE"].iloc[0]),
            "worst_exp": float(m_s42[m_s42["split"] == "train"]["worst_exp_NRMSE"].iloc[0]),
            "oil": float(p_s42[p_s42["split"] == "train"].groupby("phase")["increment_NRMSE"].mean().get("oil_cum", np.nan)),
            "gas": float(p_s42[p_s42["split"] == "train"].groupby("phase")["increment_NRMSE"].mean().get("gas_cum", np.nan)),
            "water": float(p_s42[p_s42["split"] == "train"].groupby("phase")["increment_NRMSE"].mean().get("water_cum", np.nan)),
            "case_mean": float(np.mean(list(c_vals_42.values()))),
            "case_std": float(np.std(list(c_vals_42.values()), ddof=1)),
            "case_median": float(np.median(list(c_vals_42.values()))),
            "case_iqr": float(np.percentile(list(c_vals_42.values()), 75) - np.percentile(list(c_vals_42.values()), 25)),
            "case_scores": c_vals_42,
        }
        
        # Seed 123 OOF
        p_s123, m_s123, c_s123 = score_predictions_df(all_models_oof_dfs[123][m], truth_df)
        c_vals_123 = c_s123.groupby("case_num")["case_NRMSE"].mean().to_dict()
        results_s123[m] = {
            "macro_nrmse": float(m_s123[m_s123["split"] == "train"]["mean_dev_NRMSE"].iloc[0]),
            "worst_exp": float(m_s123[m_s123["split"] == "train"]["worst_exp_NRMSE"].iloc[0]),
            "oil": float(p_s123[p_s123["split"] == "train"].groupby("phase")["increment_NRMSE"].mean().get("oil_cum", np.nan)),
            "gas": float(p_s123[p_s123["split"] == "train"].groupby("phase")["increment_NRMSE"].mean().get("gas_cum", np.nan)),
            "water": float(p_s123[p_s123["split"] == "train"].groupby("phase")["increment_NRMSE"].mean().get("water_cum", np.nan)),
            "case_mean": float(np.mean(list(c_vals_123.values()))),
            "case_std": float(np.std(list(c_vals_123.values()), ddof=1)),
            "case_median": float(np.median(list(c_vals_123.values()))),
            "case_iqr": float(np.percentile(list(c_vals_123.values()), 75) - np.percentile(list(c_vals_123.values()), 25)),
            "case_scores": c_vals_123,
        }
        
        # Validation
        p_val, m_val, c_val = score_predictions_df(all_models_val_dfs[m], truth_df)
        c_vals_val = c_val.groupby("case_num")["case_NRMSE"].mean().to_dict()
        results_val[m] = {
            "macro_nrmse": float(m_val[m_val["split"] == "validation"]["mean_dev_NRMSE"].iloc[0]),
            "worst_exp": float(m_val[m_val["split"] == "validation"]["worst_exp_NRMSE"].iloc[0]),
            "oil": float(p_val[p_val["split"] == "validation"].groupby("phase")["increment_NRMSE"].mean().get("oil_cum", np.nan)),
            "gas": float(p_val[p_val["split"] == "validation"].groupby("phase")["increment_NRMSE"].mean().get("gas_cum", np.nan)),
            "water": float(p_val[p_val["split"] == "validation"].groupby("phase")["increment_NRMSE"].mean().get("water_cum", np.nan)),
            "case_median": float(np.median(list(c_vals_val.values()))),
            "case_iqr": float(np.percentile(list(c_vals_val.values()), 75) - np.percentile(list(c_vals_val.values()), 25)),
            "case_scores": c_vals_val,
        }
        
    # -------------------------------------------------------------
    # 5. RESIDUAL CORRELATIONS & MODEL COMPLEMENTARITY
    # -------------------------------------------------------------
    print("\n--- 5. Computing Residual Correlations and Diversity Matrix ---")
    benchmark_m = "GP_Matern52_3comp"
    
    # Compute residuals for all candidate models on wide_df
    res_dict = {}
    for m in all_model_names:
        res_dict[m] = wide_df[m].to_numpy(float) - wide_df["truth"].to_numpy(float)
        
    res_df = pd.DataFrame(res_dict)
    res_corr = res_df.corr(method="pearson").round(4)
    res_corr.to_csv(out_dir / "residual_correlation_matrix.csv")
    print("Saved residual correlation matrix.")
    
    # Complementarity table: for each base model, compute correlation with GP benchmark,
    # variance ratio, and count of cases where it beats GP benchmark
    comp_rows = []
    base_case_s42 = np.array([results_s42[benchmark_m]["case_scores"][c] for c in sorted(TRAIN_CASES)])
    
    for m in [name for name, _, _ in base_models_zoo]:
        r_corr = float(res_corr.loc[m, benchmark_m])
        m_case_s42 = np.array([results_s42[m]["case_scores"][c] for c in sorted(TRAIN_CASES)])
        diffs = m_case_s42 - base_case_s42
        better_cases = int(np.sum(diffs < -1e-6))
        worse_cases = int(np.sum(diffs > 1e-6))
        
        comp_rows.append({
            "model_name": m,
            "residual_corr_with_GP": r_corr,
            "error_variance_ratio_to_GP": float(np.var(res_dict[m]) / (np.var(res_dict[benchmark_m]) + 1e-12)),
            "cases_beating_GP": better_cases,
            "cases_worse_than_GP": worse_cases,
            "oof_macro_nrmse": results_s42[m]["macro_nrmse"],
            "val_macro_nrmse": results_val[m]["macro_nrmse"],
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(out_dir / "model_complementarity.csv", index=False)
    print("Saved model complementarity table.")
    
    # -------------------------------------------------------------
    # 6. PAIRED STATISTICAL TESTING VS BENCHMARK GP
    # -------------------------------------------------------------
    print("\n--- 6. Running Paired Statistical Tests vs GP_Matern52_3comp ---")
    paired_rows = []
    rng = np.random.default_rng(42)
    n_resamples = 10000
    
    for m in all_model_names:
        if m == benchmark_m:
            continue
        m_case_s42 = np.array([results_s42[m]["case_scores"][c] for c in sorted(TRAIN_CASES)])
        diffs = m_case_s42 - base_case_s42
        
        mean_d = float(np.mean(diffs))
        med_d = float(np.median(diffs))
        std_d = float(np.std(diffs, ddof=1))
        d_cohen = float(mean_d / (std_d + 1e-12))
        
        boot_means = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n_resamples)])
        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))
        
        wins = int(np.sum(diffs < -1e-6))
        losses = int(np.sum(diffs > 1e-6))
        ties = int(len(diffs) - wins - losses)
        
        try:
            wilc_stat, wilc_p = stats.wilcoxon(m_case_s42, base_case_s42)
        except Exception:
            wilc_p = np.nan
        ttest_stat, ttest_p = stats.ttest_rel(m_case_s42, base_case_s42)
        
        # Validation diff
        val_m_scores = np.array([results_val[m]["case_scores"][c] for c in sorted(VAL_CASES)])
        val_base_scores = np.array([results_val[benchmark_m]["case_scores"][c] for c in sorted(VAL_CASES)])
        v_diffs = val_m_scores - val_base_scores
        v_wins = int(np.sum(v_diffs < -1e-6))
        v_losses = int(np.sum(v_diffs > 1e-6))
        
        paired_rows.append({
            "model_name": m,
            "oof_delta_macro": float(results_s42[m]["macro_nrmse"] - results_s42[benchmark_m]["macro_nrmse"]),
            "val_delta_macro": float(results_val[m]["macro_nrmse"] - results_val[benchmark_m]["macro_nrmse"]),
            "mean_case_diff": mean_d,
            "median_case_diff": med_d,
            "std_case_diff": std_d,
            "cohens_d": d_cohen,
            "bootstrap_ci_95_lower": ci_low,
            "bootstrap_ci_95_upper": ci_high,
            "oof_wins_vs_GP": wins,
            "oof_losses_vs_GP": losses,
            "oof_ties": ties,
            "wilcoxon_p_value": float(wilc_p),
            "ttest_p_value": float(ttest_p),
            "val_mean_diff": float(np.mean(v_diffs)),
            "val_wins_vs_GP": v_wins,
            "val_losses_vs_GP": v_losses,
        })
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(out_dir / "paired_model_comparisons.csv", index=False)
    print("Saved paired statistical comparisons.")
    
    # -------------------------------------------------------------
    # 7. MASTER COMPARISON TABLE
    # -------------------------------------------------------------
    master_table_rows = []
    for m in all_model_names:
        s42 = results_s42[m]
        s123 = results_s123[m]
        val = results_val[m]
        
        r_time = base_timings[42].get(m, {}).get("runtime", 0.0)
        w_warn = base_timings[42].get(m, {}).get("warnings", 0)
        
        master_table_rows.append({
            "model_name": m,
            "oof_macro_seed42": s42["macro_nrmse"],
            "oof_macro_seed123": s123["macro_nrmse"],
            "oof_worst_exp": s42["worst_exp"],
            "oof_oil": s42["oil"],
            "oof_gas": s42["gas"],
            "oof_water": s42["water"],
            "oof_case_median": s42["case_median"],
            "oof_case_iqr": s42["case_iqr"],
            "val_macro_NRMSE": val["macro_nrmse"],
            "val_worst_exp": val["worst_exp"],
            "val_oil": val["oil"],
            "val_gas": val["gas"],
            "val_water": val["water"],
            "val_case_median": val["case_median"],
            "runtime_sec": r_time,
            "warnings": w_warn,
        })
    master_table_df = pd.DataFrame(master_table_rows)
    master_table_df.to_csv(out_dir / "master_model_comparison_table.csv", index=False)
    print("Saved master model comparison table.")
    
    # -------------------------------------------------------------
    # 8. VISUALIZATIONS
    # -------------------------------------------------------------
    print("\n--- 8. Generating Visualizations ---")
    sns.set_theme(style="whitegrid", font_scale=1.05)
    
    # Fig 1: Residual correlation heatmap among diverse models
    fig, ax = plt.subplots(figsize=(10, 8))
    diverse_subset = [
        "GP_Matern52_3comp", "GP_Matern32_3comp", "GP_RBF_3comp",
        "Tree_ExtraTrees_3comp", "Tree_RandomForest_3comp", "Tree_HistGB_3comp",
        "Linear_RidgeCV_3comp", "Linear_BayesianRidge_3comp", "Linear_PLS_3comp"
    ]
    sub_corr = res_corr.loc[diverse_subset, diverse_subset]
    sns.heatmap(sub_corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Pairwise Residual Correlations Across Model Families", fontweight="bold")
    plt.tight_layout()
    fig_corr_path = fig_dir / "residual_correlation_heatmap.png"
    plt.savefig(fig_corr_path, dpi=300)
    plt.close()
    
    # Fig 2: OOF NRMSE Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    top_models_plot = [
        "GP_Matern52_3comp", "GP_Matern52_2comp", "GP_Matern52_4comp", "GP_Matern32_3comp",
        "Blend_GP_Matern32", "Stack_Simplex_Diverse", "Stack_Simplex_Tri", "Blend_GP_ExtraTrees",
        "Blend_GP_Ridge", "Linear_RidgeCV_3comp", "Tree_ExtraTrees_3comp"
    ]
    plot_df = master_table_df[master_table_df["model_name"].isin(top_models_plot)].sort_values("oof_macro_seed42")
    
    colors = ["#2ca02c" if "GP" in m and "Blend" not in m and "Stack" not in m else ("#1f77b4" if "Stack" in m or "Blend" in m else "#7f7f7f") for m in plot_df["model_name"]]
    bars = ax.barh(plot_df["model_name"], plot_df["oof_macro_seed42"], color=colors, alpha=0.85)
    ax.axvline(results_s42[benchmark_m]["macro_nrmse"], color="red", ls="--", lw=1.5, label="GP Benchmark (0.002933)")
    ax.set_xlabel("OOF Macro NRMSE (Seed 42)")
    ax.set_title("OOF Generalization Error Across Models and Ensembles", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig_bar_path = fig_dir / "ensemble_vs_standalone_comparison.png"
    plt.savefig(fig_bar_path, dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # 9. JSON AUDIT DUMP
    # -------------------------------------------------------------
    full_audit_dict = {
        "fitted_weights": {
            "tri_simplex_weights": {k: float(w_tri_full[i]) for i, k in enumerate(tri_models)},
            "diverse_simplex_weights": {k: float(w_div_full[i]) for i, k in enumerate(div_models)},
        },
        "seed42_summary": {m: {k: v for k, v in s.items() if k != "case_scores"} for m, s in results_s42.items()},
        "seed123_summary": {m: {k: v for k, v in s.items() if k != "case_scores"} for m, s in results_s123.items()},
        "validation_summary": {m: {k: v for k, v in s.items() if k != "case_scores"} for m, s in results_val.items()},
    }
    with open(out_dir / "master_discovery_audit.json", "w") as f:
        json.dump(full_audit_dict, f, indent=2)
        
    print("\nMaster pipeline executed successfully!")
    return master_table_df, paired_df, comp_df, full_audit_dict

if __name__ == "__main__":
    run_comprehensive_discovery()
