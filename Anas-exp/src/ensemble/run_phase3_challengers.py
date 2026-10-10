"""
Phase 3 & 4: Controlled Model Comparison Study.
Evaluates incumbent GP against candidate GP kernels, regularized linear/PLS,
tree-based response surfaces, and leak-free OOF residual hybrid models
under strictly identical 5-fold CV on Cases 1-70, with secondary validation on Cases 71-85.
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold

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

# Helper to build features and targets for an experiment and phase
def build_dataset(cases, origin, forecast_dates, phase, curves, unc_df, train_anchor_mean=None):
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

def run_challenger_study():
    output_dir = Path("outputs/eda_discovery")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    unc_df = load_uncertainty()
    curves = load_curves()
    truth_df = load_forecast_truth()
    folds_df = get_case_folds(unc_df, n_splits=5, seed=42)
    
    model_names = [
        "GP_Matern52_1comp",       # Incumbent Baseline
        "GP_Matern52_2comp",       # Higher rank GP
        "GP_Matern32_1comp",       # Challenger 1a
        "GP_RBF_1comp",            # Challenger 1b
        "Linear_RidgeCV_1comp",    # Challenger 2a
        "Linear_PLS_1comp",        # Challenger 2b
        "Tree_ExtraTrees_1comp",   # Challenger 3a
        "Tree_RandomForest_1comp", # Challenger 3b
        "Tree_HistGB_1comp",       # Challenger 3c
        "Hybrid_GP_Residual_1comp" # Challenger 4: GP + OOF Residual Ridge
    ]
    
    oof_predictions = {m: [] for m in model_names}
    val_predictions = {m: [] for m in model_names}
    
    print("=" * 80)
    print("STARTING CONTROLLED MODEL COMPARISON (5-FOLD CV ON CASES 1-70)")
    print("=" * 80)
    
    # ---------------------------------------------------------
    # PART 1: 5-FOLD CV ON CASES 1-70
    # ---------------------------------------------------------
    for fold in range(5):
        fold_start = time.time()
        val_c = sorted(folds_df[folds_df["fold"] == fold]["case_num"].tolist())
        train_c = sorted(folds_df[folds_df["fold"] != fold]["case_num"].tolist())
        print(f"Fold {fold}: Train={len(train_c)}, Val={len(val_c)}...")
        
        for origin_str, horizon in EXPERIMENTS:
            origin = pd.Timestamp(origin_str)
            forecast_dates = pd.date_range(
                origin + pd.DateOffset(months=3),
                origin + pd.DateOffset(years=horizon),
                freq="QS",
            )
            
            for phase in PHASES:
                # Build training data
                X_tr, inc_tr, anchors_tr, anchor_mean = build_dataset(
                    train_c, origin, forecast_dates, phase, curves, unc_df
                )
                X_val, inc_val, anchors_val, _ = build_dataset(
                    val_c, origin, forecast_dates, phase, curves, unc_df, train_anchor_mean=anchor_mean
                )
                
                # Scaler
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_val_s = scaler.transform(X_val)
                
                # SVD on increments
                mean_inc = inc_tr.mean(axis=0, keepdims=True)
                centered_tr = inc_tr - mean_inc
                U, S, Vt = svd(centered_tr, full_matrices=False)
                
                basis_1 = Vt[:1, :]
                coeffs_tr_1 = (centered_tr @ basis_1.T)[:, 0]
                
                basis_2 = Vt[:2, :]
                coeffs_tr_2 = (centered_tr @ basis_2.T)
                
                n_feat = X_tr_s.shape[1]
                
                # Model 1: Incumbent GP Matérn 5/2 (1 comp)
                k_m52 = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                    length_scale=np.ones(n_feat), length_scale_bounds=(1e-2, 1e2), nu=2.5
                ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
                gp_m52_1 = GaussianProcessRegressor(kernel=k_m52, alpha=1e-6, n_restarts_optimizer=2, random_state=42, normalize_y=True)
                gp_m52_1.fit(X_tr_s, coeffs_tr_1)
                pred_c_m52_1 = gp_m52_1.predict(X_val_s)
                pred_inc_m52_1 = mean_inc + pred_c_m52_1[:, None] @ basis_1
                
                # Model 2: GP Matérn 5/2 (2 comp)
                pred_c_m52_2 = np.zeros((len(val_c), 2))
                for comp_k in range(2):
                    gp_k = GaussianProcessRegressor(kernel=k_m52, alpha=1e-6, n_restarts_optimizer=2, random_state=42 + comp_k, normalize_y=True)
                    gp_k.fit(X_tr_s, coeffs_tr_2[:, comp_k])
                    pred_c_m52_2[:, comp_k] = gp_k.predict(X_val_s)
                pred_inc_m52_2 = mean_inc + pred_c_m52_2 @ basis_2
                
                # Model 3: GP Matérn 3/2 (1 comp)
                k_m32 = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                    length_scale=np.ones(n_feat), length_scale_bounds=(1e-2, 1e2), nu=1.5
                ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
                gp_m32 = GaussianProcessRegressor(kernel=k_m32, alpha=1e-6, n_restarts_optimizer=2, random_state=42, normalize_y=True)
                gp_m32.fit(X_tr_s, coeffs_tr_1)
                pred_c_m32 = gp_m32.predict(X_val_s)
                pred_inc_m32 = mean_inc + pred_c_m32[:, None] @ basis_1
                
                # Model 4: GP RBF (1 comp)
                k_rbf = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
                    length_scale=np.ones(n_feat), length_scale_bounds=(1e-2, 1e2)
                ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
                gp_rbf = GaussianProcessRegressor(kernel=k_rbf, alpha=1e-6, n_restarts_optimizer=2, random_state=42, normalize_y=True)
                gp_rbf.fit(X_tr_s, coeffs_tr_1)
                pred_c_rbf = gp_rbf.predict(X_val_s)
                pred_inc_rbf = mean_inc + pred_c_rbf[:, None] @ basis_1
                
                # Model 5: Linear RidgeCV (1 comp)
                ridge = RidgeCV(alphas=np.logspace(-2, 3, 20), cv=5)
                ridge.fit(X_tr_s, coeffs_tr_1)
                pred_c_ridge = ridge.predict(X_val_s)
                pred_inc_ridge = mean_inc + pred_c_ridge[:, None] @ basis_1
                
                # Model 6: PLS Regression (1 comp)
                pls = PLSRegression(n_components=1)
                pls.fit(X_tr_s, coeffs_tr_1)
                pred_c_pls = pls.predict(X_val_s).flatten()
                pred_inc_pls = mean_inc + pred_c_pls[:, None] @ basis_1
                
                # Model 7: ExtraTrees (1 comp)
                et = ExtraTreesRegressor(n_estimators=100, max_depth=5, min_samples_leaf=2, random_state=42)
                et.fit(X_tr_s, coeffs_tr_1)
                pred_c_et = et.predict(X_val_s)
                pred_inc_et = mean_inc + pred_c_et[:, None] @ basis_1
                
                # Model 8: RandomForest (1 comp)
                rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=2, random_state=42)
                rf.fit(X_tr_s, coeffs_tr_1)
                pred_c_rf = rf.predict(X_val_s)
                pred_inc_rf = mean_inc + pred_c_rf[:, None] @ basis_1
                
                # Model 9: HistGradientBoosting (1 comp)
                hgb = HistGradientBoostingRegressor(max_iter=50, max_depth=3, min_samples_leaf=3, random_state=42)
                hgb.fit(X_tr_s, coeffs_tr_1)
                pred_c_hgb = hgb.predict(X_val_s)
                pred_inc_hgb = mean_inc + pred_c_hgb[:, None] @ basis_1
                
                # Model 10: Hybrid GP + Inner OOF Residual Correction (1 comp)
                # Fit GP inner OOF residuals strictly on train_c
                inner_kf = KFold(n_splits=5, shuffle=True, random_state=123)
                oof_res_coeffs = np.zeros(len(train_c))
                for in_tr_idx, in_val_idx in inner_kf.split(train_c):
                    gp_inner = GaussianProcessRegressor(kernel=k_m52, alpha=1e-6, n_restarts_optimizer=1, random_state=42, normalize_y=True)
                    gp_inner.fit(X_tr_s[in_tr_idx], coeffs_tr_1[in_tr_idx])
                    oof_res_coeffs[in_val_idx] = coeffs_tr_1[in_val_idx] - gp_inner.predict(X_tr_s[in_val_idx])
                    
                res_regressor = RidgeCV(alphas=np.logspace(-1, 3, 10), cv=5)
                res_regressor.fit(X_tr_s, oof_res_coeffs)
                pred_res_c = res_regressor.predict(X_val_s)
                pred_c_hybrid = pred_c_m52_1 + pred_res_c
                pred_inc_hybrid = mean_inc + pred_c_hybrid[:, None] @ basis_1
                
                # Store all predictions
                model_preds_dict = {
                    "GP_Matern52_1comp": pred_inc_m52_1,
                    "GP_Matern52_2comp": pred_inc_m52_2,
                    "GP_Matern32_1comp": pred_inc_m32,
                    "GP_RBF_1comp": pred_inc_rbf,
                    "Linear_RidgeCV_1comp": pred_inc_ridge,
                    "Linear_PLS_1comp": pred_inc_pls,
                    "Tree_ExtraTrees_1comp": pred_inc_et,
                    "Tree_RandomForest_1comp": pred_inc_rf,
                    "Tree_HistGB_1comp": pred_inc_hgb,
                    "Hybrid_GP_Residual_1comp": pred_inc_hybrid,
                }
                
                for m_id, p_inc in model_preds_dict.items():
                    for i, c in enumerate(val_c):
                        cum_pred = anchors_val[i, 0] + p_inc[i, :]
                        for t_idx, d in enumerate(forecast_dates):
                            oof_predictions[m_id].append({
                                "model_id": m_id,
                                "case_num": c,
                                "origin": origin,
                                "cutoff": origin,
                                "horizon_years": horizon,
                                "date": d,
                                "phase": phase,
                                "prediction": float(cum_pred[t_idx]),
                                "split": "train",
                            })
                            
        print(f"Fold {fold} finished in {time.time() - fold_start:.2f}s")
        
    # ---------------------------------------------------------
    # PART 2: SECONDARY VALIDATION ON CASES 71-85 (TRAINED ON FULL CASES 1-70)
    # ---------------------------------------------------------
    print("\n--- Training on Full Cases 1-70 & Evaluating on Cases 71-85 ---")
    val_cases_full = sorted(VAL_CASES)
    train_cases_full = sorted(TRAIN_CASES)
    
    for origin_str, horizon in EXPERIMENTS:
        origin = pd.Timestamp(origin_str)
        forecast_dates = pd.date_range(
            origin + pd.DateOffset(months=3),
            origin + pd.DateOffset(years=horizon),
            freq="QS",
        )
        for phase in PHASES:
            X_tr, inc_tr, anchors_tr, anchor_mean = build_dataset(
                train_cases_full, origin, forecast_dates, phase, curves, unc_df
            )
            X_val, inc_val, anchors_val, _ = build_dataset(
                val_cases_full, origin, forecast_dates, phase, curves, unc_df, train_anchor_mean=anchor_mean
            )
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)
            
            mean_inc = inc_tr.mean(axis=0, keepdims=True)
            centered_tr = inc_tr - mean_inc
            U, S, Vt = svd(centered_tr, full_matrices=False)
            
            basis_1 = Vt[:1, :]
            coeffs_tr_1 = (centered_tr @ basis_1.T)[:, 0]
            basis_2 = Vt[:2, :]
            coeffs_tr_2 = (centered_tr @ basis_2.T)
            n_feat = X_tr_s.shape[1]
            
            # GP Matern 5/2 (1 comp)
            k_m52 = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(n_feat), length_scale_bounds=(1e-2, 1e2), nu=2.5) + WhiteKernel(1e-2, (1e-5, 1e1))
            gp_m52_1 = GaussianProcessRegressor(kernel=k_m52, alpha=1e-6, n_restarts_optimizer=2, random_state=42, normalize_y=True)
            gp_m52_1.fit(X_tr_s, coeffs_tr_1)
            pred_c_m52_1 = gp_m52_1.predict(X_val_s)
            pred_inc_m52_1 = mean_inc + pred_c_m52_1[:, None] @ basis_1
            
            # GP Matern 5/2 (2 comp)
            pred_c_m52_2 = np.zeros((len(val_cases_full), 2))
            for comp_k in range(2):
                gp_k = GaussianProcessRegressor(kernel=k_m52, alpha=1e-6, n_restarts_optimizer=2, random_state=42 + comp_k, normalize_y=True)
                gp_k.fit(X_tr_s, coeffs_tr_2[:, comp_k])
                pred_c_m52_2[:, comp_k] = gp_k.predict(X_val_s)
            pred_inc_m52_2 = mean_inc + pred_c_m52_2 @ basis_2
            
            # GP Matern 3/2 (1 comp)
            k_m32 = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(n_feat), length_scale_bounds=(1e-2, 1e2), nu=1.5) + WhiteKernel(1e-2, (1e-5, 1e1))
            gp_m32 = GaussianProcessRegressor(kernel=k_m32, alpha=1e-6, n_restarts_optimizer=2, random_state=42, normalize_y=True)
            gp_m32.fit(X_tr_s, coeffs_tr_1)
            pred_c_m32 = gp_m32.predict(X_val_s)
            pred_inc_m32 = mean_inc + pred_c_m32[:, None] @ basis_1
            
            # GP RBF (1 comp)
            k_rbf = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(n_feat), length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(1e-2, (1e-5, 1e1))
            gp_rbf = GaussianProcessRegressor(kernel=k_rbf, alpha=1e-6, n_restarts_optimizer=2, random_state=42, normalize_y=True)
            gp_rbf.fit(X_tr_s, coeffs_tr_1)
            pred_c_rbf = gp_rbf.predict(X_val_s)
            pred_inc_rbf = mean_inc + pred_c_rbf[:, None] @ basis_1
            
            # Ridge
            ridge = RidgeCV(alphas=np.logspace(-2, 3, 20), cv=5)
            ridge.fit(X_tr_s, coeffs_tr_1)
            pred_c_ridge = ridge.predict(X_val_s)
            pred_inc_ridge = mean_inc + pred_c_ridge[:, None] @ basis_1
            
            # PLS
            pls = PLSRegression(n_components=1)
            pls.fit(X_tr_s, coeffs_tr_1)
            pred_c_pls = pls.predict(X_val_s).flatten()
            pred_inc_pls = mean_inc + pred_c_pls[:, None] @ basis_1
            
            # ExtraTrees
            et = ExtraTreesRegressor(n_estimators=100, max_depth=5, min_samples_leaf=2, random_state=42)
            et.fit(X_tr_s, coeffs_tr_1)
            pred_c_et = et.predict(X_val_s)
            pred_inc_et = mean_inc + pred_c_et[:, None] @ basis_1
            
            # RandomForest
            rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=2, random_state=42)
            rf.fit(X_tr_s, coeffs_tr_1)
            pred_c_rf = rf.predict(X_val_s)
            pred_inc_rf = mean_inc + pred_c_rf[:, None] @ basis_1
            
            # HistGB
            hgb = HistGradientBoostingRegressor(max_iter=50, max_depth=3, min_samples_leaf=3, random_state=42)
            hgb.fit(X_tr_s, coeffs_tr_1)
            pred_c_hgb = hgb.predict(X_val_s)
            pred_inc_hgb = mean_inc + pred_c_hgb[:, None] @ basis_1
            
            # Hybrid
            inner_kf = KFold(n_splits=5, shuffle=True, random_state=123)
            oof_res_coeffs = np.zeros(len(train_cases_full))
            for in_tr_idx, in_val_idx in inner_kf.split(train_cases_full):
                gp_inner = GaussianProcessRegressor(kernel=k_m52, alpha=1e-6, n_restarts_optimizer=1, random_state=42, normalize_y=True)
                gp_inner.fit(X_tr_s[in_tr_idx], coeffs_tr_1[in_tr_idx])
                oof_res_coeffs[in_val_idx] = coeffs_tr_1[in_val_idx] - gp_inner.predict(X_tr_s[in_val_idx])
            res_reg = RidgeCV(alphas=np.logspace(-1, 3, 10), cv=5)
            res_reg.fit(X_tr_s, oof_res_coeffs)
            pred_res_c = res_reg.predict(X_val_s)
            pred_c_hybrid = pred_c_m52_1 + pred_res_c
            pred_inc_hybrid = mean_inc + pred_c_hybrid[:, None] @ basis_1
            
            val_model_preds = {
                "GP_Matern52_1comp": pred_inc_m52_1,
                "GP_Matern52_2comp": pred_inc_m52_2,
                "GP_Matern32_1comp": pred_inc_m32,
                "GP_RBF_1comp": pred_inc_rbf,
                "Linear_RidgeCV_1comp": pred_inc_ridge,
                "Linear_PLS_1comp": pred_inc_pls,
                "Tree_ExtraTrees_1comp": pred_inc_et,
                "Tree_RandomForest_1comp": pred_inc_rf,
                "Tree_HistGB_1comp": pred_inc_hgb,
                "Hybrid_GP_Residual_1comp": pred_inc_hybrid,
            }
            
            for m_id, p_inc in val_model_preds.items():
                for i, c in enumerate(val_cases_full):
                    cum_pred = anchors_val[i, 0] + p_inc[i, :]
                    for t_idx, d in enumerate(forecast_dates):
                        val_predictions[m_id].append({
                            "model_id": m_id,
                            "case_num": c,
                            "origin": origin,
                            "cutoff": origin,
                            "horizon_years": horizon,
                            "date": d,
                            "phase": phase,
                            "prediction": float(cum_pred[t_idx]),
                            "split": "validation",
                        })
                        
    # ---------------------------------------------------------
    # PART 3: SCORING & COMPILATION
    # ---------------------------------------------------------
    print("\n--- Scoring all candidate models ---")
    score_rows = []
    case_scores_dict = {}
    
    for m_id in model_names:
        oof_df_m = pd.DataFrame(oof_predictions[m_id])
        val_df_m = pd.DataFrame(val_predictions[m_id])
        
        # Score OOF (split='train')
        oof_p_scores, oof_m_scores, _ = score_predictions_df(oof_df_m, truth_df)
        oof_macro = float(oof_m_scores[oof_m_scores["split"] == "train"]["mean_dev_NRMSE"].iloc[0])
        
        # Per experiment OOF
        oof_by_exp = oof_p_scores[oof_p_scores["split"] == "train"].groupby(["cutoff", "horizon_years"])["increment_NRMSE"].mean().to_dict()
        # Per phase OOF
        oof_by_phase = oof_p_scores[oof_p_scores["split"] == "train"].groupby("phase")["increment_NRMSE"].mean().to_dict()
        
        # Score Validation (split='validation')
        val_p_scores, val_m_scores, val_c_scores = score_predictions_df(val_df_m, truth_df)
        val_macro = float(val_m_scores[val_m_scores["split"] == "validation"]["mean_dev_NRMSE"].iloc[0])
        val_by_exp = val_p_scores[val_p_scores["split"] == "validation"].groupby(["cutoff", "horizon_years"])["increment_NRMSE"].mean().to_dict()
        val_by_phase = val_p_scores[val_p_scores["split"] == "validation"].groupby("phase")["increment_NRMSE"].mean().to_dict()
        
        # Save case-level validation NRMSE for paired statistics
        # Case macro NRMSE: average across 4 experiments and 3 phases
        case_macro = val_c_scores.groupby("case_num")["case_NRMSE"].mean().to_dict()
        case_scores_dict[m_id] = case_macro
        
        score_rows.append({
            "model_id": m_id,
            "oof_macro_NRMSE": oof_macro,
            "oof_2003_3y": oof_by_exp.get(("2003-01-01", 3), np.nan),
            "oof_2003_5y": oof_by_exp.get(("2003-01-01", 5), np.nan),
            "oof_2004_3y": oof_by_exp.get(("2004-01-01", 3), np.nan),
            "oof_2005_3y": oof_by_exp.get(("2005-01-01", 3), np.nan),
            "oof_oil": oof_by_phase.get("oil_cum", np.nan),
            "oof_gas": oof_by_phase.get("gas_cum", np.nan),
            "oof_water": oof_by_phase.get("water_cum", np.nan),
            "val_macro_NRMSE": val_macro,
            "val_2003_3y": val_by_exp.get(("2003-01-01", 3), np.nan),
            "val_2003_5y": val_by_exp.get(("2003-01-01", 5), np.nan),
            "val_2004_3y": val_by_exp.get(("2004-01-01", 3), np.nan),
            "val_2005_3y": val_by_exp.get(("2005-01-01", 3), np.nan),
            "val_oil": val_by_phase.get("oil_cum", np.nan),
            "val_gas": val_by_phase.get("gas_cum", np.nan),
            "val_water": val_by_phase.get("water_cum", np.nan),
        })
        
    model_comp_df = pd.DataFrame(score_rows)
    model_comp_path = output_dir / "model_comparison.csv"
    model_comp_df.to_csv(model_comp_path, index=False)
    print(f"Saved model comparison table to: {model_comp_path}")
    
    # ---------------------------------------------------------
    # PART 4: PAIRED STATISTICAL COMPARISONS AGAINST INCUMBENT GP
    # ---------------------------------------------------------
    print("\n--- Paired Statistical Comparisons vs Incumbent GP (Matérn 5/2, 1-comp) ---")
    base_m_id = "GP_Matern52_1comp"
    base_case_scores = np.array([case_scores_dict[base_m_id][c] for c in sorted(VAL_CASES)])
    
    paired_rows = []
    rng = np.random.default_rng(42)
    n_resamples = 10000
    
    for m_id in model_names:
        if m_id == base_m_id:
            continue
        challenger_scores = np.array([case_scores_dict[m_id][c] for c in sorted(VAL_CASES)])
        diffs = challenger_scores - base_case_scores  # negative means challenger wins
        
        mean_diff = float(np.mean(diffs))
        median_diff = float(np.median(diffs))
        
        # 10,000 bootstrap resamples of mean difference
        boot_means = np.array([
            rng.choice(diffs, size=len(diffs), replace=True).mean()
            for _ in range(n_resamples)
        ])
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
        
        # Win / Loss / Tie count (out of 15 validation cases)
        wins = int(np.sum(diffs < -1e-6))
        losses = int(np.sum(diffs > 1e-6))
        ties = int(len(diffs) - wins - losses)
        
        # Wilcoxon signed-rank and paired t-test
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(challenger_scores, base_case_scores)
        except Exception:
            wilcoxon_stat, wilcoxon_p = np.nan, np.nan
        ttest_stat, ttest_p = stats.ttest_rel(challenger_scores, base_case_scores)
        
        # Decision rule evaluation
        # Rule 1: Lower OOF AND Val not degraded by > 10% AND statistically significant improvement (p < 0.05 and CI upper < 0)
        oof_base = model_comp_df[model_comp_df["model_id"] == base_m_id]["oof_macro_NRMSE"].iloc[0]
        oof_chal = model_comp_df[model_comp_df["model_id"] == m_id]["oof_macro_NRMSE"].iloc[0]
        val_base = model_comp_df[model_comp_df["model_id"] == base_m_id]["val_macro_NRMSE"].iloc[0]
        val_chal = model_comp_df[model_comp_df["model_id"] == m_id]["val_macro_NRMSE"].iloc[0]
        
        beats_oof = bool(oof_chal < oof_base)
        val_safe = bool(val_chal <= val_base * 1.10)
        stat_sig_better = bool(mean_diff < 0 and ci_upper < 0 and wilcoxon_p < 0.05)
        decision = "ACCEPT (Replaces Incumbent)" if (beats_oof and val_safe and stat_sig_better) else "REJECT (Retain Incumbent)"
        
        paired_rows.append({
            "challenger_model": m_id,
            "oof_delta_macro": float(oof_chal - oof_base),
            "val_delta_macro": float(val_chal - val_base),
            "mean_case_diff": mean_diff,
            "median_case_diff": median_diff,
            "bootstrap_ci_95_lower": ci_lower,
            "bootstrap_ci_95_upper": ci_upper,
            "challenger_wins": wins,
            "challenger_losses": losses,
            "ties": ties,
            "wilcoxon_p_value": float(wilcoxon_p),
            "ttest_p_value": float(ttest_p),
            "decision": decision,
        })
        
    paired_df = pd.DataFrame(paired_rows)
    paired_comp_path = output_dir / "paired_comparisons.csv"
    paired_df.to_csv(paired_comp_path, index=False)
    print(f"Saved paired statistical comparison table to: {paired_comp_path}")
    
    print("\nStudy complete! Both CSVs generated successfully.")
    return model_comp_df, paired_df

if __name__ == "__main__":
    run_challenger_study()
