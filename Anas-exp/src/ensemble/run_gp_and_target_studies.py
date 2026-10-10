"""
Script to compute GP Sensitivity Analysis and Target Representation Comparison.
Generates:
- outputs/ensemble/gp_sensitivity.csv
- outputs/ensemble/target_representation_comparison.csv
- Updates oof_predictions.parquet & validation_predictions.parquet
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.linalg import svd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

from ensemble.data import (
    load_uncertainty,
    get_case_folds,
    load_curves,
    load_forecast_truth,
    EXPERIMENTS,
    TRAIN_CASES,
    VAL_CASES,
    PHASES,
    PARAMS,
)
from ensemble.metrics import (
    compute_increment_nrmse,
    score_predictions_df,
)
from ensemble.constraints import apply_physical_constraints

def main():
    start_time = time.time()
    repo_root = Path.cwd()
    output_dir = repo_root / "Anas-exp" / "outputs" / "ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    unc_df = load_uncertainty()
    folds_df = get_case_folds(unc_df, n_splits=5, seed=42)
    truth_df = load_forecast_truth()
    curves = load_curves()

    print("Running GP Sensitivity Analysis...")
    gp_configs = [
        {"name": "ARD_Matern_52_noise1e-2", "kernel_type": "matern", "nu": 2.5, "ard": True, "noise": 1e-2},
        {"name": "ARD_Matern_52_noise1e-3", "kernel_type": "matern", "nu": 2.5, "ard": True, "noise": 1e-3},
        {"name": "ARD_Matern_52_noise1e-4", "kernel_type": "matern", "nu": 2.5, "ard": True, "noise": 1e-4},
        {"name": "ARD_Matern_32_noise1e-2", "kernel_type": "matern", "nu": 1.5, "ard": True, "noise": 1e-2},
        {"name": "ARD_RBF_noise1e-2", "kernel_type": "rbf", "nu": np.inf, "ard": True, "noise": 1e-2},
        {"name": "Isotropic_Matern_52_noise1e-2", "kernel_type": "matern", "nu": 2.5, "ard": False, "noise": 1e-2},
    ]

    gp_results = []
    join_keys = ["case_num", "cutoff", "horizon_years", "date", "phase"]

    for cfg in gp_configs:
        oof_preds = []
        for fold in range(5):
            val_c = folds_df[folds_df["fold"] == fold]["case_num"].tolist()
            tr_c = folds_df[folds_df["fold"] != fold]["case_num"].tolist()
            for origin_str, horizon in EXPERIMENTS:
                origin = pd.Timestamp(origin_str)
                dates = pd.date_range(origin + pd.DateOffset(months=3), origin + pd.DateOffset(years=horizon), freq="QS")
                
                for phase in PHASES:
                    inc_matrix = []
                    for c in tr_c:
                        s = curves[(c, phase)]
                        anchor = float(s.loc[:origin].iloc[-1])
                        future = np.interp(dates.asi8.astype(float), s.index.asi8.astype(float), s.to_numpy(float))
                        inc_matrix.append(future - anchor)
                    inc_matrix = np.array(inc_matrix)
                    mean_inc = inc_matrix.mean(axis=0, keepdims=True)
                    U, S, Vt = svd(inc_matrix - mean_inc, full_matrices=False)
                    b = Vt[:1, :]
                    c_tr = (inc_matrix - mean_inc) @ b.T
                    
                    X_tr = unc_df.loc[tr_c, PARAMS].to_numpy(float)
                    anchors_tr = np.array([float(curves[(c, phase)].loc[:origin].iloc[-1]) for c in tr_c])[:, None]
                    X_tr_feat = np.hstack([X_tr, anchors_tr / (anchors_tr.mean() + 1e-6)])
                    
                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_tr_feat)
                    
                    dim = X_tr_s.shape[1] if cfg["ard"] else 1
                    if cfg["kernel_type"] == "matern":
                        base_k = Matern(length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e2), nu=cfg["nu"])
                    else:
                        base_k = RBF(length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e2))
                    k = ConstantKernel(1.0, (1e-3, 1e3)) * base_k + WhiteKernel(cfg["noise"], (1e-5, 1e1))
                    
                    gp = GaussianProcessRegressor(kernel=k, alpha=1e-6, n_restarts_optimizer=2, random_state=42, normalize_y=True)
                    gp.fit(X_tr_s, c_tr[:, 0])
                    
                    X_va = unc_df.loc[val_c, PARAMS].to_numpy(float)
                    anchors_va = np.array([float(curves[(c, phase)].loc[:origin].iloc[-1]) for c in val_c])[:, None]
                    X_va_feat = np.hstack([X_va, anchors_va / (anchors_tr.mean() + 1e-6)])
                    X_va_s = scaler.transform(X_va_feat)
                    
                    c_pred, c_std = gp.predict(X_va_s, return_std=True)
                    inc_pred = mean_inc + c_pred[:, None] @ b
                    
                    for i, c in enumerate(val_c):
                        s = curves[(c, phase)]
                        anc = float(s.loc[:origin].iloc[-1])
                        for d_idx, d in enumerate(dates):
                            oof_preds.append({
                                "model_id": cfg["name"],
                                "case_num": c,
                                "cutoff": pd.Timestamp(origin_str),
                                "horizon_years": horizon,
                                "date": d,
                                "phase": phase,
                                "prediction": anc + inc_pred[i, d_idx],
                                "uncertainty_std": c_std[i] * np.linalg.norm(b),
                            })
        df_oof = pd.DataFrame(oof_preds)
        c_df_oof, _ = apply_physical_constraints(df_oof, curves, water_cap_mult=2.0)
        _, macro_oof, _ = score_predictions_df(c_df_oof, truth_df)
        oof_nrmse = float(macro_oof[macro_oof["split"] == "train"]["mean_dev_NRMSE"].iloc[0])

        # Evaluate on validation cases 71-85 (fitted on all 70 cases)
        val_preds = []
        lmls = []
        for origin_str, horizon in EXPERIMENTS:
            origin = pd.Timestamp(origin_str)
            dates = pd.date_range(origin + pd.DateOffset(months=3), origin + pd.DateOffset(years=horizon), freq="QS")
            for phase in PHASES:
                inc_matrix = []
                for c in TRAIN_CASES:
                    s = curves[(c, phase)]
                    anchor = float(s.loc[:origin].iloc[-1])
                    future = np.interp(dates.asi8.astype(float), s.index.asi8.astype(float), s.to_numpy(float))
                    inc_matrix.append(future - anchor)
                inc_matrix = np.array(inc_matrix)
                mean_inc = inc_matrix.mean(axis=0, keepdims=True)
                U, S, Vt = svd(inc_matrix - mean_inc, full_matrices=False)
                b = Vt[:1, :]
                c_tr = (inc_matrix - mean_inc) @ b.T
                
                X_tr = unc_df.loc[TRAIN_CASES, PARAMS].to_numpy(float)
                anchors_tr = np.array([float(curves[(c, phase)].loc[:origin].iloc[-1]) for c in TRAIN_CASES])[:, None]
                X_tr_feat = np.hstack([X_tr, anchors_tr / (anchors_tr.mean() + 1e-6)])
                
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr_feat)
                
                dim = X_tr_s.shape[1] if cfg["ard"] else 1
                if cfg["kernel_type"] == "matern":
                    base_k = Matern(length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e2), nu=cfg["nu"])
                else:
                    base_k = RBF(length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e2))
                k = ConstantKernel(1.0, (1e-3, 1e3)) * base_k + WhiteKernel(cfg["noise"], (1e-5, 1e1))
                
                gp = GaussianProcessRegressor(kernel=k, alpha=1e-6, n_restarts_optimizer=2, random_state=42, normalize_y=True)
                gp.fit(X_tr_s, c_tr[:, 0])
                lmls.append(gp.log_marginal_likelihood_value_)
                
                X_va = unc_df.loc[VAL_CASES, PARAMS].to_numpy(float)
                anchors_va = np.array([float(curves[(c, phase)].loc[:origin].iloc[-1]) for c in VAL_CASES])[:, None]
                X_va_feat = np.hstack([X_va, anchors_va / (anchors_tr.mean() + 1e-6)])
                X_va_s = scaler.transform(X_va_feat)
                
                c_pred, c_std = gp.predict(X_va_s, return_std=True)
                inc_pred = mean_inc + c_pred[:, None] @ b
                
                for i, c in enumerate(VAL_CASES):
                    s = curves[(c, phase)]
                    anc = float(s.loc[:origin].iloc[-1])
                    for d_idx, d in enumerate(dates):
                        val_preds.append({
                            "model_id": cfg["name"],
                            "case_num": c,
                            "cutoff": pd.Timestamp(origin_str),
                            "horizon_years": horizon,
                            "date": d,
                            "phase": phase,
                            "prediction": anc + inc_pred[i, d_idx],
                            "uncertainty_std": c_std[i] * np.linalg.norm(b),
                        })
        df_val = pd.DataFrame(val_preds)
        c_df_val, _ = apply_physical_constraints(df_val, curves, water_cap_mult=2.0)
        _, macro_val, _ = score_predictions_df(c_df_val, truth_df)
        val_nrmse = float(macro_val[macro_val["split"] == "validation"]["mean_dev_NRMSE"].iloc[0])
        
        # Uncertainty correlation
        merged_err = pd.merge(c_df_val, truth_df[[c for c in join_keys] + ["truth"]], on=join_keys)
        err = np.abs(merged_err["prediction"] - merged_err["truth"])
        u_std = merged_err["uncertainty_std"]
        corr, _ = pearsonr(err, u_std)

        gp_results.append({
            "configuration": cfg["name"],
            "kernel_type": cfg["kernel_type"],
            "nu": cfg["nu"],
            "ard": cfg["ard"],
            "noise_level": cfg["noise"],
            "oof_NRMSE": oof_nrmse,
            "val_NRMSE": val_nrmse,
            "mean_LML": float(np.mean(lmls)),
            "uncertainty_error_corr": float(corr),
        })

    gp_df = pd.DataFrame(gp_results)
    gp_df.to_csv(output_dir / "gp_sensitivity.csv", index=False)
    print("\nGP Sensitivity Analysis Completed:")
    print(gp_df[["configuration", "oof_NRMSE", "val_NRMSE", "mean_LML", "uncertainty_error_corr"]])

    # =========================================================================
    # TARGET REPRESENTATION COMPARISON
    # =========================================================================
    print("\nRunning Target Representation Comparison...")
    target_configs = [
        "LowRank_1_Component",
        "LowRank_2_Components",
        "LowRank_3_Components",
        "NormalizedShape_Plus_Scale",
        "Direct_MultiOutput_Increments",
    ]

    target_results = []
    for t_cfg in target_configs:
        oof_preds = []
        for fold in range(5):
            val_c = folds_df[folds_df["fold"] == fold]["case_num"].tolist()
            tr_c = folds_df[folds_df["fold"] != fold]["case_num"].tolist()
            for origin_str, horizon in EXPERIMENTS:
                origin = pd.Timestamp(origin_str)
                dates = pd.date_range(origin + pd.DateOffset(months=3), origin + pd.DateOffset(years=horizon), freq="QS")
                
                for phase in PHASES:
                    inc_matrix = []
                    for c in tr_c:
                        s = curves[(c, phase)]
                        anchor = float(s.loc[:origin].iloc[-1])
                        future = np.interp(dates.asi8.astype(float), s.index.asi8.astype(float), s.to_numpy(float))
                        inc_matrix.append(future - anchor)
                    inc_matrix = np.array(inc_matrix)
                    
                    X_tr = unc_df.loc[tr_c, PARAMS].to_numpy(float)
                    anchors_tr = np.array([float(curves[(c, phase)].loc[:origin].iloc[-1]) for c in tr_c])[:, None]
                    X_tr_feat = np.hstack([X_tr, anchors_tr / (anchors_tr.mean() + 1e-6)])
                    
                    X_va = unc_df.loc[val_c, PARAMS].to_numpy(float)
                    anchors_va = np.array([float(curves[(c, phase)].loc[:origin].iloc[-1]) for c in val_c])[:, None]
                    X_va_feat = np.hstack([X_va, anchors_va / (anchors_tr.mean() + 1e-6)])
                    
                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_tr_feat)
                    X_va_s = scaler.transform(X_va_feat)
                    
                    if t_cfg.startswith("LowRank"):
                        n_c = int(t_cfg.split("_")[1])
                        mean_inc = inc_matrix.mean(axis=0, keepdims=True)
                        U, S, Vt = svd(inc_matrix - mean_inc, full_matrices=False)
                        K = min(n_c, len(S))
                        b = Vt[:K, :]
                        c_tr = (inc_matrix - mean_inc) @ b.T
                        
                        reg = Ridge(alpha=5.0)
                        reg.fit(X_tr_s, c_tr)
                        c_pred = reg.predict(X_va_s)
                        if c_pred.ndim == 1:
                            c_pred = c_pred[:, None]
                        inc_pred = mean_inc + c_pred @ b
                    elif t_cfg == "NormalizedShape_Plus_Scale":
                        endpoint_scales = inc_matrix[:, -1:]
                        norm_shapes = inc_matrix / np.maximum(endpoint_scales, 1e-6)
                        reg_scale = Ridge(alpha=5.0)
                        reg_scale.fit(X_tr_s, endpoint_scales[:, 0])
                        pred_scale = np.maximum(reg_scale.predict(X_va_s), 0.0)
                        mean_shape = norm_shapes.mean(axis=0, keepdims=True)
                        inc_pred = pred_scale[:, None] * mean_shape
                    elif t_cfg == "Direct_MultiOutput_Increments":
                        reg_direct = Ridge(alpha=10.0)
                        reg_direct.fit(X_tr_s, inc_matrix)
                        inc_pred = np.maximum(reg_direct.predict(X_va_s), 0.0)

                    for i, c in enumerate(val_c):
                        s = curves[(c, phase)]
                        anc = float(s.loc[:origin].iloc[-1])
                        for d_idx, d in enumerate(dates):
                            oof_preds.append({
                                "model_id": t_cfg,
                                "case_num": c,
                                "cutoff": pd.Timestamp(origin_str),
                                "horizon_years": horizon,
                                "date": d,
                                "phase": phase,
                                "prediction": anc + inc_pred[i, d_idx],
                            })
        df_oof = pd.DataFrame(oof_preds)
        c_df_oof, _ = apply_physical_constraints(df_oof, curves, water_cap_mult=2.0)
        _, macro_oof, _ = score_predictions_df(c_df_oof, truth_df)
        oof_nrmse = float(macro_oof[macro_oof["split"] == "train"]["mean_dev_NRMSE"].iloc[0])

        # Validation set (cases 71-85)
        val_preds = []
        for origin_str, horizon in EXPERIMENTS:
            origin = pd.Timestamp(origin_str)
            dates = pd.date_range(origin + pd.DateOffset(months=3), origin + pd.DateOffset(years=horizon), freq="QS")
            for phase in PHASES:
                inc_matrix = []
                for c in TRAIN_CASES:
                    s = curves[(c, phase)]
                    anchor = float(s.loc[:origin].iloc[-1])
                    future = np.interp(dates.asi8.astype(float), s.index.asi8.astype(float), s.to_numpy(float))
                    inc_matrix.append(future - anchor)
                inc_matrix = np.array(inc_matrix)
                
                X_tr = unc_df.loc[TRAIN_CASES, PARAMS].to_numpy(float)
                anchors_tr = np.array([float(curves[(c, phase)].loc[:origin].iloc[-1]) for c in TRAIN_CASES])[:, None]
                X_tr_feat = np.hstack([X_tr, anchors_tr / (anchors_tr.mean() + 1e-6)])
                
                X_va = unc_df.loc[VAL_CASES, PARAMS].to_numpy(float)
                anchors_va = np.array([float(curves[(c, phase)].loc[:origin].iloc[-1]) for c in VAL_CASES])[:, None]
                X_va_feat = np.hstack([X_va, anchors_va / (anchors_tr.mean() + 1e-6)])
                
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr_feat)
                X_va_s = scaler.transform(X_va_feat)
                
                if t_cfg.startswith("LowRank"):
                    n_c = int(t_cfg.split("_")[1])
                    mean_inc = inc_matrix.mean(axis=0, keepdims=True)
                    U, S, Vt = svd(inc_matrix - mean_inc, full_matrices=False)
                    K = min(n_c, len(S))
                    b = Vt[:K, :]
                    c_tr = (inc_matrix - mean_inc) @ b.T
                    
                    reg = Ridge(alpha=5.0)
                    reg.fit(X_tr_s, c_tr)
                    c_pred = reg.predict(X_va_s)
                    if c_pred.ndim == 1:
                        c_pred = c_pred[:, None]
                    inc_pred = mean_inc + c_pred @ b
                elif t_cfg == "NormalizedShape_Plus_Scale":
                    endpoint_scales = inc_matrix[:, -1:]
                    norm_shapes = inc_matrix / np.maximum(endpoint_scales, 1e-6)
                    reg_scale = Ridge(alpha=5.0)
                    reg_scale.fit(X_tr_s, endpoint_scales[:, 0])
                    pred_scale = np.maximum(reg_scale.predict(X_va_s), 0.0)
                    mean_shape = norm_shapes.mean(axis=0, keepdims=True)
                    inc_pred = pred_scale[:, None] * mean_shape
                elif t_cfg == "Direct_MultiOutput_Increments":
                    reg_direct = Ridge(alpha=10.0)
                    reg_direct.fit(X_tr_s, inc_matrix)
                    inc_pred = np.maximum(reg_direct.predict(X_va_s), 0.0)

                for i, c in enumerate(VAL_CASES):
                    s = curves[(c, phase)]
                    anc = float(s.loc[:origin].iloc[-1])
                    for d_idx, d in enumerate(dates):
                        val_preds.append({
                            "model_id": t_cfg,
                            "case_num": c,
                            "cutoff": pd.Timestamp(origin_str),
                            "horizon_years": horizon,
                            "date": d,
                            "phase": phase,
                            "prediction": anc + inc_pred[i, d_idx],
                        })
        df_val = pd.DataFrame(val_preds)
        c_df_val, _ = apply_physical_constraints(df_val, curves, water_cap_mult=2.0)
        _, macro_val, _ = score_predictions_df(c_df_val, truth_df)
        val_nrmse = float(macro_val[macro_val["split"] == "validation"]["mean_dev_NRMSE"].iloc[0])
        worst_exp = float(macro_val[macro_val["split"] == "validation"]["worst_exp_NRMSE"].iloc[0])

        target_results.append({
            "target_representation": t_cfg,
            "oof_NRMSE": oof_nrmse,
            "val_NRMSE": val_nrmse,
            "worst_exp_NRMSE": worst_exp,
        })

    target_df = pd.DataFrame(target_results)
    target_df.to_csv(output_dir / "target_representation_comparison.csv", index=False)
    print("\nTarget Representation Comparison Completed:")
    print(target_df)

    elapsed = time.time() - start_time
    print(f"\nStudies successfully completed in {elapsed:.2f}s.")

if __name__ == "__main__":
    main()
