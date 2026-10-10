"""
Corrected Stacking Evaluation and Head-to-Head Architecture Benchmark.
Re-evaluates stacking with:
1. Normalized increment weighting (anchor removed, phase standardized)
2. Softmax-parametrized BFGS optimization (zero constraint failures)
3. Phase-specific simplex weighting
4. Direct challenge NRMSE optimization
5. GP K=3 vs GP K=4 vs GP Blend comparison
"""
import sys
import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.linalg import svd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.ensemble import ExtraTreesRegressor

from ensemble.data import (
    load_uncertainty,
    load_curves,
    load_forecast_truth,
    TRAIN_CASES,
    VAL_CASES,
    PARAMS,
    PHASES,
    EXPERIMENTS,
)
from ensemble.metrics import score_predictions_df

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
    for k in range(K):
        y_k = coeffs_tr[:, k]
        if model_type == "gp_matern52":
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=np.ones(n_feat), length_scale_bounds=(1e-2, 1e2), nu=2.5
            ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
            m = GaussianProcessRegressor(
                kernel=kernel, alpha=1e-6, n_restarts_optimizer=2,
                random_state=random_state + k * 17, normalize_y=True
            )
        elif model_type == "gp_matern32":
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=np.ones(n_feat), length_scale_bounds=(1e-2, 1e2), nu=1.5
            ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
            m = GaussianProcessRegressor(
                kernel=kernel, alpha=1e-6, n_restarts_optimizer=2,
                random_state=random_state + k * 17, normalize_y=True
            )
        elif model_type == "tree_extratrees":
            m = ExtraTreesRegressor(n_estimators=100, max_depth=6, min_samples_leaf=2, random_state=random_state + k * 17)
        elif model_type == "linear_ridge":
            m = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X_tr_s, y_k)
        pred_coeffs[:, k] = m.predict(X_val_s)
    return pred_coeffs

def fit_simplex_softmax(delta_P, delta_y):
    M = delta_P.shape[1]
    def obj(theta):
        w = np.exp(theta - np.max(theta))
        w = w / np.sum(w)
        diff = delta_P @ w - delta_y
        return np.mean(diff ** 2)
    theta0 = np.zeros(M)
    res = optimize.minimize(obj, theta0, method="BFGS")
    w = np.exp(res.x - np.max(res.x))
    return w / np.sum(w)

def main():
    out_dir = Path("outputs/targeted_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    unc_df = load_uncertainty()
    curves = load_curves()
    truth_df = load_forecast_truth()
    train_cases_arr = np.array(TRAIN_CASES)
    val_cases = sorted(VAL_CASES)

    models_zoo = [
        ("GP_Matern52_3comp", "gp_matern52", 3),
        ("GP_Matern52_4comp", "gp_matern52", 4),
        ("GP_Matern32_3comp", "gp_matern32", 3),
        ("Linear_RidgeCV_3comp", "linear_ridge", 3),
        ("Tree_ExtraTrees_3comp", "tree_extratrees", 3),
    ]

    seeds = [42, 123]
    oof_preds = {s: {m[0]: [] for m in models_zoo} for s in seeds}

    for seed in seeds:
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        folds = list(kf.split(train_cases_arr))

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

                    for name, m_type, K in models_zoo:
                        basis_K = Vt[:K, :]
                        coeffs_tr_K = centered_tr @ basis_K.T
                        p_c = fit_predict_coefficients(m_type, X_tr_s, coeffs_tr_K, X_val_s, K, random_state=seed + fold_idx * 13)
                        pred_inc = mean_inc + p_c @ basis_K

                        for i, c in enumerate(val_c):
                            cum_pred = anchors_val[i, 0] + pred_inc[i, :]
                            for t_idx, d in enumerate(forecast_dates):
                                oof_preds[seed][name].append({
                                    "model_id": name,
                                    "case_num": c,
                                    "fold": fold_idx,
                                    "cutoff": origin,
                                    "horizon_years": horizon,
                                    "date": d,
                                    "phase": phase,
                                    "prediction": float(cum_pred[t_idx]),
                                    "split": "train",
                                })

    # Also compute on holdout validation cases 71-85
    val_preds = {m[0]: [] for m in models_zoo}
    for origin_str, horizon in EXPERIMENTS:
        origin = pd.Timestamp(origin_str)
        forecast_dates = pd.date_range(origin + pd.DateOffset(months=3), origin + pd.DateOffset(years=horizon), freq="QS")
        for phase in PHASES:
            X_tr, inc_tr, anchors_tr, anchor_mean = build_slice_features(sorted(TRAIN_CASES), origin, forecast_dates, phase, curves, unc_df)
            X_val, inc_val, anchors_val, _ = build_slice_features(val_cases, origin, forecast_dates, phase, curves, unc_df, train_anchor_mean=anchor_mean)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            mean_inc = inc_tr.mean(axis=0, keepdims=True)
            centered_tr = inc_tr - mean_inc
            U, S, Vt = svd(centered_tr, full_matrices=False)

            for name, m_type, K in models_zoo:
                basis_K = Vt[:K, :]
                coeffs_tr_K = centered_tr @ basis_K.T
                p_c = fit_predict_coefficients(m_type, X_tr_s, coeffs_tr_K, X_val_s, K, random_state=42)
                pred_inc = mean_inc + p_c @ basis_K
                for i, c in enumerate(val_cases):
                    cum_pred = anchors_val[i, 0] + pred_inc[i, :]
                    for t_idx, d in enumerate(forecast_dates):
                        val_preds[name].append({
                            "model_id": name,
                            "case_num": c,
                            "cutoff": origin,
                            "horizon_years": horizon,
                            "date": d,
                            "phase": phase,
                            "prediction": float(cum_pred[t_idx]),
                            "split": "validation",
                        })

    # Build wide prediction tables
    merge_keys = ["case_num", "fold", "cutoff", "horizon_years", "date", "phase", "split"]
    val_merge_keys = ["case_num", "cutoff", "horizon_years", "date", "phase", "split"]

    all_seed_results = {}
    tri_models = ["GP_Matern52_3comp", "Linear_RidgeCV_3comp", "Tree_ExtraTrees_3comp"]

    for seed in seeds:
        wide_df = None
        for name, _, _ in models_zoo:
            sub = pd.DataFrame(oof_preds[seed][name])[merge_keys + ["prediction"]].rename(columns={"prediction": name})
            sub["date"] = pd.to_datetime(sub["date"])
            sub["cutoff"] = pd.to_datetime(sub["cutoff"])
            wide_df = sub if wide_df is None else pd.merge(wide_df, sub, on=merge_keys)

        t_sub = truth_df[truth_df["case_num"].isin(TRAIN_CASES)].copy()
        t_sub["date"] = pd.to_datetime(t_sub["date"])
        t_sub["cutoff"] = pd.to_datetime(t_sub["cutoff"])
        wide_df = pd.merge(wide_df, t_sub[["case_num", "cutoff", "horizon_years", "date", "phase", "truth"]], on=["case_num", "cutoff", "horizon_years", "date", "phase"])

        # Extract anchor to compute increments
        anchors_dict = {}
        for c in TRAIN_CASES:
            for p in PHASES:
                for origin_str, _ in EXPERIMENTS:
                    origin = pd.Timestamp(origin_str)
                    s = curves[(c, p)].loc[:origin]
                    anchors_dict[(c, p, origin)] = float(s.iloc[-1])

        wide_df["anchor"] = wide_df.apply(lambda r: anchors_dict[(r["case_num"], r["phase"], r["cutoff"])], axis=1)
        wide_df["inc_truth"] = wide_df["truth"] - wide_df["anchor"]
        for m, _, _ in models_zoo:
            wide_df[f"{m}_inc"] = wide_df[m] - wide_df["anchor"]

        # Phase std of truth increments
        phase_std_map = wide_df.groupby("phase")["inc_truth"].std().to_dict()
        wide_df["phase_std"] = wide_df["phase"].map(phase_std_map)

        # Baseline Simple Blends
        wide_df["Blend_Tri_Equal"] = (wide_df["GP_Matern52_3comp"] + wide_df["Linear_RidgeCV_3comp"] + wide_df["Tree_ExtraTrees_3comp"]) / 3.0
        wide_df["Blend_GP_Matern32_5050"] = 0.5 * wide_df["GP_Matern52_3comp"] + 0.5 * wide_df["GP_Matern32_3comp"]
        wide_df["Blend_GP_K3_K4_5050"] = 0.5 * wide_df["GP_Matern52_3comp"] + 0.5 * wide_df["GP_Matern52_4comp"]

        # 1. Corrected Simplex Stack (Normalized Increments)
        wide_df["Stack_Simplex_Corrected"] = 0.0
        wide_df["Stack_Phase_Specific"] = 0.0
        wide_df["Stack_RidgeMeta_Corrected"] = 0.0

        outer_weights_norm = []
        outer_weights_phase = {p: [] for p in PHASES}

        for fold_idx in range(5):
            tr_mask = (wide_df["fold"] != fold_idx)
            val_mask = (wide_df["fold"] == fold_idx)

            # Global normalized increment simplex
            delta_P_tr = (wide_df.loc[tr_mask, [f"{m}_inc" for m in tri_models]].to_numpy(float) / 
                          wide_df.loc[tr_mask, "phase_std"].to_numpy(float)[:, None])
            delta_y_tr = (wide_df.loc[tr_mask, "inc_truth"].to_numpy(float) / 
                          wide_df.loc[tr_mask, "phase_std"].to_numpy(float))
            
            P_val = wide_df.loc[val_mask, tri_models].to_numpy(float)
            w_norm = fit_simplex_softmax(delta_P_tr, delta_y_tr)
            outer_weights_norm.append(w_norm)
            wide_df.loc[val_mask, "Stack_Simplex_Corrected"] = P_val @ w_norm

            # Phase-specific simplex
            for phase in PHASES:
                p_tr = tr_mask & (wide_df["phase"] == phase)
                p_val = val_mask & (wide_df["phase"] == phase)
                dP_p = wide_df.loc[p_tr, [f"{m}_inc" for m in tri_models]].to_numpy(float)
                dy_p = wide_df.loc[p_tr, "inc_truth"].to_numpy(float)
                w_p = fit_simplex_softmax(dP_p, dy_p)
                outer_weights_phase[phase].append(w_p)
                P_val_p = wide_df.loc[p_val, tri_models].to_numpy(float)
                wide_df.loc[p_val, "Stack_Phase_Specific"] = P_val_p @ w_p

            # Ridge Meta on increments
            ridge_meta = RidgeCV(alphas=np.logspace(-2, 4, 15), cv=5)
            ridge_meta.fit(delta_P_tr, delta_y_tr)
            delta_P_val = (wide_df.loc[val_mask, [f"{m}_inc" for m in tri_models]].to_numpy(float) / 
                           wide_df.loc[val_mask, "phase_std"].to_numpy(float)[:, None])
            p_inc_norm = ridge_meta.predict(delta_P_val)
            pred_inc_rescaled = p_inc_norm * wide_df.loc[val_mask, "phase_std"].to_numpy(float)
            wide_df.loc[val_mask, "Stack_RidgeMeta_Corrected"] = wide_df.loc[val_mask, "anchor"] + pred_inc_rescaled

        all_seed_results[seed] = {
            "wide_df": wide_df,
            "weights_norm": outer_weights_norm,
            "weights_phase": outer_weights_phase,
        }

    # Evaluate Validation Set for holdout cases 71-85
    val_wide_df = None
    for name, _, _ in models_zoo:
        sub = pd.DataFrame(val_preds[name])[val_merge_keys + ["prediction"]].rename(columns={"prediction": name})
        sub["date"] = pd.to_datetime(sub["date"])
        sub["cutoff"] = pd.to_datetime(sub["cutoff"])
        val_wide_df = sub if val_wide_df is None else pd.merge(val_wide_df, sub, on=val_merge_keys)

    val_wide_df["Blend_Tri_Equal"] = (val_wide_df["GP_Matern52_3comp"] + val_wide_df["Linear_RidgeCV_3comp"] + val_wide_df["Tree_ExtraTrees_3comp"]) / 3.0
    val_wide_df["Blend_GP_Matern32_5050"] = 0.5 * val_wide_df["GP_Matern52_3comp"] + 0.5 * val_wide_df["GP_Matern32_3comp"]
    val_wide_df["Blend_GP_K3_K4_5050"] = 0.5 * val_wide_df["GP_Matern52_3comp"] + 0.5 * val_wide_df["GP_Matern52_4comp"]

    # Fit meta-weights on full 70 training cases (from seed 42)
    w_df_42 = all_seed_results[42]["wide_df"]
    delta_P_full = (w_df_42[[f"{m}_inc" for m in tri_models]].to_numpy(float) / w_df_42["phase_std"].to_numpy(float)[:, None])
    delta_y_full = (w_df_42["inc_truth"].to_numpy(float) / w_df_42["phase_std"].to_numpy(float))
    w_full_norm = fit_simplex_softmax(delta_P_full, delta_y_full)

    val_wide_df["Stack_Simplex_Corrected"] = val_wide_df[tri_models].to_numpy(float) @ w_full_norm

    # Phase-specific weights on full 70 cases
    w_full_phase = {}
    val_wide_df["Stack_Phase_Specific"] = 0.0
    for phase in PHASES:
        p_mask = (w_df_42["phase"] == phase)
        dP_p = w_df_42.loc[p_mask, [f"{m}_inc" for m in tri_models]].to_numpy(float)
        dy_p = w_df_42.loc[p_mask, "inc_truth"].to_numpy(float)
        w_p = fit_simplex_softmax(dP_p, dy_p)
        w_full_phase[phase] = w_p
        v_mask = (val_wide_df["phase"] == phase)
        val_wide_df.loc[v_mask, "Stack_Phase_Specific"] = val_wide_df.loc[v_mask, tri_models].to_numpy(float) @ w_p

    # Ridge meta on full
    ridge_meta_full = RidgeCV(alphas=np.logspace(-2, 4, 15), cv=5)
    ridge_meta_full.fit(delta_P_full, delta_y_full)
    # Get anchor for val
    val_anchors_dict = {}
    for c in val_cases:
        for p in PHASES:
            for origin_str, _ in EXPERIMENTS:
                origin = pd.Timestamp(origin_str)
                s = curves[(c, p)].loc[:origin]
                val_anchors_dict[(c, p, origin)] = float(s.iloc[-1])

    val_wide_df["anchor"] = val_wide_df.apply(lambda r: val_anchors_dict[(r["case_num"], r["phase"], r["cutoff"])], axis=1)
    for m in tri_models:
        val_wide_df[f"{m}_inc"] = val_wide_df[m] - val_wide_df["anchor"]
    val_wide_df["phase_std"] = val_wide_df["phase"].map(phase_std_map)
    delta_P_val = (val_wide_df[[f"{m}_inc" for m in tri_models]].to_numpy(float) / val_wide_df["phase_std"].to_numpy(float)[:, None])
    p_inc_val = ridge_meta_full.predict(delta_P_val)
    val_wide_df["Stack_RidgeMeta_Corrected"] = val_wide_df["anchor"] + p_inc_val * val_wide_df["phase_std"].to_numpy(float)

    # -------------------------------------------------------------
    # COMPILE COMPREHENSIVE BENCHMARK TABLE
    # -------------------------------------------------------------
    eval_models = [
        "GP_Matern52_3comp",
        "GP_Matern52_4comp",
        "GP_Matern32_3comp",
        "Linear_RidgeCV_3comp",
        "Tree_ExtraTrees_3comp",
        "Blend_Tri_Equal",
        "Blend_GP_Matern32_5050",
        "Blend_GP_K3_K4_5050",
        "Stack_Simplex_Corrected",
        "Stack_Phase_Specific",
        "Stack_RidgeMeta_Corrected",
    ]

    bench_rows = []
    for m in eval_models:
        # Seed 42 OOF
        s42_df = all_seed_results[42]["wide_df"][merge_keys + [m]].copy().rename(columns={m: "prediction"})
        s42_df["model_id"] = m
        p_42, m_42, c_42 = score_predictions_df(s42_df, truth_df)
        oof_nrmse_42 = float(m_42[m_42["split"] == "train"]["mean_dev_NRMSE"].iloc[0])
        oil_42 = float(p_42[p_42["split"] == "train"].groupby("phase")["increment_NRMSE"].mean().get("oil_cum", np.nan))
        gas_42 = float(p_42[p_42["split"] == "train"].groupby("phase")["increment_NRMSE"].mean().get("gas_cum", np.nan))
        water_42 = float(p_42[p_42["split"] == "train"].groupby("phase")["increment_NRMSE"].mean().get("water_cum", np.nan))
        c_vals_42 = c_42.groupby("case_num")["case_NRMSE"].mean()
        med_42 = float(c_vals_42.median())

        # Seed 123 OOF
        s123_df = all_seed_results[123]["wide_df"][merge_keys + [m]].copy().rename(columns={m: "prediction"})
        s123_df["model_id"] = m
        _, m_123, _ = score_predictions_df(s123_df, truth_df)
        oof_nrmse_123 = float(m_123[m_123["split"] == "train"]["mean_dev_NRMSE"].iloc[0])

        # Validation
        v_df = val_wide_df[val_merge_keys + [m]].copy().rename(columns={m: "prediction"})
        v_df["model_id"] = m
        p_val, m_val, c_val = score_predictions_df(v_df, truth_df)
        val_nrmse = float(m_val[m_val["split"] == "validation"]["mean_dev_NRMSE"].iloc[0])
        oil_val = float(p_val[p_val["split"] == "validation"].groupby("phase")["increment_NRMSE"].mean().get("oil_cum", np.nan))
        gas_val = float(p_val[p_val["split"] == "validation"].groupby("phase")["increment_NRMSE"].mean().get("gas_cum", np.nan))
        water_val = float(p_val[p_val["split"] == "validation"].groupby("phase")["increment_NRMSE"].mean().get("water_cum", np.nan))
        c_vals_v = c_val.groupby("case_num")["case_NRMSE"].mean()
        med_val = float(c_vals_v.median())

        bench_rows.append({
            "model_name": m,
            "oof_macro_seed42": oof_nrmse_42,
            "oof_macro_seed123": oof_nrmse_123,
            "oof_oil": oil_42,
            "oof_gas": gas_42,
            "oof_water": water_42,
            "oof_case_median": med_42,
            "val_macro_NRMSE": val_nrmse,
            "val_oil": oil_val,
            "val_gas": gas_val,
            "val_water": water_val,
            "val_case_median": med_val,
        })

    bench_df = pd.DataFrame(bench_rows)
    bench_df.to_csv(out_dir / "corrected_stacking_benchmark.csv", index=False)
    print("\nSaved: outputs/targeted_audit/corrected_stacking_benchmark.csv")
    print(bench_df[["model_name", "oof_macro_seed42", "oof_macro_seed123", "val_macro_NRMSE", "val_case_median"]])

    # Fold weights analysis
    mean_w_norm = np.mean(all_seed_results[42]["weights_norm"], axis=0)
    std_w_norm = np.std(all_seed_results[42]["weights_norm"], axis=0)
    weights_summary = {
        "global_normalized_weights": {
            "GP": float(w_full_norm[0]),
            "Ridge": float(w_full_norm[1]),
            "ExtraTrees": float(w_full_norm[2]),
        },
        "fold_normalized_weights_mean_s42": {
            "GP": float(mean_w_norm[0]),
            "Ridge": float(mean_w_norm[1]),
            "ExtraTrees": float(mean_w_norm[2]),
        },
        "fold_normalized_weights_std_s42": {
            "GP": float(std_w_norm[0]),
            "Ridge": float(std_w_norm[1]),
            "ExtraTrees": float(std_w_norm[2]),
        },
        "phase_specific_weights": {
            p: {
                "GP": float(w_full_phase[p][0]),
                "Ridge": float(w_full_phase[p][1]),
                "ExtraTrees": float(w_full_phase[p][2]),
            } for p in PHASES
        }
    }
    with open(out_dir / "corrected_weights_summary.json", "w") as f:
        json.dump(weights_summary, f, indent=2)
    print("\nSaved: outputs/targeted_audit/corrected_weights_summary.json")
    print(json.dumps(weights_summary, indent=2))

if __name__ == "__main__":
    main()
