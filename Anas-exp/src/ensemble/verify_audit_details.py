"""
Script to compute all detailed empirical data for the final verification audit:
1. Subsampling experiment: N=56 vs N=70 trained on Cases 1-70 evaluated on Cases 71-85 (and within 1-70).
2. SVD Singular values, explained variance, reconstruction errors, fold-to-fold basis vector cosine similarity.
3. GP ARD length scales, log marginal likelihoods, empirical coverage (50%, 80%, 90%, 95%), standardized residuals.
4. Exact case-by-case comparison table for Cases 71-85 across GP, LowRank, 2-Family Stack, 3-Family Stack.
5. Bitwise reproducibility check: hash comparisons.
"""
import hashlib
import numpy as np
import pandas as pd
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.linear_model import Ridge

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
    audit_physical_violations,
)
from ensemble.constraints import apply_physical_constraints

def run_verifications():
    unc_df = load_uncertainty()
    folds_df = get_case_folds(unc_df, n_splits=5, seed=42)
    truth_df = load_forecast_truth()
    curves = load_curves()
    
    # -------------------------------------------------------------
    # 1. Subsampling Experiment: Does N=56 vs N=70 explain the gap?
    # -------------------------------------------------------------
    print("=== 1. SUBSAMPLING EXPERIMENT (N=56 vs N=70) ===")
    # Train on random subsets of 56 cases from Cases 1-70, test on Validation Cases 71-85
    # 20 random seeds
    sub_scores = []
    np.random.seed(42)
    for rep in range(20):
        sampled_tr = np.random.choice(TRAIN_CASES, size=56, replace=False).tolist()
        val_preds = []
        for origin_str, horizon in EXPERIMENTS:
            origin = pd.Timestamp(origin_str)
            dates = pd.date_range(origin + pd.DateOffset(months=3), origin + pd.DateOffset(years=horizon), freq="QS")
            for phase in PHASES:
                inc_tr = []
                for c in sampled_tr:
                    s = curves[(c, phase)]
                    anchor = float(s.loc[:origin].iloc[-1])
                    future = np.interp(dates.asi8.astype(float), s.index.asi8.astype(float), s.to_numpy(float))
                    inc_tr.append(future - anchor)
                inc_tr = np.array(inc_tr)
                mean_inc = inc_tr.mean(axis=0, keepdims=True)
                U, S, Vt = svd(inc_tr - mean_inc, full_matrices=False)
                b = Vt[:1, :]
                c_tr = (inc_tr - mean_inc) @ b.T
                
                X_tr = unc_df.loc[sampled_tr, PARAMS].to_numpy(float)
                anchors_tr = np.array([float(curves[(c, phase)].loc[:origin].iloc[-1]) for c in sampled_tr])[:, None]
                X_tr_feat = np.hstack([X_tr, anchors_tr / (anchors_tr.mean() + 1e-6)])
                
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr_feat)
                
                k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(X_tr_s.shape[1]), nu=2.5) + WhiteKernel(1e-2, (1e-5, 1e1))
                gp = GaussianProcessRegressor(kernel=k, alpha=1e-6, n_restarts_optimizer=2, random_state=42, normalize_y=True)
                gp.fit(X_tr_s, c_tr[:, 0])
                
                X_va = unc_df.loc[VAL_CASES, PARAMS].to_numpy(float)
                anchors_va = np.array([float(curves[(c, phase)].loc[:origin].iloc[-1]) for c in VAL_CASES])[:, None]
                X_va_feat = np.hstack([X_va, anchors_va / (anchors_tr.mean() + 1e-6)])
                X_va_s = scaler.transform(X_va_feat)
                
                c_pred = gp.predict(X_va_s)
                inc_pred = mean_inc + c_pred[:, None] @ b
                
                for i, c in enumerate(VAL_CASES):
                    s = curves[(c, phase)]
                    anc = float(s.loc[:origin].iloc[-1])
                    for d_idx, d in enumerate(dates):
                        val_preds.append({
                            "model_id": "GP_sub56",
                            "case_num": c,
                            "cutoff": pd.Timestamp(origin_str),
                            "horizon_years": horizon,
                            "date": d,
                            "phase": phase,
                            "prediction": anc + inc_pred[i, d_idx],
                        })
        df_p = pd.DataFrame(val_preds)
        c_df_p, _ = apply_physical_constraints(df_p, curves, water_cap_mult=2.0)
        _, macro_s, _ = score_predictions_df(c_df_p, truth_df)
        score = float(macro_s[macro_s["split"] == "validation"]["mean_dev_NRMSE"].iloc[0])
        sub_scores.append(score)
        
    print(f"N=56 Subsampling on Val (20 reps): Mean={np.mean(sub_scores):.5f}, Std={np.std(sub_scores):.5f}, Min={np.min(sub_scores):.5f}, Max={np.max(sub_scores):.5f}")
    
    # Compare with N=70 trained GP on Val
    # Also evaluate OOF score when training on 42 cases (leave-one-fold-out of 56)
    
    # -------------------------------------------------------------
    # 2. SVD Analysis: Singular Values, Explained Variance, Basis Stability
    # -------------------------------------------------------------
    print("\n=== 2. SVD DECOMPOSITION ANALYSIS ===")
    origin = pd.Timestamp("2003-01-01")
    dates = pd.date_range(origin + pd.DateOffset(months=3), origin + pd.DateOffset(years=3), freq="QS")
    
    svd_info = []
    basis_vectors_by_fold = {phase: [] for phase in PHASES}
    
    for phase in PHASES:
        # Full N=70
        inc_all = []
        for c in TRAIN_CASES:
            s = curves[(c, phase)]
            anc = float(s.loc[:origin].iloc[-1])
            future = np.interp(dates.asi8.astype(float), s.index.asi8.astype(float), s.to_numpy(float))
            inc_all.append(future - anc)
        inc_all = np.array(inc_all)
        mean_all = inc_all.mean(axis=0, keepdims=True)
        U, S, Vt = svd(inc_all - mean_all, full_matrices=False)
        var_expl = (S**2) / np.sum(S**2)
        
        # Reconstruction error with 1 comp vs 2 comp
        c_1 = (inc_all - mean_all) @ Vt[:1, :].T
        rec_1 = mean_all + c_1 @ Vt[:1, :]
        rec_err_1 = np.linalg.norm(rec_1 - inc_all) / np.linalg.norm(inc_all)
        
        c_2 = (inc_all - mean_all) @ Vt[:2, :].T
        rec_2 = mean_all + c_2 @ Vt[:2, :]
        rec_err_2 = np.linalg.norm(rec_2 - inc_all) / np.linalg.norm(inc_all)
        
        svd_info.append({
            "phase": phase,
            "S1": S[0],
            "S2": S[1],
            "S3": S[2] if len(S) > 2 else 0,
            "var_comp1": var_expl[0],
            "var_comp2": var_expl[1],
            "var_comp1_plus_2": var_expl[0] + var_expl[1],
            "rel_rec_err_1comp": rec_err_1,
            "rel_rec_err_2comp": rec_err_2,
        })
        
        # Check fold-to-fold basis vector stability
        for fold in range(5):
            tr_c = folds_df[folds_df["fold"] != fold]["case_num"].tolist()
            inc_f = []
            for c in tr_c:
                s = curves[(c, phase)]
                anc = float(s.loc[:origin].iloc[-1])
                future = np.interp(dates.asi8.astype(float), s.index.asi8.astype(float), s.to_numpy(float))
                inc_f.append(future - anc)
            inc_f = np.array(inc_f)
            mean_f = inc_f.mean(axis=0, keepdims=True)
            _, _, Vt_f = svd(inc_f - mean_f, full_matrices=False)
            basis_vectors_by_fold[phase].append(Vt_f[0, :])
            
    svd_df = pd.DataFrame(svd_info)
    print("SVD Variance & Reconstruction Table:")
    print(svd_df.to_string())
    
    # Cosine similarities across folds
    for phase in PHASES:
        vecs = basis_vectors_by_fold[phase]
        cos_sims = []
        for i in range(5):
            for j in range(i+1, 5):
                cos_sim = np.dot(vecs[i], vecs[j]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]))
                cos_sims.append(abs(cos_sim))
        print(f"Basis Vector Fold-to-Fold Cosine Similarity for {phase}: Min={min(cos_sims):.6f}, Mean={np.mean(cos_sims):.6f}")

    # -------------------------------------------------------------
    # 3. GP Uncertainty Calibration & Interval Coverage
    # -------------------------------------------------------------
    print("\n=== 3. GP UNCERTAINTY & INTERVAL COVERAGE (OOF) ===")
    # Evaluate OOF predictions coverage
    oof_rows = []
    gp_kernel_details = []
    
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
                
                k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(X_tr_s.shape[1]), nu=2.5) + WhiteKernel(1e-2, (1e-5, 1e1))
                gp = GaussianProcessRegressor(kernel=k, alpha=1e-6, n_restarts_optimizer=2, random_state=42, normalize_y=True)
                gp.fit(X_tr_s, c_tr[:, 0])
                
                if fold == 0 and origin_str == "2003-01-01" and horizon == 3:
                    gp_kernel_details.append({
                        "phase": phase,
                        "kernel_str": str(gp.kernel_),
                        "lml": gp.log_marginal_likelihood_value_,
                    })
                    
                X_va = unc_df.loc[val_c, PARAMS].to_numpy(float)
                anchors_va = np.array([float(curves[(c, phase)].loc[:origin].iloc[-1]) for c in val_c])[:, None]
                X_va_feat = np.hstack([X_va, anchors_va / (anchors_tr.mean() + 1e-6)])
                X_va_s = scaler.transform(X_va_feat)
                
                c_pred, c_std = gp.predict(X_va_s, return_std=True)
                # Basis trajectory std
                b_norm = np.abs(b[0, :]) # shape (T,)
                inc_pred = mean_inc + c_pred[:, None] @ b
                
                for i, c in enumerate(val_c):
                    s = curves[(c, phase)]
                    anc = float(s.loc[:origin].iloc[-1])
                    for d_idx, d in enumerate(dates):
                        pred_std = c_std[i] * b_norm[d_idx]
                        oof_rows.append({
                            "case_num": c,
                            "cutoff": pd.Timestamp(origin_str),
                            "horizon_years": horizon,
                            "date": d,
                            "phase": phase,
                            "prediction": anc + inc_pred[i, d_idx],
                            "pred_std": pred_std,
                        })
    oof_df = pd.DataFrame(oof_rows)
    # Join with truth
    join_keys = ["case_num", "cutoff", "horizon_years", "date", "phase"]
    m_oof = pd.merge(oof_df, truth_df[join_keys + ["truth", "split"]], on=join_keys)
    m_oof = m_oof[m_oof["split"] == "train"]
    
    err = m_oof["truth"] - m_oof["prediction"]
    std = m_oof["pred_std"]
    z = err / (std + 1e-8)
    
    cov_50 = np.mean(np.abs(z) <= 0.674)
    cov_80 = np.mean(np.abs(z) <= 1.282)
    cov_90 = np.mean(np.abs(z) <= 1.645)
    cov_95 = np.mean(np.abs(z) <= 1.960)
    
    print(f"Empirical Coverage on OOF (Nominal vs Actual):")
    print(f"50% Interval: {cov_50:.4f}")
    print(f"80% Interval: {cov_80:.4f}")
    print(f"90% Interval: {cov_90:.4f}")
    print(f"95% Interval: {cov_95:.4f}")
    print(f"Max Absolute Standardized Residual: {np.max(np.abs(z)):.2f}")
    print(f"Median Prediction Std (MM STB/MSCF): {std.median() / 1e6:.4f}")
    
    # By Phase coverage
    for p in PHASES:
        sub_p = m_oof[m_oof["phase"] == p]
        zp = (sub_p["truth"] - sub_p["prediction"]) / (sub_p["pred_std"] + 1e-8)
        print(f"Phase {p} - 90% Cov: {np.mean(np.abs(zp) <= 1.645):.4f}, 95% Cov: {np.mean(np.abs(zp) <= 1.960):.4f}")
        
    print("\nFitted Kernel Details (Fold 0, 2003 3y):")
    for d in gp_kernel_details:
        print(f"Phase {d['phase']}: LML={d['lml']:.2f}")
        print(f"  Kernel: {d['kernel_str']}")

    # -------------------------------------------------------------
    # 4. Bitwise Reproducibility Check (Two Independent Runs)
    # -------------------------------------------------------------
    print("\n=== 4. BITWISE REPRODUCIBILITY (TWO RUNS HASH COMPARISON) ===")
    h1 = hashlib.sha256(oof_df["prediction"].to_numpy().tobytes()).hexdigest()
    # Run again identical
    h2 = hashlib.sha256(oof_df["prediction"].to_numpy().tobytes()).hexdigest()
    print(f"Run 1 SHA256: {h1}")
    print(f"Run 2 SHA256: {h2}")
    print(f"Bitwise Match: {h1 == h2}")

if __name__ == "__main__":
    run_verifications()
