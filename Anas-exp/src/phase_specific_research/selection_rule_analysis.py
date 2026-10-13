"""
Module: selection_rule_analysis.py
Package: phase_specific_research
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Section 3: Scientifically Rigorous Reassessment of Water-Cut Classification.
Corrects:
1. Replaces single-origin evaluation with multi-origin discriminative evaluation (255 instances).
2. Reports class balance and majority-class baseline accuracy.
3. Implements strict reservoir-grouped leave-one-out cross-validation across 15 pseudo-fields.
4. Tunes gate thresholds strictly inside training folds.
5. Computes full confusion matrix, precision, recall, specificity, F1, and ROC-AUC.
6. Tests whether water cut provides statistically significant predictive value beyond calendar age and horizon.
7. Evaluates end-to-end forecast performance improvement (Macro NRMSE).
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)

DT_DAYS = 91.3125


def extract_observable_state_features(
    case_df: pd.DataFrame,
    origin: pd.Timestamp,
) -> Dict[str, float]:
    """
    Extracts strictly pre-origin observable physical reservoir state metrics.
    No future information is accessed.
    """
    hist = case_df[case_df["date"] <= origin].sort_values("date")
    if len(hist) == 0:
        raise ValueError(f"No historical records found on or before {origin}")
        
    last_row = hist.iloc[-1]
    qo_0 = float(last_row["oil_rate_stbd"])
    qw_0 = float(last_row["water_rate_stbd"])
    q_liq_0 = qo_0 + qw_0
    
    fw_0 = float(qw_0 / max(q_liq_0, 1e-6))
    wor_0 = float(qw_0 / max(qo_0, 1.0))
    
    t_days = (origin - hist["date"].min()).days
    res_age = float(t_days / 365.25)
    
    # 4-quarter trends if available
    if len(hist) >= 5:
        qo_4q_ago = float(hist.iloc[-5]["oil_rate_stbd"])
        qw_4q_ago = float(hist.iloc[-5]["water_rate_stbd"])
        fw_4q_ago = float(qw_4q_ago / max(qo_4q_ago + qw_4q_ago, 1e-6))
        
        d_recent = float(-np.log(max(qo_0, 1.0) / max(qo_4q_ago, 1.0)))
        fw_trend = float(fw_0 - fw_4q_ago)
        
        recent_qo = hist.iloc[-min(len(hist), 8):]["oil_rate_stbd"].values
        recent_qw = hist.iloc[-min(len(hist), 8):]["water_rate_stbd"].values
        cv_qo = float(np.std(recent_qo) / max(np.mean(recent_qo), 1e-6))
        cv_qw = float(np.std(recent_qw) / max(np.mean(recent_qw), 1e-6))
    else:
        d_recent = 0.0
        fw_trend = 0.0
        cv_qo = 0.0
        cv_qw = 0.0
        
    peak_idx = hist["oil_rate_stbd"].idxmax()
    peak_date = hist.loc[peak_idx, "date"]
    quarters_since_peak = float((origin - peak_date).days / 91.3125)
    
    return {
        "res_age": res_age,
        "qo_0": qo_0,
        "qw_0": qw_0,
        "q_liq_0": q_liq_0,
        "fw_0": fw_0,
        "wor_0": wor_0,
        "d_recent": d_recent,
        "fw_trend": fw_trend,
        "cv_qo": cv_qo,
        "cv_qw": cv_qw,
        "quarters_since_peak": max(0.0, quarters_since_peak),
    }


def build_discriminative_dataset(eval_records_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs the paired discriminative dataset comparing Candidate A (Baseline)
    and Candidate C (ExtraTrees Water) across all evaluated instances.
    """
    keys = ["origin", "horizon_quarters", "case_id"]
    base_sub = eval_records_df[eval_records_df["candidate_id"] == "Candidate_A_Frozen_Baseline"].copy()
    candC_sub = eval_records_df[eval_records_df["candidate_id"] == "Candidate_C_FrozenOil_ExtraTreesWater"].copy()
    
    df = base_sub.rename(columns={
        "macro_nrmse": "base_macro",
        "oil_nrmse": "base_oil",
        "water_nrmse": "base_water",
    })
    c_cols = candC_sub[keys + ["macro_nrmse", "oil_nrmse", "water_nrmse"]].rename(columns={
        "macro_nrmse": "candC_macro",
        "oil_nrmse": "candC_oil",
        "water_nrmse": "candC_water",
    })
    df = df.merge(c_cols, on=keys)
    
    # Binary win targets
    df["et_water_wins"] = (df["candC_water"] < df["base_water"]).astype(int)
    df["candC_macro_wins"] = (df["candC_macro"] < df["base_macro"]).astype(int)
    df["macro_nrmse_diff"] = df["candC_macro"] - df["base_macro"]  # negative = Cand C wins
    return df


def evaluate_discriminative_selection_rules(
    eval_records_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Conducts comprehensive, reservoir-grouped leave-one-out cross validation
    for regime classification and selection rule evaluation.
    """
    df = build_discriminative_dataset(eval_records_df)
    
    # Target variable: does ExtraTrees water beat Baseline water?
    y = df["et_water_wins"].values
    n_total = len(y)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    p_pos = n_pos / n_total
    majority_acc = max(p_pos, 1.0 - p_pos)
    
    # Feature subsets
    feat_age_h = ["res_age", "horizon_quarters"]
    feat_fw = ["fw_0"]
    feat_combined = ["res_age", "horizon_quarters", "fw_0"]
    all_feats = ["res_age", "horizon_quarters", "fw_0", "wor_0", "d_recent", "fw_trend", "cv_qo", "cv_qw"]
    
    # Correlations
    corrs = []
    for col in all_feats:
        if col in df.columns:
            corrs.append({
                "feature": col,
                "corr_with_et_water_win": float(df[col].corr(df["et_water_wins"])),
                "corr_with_macro_win": float(df[col].corr(df["candC_macro_wins"])),
                "corr_with_macro_diff": float(df[col].corr(df["macro_nrmse_diff"])),
            })
    corr_df = pd.DataFrame(corrs)
    
    # Reservoir-grouped Leave-One-Out Cross-Validation
    unique_cases = df["case_id"].unique()
    oof_preds = {
        "Majority_Baseline": np.zeros(n_total, dtype=int),
        "Learned_Threshold_fw": np.zeros(n_total, dtype=int),
        "Logistic_Age_Horizon": np.zeros(n_total, dtype=int),
        "Logistic_WaterCut_Only": np.zeros(n_total, dtype=int),
        "Logistic_Age_Horizon_WaterCut": np.zeros(n_total, dtype=int),
        "Logistic_All_Features": np.zeros(n_total, dtype=int),
    }
    oof_probs = {
        "Logistic_Age_Horizon": np.zeros(n_total, dtype=float),
        "Logistic_WaterCut_Only": np.zeros(n_total, dtype=float),
        "Logistic_Age_Horizon_WaterCut": np.zeros(n_total, dtype=float),
        "Logistic_All_Features": np.zeros(n_total, dtype=float),
    }
    learned_thresholds = []

    for c in unique_cases:
        test_mask = (df["case_id"] == c).values
        train_mask = ~test_mask
        
        y_train = y[train_mask]
        
        # 1. Majority baseline
        maj_class = int(np.round(np.mean(y_train)))
        oof_preds["Majority_Baseline"][test_mask] = maj_class
        
        # 2. Learned Physical Threshold on fw_0
        fw_train = df.loc[train_mask, "fw_0"].values
        best_th = 0.35
        best_th_acc = -1.0
        for th in np.linspace(0.20, 0.50, 31):
            pred_th = (fw_train > th).astype(int)
            acc = np.mean(pred_th == y_train)
            if acc > best_th_acc:
                best_th_acc = acc
                best_th = th
        learned_thresholds.append({"case_id": c, "optimal_threshold": float(best_th)})
        oof_preds["Learned_Threshold_fw"][test_mask] = (df.loc[test_mask, "fw_0"].values > best_th).astype(int)
        
        # 3. Logistic Regressions
        models_to_fit = [
            ("Logistic_Age_Horizon", feat_age_h),
            ("Logistic_WaterCut_Only", feat_fw),
            ("Logistic_Age_Horizon_WaterCut", feat_combined),
            ("Logistic_All_Features", all_feats),
        ]
        for m_name, f_list in models_to_fit:
            X_tr = df.loc[train_mask, f_list].values
            X_te = df.loc[test_mask, f_list].values
            
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_tr, y_train)
            probs = clf.predict_proba(X_te)[:, 1]
            oof_probs[m_name][test_mask] = probs
            oof_preds[m_name][test_mask] = (probs >= 0.5).astype(int)

    # Metrics computation
    metrics_records = []
    for model_name, preds in oof_preds.items():
        acc = float(accuracy_score(y, preds))
        cm = confusion_matrix(y, preds)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (cm[0, 0], 0, 0, 0)
        
        prec = float(precision_score(y, preds, zero_division=0))
        rec = float(recall_score(y, preds, zero_division=0))
        spec = float(tn / max(tn + fp, 1))
        f1 = float(f1_score(y, preds, zero_division=0))
        
        auc = float(roc_auc_score(y, oof_probs[model_name])) if model_name in oof_probs else float("nan")
        loss = float(log_loss(y, oof_probs[model_name])) if model_name in oof_probs else float("nan")
        
        # End-to-end downstream performance (Macro NRMSE achieved by following gate recommendation)
        # If gate predicts 1 (ET Water favorable) -> use candC_macro; if 0 -> use base_macro
        gated_macro = np.where(preds == 1, df["candC_macro"].values, df["base_macro"].values)
        mean_gated_macro = float(np.mean(gated_macro))
        macro_diff_vs_base = float(mean_gated_macro - np.mean(df["base_macro"]))
        
        metrics_records.append({
            "model": model_name,
            "oof_accuracy": acc,
            "accuracy_gain_vs_majority": acc - majority_acc,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "precision": prec,
            "recall": rec,
            "specificity": spec,
            "f1_score": f1,
            "roc_auc": auc,
            "log_loss": loss,
            "resulting_macro_nrmse": mean_gated_macro,
            "macro_diff_vs_baseline": macro_diff_vs_base,
        })
    metrics_df = pd.DataFrame(metrics_records)
    
    # Formal Likelihood Ratio Test: Model A (Age+Horizon) vs Model C (Age+Horizon+WaterCut)
    X_reduced = df[feat_age_h].values
    X_full = df[feat_combined].values
    clf_red = LogisticRegression(max_iter=1000).fit(X_reduced, y)
    clf_full = LogisticRegression(max_iter=1000).fit(X_full, y)
    
    prob_red = np.clip(clf_red.predict_proba(X_reduced)[:, 1], 1e-12, 1.0 - 1e-12)
    prob_full = np.clip(clf_full.predict_proba(X_full)[:, 1], 1e-12, 1.0 - 1e-12)
    
    loglik_red = float(np.sum(y * np.log(prob_red) + (1 - y) * np.log(1 - prob_red)))
    loglik_full = float(np.sum(y * np.log(prob_full) + (1 - y) * np.log(1 - prob_full)))
    
    lr_stat = float(2.0 * (loglik_full - loglik_red))
    lrt_p_value = float(1.0 - chi2.cdf(lr_stat, df=1))
    
    summary_dict = {
        "n_total_instances": n_total,
        "n_positive_instances": n_pos,
        "n_negative_instances": n_neg,
        "positive_class_prevalence": p_pos,
        "majority_class_baseline_accuracy": majority_acc,
        "baseline_macro_nrmse": float(np.mean(df["base_macro"])),
        "candidate_c_macro_nrmse": float(np.mean(df["candC_macro"])),
        "best_gated_macro_nrmse": float(metrics_df["resulting_macro_nrmse"].min()),
        "likelihood_ratio_statistic": lr_stat,
        "likelihood_ratio_p_value": lrt_p_value,
        "water_cut_independent_predictive_value": bool(lrt_p_value < 0.05),
        "mean_learned_threshold": float(np.mean([t["optimal_threshold"] for t in learned_thresholds])),
    }
    return corr_df, metrics_df, summary_dict
