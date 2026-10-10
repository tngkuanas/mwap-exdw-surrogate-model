"""
Metrics module for the ensemble evaluation framework.
Guarantees bitwise consistency with the referee backtest_engine.py.
"""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

def compute_increment_nrmse(errors: np.ndarray, true_increments: np.ndarray) -> float:
    """
    Computes referee-standard increment NRMSE:
    NRMSE = RMS(prediction_error) / RMS(true_forward_increment)
    """
    if len(errors) == 0:
        return np.nan
    norm = np.sqrt(np.mean(true_increments ** 2))
    if norm <= 1e-12:
        return np.nan
    rmse = np.sqrt(np.mean(errors ** 2))
    return float(rmse / norm)

def compute_bias(errors: np.ndarray, true_increments: np.ndarray) -> float:
    """Normalized bias relative to RMS of true increment."""
    if len(errors) == 0:
        return np.nan
    norm = np.sqrt(np.mean(true_increments ** 2))
    if norm <= 1e-12:
        return np.nan
    return float(np.mean(errors) / norm)

def audit_physical_violations(df: pd.DataFrame, phase_col: str = "phase", pred_col: str = "prediction") -> Dict[str, float]:
    """
    Audits physical plausibility across predictions:
    1. Negative cumulative volumes
    2. Decreasing cumulative production steps
    3. Non-finite values (NaN / Inf)
    """
    total = len(df)
    if total == 0:
        return {"total": 0, "nan_count": 0, "negative_count": 0, "decreasing_count": 0, "violation_rate": 0.0}
    
    preds = df[pred_col].to_numpy(dtype=float)
    nan_count = int(np.isnan(preds).sum() + np.isinf(preds).sum())
    neg_count = int((preds < -1e-6).sum())
    
    # Audit monotonic steps per case, experiment, and phase
    decreasing_count = 0
    group_cols = [c for c in ["model_id", "case_num", "origin", "cutoff", "phase", "metric"] if c in df.columns]
    if "date" in df.columns and len(group_cols) > 0:
        for _, group in df.groupby(group_cols):
            sorted_group = group.sort_values("date")
            diffs = sorted_group[pred_col].diff().dropna()
            decreasing_count += int((diffs < -1e-6).sum())
            
    total_violations = nan_count + neg_count + decreasing_count
    return {
        "total": total,
        "nan_count": nan_count,
        "negative_count": neg_count,
        "decreasing_count": decreasing_count,
        "total_violations": total_violations,
        "violation_rate": float(total_violations / total),
    }

def score_predictions_df(preds: pd.DataFrame, truth_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Scores predictions against truth following the BacktestEngine contract.
    Returns:
        (phase_scores, macro_scores, case_scores)
    """
    # Merge prediction with truth
    join_cols = ["case_num", "cutoff", "horizon_years", "date", "phase"]
    # Harmonize column names
    p_copy = preds.copy()
    if "metric" in p_copy.columns and "phase" not in p_copy.columns:
        p_copy["phase"] = p_copy["metric"]
    if "origin" in p_copy.columns and "cutoff" not in p_copy.columns:
        p_copy["cutoff"] = pd.to_datetime(p_copy["origin"])
    else:
        p_copy["cutoff"] = pd.to_datetime(p_copy["cutoff"])
        
    t_copy = truth_df.copy()
    if "metric" in t_copy.columns and "phase" not in t_copy.columns:
        t_copy["phase"] = t_copy["metric"]
    t_copy["cutoff"] = pd.to_datetime(t_copy["cutoff"])
    
    p_copy["date"] = pd.to_datetime(p_copy["date"])
    t_copy["date"] = pd.to_datetime(t_copy["date"])
    
    merged = pd.merge(p_copy, t_copy, on=join_cols, suffixes=("", "_truth"))
    merged["error"] = merged["prediction"] - merged["truth"]
    
    # 1. Phase Scores
    phase_rows = []
    group_keys = ["model_id", "split", "cutoff", "horizon_years", "phase"]
    for keys, grp in merged.groupby(group_keys):
        model, split, cutoff, horizon, phase = keys
        errors = grp["error"].to_numpy(dtype=float)
        true_inc = grp["truth_increment"].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))
        nrmse = compute_increment_nrmse(errors, true_inc)
        bias = compute_bias(errors, true_inc)
        
        phase_rows.append({
            "model_id": model,
            "split": split,
            "cutoff": str(cutoff)[:10],
            "horizon_years": int(horizon),
            "phase": phase,
            "cases": grp["case_num"].nunique(),
            "RMSE": rmse,
            "MAE": mae,
            "increment_NRMSE": nrmse,
            "bias": bias,
        })
    phase_scores = pd.DataFrame(phase_rows)
    
    # 2. Macro Scores
    macro_rows = []
    for model, grp in phase_scores.groupby("model_id"):
        for split in ["train", "validation"]:
            split_grp = grp[grp["split"] == split]
            if len(split_grp) == 0:
                continue
            by_exp = split_grp.groupby(["cutoff", "horizon_years"])["increment_NRMSE"].mean()
            macro_rows.append({
                "model_id": model,
                "split": split,
                "mean_dev_NRMSE": float(by_exp.mean()),
                "worst_exp_NRMSE": float(by_exp.max()),
                "median_dev_NRMSE": float(by_exp.median()),
                "experiments_scored": len(by_exp),
            })
    macro_scores = pd.DataFrame(macro_rows)
    
    # 3. Case Scores (endpoint error and case NRMSE)
    case_rows = []
    for (model, case, cutoff, horizon, phase), grp in merged.groupby(["model_id", "case_num", "cutoff", "horizon_years", "phase"]):
        sorted_g = grp.sort_values("date")
        errs = sorted_g["error"].to_numpy(dtype=float)
        incs = sorted_g["truth_increment"].to_numpy(dtype=float)
        endpoint_err = errs[-1]
        endpoint_true = sorted_g["truth"].iloc[-1]
        case_rows.append({
            "model_id": model,
            "case_num": int(case),
            "split": grp["split"].iloc[0],
            "cutoff": str(cutoff)[:10],
            "horizon_years": int(horizon),
            "phase": phase,
            "case_NRMSE": compute_increment_nrmse(errs, incs),
            "endpoint_error": float(endpoint_err),
            "endpoint_rel_error": float(endpoint_err / endpoint_true) if endpoint_true != 0 else np.nan,
        })
    case_scores = pd.DataFrame(case_rows)
    
    return phase_scores, macro_scores, case_scores
