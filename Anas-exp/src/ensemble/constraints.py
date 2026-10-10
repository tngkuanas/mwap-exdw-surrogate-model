"""
Shared Physical Constraint and Stabilization Layer.
Implements universal physical guardrails independently of predictive models:
- Monotonic cumulative production (non-negative rates)
- Exact historical anchor preservation
- Water rate capping and material balance ceiling
"""
from typing import Dict, Tuple
import numpy as np
import pandas as pd

YEAR_DAYS = 365.25

def apply_physical_constraints(
    preds_df: pd.DataFrame,
    curves: Dict[Tuple[int, str], pd.Series] = None,
    water_cap_mult: float = 2.0,
    enforce_monotonic: bool = True,
    model_suffix: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies shared physical constraints to predictions:
    1. Floor at historical cumulative anchor: Q(t) >= Q(t_0)
    2. Monotonic cumulative steps: Q(t_{k+1}) >= Q(t_k)
    3. Water rate capping if water_cap_mult > 0
    
    Returns:
        (constrained_preds_df, constraint_audit_df)
    """
    df = preds_df.copy()
    if model_suffix:
        df["model_id"] = df["model_id"] + model_suffix
        
    audit_rows = []
    
    # Process by model, case, experiment, phase
    group_cols = [c for c in ["model_id", "case_num", "cutoff", "origin", "horizon_years", "phase"] if c in df.columns]
    grouped = df.groupby(group_cols)
    
    processed_dfs = []
    
    for keys, grp in grouped:
        # Map keys dict
        key_dict = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))
        model = key_dict["model_id"]
        case = key_dict["case_num"]
        cutoff = pd.Timestamp(key_dict.get("cutoff", key_dict.get("origin", "2003-01-01")))
        horizon = key_dict.get("horizon_years", 3)
        phase = key_dict["phase"]
        sorted_grp = grp.sort_values("date").copy()
        
        orig_preds = sorted_grp["prediction"].to_numpy(float)
        adj_preds = orig_preds.copy()
        
        # 1. Anchor Floor
        if curves is not None and (int(case), phase) in curves:
            anchor_val = float(curves[(int(case), phase)].loc[:cutoff].iloc[-1])
        else:
            anchor_val = 0.0
            
        below_anchor_count = int((adj_preds < anchor_val - 1e-6).sum())
        adj_preds = np.maximum(adj_preds, anchor_val)
        
        # 2. Water Rate Capping
        water_cap_count = 0
        if phase == "water_cum" and water_cap_mult > 0 and curves is not None:
            water_rates = curves.get((int(case), "water_rate"))
            if water_rates is not None:
                obs_water_rate = float(water_rates.loc[:cutoff].iloc[-1]) if len(water_rates.loc[:cutoff]) > 0 else 0.0
                obs_water_rate = max(100.0, obs_water_rate)  # Minimum threshold
                max_allowable_rate = water_cap_mult * obs_water_rate
                
                # dt from anchor
                forecast_dates = pd.DatetimeIndex(sorted_grp["date"])
                dt_days = np.asarray((forecast_dates - cutoff).total_seconds() / 86400.0, dtype=float)
                max_cum_water = anchor_val + max_allowable_rate * dt_days
                
                over_cap = adj_preds > max_cum_water
                water_cap_count = int(np.sum(over_cap))
                adj_preds = np.minimum(adj_preds, max_cum_water)
                
        # 3. Monotonic cumulative accumulation
        monotonic_violations = 0
        if enforce_monotonic:
            # Enforce non-decreasing
            for t_idx in range(1, len(adj_preds)):
                if adj_preds[t_idx] < adj_preds[t_idx - 1]:
                    monotonic_violations += 1
                    adj_preds[t_idx] = adj_preds[t_idx - 1]
                    
        # Record changes
        total_pts = len(orig_preds)
        diff = np.asarray(np.abs(adj_preds - orig_preds), dtype=float)
        points_altered = int((diff > 1e-4).sum())
        max_correction = float(np.max(diff)) if len(diff) > 0 else 0.0
        mean_correction = float(np.mean(diff)) if len(diff) > 0 else 0.0
        
        sorted_grp["prediction"] = adj_preds
        processed_dfs.append(sorted_grp)
        
        audit_rows.append({
            "model_id": model,
            "case_num": int(case),
            "cutoff": str(cutoff)[:10],
            "horizon_years": int(horizon),
            "phase": phase,
            "total_points": total_pts,
            "points_altered": points_altered,
            "below_anchor_count": below_anchor_count,
            "water_cap_count": water_cap_count,
            "monotonic_violations": monotonic_violations,
            "mean_correction": mean_correction,
            "max_correction": max_correction,
        })
        
    final_preds = pd.concat(processed_dfs, ignore_index=True)
    audit_df = pd.DataFrame(audit_rows)
    return final_preds, audit_df
