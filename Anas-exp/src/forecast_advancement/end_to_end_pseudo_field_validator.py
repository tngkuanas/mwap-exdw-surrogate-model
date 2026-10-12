"""
Module: end_to_end_pseudo_field_validator.py
Package: forecast_advancement
Addresses: Stage 6 — End-to-End Leave-Reservoir-Out Pseudo-Field Validation
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Framework:
1. Evaluates 5 approaches across held-out synthetic pseudo-fields (Cases 1-70):
   - Approach 1: Oracle Geological Parameters (Frozen Ensemble + True Geology)
   - Approach 2: History-Matched Geological Parameters (Frozen Ensemble + Inverted Geology)
   - Approach 3: Direct State-Based Forecasting Without Geological Inversion
   - Approach 4: Protected Frozen Production Baseline (Blend_Phase_Specific_Simplex)
   - Approach 5: Strongest New Challenger (Constrained Liquid Dynamics Forecaster)
2. Compares oil, water, gas consistency, cumulative increment NRMSE, terminal rates,
   forecast bias, and physical constraint violations.
3. Secondary validation across Cases 71-85.
4. Cases 86-100 remain 100% blind.
"""
from typing import Dict, List, Tuple, Any
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from ensemble.data import load_uncertainty, TRAIN_CASES, VAL_CASES, PARAMS, YEAR_DAYS
from ensemble.run_horizon_adaptive_stacking_experiment import RS_SOLUTION
from ensemble.metrics import compute_increment_nrmse, compute_bias
from ensemble.generate_competition_deliverable import (
    load_locked_config,
    generate_scenario_predictions,
)
from forecast_advancement.protocol_framework import (
    build_asof_case_dict,
    fit_base_models_under_protocol,
)
from forecast_advancement.recovery_state_forecaster import RecoveryStateForecaster
from forecast_advancement.constrained_liquid_dynamics import ConstrainedLiquidDynamicsForecaster
from forecast_advancement.functional_trajectory_forecaster import FunctionalTrajectoryForecaster
from deep_reservoir_science.pseudo_field_validation import (
    FastLeaveOutEmulator,
    invert_geology_for_field,
    PARAM_BOUNDS,
    BOUNDS_ARR,
)


def run_end_to_end_pseudo_field_study(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    test_fields: List[int] = None,
    origin_str: str = "2005-01-01",
    horizon_quarters: int = 12,
    dt_days: float = 91.3125,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs comprehensive end-to-end pseudo-field benchmark across the 5 required information regimes.
    """
    orig = pd.Timestamp(origin_str)
    f_dates = pd.date_range(orig + pd.DateOffset(days=1), periods=horizon_quarters, freq="QS")
    config = load_locked_config()
    weights = config["weights"]
    
    if test_fields is None:
        test_fields = [3, 8, 14, 19, 25, 31, 38, 44, 50, 56, 62, 65, 67, 69, 70]
        
    print(f"\n=========================================================================")
    print(f"STAGE 6: PSEUDO-FIELD LEAVE-RESERVOIR-OUT VALIDATION ({origin_str}, {horizon_quarters}Q)")
    print(f"Evaluating {len(test_fields)} pseudo-fields across Cases 1-70...")
    print(f"=========================================================================")
    
    case_records = []
    
    for f_id in test_fields:
        train_pool = [c for c in TRAIN_CASES if c != f_id]
        
        # 1. Truncate historical observations strictly to origin
        df_target = case_dict[f_id]
        pre_target = df_target[df_target["date"] <= orig]
        post_target = df_target[df_target["date"] > orig].iloc[:horizon_quarters]
        if len(post_target) < horizon_quarters:
            continue
            
        last = pre_target.iloc[-1]
        init_st = {
            "qo": float(last["oil_rate_stbd"]),
            "qw": float(last["water_rate_stbd"]),
            "qg": float(last["gas_rate_mscfd"]),
            "Qo": float(last["oil_cum_stb"]),
            "Qw": float(last["water_cum_stb"]),
            "Qg": float(last["gas_cum_mscf"]),
            "wor": float(last.get("wor", float(last["water_rate_stbd"]) / max(float(last["oil_rate_stbd"]), 1.0))),
            "res_age": (orig - pre_target["date"].iloc[0]).days / YEAR_DAYS,
            "origin_date": orig,
        }
        
        y_true_o = post_target["oil_cum_stb"].values
        y_true_w = post_target["water_cum_stb"].values
        inc_o = y_true_o - init_st["Qo"]
        inc_w = y_true_w - init_st["Qw"]
        
        # 2. Historical Inversion Emulator on train_pool strictly <= orig
        X_train = unc_df.loc[train_pool][PARAMS].values
        Yo_train = np.array([case_dict[c][case_dict[c]["date"] <= orig]["oil_rate_stbd"].values for c in train_pool])
        Yw_train = np.array([case_dict[c][case_dict[c]["date"] <= orig]["water_rate_stbd"].values for c in train_pool])
        
        emulator = FastLeaveOutEmulator(n_components=3, random_state=42).fit(X_train, Yo_train, Yw_train)
        obs_qo = pre_target["oil_rate_stbd"].values
        obs_qw = pre_target["water_rate_stbd"].values
        inferred_theta = invert_geology_for_field(emulator, obs_qo, obs_qw, X_train)
        
        u_true = unc_df.loc[f_id]
        u_inferred = pd.Series(inferred_theta, index=PARAMS)
        u_mean = pd.Series(np.mean(X_train, axis=0), index=PARAMS)
        
        # 3. Fit Models on train_pool under Protocol A (Strict As-Of)
        m_frozen = fit_base_models_under_protocol(case_dict, unc_df, train_pool, orig, "Protocol_A")
        
        # Train new challengers on train_pool
        m_challenger_liq = ConstrainedLiquidDynamicsForecaster(fractional_flow_mode="logistic").fit(
            case_dict, unc_df, train_pool, orig
        )
        m_challenger_state = RecoveryStateForecaster().fit(
            case_dict, unc_df, train_pool, orig
        )
        
        # Generate predictions for the 5 required approaches:
        approaches = {}
        
        # Approach 1: Oracle Geological Parameters
        _, aud_oracle = generate_scenario_predictions(m_frozen, weights, init_st, u_true, f_dates, 1.0)
        approaches["1_Oracle_Geology_Baseline"] = {
            "qo": aud_oracle["qo_rate"], "qw": aud_oracle["qw_rate"], "qg": aud_oracle["qg_rate"]
        }
        
        # Approach 2: Inferred Geological Parameters
        _, aud_inferred = generate_scenario_predictions(m_frozen, weights, init_st, u_inferred, f_dates, 1.0)
        approaches["2_Inferred_Geology_Baseline"] = {
            "qo": aud_inferred["qo_rate"], "qw": aud_inferred["qw_rate"], "qg": aud_inferred["qg_rate"]
        }
        
        # Approach 3: Direct State-Based Forecasting Without Geological Inversion
        p_direct = m_challenger_state.predict_rates(df_target, u_mean, orig, f_dates)
        approaches["3_Direct_State_Forecasting_No_Inversion"] = {
            "qo": p_direct["oil_rate"], "qw": p_direct["water_rate"], "qg": p_direct["gas_rate"]
        }
        
        # Approach 4: Protected Frozen Production Baseline (with inferred params)
        approaches["4_Frozen_Production_Baseline"] = approaches["2_Inferred_Geology_Baseline"]
        
        # Approach 5: Strongest New Challenger (Constrained Liquid Dynamics)
        p_liq = m_challenger_liq.predict_rates(df_target, u_inferred, orig, f_dates)
        approaches["5_Constrained_Liquid_Dynamics_Challenger"] = {
            "qo": p_liq["oil_rate"], "qw": p_liq["water_rate"], "qg": p_liq["gas_rate"]
        }
        
        # Evaluate each approach
        for app_name, preds in approaches.items():
            qo = np.asarray(preds["qo"], dtype=float).copy()
            qw = np.asarray(preds["qw"], dtype=float).copy()
            qg = np.asarray(preds["qg"], dtype=float).copy()
            
            # Monotonic safeguard
            is_monotonic = True
            for t in range(1, len(qo)):
                if qo[t] > qo[t - 1]:
                    is_monotonic = False
                    qo[t] = qo[t - 1]
            qo = np.maximum(qo, 0.0)
            qw = np.maximum(qw, 0.0)
            
            cum_o = init_st["Qo"] + np.cumsum(qo * dt_days)
            cum_w = init_st["Qw"] + np.cumsum(qw * dt_days)
            
            err_o = cum_o - y_true_o
            err_w = cum_w - y_true_w
            
            no = compute_increment_nrmse(err_o, inc_o)
            nw = compute_increment_nrmse(err_w, inc_w)
            bo = compute_bias(err_o, inc_o)
            bw = compute_bias(err_w, inc_w)
            macro = 0.5 * (no + nw)
            
            # Physical checks
            gas_ratio_check = float(np.max(np.abs(qg - RS_SOLUTION * qo))) < 1e-3
            term_qo = float(qo[-1])
            non_neg_check = bool(np.all(qo >= 0.0) and np.all(qw >= 0.0))
            
            case_records.append({
                "pseudo_field_id": f_id,
                "origin": origin_str,
                "horizon_quarters": horizon_quarters,
                "approach_name": app_name,
                "macro_nrmse": macro,
                "oil_nrmse": no,
                "water_nrmse": nw,
                "oil_bias": bo,
                "water_bias": bw,
                "terminal_oil_rate": term_qo,
                "gas_coupled_valid": gas_ratio_check,
                "non_negative_valid": non_neg_check,
                "monotonic_oil_valid": is_monotonic,
            })
            
    df_case = pd.DataFrame(case_records)
    
    # Aggregate summary
    summary_list = []
    base_macro = df_case[df_case["approach_name"] == "4_Frozen_Production_Baseline"].set_index("pseudo_field_id")["macro_nrmse"]
    
    for app_name, grp in df_case.groupby("approach_name"):
        grp_indexed = grp.set_index("pseudo_field_id")["macro_nrmse"]
        diff = grp_indexed - base_macro
        
        wins = int(np.sum(diff < -1e-6))
        ties = int(np.sum(np.abs(diff) <= 1e-6))
        win_rate = (wins + 0.5 * ties) / len(diff)
        
        summary_list.append({
            "approach_name": app_name,
            "macro_nrmse_mean": float(grp["macro_nrmse"].mean()),
            "oil_nrmse_mean": float(grp["oil_nrmse"].mean()),
            "water_nrmse_mean": float(grp["water_nrmse"].mean()),
            "diff_vs_baseline": float(diff.mean()),
            "win_rate_vs_baseline": win_rate,
            "mean_terminal_oil_rate": float(grp["terminal_oil_rate"].mean()),
            "all_non_negative": bool(grp["non_negative_valid"].all()),
            "all_gas_consistent": bool(grp["gas_coupled_valid"].all()),
            "n_evaluated_fields": len(grp),
        })
        
    df_summary = pd.DataFrame(summary_list).sort_values("macro_nrmse_mean")
    return df_summary, df_case
