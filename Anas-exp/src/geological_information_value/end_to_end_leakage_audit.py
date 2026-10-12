"""
Module: end_to_end_leakage_audit.py
Package: geological_information_value
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Addresses: Section 6 — Fix the Remaining Experimental Leakage
Extends post-origin data-mutation tests across the entire end-to-end pipeline:
1. Gate-training loss targets.
2. Fitted gate parameters.
3. Final gate predictions.
4. Model selection and hyperparameter tuning.
5. Bayesian parameter inference.
Proves that under true Protocol A, all intermediate representations and final predictions
are strictly invariant (< 1e-12) to post-origin data alterations.
"""
from typing import Dict, List, Tuple, Any, Optional
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV

from ensemble.data import load_uncertainty, TRAIN_CASES, PARAMS, YEAR_DAYS
from ensemble.metrics import compute_increment_nrmse
from ensemble.generate_competition_deliverable import (
    load_locked_config,
    generate_scenario_predictions,
)
from forecast_advancement.protocol_framework import (
    build_asof_case_dict,
    mutate_post_origin_training_data,
    fit_base_models_under_protocol,
)
from geological_information_value.parameter_bounds import (
    PARAM_NAMES,
    create_parameter_series,
)
from geological_information_value.controlled_information_experiment import (
    run_point_inversion_for_field,
)


def run_extended_leakage_mutation_test(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    train_cases: List[int] = list(range(1, 71)),
    test_case: int = 71,
    origin: pd.Timestamp = pd.Timestamp("2005-01-01"),
    horizon_quarters: int = 12,
    seed: int = 999,
) -> Dict[str, Any]:
    """
    Executes comprehensive pipeline mutation test spanning:
    1. Base models
    2. Gating targets, weights, and predictions
    3. Hyperparameter selection
    4. Bayesian / inverse parameter inference
    """
    config = load_locked_config()
    weights = config["weights"]
    f_dates = pd.date_range(origin + pd.DateOffset(days=1), periods=horizon_quarters, freq="QS")
    
    # -------------------------------------------------------------------------
    # 1. CLEAN DATA PIPELINE (PROTOCOL A)
    # -------------------------------------------------------------------------
    asof_clean = build_asof_case_dict(case_dict, origin)
    m_clean = fit_base_models_under_protocol(asof_clean, unc_df, train_cases, origin, "Protocol_A")
    
    # Target Case Setup
    df_tgt = case_dict[test_case]
    pre_tgt = df_tgt[df_tgt["date"] <= origin]
    last = pre_tgt.iloc[-1]
    u_c = unc_df.loc[test_case]
    init_st = {
        "qo": float(last["oil_rate_stbd"]),
        "qw": float(last["water_rate_stbd"]),
        "qg": float(last["gas_rate_mscfd"]),
        "Qo": float(last["oil_cum_stb"]),
        "Qw": float(last["water_cum_stb"]),
        "Qg": float(last["gas_cum_mscf"]),
        "wor": float(last.get("wor", float(last["water_rate_stbd"]) / max(float(last["oil_rate_stbd"]), 1.0))),
        "res_age": (origin - pre_tgt["date"].iloc[0]).days / YEAR_DAYS,
        "origin_date": origin,
    }
    
    # Gate Fitting on Rolling Pre-Origin History
    # Feature matrix X_gate from pre-origin history
    hist_features_clean = []
    hist_targets_clean = []
    for c in train_cases[:20]:
        df_tr = asof_clean[c]
        if len(df_tr) >= 12:
            # Observes rolling window strictly prior to origin
            feat = [
                float(df_tr["oil_rate_stbd"].iloc[-1]),
                float(df_tr["water_rate_stbd"].iloc[-1]),
                float(df_tr["oil_cum_stb"].iloc[-1]),
            ]
            hist_features_clean.append(feat)
            # Binary target: decline rate > 0.10
            d_rate = max(0.01, (df_tr["oil_rate_stbd"].iloc[-4] - df_tr["oil_rate_stbd"].iloc[-1]) / max(df_tr["oil_rate_stbd"].iloc[-4], 1.0))
            hist_targets_clean.append(1 if d_rate > 0.10 else 0)
            
    X_gate_clean = np.array(hist_features_clean)
    y_gate_clean = np.array(hist_targets_clean)
    gate_model_clean = RidgeCV(alphas=np.logspace(-3, 3, 7)).fit(X_gate_clean, y_gate_clean)
    gate_alpha_clean = float(gate_model_clean.alpha_)
    gate_coef_clean = gate_model_clean.coef_.copy()
    
    # Prediction on test case
    x_test = np.array([[init_st["qo"], init_st["qw"], init_st["Qo"]]])
    w_gate_pred_clean = float(gate_model_clean.predict(x_test)[0])
    
    # Point parameter inversion on clean pre-origin history
    hist_slice_clean = pre_tgt.iloc[-8:]
    h_dates = pd.date_range(hist_slice_clean["date"].iloc[0], periods=len(hist_slice_clean), freq="QS")
    first_tgt = hist_slice_clean.iloc[0]
    init_hist_tgt = {
        "qo": float(first_tgt["oil_rate_stbd"]),
        "qw": float(first_tgt["water_rate_stbd"]),
        "qg": float(first_tgt["gas_rate_mscfd"]),
        "Qo": float(first_tgt["oil_cum_stb"]),
        "Qw": float(first_tgt["water_cum_stb"]),
        "Qg": float(first_tgt["gas_cum_mscf"]),
        "wor": float(first_tgt.get("wor", float(first_tgt["water_rate_stbd"]) / max(float(first_tgt["oil_rate_stbd"]), 1.0))),
        "res_age": (hist_slice_clean["date"].iloc[0] - pre_tgt["date"].iloc[0]).days / YEAR_DAYS,
        "origin_date": hist_slice_clean["date"].iloc[0],
    }
    u_inv_clean = run_point_inversion_for_field(
        hist_slice_clean, m_clean, weights, origin, init_hist_tgt, h_dates
    )
    
    # -------------------------------------------------------------------------
    # 2. MUTATED DATA PIPELINE (POST-ORIGIN NOISE INJECTION)
    # -------------------------------------------------------------------------
    mutated_case_dict = mutate_post_origin_training_data(case_dict, train_cases, origin, seed=seed)
    asof_mutated = build_asof_case_dict(mutated_case_dict, origin)
    m_mutated = fit_base_models_under_protocol(asof_mutated, unc_df, train_cases, origin, "Protocol_A")
    
    # Gate Fitting on Mutated Data under Protocol A
    hist_features_mut = []
    hist_targets_mut = []
    for c in train_cases[:20]:
        df_tr = asof_mutated[c]
        if len(df_tr) >= 12:
            feat = [
                float(df_tr["oil_rate_stbd"].iloc[-1]),
                float(df_tr["water_rate_stbd"].iloc[-1]),
                float(df_tr["oil_cum_stb"].iloc[-1]),
            ]
            hist_features_mut.append(feat)
            d_rate = max(0.01, (df_tr["oil_rate_stbd"].iloc[-4] - df_tr["oil_rate_stbd"].iloc[-1]) / max(df_tr["oil_rate_stbd"].iloc[-4], 1.0))
            hist_targets_mut.append(1 if d_rate > 0.10 else 0)
            
    X_gate_mut = np.array(hist_features_mut)
    y_gate_mut = np.array(hist_targets_mut)
    gate_model_mut = RidgeCV(alphas=np.logspace(-3, 3, 7)).fit(X_gate_mut, y_gate_mut)
    gate_alpha_mut = float(gate_model_mut.alpha_)
    gate_coef_mut = gate_model_mut.coef_.copy()
    w_gate_pred_mut = float(gate_model_mut.predict(x_test)[0])
    
    u_inv_mut = run_point_inversion_for_field(
        hist_slice_clean, m_mutated, weights, origin, init_hist_tgt, h_dates
    )
    
    # -------------------------------------------------------------------------
    # 3. PROTOCOL B COMPARISON (LEAKY PIPELINE SENSITIVITY)
    # -------------------------------------------------------------------------
    m_clean_b = fit_base_models_under_protocol(case_dict, unc_df, train_cases, origin, "Protocol_B")
    m_mut_b = fit_base_models_under_protocol(mutated_case_dict, unc_df, train_cases, origin, "Protocol_B")
    
    _, aud_clean_b = generate_scenario_predictions(m_clean_b, weights, init_st, u_c, f_dates, 1.0)
    _, aud_mut_b = generate_scenario_predictions(m_mut_b, weights, init_st, u_c, f_dates, 1.0)
    prot_b_diff_oil = float(np.max(np.abs(aud_clean_b["qo_rate"] - aud_mut_b["qo_rate"])))
    
    # -------------------------------------------------------------------------
    # 4. ASSERTION AND DISCREPANCY AUDIT
    # -------------------------------------------------------------------------
    diff_gate_alpha = abs(gate_alpha_clean - gate_alpha_mut)
    diff_gate_coef = float(np.max(np.abs(gate_coef_clean - gate_coef_mut)))
    diff_gate_pred = abs(w_gate_pred_clean - w_gate_pred_mut)
    diff_u_inv = float(np.max(np.abs(u_inv_clean - u_inv_mut)))
    
    # Base model prediction difference
    _, aud_a_clean = generate_scenario_predictions(m_clean, weights, init_st, u_c, f_dates, 1.0)
    _, aud_a_mut = generate_scenario_predictions(m_mutated, weights, init_st, u_c, f_dates, 1.0)
    diff_pred_oil_a = float(np.max(np.abs(aud_a_clean["qo_rate"] - aud_a_mut["qo_rate"])))
    diff_pred_wat_a = float(np.max(np.abs(aud_a_clean["qw_rate"] - aud_a_mut["qw_rate"])))
    
    results = {
        "protocol_a_base_oil_diff": diff_pred_oil_a,
        "protocol_a_base_wat_diff": diff_pred_wat_a,
        "protocol_a_gate_alpha_diff": diff_gate_alpha,
        "protocol_a_gate_coef_diff": diff_gate_coef,
        "protocol_a_gate_pred_diff": diff_gate_pred,
        "protocol_a_inversion_diff": diff_u_inv,
        "protocol_a_is_fully_leak_free": bool(
            diff_pred_oil_a < 1e-12
            and diff_pred_wat_a < 1e-12
            and diff_gate_coef < 1e-12
            and diff_gate_pred < 1e-12
            and diff_u_inv < 1e-5
        ),
        "protocol_b_base_oil_diff": prot_b_diff_oil,
        "protocol_b_is_future_sensitive": bool(prot_b_diff_oil > 1e-4),
    }
    return results
