"""
Module: protocol_framework.py
Package: forecast_advancement
Addresses: Stage 1 — Correct the Strict As-Of Protocol and Leakage Mutation Tests
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Provides:
1. build_asof_case_dict(): Enforces strict historical data truncation <= origin.
2. fit_base_models_under_protocol(): Fits production baseline under Protocol A or B.
3. run_data_mutation_leakage_test(): Mandatory test proving Protocol A is 100% leak-free
   and Protocol B is sensitive to post-origin training data.
"""
from typing import Dict, List, Tuple, Any, Optional
import copy
import numpy as np
import pandas as pd

from ensemble.run_horizon_adaptive_stacking_experiment import (
    RecursiveTransitionForecaster,
    LearnedDCARateForecaster,
    HybridDynamicForecaster,
    extract_transition_training_data,
    YEAR_DAYS,
    RS_SOLUTION,
)
from ensemble.generate_competition_deliverable import (
    load_locked_config,
    generate_scenario_predictions,
)
from conditional_challengers.corrected_residual_dca import (
    CorrectedResidualForecaster,
    fit_historical_dca_at_origin,
    extract_state_features,
)


def build_asof_case_dict(
    case_dict: Dict[int, pd.DataFrame],
    origin: pd.Timestamp,
) -> Dict[int, pd.DataFrame]:
    """
    Constructs a strictly truncated dictionary where every case DataFrame
    contains strictly observations with date <= origin.
    Prevents any model or preprocessing step from accessing post-origin data.
    """
    return {
        c: df[df["date"] <= origin].copy()
        for c, df in case_dict.items()
    }


def fit_base_models_under_protocol(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    train_cases: List[int],
    origin: pd.Timestamp,
    protocol: str = "Protocol_A",
    horizon_quarters: int = 80,
) -> Dict[str, Any]:
    """
    Fits the locked production base models under either Protocol A or Protocol B.
    
    Protocol A (Strict Historical As-Of):
      - Truncates training data strictly to date <= origin before passing to models.
      - DCA learns from historical decline window pre-origin.
      - Hybrid forecaster uses historical transition + historical DCA.
      
    Protocol B (Cross-Case Forward-Supervised):
      - Training cases (1-70) expose post-origin trajectories.
      - DCA learns from forward-looking simulator trajectories post-origin.
    """
    if protocol == "Protocol_A":
        tr_dict = build_asof_case_dict(case_dict, origin)
    elif protocol == "Protocol_B":
        tr_dict = case_dict
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
        
    m_et = RecursiveTransitionForecaster(model_type="extratrees").fit(
        extract_transition_training_data(tr_dict, unc_df, train_cases, origin)
    )
    m_exp = LearnedDCARateForecaster(decline_type="exponential").fit(
        tr_dict, unc_df, train_cases, origin, horizon_quarters
    )
    m_hyp = LearnedDCARateForecaster(decline_type="hyperbolic").fit(
        tr_dict, unc_df, train_cases, origin, horizon_quarters
    )
    m_hyb = HybridDynamicForecaster(switch_quarters=12).fit(
        tr_dict, unc_df, train_cases, origin, horizon_quarters
    )
    
    return {
        "Recursive_ExtraTrees": m_et,
        "Learned_Exponential_DCA": m_exp,
        "Learned_Hyperbolic_DCA": m_hyp,
        "Hybrid_Dynamic": m_hyb,
        "protocol": protocol,
        "origin": origin,
    }


def mutate_post_origin_training_data(
    case_dict: Dict[int, pd.DataFrame],
    train_cases: List[int],
    origin: pd.Timestamp,
    seed: int = 999,
) -> Dict[int, pd.DataFrame]:
    """
    Creates a copy of case_dict where all observations post-origin
    for permitted training cases are replaced with random noise.
    """
    rng = np.random.RandomState(seed)
    mutated = {}
    for c, df in case_dict.items():
        df_copy = df.copy()
        if c in train_cases:
            post_mask = df_copy["date"] > origin
            n_post = post_mask.sum()
            if n_post > 0:
                # Randomize rates and cumulatives heavily
                df_copy.loc[post_mask, "oil_rate_stbd"] = rng.uniform(10.0, 50000.0, n_post)
                df_copy.loc[post_mask, "water_rate_stbd"] = rng.uniform(5.0, 40000.0, n_post)
                df_copy.loc[post_mask, "gas_rate_mscfd"] = rng.uniform(10.0, 20000.0, n_post)
                df_copy.loc[post_mask, "oil_cum_stb"] = rng.uniform(1e6, 1e8, n_post)
                df_copy.loc[post_mask, "water_cum_stb"] = rng.uniform(1e6, 1e8, n_post)
                df_copy.loc[post_mask, "gas_cum_mscf"] = rng.uniform(1e6, 1e8, n_post)
        mutated[c] = df_copy
    return mutated


def run_data_mutation_leakage_test(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    train_cases: List[int],
    test_case: int,
    origin: pd.Timestamp,
    horizon_quarters: int = 12,
) -> Dict[str, Any]:
    """
    Executes the mandatory leakage acceptance test:
    1. Fit and predict under Protocol A on clean data.
    2. Randomize every post-origin training-case observation.
    3. Fit and predict again under Protocol A.
    4. Assert identical predictions within numerical tolerance (< 1e-12).
    5. Fit and predict under Protocol B on clean vs mutated data.
    6. Assert Protocol B changes due to future dependency.
    """
    config = load_locked_config()
    weights = config["weights"]
    f_dates = pd.date_range(origin + pd.DateOffset(days=1), periods=horizon_quarters, freq="QS")
    
    # Target case initial state
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
    
    # 1. Protocol A on Clean Data
    m_clean_a = fit_base_models_under_protocol(case_dict, unc_df, train_cases, origin, "Protocol_A")
    _, aud_clean_a = generate_scenario_predictions(m_clean_a, weights, init_st, u_c, f_dates, 1.0)
    pred_clean_a_oil = aud_clean_a["qo_rate"]
    pred_clean_a_wat = aud_clean_a["qw_rate"]
    
    # 2. Mutate Post-Origin Data
    mutated_dict = mutate_post_origin_training_data(case_dict, train_cases, origin)
    
    # 3. Protocol A on Mutated Data
    m_mut_a = fit_base_models_under_protocol(mutated_dict, unc_df, train_cases, origin, "Protocol_A")
    _, aud_mut_a = generate_scenario_predictions(m_mut_a, weights, init_st, u_c, f_dates, 1.0)
    pred_mut_a_oil = aud_mut_a["qo_rate"]
    pred_mut_a_wat = aud_mut_a["qw_rate"]
    
    diff_a_oil = float(np.max(np.abs(pred_clean_a_oil - pred_mut_a_oil)))
    diff_a_wat = float(np.max(np.abs(pred_clean_a_wat - pred_mut_a_wat)))
    
    # 4. Protocol B on Clean vs Mutated Data
    m_clean_b = fit_base_models_under_protocol(case_dict, unc_df, train_cases, origin, "Protocol_B")
    _, aud_clean_b = generate_scenario_predictions(m_clean_b, weights, init_st, u_c, f_dates, 1.0)
    pred_clean_b_oil = aud_clean_b["qo_rate"]
    
    m_mut_b = fit_base_models_under_protocol(mutated_dict, unc_df, train_cases, origin, "Protocol_B")
    _, aud_mut_b = generate_scenario_predictions(m_mut_b, weights, init_st, u_c, f_dates, 1.0)
    pred_mut_b_oil = aud_mut_b["qo_rate"]
    
    diff_b_oil = float(np.max(np.abs(pred_clean_b_oil - pred_mut_b_oil)))
    
    # Check residual forecaster independently
    tr_asof = build_asof_case_dict(case_dict, origin)
    tr_asof_mut = build_asof_case_dict(mutated_dict, origin)
    
    res_clean = CorrectedResidualForecaster(architecture="gp").fit(
        tr_asof, unc_df, train_cases, cutoff_date=origin, max_horizon_quarters=horizon_quarters
    )
    res_mut = CorrectedResidualForecaster(architecture="gp").fit(
        tr_asof_mut, unc_df, train_cases, cutoff_date=origin, max_horizon_quarters=horizon_quarters
    )
    dca_base = fit_historical_dca_at_origin(tr_asof, unc_df, train_cases, origin)
    
    pred_res_clean = res_clean.predict_rates(df_tgt, u_c, origin, f_dates, dca_base)
    pred_res_mut = res_mut.predict_rates(df_tgt, u_c, origin, f_dates, dca_base)
    diff_res_oil = float(np.max(np.abs(pred_res_clean["oil_rate"] - pred_res_mut["oil_rate"])))
    
    return {
        "protocol_a_oil_max_diff": diff_a_oil,
        "protocol_a_wat_max_diff": diff_a_wat,
        "protocol_a_is_leak_free": (diff_a_oil < 1e-12 and diff_a_wat < 1e-12),
        "protocol_b_oil_max_diff": diff_b_oil,
        "protocol_b_is_sensitive": (diff_b_oil > 1e-4),
        "residual_forecaster_asof_diff": diff_res_oil,
        "residual_forecaster_is_leak_free": (diff_res_oil < 1e-12),
    }
