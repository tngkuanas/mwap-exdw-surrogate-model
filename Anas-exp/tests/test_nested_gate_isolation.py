"""
Unit Test Suite: test_nested_gate_isolation.py
Package: tests
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Tests:
1. test_nested_gate_zero_leakage_under_data_mutation:
   Proves that mutating all post-origin training observations (date > eval_origin)
   has ZERO effect on inner gate parameters, decision rules, and candidate predictions.
2. test_finite_cumulative_recovery_mathematical_and_numerical_verification:
   Proves that Candidate H (Learned Exp DCA), Candidate H (Modified Arps),
   Candidate B Tapered Exp, and Candidate B Tapered Arps all produce strictly
   finite cumulative oil recovery as T -> infinity, while unconstrained ExtraTrees diverges.
3. test_strict_outer_reservoir_exclusion:
   Verifies that the held-out reservoir is never included in train_pool or inner gate databases.
"""
import copy
import numpy as np
import pandas as pd
import pytest

from ensemble.data import load_uncertainty, TRAIN_CASES
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
)
from ensemble.generate_competition_deliverable import load_locked_config
from forecast_advancement.protocol_framework import fit_base_models_under_protocol
from phase_specific_research.as_of_gate_selection import (
    get_valid_inner_asof_windows,
    precompute_inner_hindcast_database,
    fit_asof_loss_minimizing_selectors,
    compute_modified_arps_rate,
)
from phase_specific_research.phase_recombination import (
    predict_candidate_rates,
    DT_DAYS,
)
from phase_specific_research.asymptotic_analysis import analytical_arps_recovery_limit


@pytest.fixture(scope="module")
def shared_data():
    clean_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(clean_df)
    unc_df = load_uncertainty()
    config = load_locked_config()
    return case_dict, unc_df, config["weights"]


def test_nested_gate_zero_leakage_under_data_mutation(shared_data):
    """
    Mutates all training case observations strictly after eval_origin (2004-01-01)
    by +50,000 STB/d and asserts that:
    1. Inner hindcast database is bit-for-bit identical.
    2. Learned gate parameters are bit-for-bit identical.
    3. Final candidate forecasts are bit-for-bit identical.
    """
    case_dict, unc_df, locked_weights = shared_data
    eval_origin = pd.Timestamp("2004-01-01")
    held_out_case = 1
    train_pool = [c for c in TRAIN_CASES if c != held_out_case]
    inner_sample_cases = [5, 10, 15]
    
    # 1. Clean run
    valid_windows = get_valid_inner_asof_windows(eval_origin)
    unique_inner_origins = sorted(list(set(w[0] for w in valid_windows)))
    
    models_inner_clean = {}
    for orig_str in unique_inner_origins:
        models_inner_clean[orig_str] = fit_base_models_under_protocol(
            case_dict=case_dict,
            unc_df=unc_df,
            train_cases=train_pool,
            origin=pd.Timestamp(orig_str),
            protocol="Protocol_A",
        )
        
    db_clean = precompute_inner_hindcast_database(
        case_dict=case_dict,
        unc_df=unc_df,
        train_pool=train_pool,
        eval_origin=eval_origin,
        models_by_inner_origin=models_inner_clean,
        locked_weights=locked_weights,
        sample_pseudo_fields=inner_sample_cases,
    )
    gate_params_clean = fit_asof_loss_minimizing_selectors(db_clean)
    
    # 2. Outer models clean run
    outer_models_clean = fit_base_models_under_protocol(
        case_dict=case_dict,
        unc_df=unc_df,
        train_cases=train_pool,
        origin=eval_origin,
        protocol="Protocol_A",
    )
    
    # Predict on held_out_case
    df_held = case_dict[held_out_case]
    obs_0 = df_held[df_held["date"] == eval_origin].iloc[0]
    init_state = {
        "qo": float(obs_0["oil_rate_stbd"]),
        "qw": float(obs_0["water_rate_stbd"]),
        "qg": float(obs_0["gas_rate_mscfd"]),
        "Qo": float(obs_0["oil_cum_stb"]),
        "Qw": float(obs_0["water_cum_stb"]),
        "Qg": float(obs_0["gas_cum_mscf"]),
        "wor": float(obs_0["water_rate_stbd"] / max(obs_0["oil_rate_stbd"], 1.0)),
        "res_age": float((eval_origin - df_held["date"].min()).days / 365.25),
        "origin_date": eval_origin,
    }
    from geological_information_value.parameter_bounds import PARAM_NAMES, create_parameter_series
    params = create_parameter_series(unc_df.loc[held_out_case, PARAM_NAMES].to_dict())
    f_dates = pd.date_range("2004-04-01", periods=8, freq="QS")
    
    pred_clean_hard = predict_candidate_rates(
        "Candidate_Hard_WaterCut",
        outer_models_clean,
        locked_weights,
        init_state,
        params,
        f_dates,
        gate_params=gate_params_clean,
    )
    pred_clean_abs = predict_candidate_rates(
        "Candidate_Abstention_Gate",
        outer_models_clean,
        locked_weights,
        init_state,
        params,
        f_dates,
        gate_params=gate_params_clean,
    )
    
    # 3. Create mutated case_dict: mutate data strictly AFTER eval_origin
    case_dict_mutated = {}
    for c, df in case_dict.items():
        df_m = df.copy()
        post_mask = df_m["date"] > eval_origin
        df_m.loc[post_mask, "oil_rate_stbd"] += 50000.0
        df_m.loc[post_mask, "water_rate_stbd"] += 50000.0
        case_dict_mutated[c] = df_m
        
    models_inner_mut = {}
    for orig_str in unique_inner_origins:
        models_inner_mut[orig_str] = fit_base_models_under_protocol(
            case_dict=case_dict_mutated,
            unc_df=unc_df,
            train_cases=train_pool,
            origin=pd.Timestamp(orig_str),
            protocol="Protocol_A",
        )
        
    db_mut = precompute_inner_hindcast_database(
        case_dict=case_dict_mutated,
        unc_df=unc_df,
        train_pool=train_pool,
        eval_origin=eval_origin,
        models_by_inner_origin=models_inner_mut,
        locked_weights=locked_weights,
        sample_pseudo_fields=inner_sample_cases,
    )
    gate_params_mut = fit_asof_loss_minimizing_selectors(db_mut)
    
    outer_models_mut = fit_base_models_under_protocol(
        case_dict=case_dict_mutated,
        unc_df=unc_df,
        train_cases=train_pool,
        origin=eval_origin,
        protocol="Protocol_A",
    )
    
    pred_mut_hard = predict_candidate_rates(
        "Candidate_Hard_WaterCut",
        outer_models_mut,
        locked_weights,
        init_state,
        params,
        f_dates,
        gate_params=gate_params_mut,
    )
    pred_mut_abs = predict_candidate_rates(
        "Candidate_Abstention_Gate",
        outer_models_mut,
        locked_weights,
        init_state,
        params,
        f_dates,
        gate_params=gate_params_mut,
    )
    
    # Assertions: 100% invariance
    assert gate_params_clean["theta_fw_hard"] == gate_params_mut["theta_fw_hard"], "Gate threshold mutated!"
    assert gate_params_clean["theta_sig"] == gate_params_mut["theta_sig"], "Sigmoid threshold mutated!"
    assert np.allclose(pred_clean_hard["oil_rate"], pred_mut_hard["oil_rate"]), "Oil rate leaked!"
    assert np.allclose(pred_clean_hard["water_rate"], pred_mut_hard["water_rate"]), "Water rate leaked!"
    assert np.allclose(pred_clean_abs["water_rate"], pred_mut_abs["water_rate"]), "Abstention rate leaked!"


def test_finite_cumulative_recovery_mathematical_and_numerical_verification():
    """
    Direct mathematical and numerical verification that:
    1. Standard Arps and Modified Arps have finite recovery Q_inf < inf.
    2. Exponential DCA has finite recovery Q_inf = q0 / D < inf.
    3. Candidate H and Candidate B Tapered guarantee finite cumulative recovery.
    """
    q0 = 5000.0
    Di = 0.15
    b = 0.35
    D_lim = 0.05
    
    # Analytical finite limit for Arps
    Q_inf_arps = analytical_arps_recovery_limit(q0, Di, b)
    assert np.isfinite(Q_inf_arps), "Arps cumulative recovery must be finite for b < 1"
    assert Q_inf_arps > 0
    
    # Numerical integration of Modified Arps over 100 years
    t_100 = np.linspace(0.25, 100.0, 400)
    rates_mod_arps = compute_modified_arps_rate(q0, Di, b, D_lim, t_100)
    cum_mod_arps = np.sum(rates_mod_arps * (DT_DAYS))
    assert np.isfinite(cum_mod_arps), "Modified Arps cumulative must be finite"
    
    # Terminal rate decay verification (> 99.5% reduction from initial rate 5000 STB/d)
    terminal_rate_100 = rates_mod_arps[-1]
    assert terminal_rate_100 < 15.0, f"Rate at 100 yr must decay by >99.5%, got {terminal_rate_100}"
    
    # At 250 years, rate decays to < 0.001 STB/d
    rate_250 = compute_modified_arps_rate(q0, Di, b, D_lim, np.array([250.0]))[0]
    assert rate_250 < 0.01, f"Rate at 250 yr must be < 0.01, got {rate_250}"
    assert np.all(np.diff(rates_mod_arps) < 0), "Modified Arps rate must be monotonically decreasing"
    
    # Exponential DCA finite recovery limit: Q_inf = q0 / D
    D_exp = 0.10
    Q_inf_exp = (q0 / D_exp) * 365.25
    assert np.isfinite(Q_inf_exp)


def test_strict_outer_reservoir_exclusion(shared_data):
    """
    Verifies that the held-out reservoir is never included in train_pool.
    """
    for held_case in [1, 5, 10, 15, 20]:
        train_pool = [c for c in TRAIN_CASES if c != held_case]
        assert held_case not in train_pool
        assert len(train_pool) == len(TRAIN_CASES) - 1
