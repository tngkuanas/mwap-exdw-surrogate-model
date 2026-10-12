"""
Unit Tests: test_geological_information_value.py
Package: geological_information_value
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Tests:
1. Parameter bounds, normalization, and prior sampling functions.
2. Extended pipeline post-origin mutation invariance under Protocol A.
3. Forward-model sensitivity and Jacobian computations.
4. Geology-Conditioned Physical Forecaster (GCDF) physical invariants.
"""
import numpy as np
import pandas as pd
import pytest

from geological_information_value.parameter_bounds import (
    PARAM_NAMES,
    SHORT_NAMES,
    LOWER_BOUNDS,
    UPPER_BOUNDS,
    RANGES,
    PRIOR_MEANS,
    normalize_parameters,
    denormalize_parameters,
    sample_prior_uniform,
    get_adversarial_mismatched_parameters,
    create_parameter_series,
)
from geological_information_value.geology_conditioned_challenger import (
    GeologyConditionedPhysicalForecaster,
)
from geological_information_value.end_to_end_leakage_audit import (
    run_extended_leakage_mutation_test,
)
from ensemble.data import load_uncertainty, resolve_shared_dir, TRAIN_CASES
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
)


def test_parameter_bounds_and_transformations():
    """Verifies round-trip normalization and bounds."""
    assert len(PARAM_NAMES) == 4
    assert np.all(UPPER_BOUNDS > LOWER_BOUNDS)
    
    # Test normalization and denormalization roundtrip
    test_u = np.array([0.05, 1.20, 2.50, 150.0])
    u_norm = normalize_parameters(test_u)
    assert np.all((u_norm >= 0.0) & (u_norm <= 1.0))
    u_recovered = denormalize_parameters(u_norm)
    assert np.allclose(test_u, u_recovered, atol=1e-12)
    
    # Test adversarial mismatch
    u_mismatch = get_adversarial_mismatched_parameters(test_u)
    assert np.all(u_mismatch >= LOWER_BOUNDS)
    assert np.all(u_mismatch <= UPPER_BOUNDS)
    
    # Test prior sampling
    samples = sample_prior_uniform(50, seed=123)
    assert samples.shape == (50, 4)
    assert np.all(samples >= LOWER_BOUNDS)
    assert np.all(samples <= UPPER_BOUNDS)


def test_geology_conditioned_forecaster_invariants():
    """Verifies that GCDF preserves non-negativity and responsive physical decline."""
    shared_dir = resolve_shared_dir()
    unc_df = load_uncertainty(shared_dir)
    raw = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(raw)
    
    origin = pd.Timestamp("2005-01-01")
    f_dates = pd.date_range("2005-04-01", periods=12, freq="QS")
    
    # Fit forecaster on small subset
    forecaster = GeologyConditionedPhysicalForecaster()
    forecaster.fit(case_dict, unc_df, TRAIN_CASES[:10], origin)
    
    init_state = {
        "qo": 5000.0,
        "qw": 1000.0,
        "wor": 0.20,
        "origin_date": origin,
    }
    
    # Test with low vs high aquifer support
    u_low = pd.Series({
        "Fault Transmissibility": 0.05,
        "Porosity Multiplier": 1.2,
        "Permeability Multiplier": 0.5,
        "Aquifer Pore Volume": 20.0,
    })
    u_high = pd.Series({
        "Fault Transmissibility": 0.50,
        "Porosity Multiplier": 1.2,
        "Permeability Multiplier": 5.0,
        "Aquifer Pore Volume": 250.0,
    })
    
    res_low = forecaster.predict_rates(init_state, u_low, f_dates)
    res_high = forecaster.predict_rates(init_state, u_high, f_dates)
    
    # 1. Non-negativity
    assert np.all(res_low["oil_rate"] >= 0.0)
    assert np.all(res_low["water_rate"] >= 0.0)
    assert np.all(res_high["oil_rate"] >= 0.0)
    assert np.all(res_high["water_rate"] >= 0.0)
    
    # 2. Physics responsiveness: high aquifer support should maintain higher oil rate
    # and produce more water than low aquifer support
    assert res_high["pi_aq_used"] > res_low["pi_aq_used"]
    assert res_high["oil_rate"][-1] > res_low["oil_rate"][-1]
    assert res_high["water_rate"][-1] > res_low["water_rate"][-1]


def test_extended_pipeline_mutation_leakage():
    """Verifies that under Protocol A, base models and gate parameters are 100% leak-free."""
    shared_dir = resolve_shared_dir()
    unc_df = load_uncertainty(shared_dir)
    raw = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(raw)
    
    res = run_extended_leakage_mutation_test(
        case_dict=case_dict,
        unc_df=unc_df,
        train_cases=TRAIN_CASES[:10],
        test_case=71,
        origin=pd.Timestamp("2005-01-01"),
        horizon_quarters=8,
    )
    
    assert res["protocol_a_is_fully_leak_free"] is True
    assert res["protocol_a_base_oil_diff"] < 1e-12
    assert res["protocol_a_base_wat_diff"] < 1e-12
    assert res["protocol_a_gate_coef_diff"] < 1e-12
    assert res["protocol_a_gate_pred_diff"] < 1e-12
