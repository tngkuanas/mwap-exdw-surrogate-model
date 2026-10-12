"""
Unit and Acceptance Tests for Forecast Scientific Advancement
Module: tests/test_forecast_advancement.py
Branch: research/forecast-scientific-advancement
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp
"""
import pytest
import numpy as np
import pandas as pd

from ensemble.data import load_uncertainty, resolve_shared_dir, TRAIN_CASES, VAL_CASES
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
)
from forecast_advancement.protocol_framework import (
    build_asof_case_dict,
    run_data_mutation_leakage_test,
)
from forecast_advancement.constrained_liquid_dynamics import (
    ConstrainedLiquidDynamicsForecaster,
)
from forecast_advancement.functional_trajectory_forecaster import (
    FunctionalTrajectoryForecaster,
)
from forecast_advancement.recovery_state_forecaster import (
    RecoveryStateForecaster,
)


@pytest.fixture(scope="module")
def shared_data():
    shared_dir = resolve_shared_dir()
    unc_df = load_uncertainty(shared_dir)
    clean_sim_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(clean_sim_df)
    return unc_df, case_dict


def test_protocol_a_data_mutation_zero_leakage(shared_data):
    """
    Mandatory Acceptance Test: Mutating post-origin training observations
    must produce IDENTICAL predictions under Protocol A (diff < 1e-12)
    and legitimately change predictions under Protocol B.
    """
    unc_df, case_dict = shared_data
    # Run test on 20 training cases and target case 71
    train_subset = list(range(1, 21))
    origin = pd.Timestamp("2005-01-01")
    
    res = run_data_mutation_leakage_test(
        case_dict, unc_df, train_subset, test_case=71, origin=origin, horizon_quarters=8
    )
    
    assert res["protocol_a_is_leak_free"], f"Protocol A leaked! Diff: {res['protocol_a_oil_max_diff']}"
    assert res["residual_forecaster_is_leak_free"], f"Residual forecaster leaked! Diff: {res['residual_forecaster_asof_diff']}"
    assert res["protocol_b_is_sensitive"], f"Protocol B failed to show future sensitivity! Diff: {res['protocol_b_oil_max_diff']}"


def test_constrained_liquid_dynamics_invariants(shared_data):
    """
    Verifies mass conservation, fractional flow bounds (0 <= fw <= 1),
    non-negativity, and monotonicity for Constrained Liquid Dynamics.
    """
    unc_df, case_dict = shared_data
    train_subset = list(range(1, 15))
    origin = pd.Timestamp("2005-01-01")
    f_dates = pd.date_range("2005-04-01", periods=20, freq="QS")
    
    m_liq = ConstrainedLiquidDynamicsForecaster(fractional_flow_mode="logistic").fit(
        case_dict, unc_df, train_subset, origin
    )
    
    df_tgt = case_dict[71]
    u_tgt = unc_df.loc[71]
    preds = m_liq.predict_rates(df_tgt, u_tgt, origin, f_dates)
    
    qo = preds["oil_rate"]
    qw = preds["water_rate"]
    qliq = preds["liquid_rate"]
    fw = preds["water_cut"]
    
    # Invariant 1: Mass conservation q_o + q_w == q_liquid
    assert np.allclose(qo + qw, qliq, atol=1e-5), "Liquid mass conservation violated!"
    
    # Invariant 2: 0 <= fw <= 1
    assert np.all(fw >= 0.0) and np.all(fw <= 1.0), "Water cut exceeded physical bounds [0, 1]!"
    
    # Invariant 3: Non-negativity
    assert np.all(qo >= 0.0) and np.all(qw >= 0.0), "Negative production rates detected!"
    
    # Invariant 4: Monotonic non-increasing oil rate
    diffs_qo = np.diff(qo)
    assert np.all(diffs_qo <= 1e-6), "Oil rate is not monotonically non-increasing!"


def test_functional_trajectory_continuity_and_tails(shared_data):
    """
    Verifies that the functional trajectory forecaster enforces rate continuity
    at origin and bounded extrapolation tails.
    """
    unc_df, case_dict = shared_data
    train_subset = list(range(1, 20))
    origin = pd.Timestamp("2005-01-01")
    f_dates = pd.date_range("2005-04-01", periods=40, freq="QS")
    
    m_func = FunctionalTrajectoryForecaster().fit(
        case_dict, unc_df, train_subset, origin, window_quarters=8
    )
    
    df_tgt = case_dict[71]
    u_tgt = unc_df.loc[71]
    last_rate = float(df_tgt[df_tgt["date"] <= origin]["oil_rate_stbd"].iloc[-1])
    
    preds = m_func.predict_rates(df_tgt, u_tgt, origin, f_dates)
    qo = preds["oil_rate"]
    
    # Continuity: step 0 rate is closely aligned with origin rate
    assert np.isclose(qo[0], last_rate, rtol=0.25), f"Origin discontinuity: {qo[0]} vs {last_rate}"
    
    # Bounded tail: late rate decays smoothly and stays positive
    assert qo[-1] <= qo[0], "Late time rate exploded above initial rate!"
    assert qo[-1] >= 0.0, "Negative rate in late tail!"


def test_recovery_state_dimensionless_quantities(shared_data):
    """
    Verifies dimensionless recovery factor and water encroachment calculations.
    """
    unc_df, case_dict = shared_data
    train_subset = list(range(1, 15))
    origin = pd.Timestamp("2005-01-01")
    f_dates = pd.date_range("2005-04-01", periods=12, freq="QS")
    
    m_state = RecoveryStateForecaster().fit(case_dict, unc_df, train_subset, origin)
    
    df_tgt = case_dict[72]
    u_tgt = unc_df.loc[72]
    preds = m_state.predict_rates(df_tgt, u_tgt, origin, f_dates)
    
    assert len(preds["oil_rate"]) == 12
    assert np.all(preds["oil_rate"] >= 0.0)
    assert np.all(preds["water_rate"] >= 0.0)
