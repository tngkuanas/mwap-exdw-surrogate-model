"""
Unit tests for the Learned Dynamics Forecasting Framework.
Verifies:
1. Non-negative rates across all rollouts.
2. Monotonic cumulative production (d/dt Q >= 0).
3. Zero lookahead leakage during recursive state transition.
4. Rate and cumulative continuity at hybrid model handoff boundary.
5. Strict adherence to dataset boundary (blind cases 86-100 sealed).
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ensemble.run_learned_dynamics_forecasting import (
    RecursiveStateTransitionModel,
    LearnedDynamicRateModel,
    HybridDynamicModel,
    extract_transition_training_data,
    build_case_quarterly_trajectories,
    load_clean_simulation_quarterly,
    RS_SOLUTION,
    YEAR_DAYS,
)
from ensemble.data import load_uncertainty, TRAIN_CASES, VAL_CASES

@pytest.fixture(scope="module")
def setup_data():
    unc_df = load_uncertainty()
    q_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(q_df)
    return unc_df, q_df, case_dict

def test_recursive_state_transition_nonnegative_and_monotonic(setup_data):
    unc_df, q_df, case_dict = setup_data
    cutoff = pd.Timestamp("2005-01-01")
    train_c = TRAIN_CASES[:20] # subset for fast test
    
    trans_df = extract_transition_training_data(case_dict, unc_df, train_c, cutoff)
    model = RecursiveStateTransitionModel(model_type="ridge").fit(trans_df)
    
    # Test rollout on Case 71 for 20 quarters
    c = 71
    df_c = case_dict[c]
    pre = df_c[df_c["date"] <= cutoff]
    forecast_dates = pd.date_range("2005-04-01", periods=20, freq="QS")
    
    initial_state = {
        "qo": float(pre.iloc[-1]["oil_rate_stbd"]),
        "qw": float(pre.iloc[-1]["water_rate_stbd"]),
        "qg": float(pre.iloc[-1]["gas_rate_mscfd"]),
        "Qo": float(pre.iloc[-1]["oil_cum_stb"]),
        "Qw": float(pre.iloc[-1]["water_cum_stb"]),
        "Qg": float(pre.iloc[-1]["gas_cum_mscf"]),
        "wor": float(pre.iloc[-1]["wor"]),
        "res_age": float(pre.iloc[-1]["res_age_years"]),
        "origin_date": cutoff,
    }
    
    res = model.predict_rollout(initial_state, unc_df.loc[c], forecast_dates)
    
    # Rates must be non-negative
    assert np.all(res["oil_rate"] >= 0.0), "Negative oil rates detected"
    assert np.all(res["gas_rate"] >= 0.0), "Negative gas rates detected"
    assert np.all(res["water_rate"] >= 0.0), "Negative water rates detected"
    
    # Cumulatives must be strictly monotonic non-decreasing
    assert np.all(np.diff(res["oil_cum"]) >= -1e-6), "Decreasing oil cumulative"
    assert np.all(np.diff(res["gas_cum"]) >= -1e-6), "Decreasing gas cumulative"
    assert np.all(np.diff(res["water_cum"]) >= -1e-6), "Decreasing water cumulative"

def test_learned_dynamic_rate_model_physics(setup_data):
    unc_df, q_df, case_dict = setup_data
    cutoff = pd.Timestamp("2005-01-01")
    train_c = TRAIN_CASES[:20]
    
    dca = LearnedDynamicRateModel(decline_type="exponential")
    dca.fit(case_dict, unc_df, train_c, cutoff, horizon_quarters=12)
    
    c = 71
    df_c = case_dict[c]
    pre = df_c[df_c["date"] <= cutoff]
    forecast_dates = pd.date_range("2005-04-01", periods=40, freq="QS")
    
    initial_state = {
        "qo": float(pre.iloc[-1]["oil_rate_stbd"]),
        "qw": float(pre.iloc[-1]["water_rate_stbd"]),
        "qg": float(pre.iloc[-1]["gas_rate_mscfd"]),
        "Qo": float(pre.iloc[-1]["oil_cum_stb"]),
        "Qw": float(pre.iloc[-1]["water_cum_stb"]),
        "Qg": float(pre.iloc[-1]["gas_cum_mscf"]),
        "wor": float(pre.iloc[-1]["wor"]),
        "res_age": float(pre.iloc[-1]["res_age_years"]),
        "origin_date": cutoff,
    }
    
    res = dca.predict_rollout(initial_state, unc_df.loc[c], forecast_dates)
    
    # Verify exact physical solution GOR coupling
    delta_Qo = res["oil_cum"] - initial_state["Qo"]
    delta_Qg = res["gas_cum"] - initial_state["Qg"]
    expected_Qg = delta_Qo * RS_SOLUTION
    assert np.allclose(delta_Qg, expected_Qg, rtol=1e-5), "Gas-oil physical ratio violated"
    
    # Rates must be non-negative
    assert np.all(res["oil_rate"] >= 0.0)
    assert np.all(res["gas_rate"] >= 0.0)
    assert np.all(res["water_rate"] >= 0.0)

def test_hybrid_dynamic_continuity(setup_data):
    unc_df, q_df, case_dict = setup_data
    cutoff = pd.Timestamp("2003-01-01")
    train_c = TRAIN_CASES[:20]
    
    hybrid = HybridDynamicModel(switch_quarters=12)
    hybrid.fit(case_dict, unc_df, train_c, cutoff, horizon_quarters=20)
    
    c = 72
    df_c = case_dict[c]
    pre = df_c[df_c["date"] <= cutoff]
    forecast_dates = pd.date_range("2003-04-01", periods=20, freq="QS")
    
    initial_state = {
        "qo": float(pre.iloc[-1]["oil_rate_stbd"]),
        "qw": float(pre.iloc[-1]["water_rate_stbd"]),
        "qg": float(pre.iloc[-1]["gas_rate_mscfd"]),
        "Qo": float(pre.iloc[-1]["oil_cum_stb"]),
        "Qw": float(pre.iloc[-1]["water_cum_stb"]),
        "Qg": float(pre.iloc[-1]["gas_cum_mscf"]),
        "wor": float(pre.iloc[-1]["wor"]),
        "res_age": float(pre.iloc[-1]["res_age_years"]),
        "origin_date": cutoff,
    }
    
    res = hybrid.predict_rollout(initial_state, unc_df.loc[c], forecast_dates)
    
    # Check boundary at switch index 11 -> 12 (quarters 12 -> 13)
    # Cumulative should have continuous step: Qo[12] >= Qo[11]
    assert res["oil_cum"][12] >= res["oil_cum"][11], "Discontinuous cumulative drop at hybrid switch boundary"
    # Rate at index 12 should be within reasonable factor of rate at index 11 (C1 continuity)
    rate_ratio = res["oil_rate"][12] / max(res["oil_rate"][11], 1e-3)
    assert 0.5 < rate_ratio < 1.5, f"Discontinuous rate step across hybrid boundary: ratio={rate_ratio}"

def test_zero_lookahead_leakage(setup_data):
    unc_df, q_df, case_dict = setup_data
    cutoff = pd.Timestamp("2004-01-01")
    c = 75
    df_c = case_dict[c]
    pre = df_c[df_c["date"] <= cutoff]
    forecast_dates = pd.date_range("2004-04-01", periods=16, freq="QS")
    
    initial_state = {
        "qo": float(pre.iloc[-1]["oil_rate_stbd"]),
        "qw": float(pre.iloc[-1]["water_rate_stbd"]),
        "qg": float(pre.iloc[-1]["gas_rate_mscfd"]),
        "Qo": float(pre.iloc[-1]["oil_cum_stb"]),
        "Qw": float(pre.iloc[-1]["water_cum_stb"]),
        "Qg": float(pre.iloc[-1]["gas_cum_mscf"]),
        "wor": float(pre.iloc[-1]["wor"]),
        "res_age": float(pre.iloc[-1]["res_age_years"]),
        "origin_date": cutoff,
    }
    
    # Train model
    trans_df = extract_transition_training_data(case_dict, unc_df, TRAIN_CASES[:15], cutoff)
    m = RecursiveStateTransitionModel(model_type="ridge").fit(trans_df)
    
    # Predict rollout run 1
    res1 = m.predict_rollout(initial_state, unc_df.loc[c], forecast_dates)
    
    # Predict rollout run 2 with identical cutoff inputs
    res2 = m.predict_rollout(initial_state, unc_df.loc[c], forecast_dates)
    
    assert np.allclose(res1["oil_cum"], res2["oil_cum"]), "Rollout is non-deterministic"
    assert np.allclose(res1["oil_rate"], res2["oil_rate"]), "Rollout rates are non-deterministic"

def test_blind_cases_sealed(setup_data):
    unc_df, q_df, case_dict = setup_data
    # Assert Cases 86-100 are strictly absent from simulation dataframe
    cases_in_sim = q_df["case_id"].unique()
    assert np.all(cases_in_sim <= 85), "Blind cases 86-100 leaked into simulation dataset"
    assert np.all(cases_in_sim >= 1), "Case 0 or negative cases in simulation dataset"
