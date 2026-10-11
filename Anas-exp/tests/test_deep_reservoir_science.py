"""
Regression Test Suite for Deep Reservoir Science Research.
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ensemble.data import load_uncertainty, TRAIN_CASES, VAL_CASES, PARAMS
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
    LearnedDCARateForecaster,
)
from deep_reservoir_science.pseudo_field_validation import FastLeaveOutEmulator, invert_geology_for_field, PARAM_BOUNDS
from deep_reservoir_science.direct_forecasting_satija import SatijaDirectForecaster
from deep_reservoir_science.physics_normalized_state import extract_physics_normalized_panel, PhysicsNormalizedSVR
from deep_reservoir_science.physics_constrained_residuals import ConstrainedResidualForecaster
from deep_reservoir_science.forecast_scenario_selection import select_scenarios_by_policy, compute_historical_misfits
from deep_reservoir_science.reduced_order_compartment import compute_identifiability_fisher_matrix, ReducedOrderCompartmentModel


@pytest.fixture(scope="module")
def shared_data():
    unc = load_uncertainty()
    qdf = load_clean_simulation_quarterly()
    cdict = build_case_quarterly_trajectories(qdf)
    return unc, qdf, cdict


def test_pseudo_field_emulator_and_inversion(shared_data):
    unc, _, cdict = shared_data
    train_subset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    orig = pd.Timestamp("2005-01-01")
    
    X_tr = unc.loc[train_subset][PARAMS].values
    Yo_tr = np.array([cdict[c][cdict[c]["date"] <= orig]["oil_rate_stbd"].values for c in train_subset])
    Yw_tr = np.array([cdict[c][cdict[c]["date"] <= orig]["water_rate_stbd"].values for c in train_subset])
    
    emulator = FastLeaveOutEmulator(n_components=2, random_state=42).fit(X_tr, Yo_tr, Yw_tr)
    yo_pred, yw_pred = emulator.predict(X_tr[:2])
    assert yo_pred.shape == (2, Yo_tr.shape[1])
    assert yw_pred.shape == (2, Yw_tr.shape[1])
    
    # Inversion within bounds
    target_qo = cdict[11][cdict[11]["date"] <= orig]["oil_rate_stbd"].values
    target_qw = cdict[11][cdict[11]["date"] <= orig]["water_rate_stbd"].values
    theta_inv = invert_geology_for_field(emulator, target_qo, target_qw, X_tr)
    assert len(theta_inv) == 4
    for k, (low, high) in enumerate(PARAM_BOUNDS):
        assert low <= theta_inv[k] <= high


def test_satija_direct_forecaster_constraints(shared_data):
    unc, _, cdict = shared_data
    train_subset = [1, 2, 3, 4, 5]
    orig = pd.Timestamp("2005-01-01")
    hq = 12
    f_dates = pd.date_range(orig + pd.DateOffset(days=1), periods=hq, freq="QS")
    
    forecaster = SatijaDirectForecaster(n_hist_comp=3, n_fut_comp=2, method="ridge")
    forecaster.fit(cdict, train_subset, orig, hq)
    
    df_target = cdict[71]
    pre = df_target[df_target["date"] <= orig]
    last = pre.iloc[-1]
    init_st = {
        "qo": float(last["oil_rate_stbd"]),
        "qw": float(last["water_rate_stbd"]),
        "Qo": float(last["oil_cum_stb"]),
        "Qw": float(last["water_cum_stb"]),
        "origin_date": orig,
    }
    
    preds = forecaster.predict(df_target, orig, init_st, f_dates)
    assert len(preds["oil_rate"]) == hq
    assert len(preds["water_rate"]) == hq
    # Non-negativity
    assert (preds["oil_rate"] >= 0.0).all()
    assert (preds["water_rate"] >= 0.0).all()
    # Monotonic oil decline
    diffs = np.diff(preds["oil_rate"])
    assert (diffs <= 1e-4).all()


def test_physics_normalized_panel_features(shared_data):
    unc, _, cdict = shared_data
    sub_dict = {1: cdict[1], 2: cdict[2]}
    panel = extract_physics_normalized_panel(sub_dict, unc)
    
    expected_cols = ["rf", "wf", "wc", "logit_wc", "D_recent", "B_recent"]
    for col in expected_cols:
        assert col in panel.columns
    # Recovery factor and water cut bounds
    assert (panel["rf"] >= 0.0).all()
    assert (panel["wc"] >= 0.0).all()
    assert (panel["wc"] <= 1.0).all()


def test_constrained_residual_decay_envelope(shared_data):
    unc, _, cdict = shared_data
    orig = pd.Timestamp("2004-01-01")
    f_dates_long = pd.date_range(orig + pd.DateOffset(days=1), periods=80, freq="QS")
    
    m_dca = LearnedDCARateForecaster(decline_type="exponential").fit(
        {c: cdict[c][cdict[c]["date"] <= orig] for c in [1, 2, 3, 4, 5]},
        unc, [1, 2, 3, 4, 5], orig, 16
    )
    
    m_res = ConstrainedResidualForecaster(residual_type="damped_linear", gamma_decay=0.25).fit(
        cdict, unc, [1, 2, 3, 4, 5], orig, 16, m_dca
    )
    
    last = cdict[71][cdict[71]["date"] <= orig].iloc[-1]
    u_c = unc.loc[71]
    init_st = {
        "qo": float(last["oil_rate_stbd"]),
        "qw": float(last["water_rate_stbd"]),
        "qg": float(last["gas_rate_mscfd"]),
        "Qo": float(last["oil_cum_stb"]),
        "Qw": float(last["water_cum_stb"]),
        "Qg": float(last["gas_cum_mscf"]),
        "wor": float(last["wor"]),
        "res_age": float(last["res_age_years"]),
        "origin_date": orig,
    }
    
    preds = m_res.predict_rates(init_st, u_c, f_dates_long, m_dca)
    # Check that residual damping approaches 0 at late times
    assert preds["residual_damping"][-1] < 0.01  # < 1% of initial residual at year 20
    # Rates remain positive and finite
    assert (preds["oil_rate"] >= 0.0).all()
    assert np.isfinite(preds["oil_rate"]).all()


def test_scenario_selection_policies():
    df_misfits = pd.DataFrame({
        "candidate_id": list(range(1, 31)),
        "total_misfit": np.linspace(0.1, 3.0, 30),
    })
    cand_qo = {c: np.ones(20) * (100 - c) for c in range(1, 31)}
    
    policies = [
        "ten_best_historical",
        "posterior_l2_weighting",
        "forecast_validated_weighting",
        "representative_clustering",
    ]
    
    for pol in policies:
        cands, weights = select_scenarios_by_policy(df_misfits, cand_qo, pol, n_scenarios=10)
        assert len(cands) == 10
        assert len(weights) == 10
        assert np.isclose(np.sum(weights), 1.0)
        assert (weights >= 0.0).all()


def test_reduced_order_identifiability_conditioning():
    res = compute_identifiability_fisher_matrix()
    # Confirm rank deficiency and extreme ill-conditioning
    assert res["condition_number"] > 1e6
    assert res["effective_rank"] < res["total_parameters"]


def test_generated_tables_and_figures_exist():
    tab_dir = Path("outputs/research/deep_reservoir_science/tables")
    fig_dir = Path("outputs/research/deep_reservoir_science/figures")
    
    expected_tables = [
        "pseudo_field_inversion_vs_forecasting_audit.csv",
        "direct_forecasting_satija_results.csv",
        "physics_normalized_state_comparison.csv",
        "physics_constrained_residual_comparison.csv",
        "residual_20year_stability_audit.csv",
        "scenario_selection_comparison.csv",
        "reduced_order_identifiability_analysis.csv",
        "unified_challengers_master_table.csv",
    ]
    
    for t_name in expected_tables:
        p = tab_dir / t_name
        assert p.exists(), f"Missing table: {p}"
        assert p.stat().st_size > 0, f"Empty table: {p}"
        
    expected_figures = [
        "fig1_pseudo_field_error_decomposition.png",
        "fig2_direct_forecasting_satija_comparison.png",
        "fig3_physics_normalized_state_audit.png",
        "fig4_constrained_residuals_and_stability.png",
        "fig5_scenario_selection_tradeoffs.png",
        "fig6_reduced_order_identifiability.png",
    ]
    
    for f_name in expected_figures:
        p = fig_dir / f_name
        assert p.exists(), f"Missing figure: {p}"
        assert p.stat().st_size > 1000, f"Invalid figure: {p}"
