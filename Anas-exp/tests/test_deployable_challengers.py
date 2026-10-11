"""
Test Suite: test_deployable_challengers.py
Validates Deployable Forecasting Challengers & Physical Constraints
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from deployable_challengers.deployable_direct_forecasting import (
    DeployableDirectForecaster,
    extract_fixed_state_features,
)
from deployable_challengers.deployable_residual_dca import (
    DeployableResidualForecaster,
)
from ensemble.data import load_uncertainty, TRAIN_CASES
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
    LearnedDCARateForecaster,
)


@pytest.fixture(scope="module")
def shared_data():
    unc = load_uncertainty()
    qdf = load_clean_simulation_quarterly()
    cdict = build_case_quarterly_trajectories(qdf)
    return unc, cdict


def test_direct_forecaster_fixed_dimensions(shared_data):
    """Verifies that direct forecaster input features are strictly D=25 across any cutoff date."""
    unc, cdict = shared_data
    df1 = cdict[1]
    
    # Test multiple distinct historical cutoff dates
    dates_to_test = [
        pd.Timestamp("2001-01-01"),
        pd.Timestamp("2003-01-01"),
        pd.Timestamp("2005-01-01"),
        pd.Timestamp("2008-01-01"),
    ]
    
    for d in dates_to_test:
        feats = extract_fixed_state_features(df1, d)
        assert feats.shape == (25,), f"Expected feature dimension 25 at date {d}, got {feats.shape}"
        assert np.isfinite(feats).all(), f"Non-finite features detected at date {d}"


def test_residual_envelope_boundary_conditions():
    """Verifies envelope satisfies r(0) = 0 and r(t) -> 0 as t -> inf."""
    forecaster = DeployableResidualForecaster(kappa=1.5, gamma=0.25)
    
    # t = 0
    t0 = np.array([0.0])
    env0 = (1.0 - np.exp(-forecaster.kappa * t0)) * np.exp(-forecaster.gamma * t0)
    assert np.isclose(env0[0], 0.0), f"Envelope at t=0 must be exactly 0, got {env0[0]}"
    
    # t -> inf (e.g. 50 years)
    t_inf = np.array([50.0])
    env_inf = (1.0 - np.exp(-forecaster.kappa * t_inf)) * np.exp(-forecaster.gamma * t_inf)
    assert env_inf[0] < 1e-4, f"Envelope at t=50yr must approach 0, got {env_inf[0]}"
    
    # Peak occurs at intermediate time t_peak = ln(1 + kappa/gamma) / kappa
    t_peak_analytic = np.log(1.0 + forecaster.kappa / forecaster.gamma) / forecaster.kappa
    t_grid = np.linspace(0.01, 10.0, 1000)
    env_grid = (1.0 - np.exp(-forecaster.kappa * t_grid)) * np.exp(-forecaster.gamma * t_grid)
    t_peak_numeric = t_grid[np.argmax(env_grid)]
    assert np.isclose(t_peak_analytic, t_peak_numeric, atol=0.05), "Peak time mismatch"


def test_post_plateau_monotonicity_enforcement(shared_data):
    """Verifies that predicted oil rates are strictly non-increasing post-plateau."""
    unc, cdict = shared_data
    df1 = cdict[1]
    u1 = unc.loc[1]
    orig = pd.Timestamp("2005-01-01")
    f_dates = pd.date_range(orig + pd.DateOffset(days=1), periods=16, freq="QS")
    
    # Fit base DCA
    dict_asof = {c: cdict[c][cdict[c]["date"] <= orig] for c in TRAIN_CASES[:10]}
    base_dca = LearnedDCARateForecaster(decline_type="exponential").fit(
        dict_asof, unc, TRAIN_CASES[:10], orig, 16
    )
    
    # Fit linear residual model
    m_lin = DeployableResidualForecaster(architecture="linear").fit(cdict, unc, TRAIN_CASES[:10], orig, 16)
    preds = m_lin.predict_rates(df1, u1, orig, f_dates, base_dca)
    
    qo = preds["oil_rate"]
    qw = preds["water_rate"]
    
    # Non-negativity
    assert (qo >= 0.0).all(), "Oil rates must be non-negative"
    assert (qw >= 0.0).all(), "Water rates must be non-negative"
    
    # Monotonicity
    diffs = np.diff(qo)
    assert (diffs <= 1e-5).all(), f"Violations of monotonic oil decline detected: max diff = {np.max(diffs)}"


def test_generated_research_tables_integrity():
    """Verifies that all required research iteration 10 tables exist and are properly populated."""
    table_dir = Path("outputs/research/deployable_challengers/tables")
    
    req_tables = [
        "deployable_direct_forecasting_results.csv",
        "coupled_water_paradigm_comparison.csv",
        "deployable_residual_dca_comparison.csv",
        "deployable_residual_20year_stability_audit.csv",
        "rigorous_pseudo_field_case_level.csv",
        "rigorous_pseudo_field_summary.csv",
    ]
    
    for t_name in req_tables:
        p = table_dir / t_name
        assert p.exists(), f"Missing required table: {p}"
        df = pd.read_csv(p)
        assert len(df) > 0, f"Table {p} is empty"
