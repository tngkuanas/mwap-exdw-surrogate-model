"""
Test Suite: test_forecast_reliability_challenge.py
Verifies numerical integrity, accounting semantics, physical consistency, and non-regression
for the Final Research Sprint: Multi-Angle Forecast Reliability Challenge.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ensemble.data import load_uncertainty, TRAIN_CASES, PARAMS
from ensemble.metrics import compute_increment_nrmse, compute_bias
from forecast_reliability_challenge.waterflood_coupled_challenger import (
    CoupledLogitWaterfloodForecaster,
    CoupledWORWaterfloodForecaster,
)


def test_locked_ensemble_config_integrity():
    """Verifies that locked ensemble weights and structure remain strictly untouched."""
    cfg_path = Path("src/ensemble/locked_ensemble_config.json")
    assert cfg_path.exists(), "Locked ensemble config missing!"
    with open(cfg_path) as f:
        cfg = json.load(f)
        
    assert cfg["model_name"] == "Blend_Phase_Specific_Simplex"
    assert cfg["status"] == "FROZEN"
    assert abs(sum(cfg["weights"]["oil"].values()) - 1.0) < 1e-6
    assert abs(sum(cfg["weights"]["water"].values()) - 1.0) < 1e-6
    assert cfg["physics"]["solution_gas_oil_ratio_rs_mscf_per_stb"] == 0.3633


def test_production_accounting_quadrature_reconciliation():
    """Verifies that right-endpoint integration aligns with simulator cumulative production to within 1%."""
    audit_table = Path("outputs/research/forecast_reliability_challenge/tables/accounting_audit_simulation_reconciliation.csv")
    assert audit_table.exists(), "Accounting reconciliation table not found!"
    df = pd.read_csv(audit_table)
    
    oil = df[df["phase"] == "oil"]
    mean_re_err = oil["right_endpoint_mape_pct"].mean()
    mean_le_err = oil["left_endpoint_mape_pct"].mean()
    mean_trap_err = oil["trapezoidal_mape_pct"].mean()
    
    # Right-endpoint error across all 85 cases must be < 1% (MAPE = 0.8356%)
    assert mean_re_err < 1.0, f"Right-endpoint error {mean_re_err:.4f}% exceeds 1% threshold!"
    
    # Right-endpoint error must be substantially smaller than left-endpoint and trapezoidal
    assert mean_re_err < mean_le_err, "Right-endpoint should be more accurate than left-endpoint!"
    assert mean_re_err < mean_trap_err, "Right-endpoint should be more accurate than trapezoidal!"


def test_calendar_vs_fixed_quarter_days():
    """Verifies that fixed dt = 91.3125 matches actual calendar days over 80 quarters to 0.0 days."""
    dates = pd.date_range("2008-01-01", "2028-01-01", freq="QS")
    total_calendar_days = (dates[-1] - dates[0]).total_seconds() / 86400.0
    total_fixed_days = 80 * 91.3125
    assert abs(total_calendar_days - total_fixed_days) < 1e-4, (
        f"Drift detected: calendar={total_calendar_days}, fixed={total_fixed_days}"
    )


def test_waterflood_mass_conservation_and_bounds():
    """Verifies that coupled waterflood models strictly conserve mass and bound water cut."""
    model_logit = CoupledLogitWaterfloodForecaster(fw_max=0.98)
    initial_state = {
        "qo": 5000.0,
        "qw": 500.0,
        "res_age": 10.0,
        "origin_date": pd.Timestamp("2008-01-01"),
    }
    u_c = pd.Series({
        "Fault Transmissibility": 0.5,
        "Porosity Multiplier": 1.0,
        "Permeability Multiplier": 1.0,
        "Aquifer Pore Volume": 100.0,
    })
    f_dates = pd.date_range("2008-04-01", periods=40, freq="QS")
    
    # Mock model weights with 1D coefficients
    model_logit.model_DL.coef_ = np.zeros(7)
    model_logit.model_DL.intercept_ = 0.05
    model_logit.model_alpha.coef_ = np.zeros(7)
    model_logit.model_alpha.intercept_ = 0.10
    
    res = model_logit.predict_rates(initial_state, u_c, f_dates)
    
    # Mass conservation: q_o + q_w == q_L
    sum_rates = res["oil_rate"] + res["water_rate"]
    np.testing.assert_allclose(sum_rates, res["liquid_rate"], rtol=1e-5)
    
    # Water cut bounds: 0 <= f_w <= 0.98
    assert np.all(res["water_cut"] >= 0.0)
    assert np.all(res["water_cut"] <= 0.98)
    
    # Gas coupling
    np.testing.assert_allclose(res["gas_rate"], 0.3633 * res["oil_rate"], rtol=1e-5)


def test_nrmse_metric_function_behavior():
    """Verifies referee NRMSE metric calculation with true increments."""
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([105.0, 195.0, 310.0, 390.0])
    err = y_pred - y_true
    
    nrmse = compute_increment_nrmse(err, y_true)
    norm = np.sqrt(np.mean(y_true ** 2))
    rmse = np.sqrt(np.mean(err ** 2))
    expected = rmse / norm
    
    assert abs(nrmse - expected) < 1e-6
    assert nrmse < 0.05, "Small perturbation should yield low NRMSE"
