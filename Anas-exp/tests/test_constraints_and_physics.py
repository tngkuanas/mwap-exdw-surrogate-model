"""
Tests for physical constraints: anchor preservation, monotonicity, and water caps.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ensemble.constraints import apply_physical_constraints
from ensemble.metrics import audit_physical_violations

def test_anchor_preservation():
    """Tests that physical constraints enforce cumulative floor at historical anchor."""
    dates = pd.date_range("2003-04-01", "2006-01-01", freq="QS")
    fake_preds = pd.DataFrame({
        "model_id": "test",
        "case_num": 1,
        "cutoff": pd.Timestamp("2003-01-01"),
        "origin": pd.Timestamp("2003-01-01"),
        "horizon_years": 3,
        "date": dates,
        "phase": "oil_cum",
        "prediction": [10.0, 5.0, 2.0, 1.0, 0.0, -5.0, -10.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    
    # Fake curves dict with anchor = 50.0
    curves = {(1, "oil_cum"): pd.Series([50.0], index=[pd.Timestamp("2003-01-01")])}
    
    constrained, audit = apply_physical_constraints(fake_preds, curves=curves, enforce_monotonic=True)
    
    # All predictions should be at least 50.0
    assert (constrained["prediction"] >= 50.0).all(), "Predictions breached anchor floor!"
    # Must be monotonically non-decreasing
    assert (constrained["prediction"].diff().dropna() >= -1e-8).all(), "Predictions are not monotonic!"

def test_water_rate_capping():
    """Tests that runaway water rates are clamped at allowable rate ceiling."""
    cutoff = pd.Timestamp("2008-01-01")
    dates = pd.date_range("2008-04-01", "2028-01-01", freq="QS")
    dt_days = (dates - cutoff).total_seconds() / 86400.0
    
    # Exploding water prediction
    fake_water = pd.DataFrame({
        "model_id": "test_water",
        "case_num": 1,
        "cutoff": cutoff,
        "origin": cutoff,
        "horizon_years": 20,
        "date": dates,
        "phase": "water_cum",
        "prediction": 1000.0 + 1e6 * dt_days,  # 1,000,000 STB/day
    })
    
    # Observed water rate = 500 STB/day
    curves = {
        (1, "water_cum"): pd.Series([1000.0], index=[cutoff]),
        (1, "water_rate"): pd.Series([500.0], index=[cutoff]),
    }
    
    # Cap 2x = 1,000 STB/day
    constrained, audit = apply_physical_constraints(fake_water, curves=curves, water_cap_mult=2.0)
    
    max_allowed = 1000.0 + (2.0 * 500.0) * dt_days
    assert (constrained["prediction"] <= max_allowed + 1e-4).all(), "Water cap failed to bind!"
    assert audit["water_cap_count"].iloc[0] > 0, "Water cap was not triggered!"

def test_validation_predictions_clean():
    """Verifies that generated validation predictions have zero negative values."""
    val_path = Path("outputs/ensemble/validation_predictions.parquet") if Path("outputs/ensemble/validation_predictions.parquet").exists() else Path("Anas-exp/outputs/ensemble/validation_predictions.parquet")
    assert val_path.exists(), "Missing validation_predictions.parquet"
    val = pd.read_parquet(val_path)
    viol = audit_physical_violations(val)
    assert viol["negative_count"] == 0, f"Found negative values in validation: {viol['negative_count']}"
    assert viol["nan_count"] == 0, f"Found NaN values in validation: {viol['nan_count']}"
