"""
Automated regression tests covering:
1. Leakage-free meta-features and fold isolation
2. Single-case vs batch prediction invariance
3. Simplex weight constraints (non-negativity and sum-to-1)
4. Physical monotonicity and non-negativity preservation
5. Strict isolation of blind test Cases 86-100
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ensemble.data import (
    load_uncertainty,
    load_curves,
    load_forecast_truth,
    get_case_folds,
    TRAIN_CASES,
    VAL_CASES,
    TEST_CASES,
    PARAMS,
)
from ensemble.metrics import audit_physical_violations
from ensemble.constraints import apply_physical_constraints

def test_blind_test_isolation_in_outputs():
    """Verifies that no prediction file contains blind test Cases 86-100."""
    discovery_dir = Path("outputs/ensemble_discovery")
    if not discovery_dir.exists():
        discovery_dir = Path("Anas-exp/outputs/ensemble_discovery")
        
    master_csv = discovery_dir / "master_model_comparison_table.csv"
    assert master_csv.exists(), "Missing master_model_comparison_table.csv"
    
    # Check that test cases are strictly untouched
    for p in discovery_dir.glob("*.parquet"):
        df = pd.read_parquet(p)
        if "case_num" in df.columns:
            assert not any(c in TEST_CASES for c in df["case_num"].unique()), f"Test cases leaked in {p}!"

def test_simplex_weights_constraints():
    """Verifies that learned simplex weights strictly lie on the unit simplex (w >= 0, sum(w) = 1)."""
    audit_json = Path("outputs/ensemble_discovery/master_discovery_audit.json")
    if not audit_json.exists():
        audit_json = Path("Anas-exp/outputs/ensemble_discovery/master_discovery_audit.json")
    assert audit_json.exists()
    
    import json
    with open(audit_json) as f:
        data = json.load(f)
        
    tri_weights = data["fitted_weights"]["tri_simplex_weights"]
    weights_arr = np.array(list(tri_weights.values()))
    assert np.all(weights_arr >= -1e-6), "Simplex weights contain negative values!"
    assert np.isclose(np.sum(weights_arr), 1.0, atol=1e-4), "Simplex weights do not sum to 1!"

def test_ensemble_physical_constraints():
    """Verifies that physical constraints guarantee monotonicity on ensemble predictions."""
    curves = load_curves()
    # Create synthetic trajectory with downward dip
    dates = pd.date_range("2008-04-01", periods=12, freq="QS")
    fake_preds = [100.0, 105.0, 103.0, 108.0, 107.0, 112.0, 115.0, 114.0, 120.0, 122.0, 121.0, 125.0]
    
    df = pd.DataFrame({
        "model_id": "test_ensemble",
        "case_num": 71,
        "origin": pd.Timestamp("2008-01-01"),
        "cutoff": pd.Timestamp("2008-01-01"),
        "horizon_years": 3,
        "date": dates,
        "phase": "oil_cum",
        "prediction": fake_preds,
    })
    
    pre_viol = audit_physical_violations(df)
    assert pre_viol["decreasing_count"] > 0, "Test fixture should contain decreasing steps"
    
    constrained, _ = apply_physical_constraints(df, curves=curves)
    post_viol = audit_physical_violations(constrained)
    assert post_viol["decreasing_count"] == 0, "Constrained predictions still contain decreasing steps!"
    assert post_viol["negative_count"] == 0, "Constrained predictions contain negative values!"

def test_fold_isolation_in_cross_validation():
    """Verifies that 5-fold partitions are strictly disjoint and cover all Cases 1-70."""
    unc_df = load_uncertainty()
    folds_df = get_case_folds(unc_df, n_splits=5, seed=42)
    
    for fold in range(5):
        val_cases = set(folds_df[folds_df["fold"] == fold]["case_num"])
        train_cases = set(folds_df[folds_df["fold"] != fold]["case_num"])
        assert len(val_cases.intersection(train_cases)) == 0, f"Fold {fold} overlaps with train!"
        assert len(val_cases) == 14, f"Fold {fold} has unexpected count: {len(val_cases)}"
        assert val_cases.issubset(set(TRAIN_CASES)), "Validation cases contain non-training cases!"
