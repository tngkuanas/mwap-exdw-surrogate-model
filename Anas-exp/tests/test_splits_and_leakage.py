"""
Tests for case-level split integrity and temporal leakage protection.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ensemble.data import (
    load_uncertainty,
    get_case_folds,
    load_forecast_truth,
    TRAIN_CASES,
    VAL_CASES,
    TEST_CASES,
)

def test_case_splits_mutually_exclusive():
    """Verifies that train, validation, and test splits are strictly disjoint."""
    train_set = set(TRAIN_CASES)
    val_set = set(VAL_CASES)
    test_set = set(TEST_CASES)
    
    assert len(train_set) == 70
    assert len(val_set) == 15
    assert len(test_set) == 15
    
    assert train_set.isdisjoint(val_set), "Train and validation sets overlap!"
    assert train_set.isdisjoint(test_set), "Train and test sets overlap!"
    assert val_set.isdisjoint(test_set), "Validation and test sets overlap!"

def test_five_fold_partition():
    """Verifies that 5-fold CV partitions Cases 1-70 with 14 cases per fold."""
    folds_df = get_case_folds(n_splits=5, seed=42)
    assert len(folds_df) == 70
    assert set(folds_df["fold"]) == {0, 1, 2, 3, 4}
    
    counts = folds_df["fold"].value_counts()
    for count in counts:
        assert count == 14, f"Fold count not 14: {count}"

def test_no_future_leakage_in_forecast_truth():
    """Verifies that forecast truth timestamps are strictly after the cutoff origin."""
    truth = load_forecast_truth()
    diffs = (truth["date"] - truth["cutoff"]).dt.total_seconds()
    assert (diffs > 0).all(), "Truth contains timestamps at or before cutoff date!"

def test_oof_meta_features_leakage_free():
    """Verifies that OOF predictions file only contains training cases 1-70."""
    oof_path = Path("outputs/ensemble/oof_predictions.parquet") if Path("outputs/ensemble/oof_predictions.parquet").exists() else Path("Anas-exp/outputs/ensemble/oof_predictions.parquet")
    assert oof_path.exists(), "Missing oof_predictions.parquet"
    oof = pd.read_parquet(oof_path)
    
    assert set(oof["case_num"]).issubset(set(TRAIN_CASES)), "OOF predictions contain non-training cases!"
