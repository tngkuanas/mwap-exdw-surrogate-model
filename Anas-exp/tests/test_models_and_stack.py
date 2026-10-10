"""
Tests for candidate models, meta-learners, and reproducibility.
"""
import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path

from ensemble.stacking import SimpleAverageEnsemble, NonNegativeWeightedEnsemble, RidgeMetaEnsemble

def test_meta_learner_simplex_weights():
    """Verifies that non-negative weighted meta-learner produces simplex weights (sum=1, w>=0)."""
    np.random.seed(42)
    N = 100
    truth = np.random.uniform(10, 50, N)
    pred1 = truth + np.random.normal(0, 1, N)
    pred2 = truth + np.random.normal(0, 2, N)
    pred3 = truth + np.random.normal(0, 3, N)
    
    meta_df = pd.DataFrame({
        "phase": ["oil_cum"] * N,
        "truth": truth,
        "m1": pred1,
        "m2": pred2,
        "m3": pred3,
    })
    
    ensemble = NonNegativeWeightedEnsemble()
    ensemble.fit(meta_df, ["m1", "m2", "m3"], truth_col="truth")
    
    weights = ensemble.weights["oil_cum"]
    assert len(weights) == 3
    assert np.isclose(np.sum(weights), 1.0, atol=1e-4), "Weights do not sum to 1!"
    assert (weights >= -1e-6).all(), "Negative weights detected!"

def test_artifacts_exist_and_nonempty():
    """Verifies that all required artifacts from master task exist and have content."""
    base_dir = Path("outputs/ensemble") if Path("outputs/ensemble").exists() else Path("Anas-exp/outputs/ensemble")
    required_files = [
        "eda_audit.md",
        "model_configurations.json",
        "oof_predictions.parquet",
        "validation_predictions.parquet",
        "base_model_metrics.csv",
        "horizon_metrics.csv",
        "residual_diversity.csv",
        "constraint_sensitivity.csv",
        "stack_comparison.csv",
        "validation_report.md",
        "reproducibility.sh",
    ]
    for f in required_files:
        p = base_dir / f
        assert p.exists(), f"Missing required file: {f}"
        assert p.stat().st_size > 0, f"File {f} is empty!"

def test_model_configurations_valid_json():
    """Verifies that model_configurations.json is valid and contains core four settings."""
    cfg_path = Path("outputs/ensemble/model_configurations.json") if Path("outputs/ensemble/model_configurations.json").exists() else Path("Anas-exp/outputs/ensemble/model_configurations.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    assert "core_four_families" in cfg
    assert len(cfg["core_four_families"]) == 4
    assert cfg["training_cases"] == 70
    assert cfg["validation_cases"] == 15
