"""
Regression tests for targeted audit fixes:
1. Simplex weight optimization convergence (no silent SLSQP line-search failure).
2. Physical violation audit grouping isolation (no cross-model date interleaving).
3. Softmax simplex optimization recovery on normalized increments.
"""
import pytest
import numpy as np
import pandas as pd
from ensemble.metrics import audit_physical_violations
from ensemble.correct_and_audit_stacking import fit_simplex_softmax

def test_audit_physical_violations_model_isolation():
    """Ensure audit_physical_violations isolates different models and ranks so tied dates do not register negative jumps."""
    dates = pd.date_range("2008-04-01", periods=10, freq="QS")
    rows = []
    # Model A has high values, Model B has lower values
    for t_idx, d in enumerate(dates):
        rows.append({
            "model_id": "Model_A",
            "case_num": 1,
            "phase": "oil_cum",
            "date": d,
            "prediction": 100.0 + t_idx * 5.0,
        })
        rows.append({
            "model_id": "Model_B",
            "case_num": 1,
            "phase": "oil_cum",
            "date": d,
            "prediction": 50.0 + t_idx * 5.0,
        })
    df = pd.DataFrame(rows)
    # When grouped by model_id, each model is strictly increasing: 0 violations
    res = audit_physical_violations(df)
    assert res["decreasing_count"] == 0, f"Expected 0 violations with model isolation, got {res['decreasing_count']}"

def test_fit_simplex_softmax_convergence():
    """Ensure softmax simplex optimization stably recovers known weights on normalized increments."""
    rng = np.random.default_rng(42)
    N = 200
    # Two base models: Model 1 is accurate, Model 2 is noisy
    true_y = rng.normal(0, 1, size=N)
    pred_1 = true_y + rng.normal(0, 0.05, size=N)
    pred_2 = true_y + rng.normal(0, 0.50, size=N)
    P = np.column_stack([pred_1, pred_2])

    w = fit_simplex_softmax(P, true_y)
    assert len(w) == 2
    assert np.isclose(np.sum(w), 1.0, atol=1e-5)
    assert np.all(w >= 0.0)
    # Model 1 should receive > 80% weight
    assert w[0] > 0.80, f"Expected Model 1 to receive > 80% weight, got {w[0]:.3f}"
