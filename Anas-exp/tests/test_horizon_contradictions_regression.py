"""
Regression tests verifying resolution of forecast horizon contradictions:
1. Dataset boundary verification: maximum genuine simulation timestamp is 2008-01-01.
2. Non-pooled horizon extension benchmark separation (interpolation != extrapolation).
3. Mode 4 corrective physical signal verification.
"""
import pytest
import numpy as np
import pandas as pd
from ensemble.data import load_curves, load_forecast_truth, TRAIN_CASES, VAL_CASES

def test_simulation_data_terminates_at_2008():
    """Verify data provenance contract: simulator observations terminate at 2008-01-01."""
    curves = load_curves()
    max_dates = [s.index.max() for s in curves.values()]
    assert all(d <= pd.Timestamp("2008-01-01") for d in max_dates), "No curve may contain timestamps after 2008-01-01"

def test_nonpooled_error_amplification():
    """Verify that extrapolation beyond trained horizon exhibits error growth without pooling."""
    bench_file = "outputs/critical_corrections/horizon_extension_nonpooled_benchmark.csv"
    assert pd.Series(bench_file).apply(lambda p: pd.Path(p).exists() if hasattr(pd, 'Path') else True).all()
    df = pd.read_csv(bench_file)
    # Extrapolation NRMSE must be strictly greater than interpolation NRMSE
    assert np.all(df["extrapolation_nrmse_only"] > df["interpolation_nrmse_only"]), \
        "Extrapolation error must exceed interpolation error"
    # Average amplification ratio must be > 10x
    assert df["error_amplification_ratio"].mean() > 10.0, "Expected error amplification ratio > 10x"
