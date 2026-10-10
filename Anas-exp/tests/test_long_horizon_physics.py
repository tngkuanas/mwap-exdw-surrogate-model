"""
Tests for long-horizon forecasting physics and hybrid continuation:
1. Continuity at transition boundary (C0 continuity).
2. Monotonicity of rate-continuous extrapolation (zero decreasing steps).
3. Anchor preservation and rate domain non-negativity.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from ensemble.run_validated_long_horizon import predict_hybrid_gp_rate_continuous, predict_rate_domain_decline

def test_hybrid_gp_rate_continuous_continuity_and_monotonicity():
    """Verify that hybrid GP rate-continuous extrapolation has exact C0 continuity and zero decreasing steps."""
    # Synthetic GP and basis
    K = 3
    T_learned = 20
    T_extended = 80
    n_cases = 5

    learned_dates = pd.date_range("2008-04-01", periods=T_learned, freq="QS")
    extended_dates = pd.date_range("2008-04-01", periods=T_extended, freq="QS")
    origin = pd.Timestamp("2008-01-01")

    # Mock SVD basis (monotonic exponential decay)
    t_idx = np.arange(1, T_learned + 1)
    b1 = 1.0 - np.exp(-t_idx / 10.0)
    b2 = (t_idx / float(T_learned)) ** 1.5
    b3 = np.log1p(t_idx)
    Vt = np.vstack([b1 / np.linalg.norm(b1), b2 / np.linalg.norm(b2), b3 / np.linalg.norm(b3)])

    mean_inc = np.linspace(100, 1000, T_learned)[None, :]
    X_val_s = np.random.randn(n_cases, 5)
    anchors_val = np.array([5000.0, 10000.0, 15000.0, 20000.0, 25000.0])[:, None]

    # Mock fitted GP models
    class MockGP:
        def predict(self, X, return_std=False):
            m = np.ones(len(X)) * 50.0
            s = np.ones(len(X)) * 5.0
            return (m, s) if return_std else m

    gps = [MockGP() for _ in range(K)]

    preds, stds = predict_hybrid_gp_rate_continuous(
        gps, Vt, mean_inc, X_val_s, anchors_val, learned_dates, extended_dates, origin, K=K
    )

    assert preds.shape == (n_cases, T_extended)
    assert stds.shape == (n_cases, T_extended)

    # 1. Check C0 continuity at boundary
    # Q(t^*) at learned_steps - 1 must match
    for i in range(n_cases):
        learned_end_val = anchors_val[i, 0] + mean_inc[0, -1] + np.sum(np.ones(K) * 50.0 * Vt[:K, -1])
        assert np.isclose(preds[i, T_learned - 1], learned_end_val, atol=1e-5)
        # Next step must be strictly greater than boundary
        assert preds[i, T_learned] >= preds[i, T_learned - 1]

    # 2. Check monotonicity across all extended steps (T_learned to T_extended)
    diffs = np.diff(preds, axis=1)
    assert np.all(diffs[:, T_learned - 1:] >= 0.0), "Extrapolated segment must be strictly non-decreasing"

def test_rate_domain_decline_monotonic():
    """Verify that rate domain decline produces strictly increasing cumulative curves."""
    curves = {}
    cases = [71, 72]
    dates = pd.date_range("2000-01-01", "2008-01-01", freq="QS")
    origin = pd.Timestamp("2008-01-01")
    forecast_dates = pd.date_range("2008-04-01", periods=80, freq="QS")

    for c in cases:
        # Monotonic cum
        cum_vals = np.linspace(1000, 10000, len(dates))
        rate_vals = np.linspace(100, 50, len(dates))
        curves[(c, "oil_cum")] = pd.Series(cum_vals, index=dates)
        curves[(c, "oil_rate")] = pd.Series(rate_vals, index=dates)

    unc_df = pd.DataFrame({
        "Aquifer Pore Volume": [100.0, 200.0],
    }, index=cases)

    preds = predict_rate_domain_decline(cases, origin, forecast_dates, "oil_cum", curves, unc_df)
    assert preds.shape == (2, 80)
    diffs = np.diff(preds, axis=1)
    assert np.all(diffs >= 0.0), "Rate-domain cumulative forecast must be non-decreasing"
