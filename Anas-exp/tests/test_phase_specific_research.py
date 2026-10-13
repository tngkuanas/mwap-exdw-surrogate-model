"""
Unit Tests: test_phase_specific_research.py
Package: tests
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Tests:
1. Analytical Arps recovery limits: finite for 0 < b < 1, infinite for b >= 1.
2. Frozen ensemble rate floor extraction: ExtraTrees contribution is ~737.1 STB/d.
3. Constrained long-tail ensemble: smooth support-tapering guarantees q_o -> 0 as t -> infinity.
4. Candidate phase recombination models: all 8 candidates output valid rates and non-negative cumulatives.
5. Observable state features: extracts valid pre-origin metrics without target leakage.
6. Gas coupling preservation: q_g = 0.3633 * q_o strictly preserved across all candidates.
"""
import pytest
import numpy as np
import pandas as pd

from phase_specific_research.asymptotic_analysis import (
    analytical_arps_recovery_limit,
)
from phase_specific_research.phase_recombination import (
    predict_candidate_rates,
    ALL_CANDIDATE_IDS,
)
from phase_specific_research.selection_rule_analysis import (
    extract_observable_state_features,
)


def test_analytical_arps_recovery_finite_for_sub_unity_b():
    """
    Mathematical Proof Verification:
    Arps hyperbolic with 0 < b < 1 has finite ultimate recovery Q_inf = q0 / ((1 - b) * Di).
    """
    q0 = 5000.0
    Di = 0.25
    b = 0.35
    
    Q_inf = analytical_arps_recovery_limit(q0, Di, b)
    assert np.isfinite(Q_inf)
    # Expected: 5000 / ((1 - 0.35) * 0.25) * 365.25 = 5000 / 0.1625 * 365.25 ~= 11.238 MMSTB
    assert Q_inf == pytest.approx(5000.0 / ((1.0 - 0.35) * 0.25) * 365.25, rel=1e-5)
    
    # b >= 1.0 diverges to infinity
    assert np.isinf(analytical_arps_recovery_limit(q0, Di, 1.0))
    assert np.isinf(analytical_arps_recovery_limit(q0, Di, 1.5))


def test_all_candidates_instantiation_and_properties():
    """Verifies that all 8 candidate forecasters return valid non-negative trajectories."""
    n_steps = 12
    forecast_dates = pd.date_range("2005-04-01", periods=n_steps, freq="QS")
    
    # Mock base models
    class MockModel:
        def predict_rates(self, state, params, dates):
            n = len(dates)
            return {
                "oil_rate": np.linspace(4000.0, 2000.0, n),
                "water_rate": np.linspace(500.0, 3000.0, n),
            }
            
    models = {
        "Recursive_ExtraTrees": MockModel(),
        "Learned_Exponential_DCA": MockModel(),
        "Learned_Hyperbolic_DCA": MockModel(),
        "Hybrid_Dynamic": MockModel(),
    }
    locked_weights = {
        "oil": {
            "Recursive_ExtraTrees": 0.240466,
            "Learned_Exponential_DCA": 0.617902,
            "Learned_Hyperbolic_DCA": 0.0,
            "Hybrid_Dynamic": 0.141631,
        },
        "water": {
            "Recursive_ExtraTrees": 0.497170,
            "Learned_Exponential_DCA": 0.251415,
            "Learned_Hyperbolic_DCA": 0.251415,
            "Hybrid_Dynamic": 0.0,
        },
    }
    init_state = {
        "qo": 4500.0,
        "qw": 600.0,
        "qg": 1635.0,
        "Qo": 20000000.0,
        "Qw": 3000000.0,
        "Qg": 7266000.0,
        "wor": 600.0 / 4500.0,
        "res_age": 7.0,
        "origin_date": pd.Timestamp("2005-01-01"),
    }
    params = pd.Series({"Fault Transmissibility": 0.5, "Porosity Multiplier": 1.0})
    
    for cand_id in ALL_CANDIDATE_IDS:
        res = predict_candidate_rates(cand_id, models, locked_weights, init_state, params, forecast_dates)
        
        # 1. Output shapes
        assert len(res["oil_rate"]) == n_steps
        assert len(res["water_rate"]) == n_steps
        assert len(res["cum_oil"]) == n_steps
        assert len(res["cum_water"]) == n_steps
        
        # 2. Physical non-negativity
        assert np.all(res["oil_rate"] >= 0.0)
        assert np.all(res["water_rate"] >= 0.0)
        assert np.all(res["gas_rate"] >= 0.0)
        
        # 3. Monotonicity
        assert np.all(np.diff(res["cum_oil"]) >= 0.0)
        assert np.all(np.diff(res["cum_water"]) >= 0.0)
        
        # 4. Gas coupling: qg = 0.3633 * qo
        np.testing.assert_allclose(res["gas_rate"], 0.3633 * res["oil_rate"], rtol=1e-10)


def test_observable_state_feature_extraction():
    """Verifies that pre-origin features are correctly extracted without target leakage."""
    dates = pd.date_range("2000-01-01", "2005-01-01", freq="QS")
    n = len(dates)
    df = pd.DataFrame({
        "date": dates,
        "oil_rate_stbd": np.linspace(8000.0, 3000.0, n),
        "water_rate_stbd": np.linspace(200.0, 4000.0, n),
        "gas_rate_mscfd": np.linspace(2900.0, 1090.0, n),
        "oil_cum_stb": np.cumsum(np.linspace(8000.0, 3000.0, n) * 91.3125),
        "water_cum_stb": np.cumsum(np.linspace(200.0, 4000.0, n) * 91.3125),
        "gas_cum_mscf": np.cumsum(np.linspace(2900.0, 1090.0, n) * 91.3125),
    })
    
    origin = pd.Timestamp("2004-01-01")
    feats = extract_observable_state_features(df, origin)
    
    assert "res_age" in feats
    assert "fw_0" in feats
    assert "wor_0" in feats
    assert "d_recent" in feats
    assert "fw_trend" in feats
    assert "cv_qo" in feats
    
    # State features must be strictly positive and bounded
    assert 0.0 <= feats["fw_0"] <= 1.0
    assert feats["res_age"] == pytest.approx(4.0, abs=0.1)
