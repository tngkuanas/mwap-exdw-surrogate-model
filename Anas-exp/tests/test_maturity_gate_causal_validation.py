"""
Unit & Integration Tests for Causal Maturity Validation & Model Development
Module: tests/test_maturity_gate_causal_validation.py
Branch: research/maturity-gate-causal-validation
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp
"""
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from causal_maturity.sustained_decline_detector import (
    SustainedDeclineDetector,
    extract_robust_case_maturity,
)
from causal_maturity.exact_time_varying_integration import (
    ExactDeclineIntegrator,
    evaluate_raw_decline,
)
from causal_maturity.run_causal_validation_experiments import (
    generate_baseline_reconciliation_table,
)
from causal_maturity.cross_fitted_maturity_gates import (
    compute_case_real_macro_nrmse,
)


def test_sustained_decline_detector_synthetic_profiles():
    """Verify physical decline detector on startup, plateau, temporary fluctuations, and mature depletion."""
    detector = SustainedDeclineDetector()
    dates = pd.date_range("2000-01-01", periods=20, freq="QS")
    
    # 1. Startup Ramp-up: rate increases from 0 to 6000 (pre-plateau)
    qo_startup = np.array([0.0, 2000.0, 6000.0, 10000.0] + [10000.0] * 16)
    res_ramp = detector.analyze(dates[:3], qo_startup[:3])
    assert not res_ramp["is_declining"]
    assert res_ramp["quarters_in_decline"] == 0
    assert res_ramp["mature_decline_stage"] in ["PRE_PRODUCTION", "RAMP_UP"]

    res_start = detector.analyze(dates[:4], qo_startup[:4])
    assert not res_start["is_declining"]
    assert res_start["quarters_in_decline"] == 0
    assert res_start["mature_decline_stage"] in ["RAMP_UP", "PLATEAU"]

    # 2. Flat Plateau: rate steady at 15000 for 12 quarters
    qo_plat = np.full(12, 15000.0)
    res_plat = detector.analyze(dates[:12], qo_plat)
    assert not res_plat["is_declining"]
    assert res_plat["quarters_in_decline"] == 0
    assert res_plat["mature_decline_stage"] == "PLATEAU"
    assert pytest.approx(res_plat["plateau_rate"], rel=1e-3) == 15000.0

    # 3. Temporary Shut-In / Workover with Full Recovery
    # Plateau 15000, drops to 8000 for 1 quarter, rebounds back to 14800
    qo_fluc = np.array([15000.0] * 6 + [8000.0] + [14800.0] * 5)
    res_fluc = detector.analyze(dates[:12], qo_fluc)
    # Must NOT treat temporary dip as permanent decline because rate recovered into plateau band
    assert not res_fluc["is_declining"]
    assert res_fluc["quarters_in_decline"] == 0
    assert res_fluc["mature_decline_stage"] == "PLATEAU"

    # 4. Clear Decline Onset after Plateau
    # 6 quarters plateau at 15000, then steady decline: 13000, 11500, 10000, 9000
    qo_decl = np.array([15000.0] * 6 + [13000.0, 11500.0, 10000.0, 9000.0])
    res_decl = detector.analyze(dates[:10], qo_decl)
    assert res_decl["is_declining"]
    assert res_decl["quarters_in_decline"] >= 3
    assert res_decl["decline_onset_date"] == dates[6]
    assert res_decl["decline_depth"] > 0.30
    assert res_decl["mature_decline_stage"] in ["EARLY_DECLINE", "ESTABLISHED_DECLINE"]

    # 5. Deep Mature Depletion (plateau 15000, current rate 4000 -> depth > 0.70)
    qo_depleted = np.array([15000.0] * 6 + [12000.0, 9000.0, 6000.0, 4000.0])
    res_dep = detector.analyze(dates[:10], qo_depleted)
    assert res_dep["is_declining"]
    assert res_dep["decline_depth"] > 0.70
    assert res_dep["mature_decline_stage"] == "MATURE_DEPLETION"


def test_exact_decline_integrator_multiple_floor_crossings():
    """Verify exact numerical quadrature vs flawed approximation when raw decline crosses floor multiple times."""
    d_min = 0.025
    d_base = 0.012
    decay_rates = np.array([0.4, 1.2, 3.0])
    c = np.array([0.06, -0.09, 0.05]) # Crosses d_min floor multiple times
    
    integrator = ExactDeclineIntegrator(d_base, decay_rates, c, d_min)
    t_grid = np.linspace(0.1, 15.0, 31)
    
    # 1. Compare dense array trapezoid vs adaptive Gauss-Kronrod quad
    dense_I = integrator.integrate_dense_array(t_grid, n_points_per_year=400)
    quad_I = np.array([integrator.integrate_quad(t)[0] for t in t_grid])
    
    max_quad_diff = np.max(np.abs(dense_I - quad_I))
    assert max_quad_diff < 1e-6, f"Dense grid differs from adaptive quad by {max_quad_diff}"
    
    # 2. Compare against flawed Iteration 10 approximation
    flawed_I = integrator.flawed_previous_approximation(t_grid)
    max_flawed_diff = np.max(np.abs(dense_I - flawed_I))
    # Flawed approximation under-integrates when raw dips below floor and rises above
    assert max_flawed_diff > 1e-3, "Flawed approximation unexpectedly matched exact integral"
    
    # 3. Verify strict mathematical monotonicity of cumulative integral and rates
    diffs_I = np.diff(dense_I)
    assert np.all(diffs_I > 0.0), "Cumulative integrated decline must be strictly monotonically increasing"
    
    qo_0 = 5000.0
    rates_o = qo_0 * np.exp(-dense_I)
    diffs_qo = np.diff(rates_o)
    assert np.all(diffs_qo < 0.0), "Rate must be strictly monotonically decreasing everywhere"


def test_baseline_reconciliation_exact_values():
    """Verify that earlier baseline score variations are exactly reproduced and reconciled."""
    df_rec = generate_baseline_reconciliation_table()
    assert len(df_rec) == 4
    
    # 2004 Origin
    rec_2004_asof = df_rec[(df_rec["origin"] == "2004-01-01") & (df_rec["training_protocol"].str.contains("Protocol A"))].iloc[0]
    rec_2004_fwd = df_rec[(df_rec["origin"] == "2004-01-01") & (df_rec["training_protocol"].str.contains("Protocol B"))].iloc[0]
    assert np.isclose(rec_2004_asof["macro_nrmse"], 0.039083, atol=1e-4)
    assert np.isclose(rec_2004_fwd["macro_nrmse"], 0.053032, atol=1e-4)
    
    # 2005 Origin
    rec_2005_asof = df_rec[(df_rec["origin"] == "2005-01-01") & (df_rec["training_protocol"].str.contains("Protocol A"))].iloc[0]
    rec_2005_fwd = df_rec[(df_rec["origin"] == "2005-01-01") & (df_rec["training_protocol"].str.contains("Protocol B"))].iloc[0]
    assert np.isclose(rec_2005_asof["macro_nrmse"], 0.038780, atol=1e-4)
    assert np.isclose(rec_2005_fwd["macro_nrmse"], 0.027557, atol=1e-4)


def test_real_macro_nrmse_objective_calculation():
    """Verify compute_case_real_macro_nrmse handles exact cum calculation and non-negativity."""
    df_post = pd.DataFrame({
        "oil_cum_stb": [10.8e6, 11.5e6, 12.1e6, 12.6e6],
        "water_cum_stb": [3.2e6, 3.8e6, 4.5e6, 5.3e6],
    })
    init_st = {"Qo": 10.0e6, "Qw": 3.0e6}
    
    # Perfect rates that match post cumulatives exactly
    dt_days = 91.3125
    inc_o = np.diff([init_st["Qo"]] + df_post["oil_cum_stb"].tolist())
    inc_w = np.diff([init_st["Qw"]] + df_post["water_cum_stb"].tolist())
    perf_qo = inc_o / dt_days
    perf_qw = inc_w / dt_days
    
    score = compute_case_real_macro_nrmse(perf_qo, perf_qw, df_post, init_st, dt_days)
    assert score < 1e-6, f"Expected near zero error for perfect forecast, got {score}"
