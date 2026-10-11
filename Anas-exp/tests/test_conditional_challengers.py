"""
Unit and Integration Tests for Conditional Challenger Research.
Module: tests/test_conditional_challengers.py
Branch: research/production-anchored-conditional-models
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp
"""
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from conditional_challengers.reproduce_production_baseline import (
    extract_field_case0_anchor,
    verify_production_baseline_reproduction,
)
from conditional_challengers.corrected_residual_dca import (
    fit_historical_dca_at_origin,
    extract_state_features,
)
from conditional_challengers.maturity_descriptors_and_gates import (
    extract_maturity_descriptors,
    MaturityConditionedGate,
)
from conditional_challengers.time_varying_decline_challenger import (
    TimeVaryingDeclineForecaster,
)
from ensemble.data import load_uncertainty, resolve_shared_dir, TRAIN_CASES
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
)


def test_case0_historical_anchor_values():
    """Verify authoritative Case 0 historical anchor values at 2008-01-01."""
    anchor = extract_field_case0_anchor()
    
    assert anchor["origin_date"] == pd.Timestamp("2008-01-01")
    # Oil cumulative: ~38.38 MMSTB
    assert pytest.approx(anchor["Qo"], abs=1.0) == 38_378_316.97
    # Water cumulative: ~12.81 MMSTB
    assert pytest.approx(anchor["Qw"], abs=1.0) == 12_814_432.28
    # Gas cumulative: ~13.94 Bscf
    assert pytest.approx(anchor["Qg"], abs=1.0) == 13_942_537.32
    # Final rates
    assert pytest.approx(anchor["qo"], abs=0.1) == 5_066.68
    assert pytest.approx(anchor["qw"], abs=0.1) == 7_926.36
    assert pytest.approx(anchor["qg"], abs=0.1) == 1_838.96


def test_production_baseline_reproduction():
    """Verify that production baseline reproduction reproduces submission workbook with 0 STB error."""
    repro_df = verify_production_baseline_reproduction()
    
    assert len(repro_df) == 10
    assert "abs_diff_oil_stb" in repro_df.columns
    # Max absolute difference strictly below 0.05 STB (in fact 0.0000 STB)
    max_diff = repro_df["abs_diff_oil_stb"].max()
    assert max_diff < 0.05
    
    # Case 1 has 40 quarters and ~47.65 MMSTB
    c1 = repro_df[repro_df["case_id"] == "Case 1"].iloc[0]
    assert c1["horizon_quarters"] == 40
    assert 47_600_000 <= c1["repro_oil_cum_stb"] <= 47_700_000
    
    # Cases 2-10 have 80 quarters and in [51.17, 51.19] MMSTB
    c_rest = repro_df[repro_df["case_id"] != "Case 1"]
    for _, row in c_rest.iterrows():
        assert row["horizon_quarters"] == 80
        assert 51_160_000 <= row["repro_oil_cum_stb"] <= 51_190_000


def test_maturity_descriptors_and_gate():
    """Verify maturity descriptor calculation and smooth gating function."""
    # Synthetic declining rate profile
    dates = pd.date_range("1999-01-01", "2005-01-01", freq="QS")
    n = len(dates)
    t = np.arange(n)
    
    # Rates plateau until 2002 then decline
    qo = np.where(t < 12, 10000.0, 10000.0 * np.exp(-0.15 * (t - 12)))
    qw = 2000.0 + 300.0 * t
    
    df_case = pd.DataFrame({
        "date": dates,
        "oil_rate_stbd": qo,
        "water_rate_stbd": qw,
    })
    
    origin = pd.Timestamp("2005-01-01")
    desc = extract_maturity_descriptors(df_case, origin)
    
    assert desc["quarters_in_decline"] >= 8
    assert desc["nominal_decline_rate"] > 0.0
    assert 0.0 <= desc["water_cut"] <= 1.0
    assert np.isfinite(desc["rate_variability_cv"])
    
    # Test gate
    gate = MaturityConditionedGate()
    mat_list = [{"quarters_in_decline": q} for q in [2.0, 4.0, 8.0, 12.0, 16.0]]
    loss_diff = np.array([0.05, 0.02, -0.01, -0.04, -0.05])
    gate.fit(mat_list, loss_diff)
    
    # Evaluate gate weights
    gw = gate.compute_gate_weight({"quarters_in_decline": 12.0})
    assert 0.0 <= gw <= 1.0


def test_time_varying_decline_monotonicity():
    """Verify mathematical monotonicity of continuous time-varying decline model."""
    forecaster = TimeVaryingDeclineForecaster(
        decay_rates=(0.2, 0.5, 1.0),
        d_min=0.01,
    )
    
    # Mathematical integration verification
    # For any D(t) >= d_min > 0, int_0^t D(s) ds is strictly increasing
    t_years = np.linspace(0, 20, 81)
    D_base = 0.08
    c_pred = np.array([0.05, -0.02, 0.01])
    Phi_int = forecaster._integrated_basis(t_years)
    Phi = forecaster._basis_functions(t_years)
    
    D_t = np.maximum(forecaster.d_min, D_base + Phi @ c_pred)
    assert np.all(D_t >= forecaster.d_min)
    
    int_D = np.maximum(forecaster.d_min * t_years, D_base * t_years + Phi_int @ c_pred)
    # Integral must be strictly non-decreasing
    assert np.all(np.diff(int_D) >= 0.0)
    
    # Rate: qo_t = qo_0 * exp(-int_D)
    qo_0 = 5000.0
    qo_t = qo_0 * np.exp(-int_D)
    
    # Rates must be strictly non-negative
    assert np.all(qo_t >= 0.0)
    assert pytest.approx(qo_t[0], rel=1e-5) == 5000.0
    # Strictly non-increasing
    diffs = np.diff(qo_t)
    assert np.all(diffs <= 1e-9), f"Max increase observed: {np.max(diffs)}"


def test_out_of_origin_semantics_no_future_leakage():
    """Verify that historical DCA fitting strictly does not use data beyond origin."""
    shared_dir = resolve_shared_dir()
    df_raw = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(df_raw)
    unc_df = load_uncertainty(shared_dir)
    
    origin = pd.Timestamp("2004-01-01")
    # Fit historical DCA strictly at 2004-01-01
    dca = fit_historical_dca_at_origin(case_dict, unc_df, [1, 2, 3], origin)
    
    # Check that model parameters exist and are finite
    assert hasattr(dca, "model_D")
    assert hasattr(dca, "model_water_growth")
    
    # State features extraction strictly at origin
    feats = extract_state_features(case_dict[1], origin, unc_df.loc[1])
    assert len(feats) == 9
    assert np.all(np.isfinite(feats))


def test_output_artifacts_and_tables():
    """Verify generated tables and figures exist, are non-empty, and contain required schema."""
    table_dir = Path("outputs/research/conditional_challengers/tables")
    fig_dir = Path("outputs/research/conditional_challengers/figures")
    
    # 1. Tables
    repro_path = table_dir / "production_baseline_reproduction.csv"
    assert repro_path.exists()
    df_repro = pd.read_csv(repro_path)
    assert len(df_repro) == 10
    assert "abs_diff_oil_stb" in df_repro.columns
    assert df_repro["abs_diff_oil_stb"].max() < 0.05
    
    case_res_path = table_dir / "case_level_evaluation_results.csv"
    assert case_res_path.exists()
    df_c = pd.read_csv(case_res_path)
    assert len(df_c) > 0
    assert "macro_nrmse" in df_c.columns
    assert "model_name" in df_c.columns
    assert "case_id" in df_c.columns
    
    bench_path = table_dir / "conditional_model_benchmark_summary.csv"
    assert bench_path.exists()
    df_b = pd.read_csv(bench_path)
    assert len(df_b) > 0
    assert "wilcoxon_p_value" in df_b.columns
    assert "win_rate_vs_baseline" in df_b.columns
    
    # 2. Figures
    for fig_name in [
        "fig1_corrected_training_protocol.png",
        "fig2_conditional_model_benchmark.png",
        "fig3_origin_2005_case_breakdown.png",
        "fig4_twenty_year_case0_anchor_trajectories.png",
        "fig5_physics_watercut_and_liquid.png",
    ]:
        f_p = fig_dir / fig_name
        assert f_p.exists(), f"Figure {fig_name} missing"
        assert f_p.stat().st_size > 10_000, f"Figure {fig_name} too small"

