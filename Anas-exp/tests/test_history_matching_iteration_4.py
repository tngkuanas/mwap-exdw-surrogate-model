"""
Test Suite: test_history_matching_iteration_4.py
Validates the scientific integrity, provenance, calibration, and physical sanity of Research Iteration 4.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def base_dir():
    return Path("/Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp")


@pytest.fixture
def tables_dir(base_dir):
    return base_dir / "outputs/research/history_matching_iteration_4/tables"


def test_provenance_audit_no_missing_files(tables_dir):
    """Verifies that all input artifacts are found and cryptographically verified."""
    audit_file = tables_dir / "provenance_audit_v4.csv"
    assert audit_file.exists(), f"Missing audit file: {audit_file}"

    df = pd.read_csv(audit_file)
    assert len(df) >= 5, "Audit file must contain at least 5 tracked artifacts"
    assert not (df["audit_status"] == "FILE_NOT_FOUND").any(), "Found FILE_NOT_FOUND in provenance audit!"
    assert (df["audit_status"] == "VERIFIED_OK").all(), "All tracked artifacts must be VERIFIED_OK"


def test_parameter_bounds_hierarchy(tables_dir):
    """Verifies that inference bounds are contained within physical bounds and encompass all observed data."""
    bounds_file = tables_dir / "parameter_bounds_hierarchy_v4.csv"
    assert bounds_file.exists(), f"Missing bounds hierarchy file: {bounds_file}"

    df = pd.read_csv(bounds_file)
    assert len(df) == 4, "Must cover all 4 canonical reservoir parameters"
    assert df["inference_within_physical"].all(), "Inference bounds must be strictly within physical bounds"
    assert df["observed_within_inference"].all(), "Observed data (Cases 1-85) must be within inference prior bounds"


def test_sbc_coverage_metrics(tables_dir):
    """Verifies that simulation-based calibration outputs exist and contain valid Clopper-Pearson CIs."""
    sbc_file = tables_dir / "sbc_coverage_comparison_v4.csv"
    assert sbc_file.exists(), f"Missing SBC file: {sbc_file}"

    df = pd.read_csv(sbc_file)
    assert "fixed_scale" in df["experiment"].values
    assert "state_dependent" in df["experiment"].values

    # Check Clopper-Pearson CIs
    assert (df["cp_ci_lower_95"] <= df["empirical_coverage"]).all()
    assert (df["empirical_coverage"] <= df["cp_ci_upper_95"]).all()

    # Fixed scale at nominal 90% must encompass nominal level for all 4 parameters
    fixed_90 = df[(df["experiment"] == "fixed_scale") & (df["nominal_level"] == 0.90)]
    assert fixed_90["nominal_in_ci"].all(), "Fixed scale 90% CIs must encompass nominal 0.90"


def test_quantile_stability_monotonicity_and_se(tables_dir):
    """Verifies quantile ordering P10 >= P50 >= P90 and standard error reduction with candidate pool size."""
    stab_file = tables_dir / "quantile_stability_v4.csv"
    assert stab_file.exists(), f"Missing quantile stability file: {stab_file}"

    df = pd.read_csv(stab_file).sort_values("candidate_pool_size").reset_index(drop=True)
    assert set(df["candidate_pool_size"]) == {80, 300, 600}

    # Petroleum exceedance monotonicity: P10 >= P50 >= P90
    for _, row in df.iterrows():
        assert row["p10_eur_stb"] >= row["p50_eur_stb"], "P10 must be >= P50"
        assert row["p50_eur_stb"] >= row["p90_eur_stb"], "P50 must be >= P90"
        assert row["eur_spread_stb"] > 0, "Spread must be positive"

    # Standard error reduction from N=80 to N=600
    se_80 = df.loc[df["candidate_pool_size"] == 80, "spread_boot_se_stb"].iloc[0]
    se_600 = df.loc[df["candidate_pool_size"] == 600, "spread_boot_se_stb"].iloc[0]
    assert se_600 < se_80, f"SE at N=600 ({se_600:.1f}) must be smaller than at N=80 ({se_80:.1f})"


def test_selection_policy_quality_filter(tables_dir):
    """Verifies that Strategy C scenarios strictly obey historical match quality filters."""
    scen_file = tables_dir / "selected_scenarios_v4.csv"
    assert scen_file.exists(), f"Missing selected scenarios file: {scen_file}"

    df = pd.read_csv(scen_file)
    strat_c = df[df["strategy"] == "Strategy C: Quality-Filtered Quantiles"]
    assert len(strat_c) == 10, "Strategy C must select exactly 10 cases"

    # Check engineering quality tolerances
    assert (strat_c["oil_cum_nrmse"] <= 0.015).all(), "Strategy C Oil Cum NRMSE must be <= 1.5%"
    assert (strat_c["water_cum_nrmse"] <= 0.080).all(), "Strategy C Water Cum NRMSE must be <= 8.0%"
    assert (strat_c["terminal_oil_err"] <= 0.010).all(), "Strategy C Terminal Oil Error must be <= 1.0%"
    assert strat_c["passes_quality_filter"].all(), "All Strategy C cases must pass quality filter"


def test_reservoir_stress_test_physics(tables_dir):
    """Verifies that reservoir stress test metrics pass physical sanity criteria."""
    stress_file = tables_dir / "reservoir_stress_test_v4.csv"
    assert stress_file.exists(), f"Missing reservoir stress test file: {stress_file}"

    df = pd.read_csv(stress_file).set_index("test_number")

    # Test 1: Water cut monotonicity & bounds
    assert df.loc[1, "pass_rate_pct"] == 100.0, "Test 1 water cut must have 100% pass rate"
    assert df.loc[1, "verdict"] == "PASS"

    # Test 2: Oil rate decline monotonicity
    assert df.loc[2, "pass_rate_pct"] == 100.0, "Test 2 oil rate decline must have 100% pass rate"
    assert df.loc[2, "verdict"] == "PASS"

    # Test 5: Recovery Factor bounds (25% - 35%)
    assert df.loc[5, "pass_rate_pct"] == 100.0, "Test 5 recovery factor must have 100% pass rate"
    assert df.loc[5, "verdict"] == "PASS"

    # Test 6: DCA hyperbolic decline tail sensitivity (< 2.5%)
    assert df.loc[6, "pass_rate_pct"] == 100.0, "Test 6 decline tail sensitivity must have 100% pass rate"
    assert df.loc[6, "verdict"] == "PASS"
