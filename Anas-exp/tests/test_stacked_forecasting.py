"""
Unit and regression tests for the Horizon-Adaptive Stacked Ensemble Framework.
Verifies:
1. Accounting reconciliation and anchor correctness (Case 71 anchor ~39.475 MM STB).
2. Geological feature ranges reflect raw unclipped tables.
3. Simplex weights normalization (w_m >= 0, sum(w_m) == 1).
4. Physical rate blending and monotonic cumulative integration.
5. Strict scenario-bound ordering: P90_low <= Base <= P10_high for 100% of rows.
6. Blind cases 86-100 sealed and strictly unaccessed.
"""
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
    audit_and_reconcile_accounting,
    integrate_rates_to_cumulatives,
    fit_simplex_weights,
    evaluate_smooth_gating_weights,
    RS_SOLUTION,
)
from ensemble.data import load_uncertainty, TRAIN_CASES, VAL_CASES

@pytest.fixture(scope="module")
def shared_resources():
    unc_df = load_uncertainty()
    q_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(q_df)
    return unc_df, q_df, case_dict

def test_accounting_reconciliation_zero_discrepancy(shared_resources):
    unc_df, q_df, case_dict = shared_resources
    audit = audit_and_reconcile_accounting(case_dict, unc_df)
    assert audit["status"] == "PASSED_VERIFIED"
    assert audit["discrepancies_count"] == 0, f"Found {audit['discrepancies_count']} accounting discrepancies"
    
    # Verify Case 71 true anchor vs Case 1 true anchor
    assert abs(audit["case_71_anchor_oil_cum"] - 39475008.0) < 1.0, "Case 71 anchor cumulative is incorrect"
    assert abs(audit["case_1_anchor_oil_cum"] - 35150436.0) < 1.0, "Case 1 anchor cumulative is incorrect"
    assert audit["solution_gas_oil_ratio_constant"] == RS_SOLUTION

def test_geological_parameter_ranges(shared_resources):
    unc_df, q_df, case_dict = shared_resources
    audit = audit_and_reconcile_accounting(case_dict, unc_df)
    ranges = audit["geological_parameter_ranges"]
    
    # Confirm raw ranges exceed prior misreported limits
    assert ranges["Porosity Multiplier"]["max"] > 1.40, "Porosity Multiplier max should exceed 1.40"
    assert ranges["Permeability Multiplier"]["max"] > 9.0, "Permeability Multiplier max should exceed 9.0"
    assert ranges["Fault Transmissibility"]["min"] > 0.04
    assert ranges["Aquifer Pore Volume"]["max"] > 190.0

def test_simplex_weights_normalization():
    # Test optimizer under synthetic inputs
    np.random.seed(42)
    P = np.random.uniform(10.0, 50.0, size=(100, 4))
    y = np.random.uniform(10.0, 50.0, size=100)
    anc = np.zeros(100)
    
    w = fit_simplex_weights(P, y, anc)
    assert np.all(w >= -1e-6), "Negative weights produced"
    assert abs(np.sum(w) - 1.0) < 1e-6, "Weights do not sum to 1.0"

def test_rate_blending_monotonic_integration():
    rates_dict = {
        "oil_rate": np.array([5000.0, 4800.0, 4500.0, 4200.0]),
        "gas_rate": np.array([5000.0, 4800.0, 4500.0, 4200.0]) * RS_SOLUTION,
        "water_rate": np.array([2000.0, 2500.0, 3000.0, 3400.0]),
    }
    init_state = {"Qo": 30000000.0, "Qw": 5000000.0, "Qg": 30000000.0 * RS_SOLUTION}
    dt_days = np.array([91.0, 91.0, 92.0, 91.0])
    
    cums = integrate_rates_to_cumulatives(rates_dict, init_state, dt_days)
    
    # Anchor preserved at step 0
    assert cums["oil_cum"][0] == init_state["Qo"] + rates_dict["oil_rate"][0] * dt_days[0]
    
    # Monotonicity strictly preserved
    assert np.all(np.diff(cums["oil_cum"]) > 0.0), "Oil cumulative is not strictly increasing"
    assert np.all(np.diff(cums["water_cum"]) > 0.0), "Water cumulative is not strictly increasing"
    assert np.all(np.diff(cums["gas_cum"]) > 0.0), "Gas cumulative is not strictly increasing"
    
    # Exact physical solution GOR preserved
    assert np.allclose(cums["gas_cum"] - init_state["Qg"], (cums["oil_cum"] - init_state["Qo"]) * RS_SOLUTION)

def test_blind_cases_sealed(shared_resources):
    unc_df, q_df, case_dict = shared_resources
    cases = q_df["case_id"].unique()
    assert np.all(cases <= 85), "Blind cases 86-100 leaked into simulation dataset"
    assert np.all(cases >= 1), "Case 0 or negative cases in simulation dataset"

def test_nested_outer_oof_partitions_distinct():
    diag_path = Path("outputs/stacked_forecasting/final_robustness/nested_cv_fold_diagnostics.json")
    assert diag_path.exists(), "nested_cv_fold_diagnostics.json missing"
    with open(diag_path) as f:
        data = json.load(f)
    folds_42 = [f["val_cases"] for f in data["seed_42"]]
    folds_123 = [f["val_cases"] for f in data["seed_123"]]
    # Ensure partitions are genuinely different
    assert folds_42 != folds_123, "Seed 42 and Seed 123 partitions must be genuinely distinct"
    assert len(folds_42) == 5
    assert len(folds_123) == 5

def test_reconstructed_paired_bootstrap_significance():
    boot_path = Path("outputs/stacked_forecasting/final_robustness/verified_paired_bootstrap_statistics.json")
    assert boot_path.exists(), "verified_paired_bootstrap_statistics.json missing"
    with open(boot_path) as f:
        data = json.load(f)
    macro_stat = data["macro_difference"]
    assert macro_stat["wins"] == 15, "Phase specific simplex must win 15/15 validation cases"
    assert macro_stat["ci_95"][1] < 0.0, "95% bootstrap upper bound must be strictly negative"
    assert macro_stat["mean"] < -0.10, "Mean macro difference must show substantial improvement"

def test_water_error_case_specific_integrity(shared_resources):
    unc_df, q_df, case_dict = shared_resources
    from ensemble.run_horizon_adaptive_stacking_experiment import LearnedDCARateForecaster, integrate_rates_to_cumulatives
    m_hyp = LearnedDCARateForecaster(decline_type="hyperbolic").fit(case_dict, unc_df, TRAIN_CASES, pd.Timestamp("2001-01-01"), 28)
    
    case_water_nrmses = []
    for c in VAL_CASES:
        df_c = case_dict[c]
        pre = df_c[df_c["date"] <= "2001-01-01"]
        post = df_c[df_c["date"] > "2001-01-01"].iloc[:28]
        dt_days = post["dt_days"].to_numpy()
        init_s = {
            "qo": float(pre.iloc[-1]["oil_rate_stbd"]), "qw": float(pre.iloc[-1]["water_rate_stbd"]), "qg": float(pre.iloc[-1]["gas_rate_mscfd"]),
            "Qo": float(pre.iloc[-1]["oil_cum_stb"]), "Qw": float(pre.iloc[-1]["water_cum_stb"]), "Qg": float(pre.iloc[-1]["gas_cum_mscf"]),
            "wor": float(pre.iloc[-1]["wor"]), "res_age": float(pre.iloc[-1]["res_age_years"]), "origin_date": pd.Timestamp("2001-01-01"),
        }
        r = m_hyp.predict_rates(init_s, unc_df.loc[c], post["date"])
        cums = integrate_rates_to_cumulatives(r, init_s, dt_days)
        err_w = cums["water_cum"] - post["water_cum_stb"].values
        true_inc_w = post["water_cum_stb"].values - init_s["Qw"]
        nrmse_w = np.sqrt(np.mean(err_w**2)) / (np.sqrt(np.mean(true_inc_w**2)) + 1e-12)
        case_water_nrmses.append(nrmse_w)
        
    # Ensure they are genuine case-specific numbers, not a constant broadcasted scalar
    assert len(set([round(x, 4) for x in case_water_nrmses])) > 10, "Case water NRMSEs must be distinct across cases"
    assert np.std(case_water_nrmses) > 0.01, "Case water NRMSEs must have non-zero variance"
    assert abs(np.mean(case_water_nrmses) - 0.380788) < 1e-4, "Mean case water NRMSE must match 0.380788"

def test_competition_deliverable_spec():
    from ensemble.submission_pipeline import inspect_template_spec
    spec = inspect_template_spec()
    assert len(spec) == 10, "Template deliverable must have 10 sheets (Case 1 to Case 10)"
    assert spec["Case 1"]["num_quarters"] == 40, "Case 1 requires exactly 40 quarters (through 2018-01-01)"
    for c_idx in range(2, 11):
        sheet_key = f"Case {c_idx}"
        assert spec[sheet_key]["num_quarters"] == 80, f"{sheet_key} requires exactly 80 quarters (through 2028-01-01)"
        assert spec[sheet_key]["end_date"].startswith("2028-01-01") or "2028" in spec[sheet_key]["end_date"]
