"""
Submission Readiness and End-to-End Delivery Verification Tests.
ExxonMobil DataWorks Challenge 2026.
"""
import json
from pathlib import Path
import numpy as np
import openpyxl
import pytest

from ensemble.submission_pipeline import (
    validate_deliverable_workbook,
    audit_rate_reconciliation,
    RS_SOLUTION,
    EXPECTED_COLUMNS,
)


def test_locked_config_integrity():
    config_path = Path("src/ensemble/locked_ensemble_config.json")
    assert config_path.exists(), "locked_ensemble_config.json does not exist!"
    
    with open(config_path) as f:
        config = json.load(f)
        
    assert config["status"] == "FROZEN"
    assert config["model_name"] == "Blend_Phase_Specific_Simplex"
    
    w_oil = config["weights"]["oil"]
    w_wat = config["weights"]["water"]
    
    assert abs(sum(w_oil.values()) - 1.0) < 1e-6
    assert abs(sum(w_wat.values()) - 1.0) < 1e-6
    assert all(v >= 0.0 for v in w_oil.values())
    assert all(v >= 0.0 for v in w_wat.values())
    
    # Specific frozen weights
    assert abs(w_oil["Recursive_ExtraTrees"] - 0.240466) < 1e-4
    assert abs(w_oil["Learned_Exponential_DCA"] - 0.617902) < 1e-4
    assert abs(w_oil["Learned_Hyperbolic_DCA"] - 0.0) < 1e-6
    assert abs(w_oil["Hybrid_Dynamic"] - 0.141631) < 1e-4
    
    assert abs(w_wat["Recursive_ExtraTrees"] - 0.497170) < 1e-4
    assert abs(w_wat["Learned_Exponential_DCA"] - 0.251415) < 1e-4
    assert abs(w_wat["Learned_Hyperbolic_DCA"] - 0.251415) < 1e-4
    
    # Physics constants
    assert config["physics"]["solution_gas_oil_ratio_rs_mscf_per_stb"] == RS_SOLUTION
    
    # Horizons
    assert config["forecast_horizons"]["Case 1"]["num_quarters"] == 40
    for i in range(2, 11):
        assert config["forecast_horizons"][f"Case {i}"]["num_quarters"] == 80


def test_official_deliverable_workbook_verification():
    file_path = Path("outputs/submission/09_Template_Deliverable_Submission.xlsx")
    assert file_path.exists(), "Official deliverable file missing!"
    
    audit = validate_deliverable_workbook(file_path)
    assert audit["all_checks_passed"] is True
    assert audit["total_sheets"] == 10
    assert audit["nan_count"] == 0
    assert audit["negative_values_count"] == 0
    assert audit["monotonicity_violations"] == 0
    assert audit["gor_discrepancies"] == 0
    
    # Check Case 1 horizon
    assert audit["sheets"]["Case 1"]["num_quarters"] == 40
    assert audit["sheets"]["Case 1"]["oil_monotonic"] is True
    assert audit["sheets"]["Case 1"]["water_monotonic"] is True
    assert audit["sheets"]["Case 1"]["gas_monotonic"] is True
    
    # Check Cases 2-10 horizons
    for i in range(2, 11):
        sheet_key = f"Case {i}"
        assert audit["sheets"][sheet_key]["num_quarters"] == 80
        assert audit["sheets"][sheet_key]["oil_monotonic"] is True
        assert audit["sheets"][sheet_key]["water_monotonic"] is True
        assert audit["sheets"][sheet_key]["gas_monotonic"] is True
        assert audit["sheets"][sheet_key]["max_gor_residual"] < 1e-4


def test_smoke_test_deliverable_verification():
    file_path = Path("outputs/submission/smoke_test_dev_cases_1_to_10.xlsx")
    assert file_path.exists(), "Smoke test deliverable file missing!"
    
    audit = validate_deliverable_workbook(file_path)
    assert audit["all_checks_passed"] is True
    assert audit["total_sheets"] == 10
    assert audit["nan_count"] == 0
    assert audit["negative_values_count"] == 0
    assert audit["monotonicity_violations"] == 0


def test_rate_reconciliation_exactness():
    rates = np.array([5000.0, 4800.0, 4600.0, 4400.0, 4200.0])
    anchor_cum = 38000000.0
    dt_days = 91.3125
    cums = anchor_cum + np.cumsum(rates * dt_days)
    
    err = audit_rate_reconciliation(rates, cums, anchor_cum, dt_days)
    assert err < 1e-10, f"Rate reconciliation error exceeds tolerance: {err}"


def test_submission_audit_report_contents():
    report_path = Path("outputs/submission/submission_audit_report.json")
    assert report_path.exists(), "submission_audit_report.json missing!"
    
    with open(report_path) as f:
        report = json.load(f)
        
    assert report["status"] == "VERIFIED_SUBMISSION_READY"
    assert report["rate_reconciliation_max_error"] == 0.0
    assert report["solution_gor_residual_max"] == 0.0
    assert report["tail_stabilization_audit"]["case_2_q80_cum_stb"] > report["tail_stabilization_audit"]["case_2_q40_cum_stb"]
