"""
Regression Tests: test_deployment_faithful_validation.py
Addresses: EXXONMOBIL DATAWORKS 2026 — DEPLOYMENT-FAITHFUL FORECAST VALIDATION
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Ensures report tables, backing CSV/JSON files, and mathematical models maintain
strict numerical integrity and do not drift or disagree.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest


TABLE_DIR = Path("outputs/research/deployment_faithful_validation/tables")
AUDIT_JSON_PATH = Path("outputs/submission/submission_audit_report.json")


class TestDeploymentFaithfulValidation:
    """Regression test suite verifying backing source data matches validated results."""

    def test_deployed_estimator_audit_integrity(self):
        """Verifies Phase 1: Deployed estimator at January 2008 uses retrospective fallback."""
        audit_file = TABLE_DIR / "deployed_estimator_audit.json"
        assert audit_file.exists(), f"Missing {audit_file}"
        
        with open(audit_file, "r") as f:
            audit = json.load(f)
            
        assert audit["cutoff_date"] == "2008-01-01 00:00:00"
        # Verify Exponential DCA target taxonomy is retrospective
        exp_tax = audit["target_taxonomy"]["Learned_Exponential_DCA"]
        assert exp_tax["future_data_used"] is False
        assert "Retrospective" in exp_tax["target_nature"]
        
        # Verify Ridge parameters exist and are finite
        ridge_D = audit["ridge_D_params"]
        assert np.isfinite(ridge_D["alpha"])
        assert np.isfinite(ridge_D["intercept"])
        assert len(ridge_D["coefficients"]) == 8

    def test_defective_2001_challenger_root_cause(self):
        """Verifies Phase 3: All 15 validation cases floored at 0.01 in the 2001 origin."""
        diag_file = TABLE_DIR / "defective_2001_challenger_diagnosis.csv"
        assert diag_file.exists(), f"Missing {diag_file}"
        
        df_diag = pd.read_csv(diag_file)
        assert len(df_diag) == 15
        
        # In 2001 origin, 100% of cases must have been floored due to startup ramp-up
        assert df_diag["floored_flag"].all()
        assert np.allclose(df_diag["D_used_pred_B"], 0.01, atol=1e-5)
        # All raw predictions must be below 0.01
        assert (df_diag["D_raw_pred_B"] < 0.01).all()

    def test_maturity_matched_experiments_reconciliation(self):
        """Verifies Phase 2: Maturity-matched retrospective evaluation at 2004 and 2005."""
        summary_file = TABLE_DIR / "maturity_matched_origin_comparison.csv"
        assert summary_file.exists(), f"Missing {summary_file}"
        
        df_sum = pd.read_csv(summary_file)
        
        # 1. Verify 2005 origin (Deployment-Matched Mature)
        row_2005_B = df_sum[(df_sum["origin"] == "2005-01-01 00:00:00") & (df_sum["model_key"] == "B")].iloc[0]
        row_2005_A = df_sum[(df_sum["origin"] == "2005-01-01 00:00:00") & (df_sum["model_key"] == "A")].iloc[0]
        row_2005_E = df_sum[(df_sum["origin"] == "2005-01-01 00:00:00") & (df_sum["model_key"] == "E")].iloc[0]
        
        # Model B (Deployment-Faithful) beats Model A (Legacy Forward) at 2005 origin
        assert row_2005_B["macro_nrmse"] < 0.05
        assert row_2005_B["macro_nrmse"] < row_2005_A["macro_nrmse"]
        
        # Model E (Frozen Ensemble) outperforms all standalone models
        assert row_2005_E["macro_nrmse"] < 0.03
        assert row_2005_E["macro_nrmse"] < row_2005_B["macro_nrmse"]
        
        # 2. Verify 2004 origin: Decline error |D_err| is well below 0.05 (mature decline)
        row_2004_B = df_sum[(df_sum["origin"] == "2004-01-01 00:00:00") & (df_sum["model_key"] == "B")].iloc[0]
        assert row_2004_B["mean_D_error"] < 0.03  # Actual is ~0.018

    def test_authoritative_geological_parameters_reconciliation(self):
        """Verifies Phase 5: Reconciled scenario CSV matches authoritative audit JSON."""
        reconciled_file = TABLE_DIR / "authoritative_submission_scenarios_reconciled.csv"
        assert reconciled_file.exists(), f"Missing {reconciled_file}"
        assert AUDIT_JSON_PATH.exists(), f"Missing {AUDIT_JSON_PATH}"
        
        df_rec = pd.read_csv(reconciled_file)
        with open(AUDIT_JSON_PATH, "r") as f:
            audit = json.load(f)
            
        scenarios = audit["scenarios"]
        
        # Check Case 2
        case2_audit = next(s for s in scenarios if s["case_id"] == "Case 2")
        case2_rec = df_rec[df_rec["case_id"] == "Case 2"].iloc[0]
        assert case2_rec["candidate_id"] == 440
        assert np.isclose(case2_rec["fault_trans"], case2_audit["parameters"]["Fault Transmissibility"])
        assert np.isclose(case2_rec["porosity_mult"], case2_audit["parameters"]["Porosity Multiplier"])
        assert np.isclose(case2_rec["perm_mult"], case2_audit["parameters"]["Permeability Multiplier"])
        assert np.isclose(case2_rec["aquifer_pv"], case2_audit["parameters"]["Aquifer Pore Volume"])
        
        # Check Case 10
        case10_audit = next(s for s in scenarios if s["case_id"] == "Case 10")
        case10_rec = df_rec[df_rec["case_id"] == "Case 10"].iloc[0]
        assert case10_rec["candidate_id"] == 144
        assert np.isclose(case10_rec["fault_trans"], case10_audit["parameters"]["Fault Transmissibility"])
        assert np.isclose(case10_rec["porosity_mult"], case10_audit["parameters"]["Porosity Multiplier"])
        assert np.isclose(case10_rec["perm_mult"], case10_audit["parameters"]["Permeability Multiplier"])
        assert np.isclose(case10_rec["aquifer_pv"], case10_audit["parameters"]["Aquifer Pore Volume"])

    def test_petroleum_exceedance_and_cdf_ordering(self):
        """Verifies Phase 4: Proper P10/P50/P90 petroleum exceedance vs CDF quantiles."""
        framework_file = TABLE_DIR / "corrected_posterior_uncertainty_framework.csv"
        assert framework_file.exists(), f"Missing {framework_file}"
        
        df_fw = pd.read_csv(framework_file).set_index("category")
        
        cdf_10 = df_fw.loc["Ordinary CDF 10th Percentile [q_0.10]", "value_stb"]
        cdf_50 = df_fw.loc["Posterior Median / Exceedance P50 (2P)", "value_stb"]
        cdf_90 = df_fw.loc["Ordinary CDF 90th Percentile [q_0.90]", "value_stb"]
        
        spe_p90 = df_fw.loc["Petroleum Exceedance P90 (1P)", "value_stb"]
        spe_p50 = df_fw.loc["Posterior Median / Exceedance P50 (2P)", "value_stb"]
        spe_p10 = df_fw.loc["Petroleum Exceedance P10 (3P)", "value_stb"]
        
        # In petroleum engineering, P90 is proved (conservative), P10 is possible (optimistic)
        assert np.isclose(spe_p90, cdf_10)
        assert np.isclose(spe_p50, cdf_50)
        assert np.isclose(spe_p10, cdf_90)
        
        # Numerical order
        assert spe_p90 < spe_p50 < spe_p10
        assert cdf_10 < cdf_50 < cdf_90
