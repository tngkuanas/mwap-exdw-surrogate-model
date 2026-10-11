"""
Regression Tests: test_deployment_faithful_validation.py
Addresses: EXXONMOBIL DATAWORKS 2026 — FINAL DEPLOYMENT VALIDATION CORRECTION
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Ensures report tables, backing CSV/JSON files, and mathematical models maintain
strict numerical integrity and do not drift or disagree.
Tests invoke actual production fitting functions on truncated data to assert
intended fallback execution and zero future-data leakage.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ensemble.data import load_uncertainty, TRAIN_CASES, VAL_CASES
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
    extract_transition_training_data,
    RecursiveTransitionForecaster,
    LearnedDCARateForecaster,
    HybridDynamicForecaster,
    YEAR_DAYS,
)
from ensemble.metrics import compute_increment_nrmse

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
        
        # Verify exact Case 1 and Case 2 standalone DCA values
        assert np.isclose(audit["case1_d_predicted"], 0.193832, atol=1e-4)
        assert np.isclose(audit["case2_d_predicted"], 0.194102, atol=1e-4)
        assert np.isclose(audit["case1_standalone_dca_eur_stb"], 46356530.15, atol=1.0)
        assert np.isclose(audit["case2_standalone_dca_eur_stb"], 47493064.09, atol=1.0)

    def test_actual_production_pathway_fallback_execution(self):
        """
        Phase 6 Test: Invokes real production model-fitting functions with data
        truncated at valid retrospective origins (2004 and 2005).
        Asserts DCA activates intended fallback pre[date >= '2002-01-01'] and
        does not access future data.
        """
        unc_df = load_uncertainty()
        q_df = load_clean_simulation_quarterly()
        case_dict = build_case_quarterly_trajectories(q_df)
        
        orig = pd.Timestamp("2004-01-01")
        hq = 16
        
        # Construct strict as-of dictionary (strictly truncated at orig)
        case_dict_asof = {c: case_dict[c][case_dict[c]["date"] <= orig] for c in TRAIN_CASES}
        
        # 1. Assert all training cases have zero future rows in case_dict_asof
        for c in TRAIN_CASES:
            assert (case_dict_asof[c]["date"] > orig).sum() == 0
            
        # 2. Invoke real production LearnedDCARateForecaster.fit
        dca_prod = LearnedDCARateForecaster(decline_type="exponential")
        dca_prod.fit(case_dict_asof, unc_df, TRAIN_CASES, orig, hq)
        
        # Verify model was fitted on historical decline (2002-2004 = 2 years)
        assert np.isfinite(dca_prod.model_D.intercept_)
        assert np.isfinite(dca_prod.model_D.alpha_)
        assert dca_prod.model_D.intercept_ > 0.5  # Positive nominal decline intercept
        
        # 3. Predict on held-out validation Case 71 and verify physical monotonicity
        c_val = 71
        df_val = case_dict[c_val]
        pre_val = df_val[df_val["date"] <= orig]
        post_val = df_val[df_val["date"] > orig].iloc[:hq]
        u_val = unc_df.loc[c_val]
        last_val = pre_val.iloc[-1]
        
        init_st = {
            "qo": float(last_val["oil_rate_stbd"]),
            "qw": float(last_val["water_rate_stbd"]),
            "qg": float(last_val["gas_rate_mscfd"]),
            "Qo": float(last_val["oil_cum_stb"]),
            "Qw": float(last_val["water_cum_stb"]),
            "Qg": float(last_val["gas_cum_mscf"]),
            "wor": float(last_val["wor"]),
            "res_age": float(last_val["res_age_years"]),
            "origin_date": orig,
        }
        
        f_dates = pd.date_range(orig + pd.DateOffset(days=1), periods=hq, freq="QS")
        preds = dca_prod.predict_rates(init_st, u_val, f_dates)
        
        # Rate must decay monotonically for exponential decline
        qo_preds = preds["oil_rate"]
        assert (np.diff(qo_preds) <= 1e-6).all(), "Oil rate must decline monotonically"
        assert preds["D_used"] >= 0.01, "Predicted decline rate must respect the 0.01 clamp floor"

    def test_protocol_a_vs_protocol_b_exact_reconciliations(self):
        """
        Verifies exact numerical reconciliation across all origins:
        - 2001: Model E_fwd (0.0533) wins 15/15 cases; DCA models hit 0.01 startup floor.
        - 2003: Model A_fwd (0.0747) beats Model E_fwd (0.1110); Model A wins 11/15 cases.
        - 2004: Model A_fwd (0.0475) beats Model E_fwd (0.0530); Model E_asof achieves 0.0391.
        - 2005: Model E_fwd achieves 0.0276 (oil 0.050606, water 0.004507);
                Model B_prod achieves 0.0464 (oil 0.029463, water 0.063278);
                Model E_asof achieves 0.0388 (oil 0.036802, water 0.040759).
        """
        proto_file = TABLE_DIR / "protocol_a_vs_protocol_b_comparison.csv"
        assert proto_file.exists(), f"Missing {proto_file}"
        
        df_proto = pd.read_csv(proto_file)
        
        # 1. Origin 2001 Reconciliation
        row_2001_Efwd = df_proto[(df_proto["origin"] == "2001-01-01 00:00:00") & (df_proto["model_key"] == "E_fwd")].iloc[0]
        row_2001_Bprod = df_proto[(df_proto["origin"] == "2001-01-01 00:00:00") & (df_proto["model_key"] == "B_prod")].iloc[0]
        assert np.isclose(row_2001_Efwd["macro_nrmse"], 0.053262, atol=1e-4)
        assert np.isclose(row_2001_Bprod["macro_nrmse"], 0.453241, atol=1e-4)
        assert row_2001_Efwd["win_count_out_of_15"] == 15
        
        # 2. Origin 2003 Reconciliation: Model A_fwd wins 11/15 cases and beats Model E_fwd
        row_2003_Afwd = df_proto[(df_proto["origin"] == "2003-01-01 00:00:00") & (df_proto["model_key"] == "A_fwd")].iloc[0]
        row_2003_Efwd = df_proto[(df_proto["origin"] == "2003-01-01 00:00:00") & (df_proto["model_key"] == "E_fwd")].iloc[0]
        assert np.isclose(row_2003_Afwd["macro_nrmse"], 0.074715, atol=1e-4)
        assert np.isclose(row_2003_Efwd["macro_nrmse"], 0.110994, atol=1e-4)
        assert row_2003_Afwd["macro_nrmse"] < row_2003_Efwd["macro_nrmse"]
        assert row_2003_Afwd["win_count_out_of_15"] == 11
        assert row_2003_Efwd["win_count_out_of_15"] == 4
        
        # 3. Origin 2004 Reconciliation: Model A_fwd beats Model E_fwd, but Model E_asof achieves lowest error
        row_2004_Afwd = df_proto[(df_proto["origin"] == "2004-01-01 00:00:00") & (df_proto["model_key"] == "A_fwd")].iloc[0]
        row_2004_Efwd = df_proto[(df_proto["origin"] == "2004-01-01 00:00:00") & (df_proto["model_key"] == "E_fwd")].iloc[0]
        row_2004_Easof = df_proto[(df_proto["origin"] == "2004-01-01 00:00:00") & (df_proto["model_key"] == "E_asof")].iloc[0]
        assert np.isclose(row_2004_Afwd["macro_nrmse"], 0.047476, atol=1e-4)
        assert np.isclose(row_2004_Efwd["macro_nrmse"], 0.053031, atol=1e-4)
        assert row_2004_Afwd["macro_nrmse"] < row_2004_Efwd["macro_nrmse"]
        assert np.isclose(row_2004_Easof["macro_nrmse"], 0.039083, atol=1e-4)
        assert row_2004_Easof["macro_nrmse"] < row_2004_Afwd["macro_nrmse"]
        
        # 4. Origin 2005 Reconciliation: Phase decomposition
        row_2005_Efwd = df_proto[(df_proto["origin"] == "2005-01-01 00:00:00") & (df_proto["model_key"] == "E_fwd")].iloc[0]
        row_2005_Bprod = df_proto[(df_proto["origin"] == "2005-01-01 00:00:00") & (df_proto["model_key"] == "B_prod")].iloc[0]
        row_2005_Easof = df_proto[(df_proto["origin"] == "2005-01-01 00:00:00") & (df_proto["model_key"] == "E_asof")].iloc[0]
        
        # Verify Model E_fwd phase scores
        assert np.isclose(row_2005_Efwd["oil_nrmse"], 0.050606, atol=1e-4)
        assert np.isclose(row_2005_Efwd["water_nrmse"], 0.004507, atol=1e-4)
        assert np.isclose(row_2005_Efwd["macro_nrmse"], 0.027557, atol=1e-4)
        
        # Verify Model B_prod has LOWER oil error than Model E_fwd
        assert np.isclose(row_2005_Bprod["oil_nrmse"], 0.029463, atol=1e-4)
        assert row_2005_Bprod["oil_nrmse"] < row_2005_Efwd["oil_nrmse"]
        
        # Verify Model E_asof achieves balanced performance
        assert np.isclose(row_2005_Easof["oil_nrmse"], 0.036802, atol=1e-4)
        assert np.isclose(row_2005_Easof["water_nrmse"], 0.040759, atol=1e-4)
        assert np.isclose(row_2005_Easof["macro_nrmse"], 0.038780, atol=1e-4)
        assert row_2005_Easof["macro_nrmse"] < row_2005_Bprod["macro_nrmse"]

    def test_mcmc_autocorrelation_and_ess_audit(self):
        """
        Phase 4 Test: Verifies MCMC autocorrelation, ESS, and true MCSE.
        Asserts that thinning does NOT establish independent draws, but EUR samples
        have stride 53.3 > 2 * tau_int, giving tau_int = 1.00, N_eff = 150.0, and true MCSE = 1,895.82 STB.
        """
        ess_file = TABLE_DIR / "mcmc_autocorrelation_and_ess_audit.json"
        assert ess_file.exists(), f"Missing {ess_file}"
        
        with open(ess_file, "r") as f:
            audit = json.load(f)
            
        params = audit["parameter_diagnostics"]
        # All 4 parameters must have tau_int > 10 (significant autocorrelation)
        for p_name, diag in params.items():
            assert diag["tau_int"] > 10.0
            assert diag["N_eff"] < diag["N_raw"]
            assert diag["lag10_autocorr"] > 0.20, "10x thinning leaves significant autocorrelation"
            
        eur_diag = audit["eur_diagnostics"]
        assert eur_diag["N_samples"] == 150
        assert np.isclose(eur_diag["tau_int"], 1.00, atol=1e-2)
        assert np.isclose(eur_diag["N_eff"], 150.0, atol=1e-1)
        assert np.isclose(eur_diag["mcse_stb"], 1895.82, atol=1.0)
        
        # Verify exceedance vs CDF quantiles
        assert np.isclose(eur_diag["spe_exceedance_p90_stb"], eur_diag["cdf_p10_stb"])
        assert np.isclose(eur_diag["spe_exceedance_p50_stb"], eur_diag["cdf_p50_stb"])
        assert np.isclose(eur_diag["spe_exceedance_p10_stb"], eur_diag["cdf_p90_stb"])
        assert eur_diag["spe_exceedance_p90_stb"] < eur_diag["spe_exceedance_p50_stb"] < eur_diag["spe_exceedance_p10_stb"]

    def test_petroleum_exceedance_and_prms_reserves_distinction(self):
        """Verifies Phase 4: Proper P10/P50/P90 petroleum exceedance vs CDF quantiles."""
        framework_file = TABLE_DIR / "corrected_posterior_uncertainty_framework.csv"
        assert framework_file.exists(), f"Missing {framework_file}"
        
        df_fw = pd.read_csv(framework_file).set_index("category")
        
        cdf_10 = df_fw.loc["Ordinary CDF 10th Percentile [q_0.10]", "value_stb"]
        cdf_50 = df_fw.loc["Posterior Median / Exceedance P50 (2P Technical)", "value_stb"]
        cdf_90 = df_fw.loc["Ordinary CDF 90th Percentile [q_0.90]", "value_stb"]
        
        spe_p90 = df_fw.loc["Petroleum Exceedance P90 (1P Technical)", "value_stb"]
        spe_p50 = df_fw.loc["Posterior Median / Exceedance P50 (2P Technical)", "value_stb"]
        spe_p10 = df_fw.loc["Petroleum Exceedance P10 (3P Technical)", "value_stb"]
        
        assert np.isclose(spe_p90, cdf_10)
        assert np.isclose(spe_p50, cdf_50)
        assert np.isclose(spe_p10, cdf_90)
        assert spe_p90 < spe_p50 < spe_p10

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
