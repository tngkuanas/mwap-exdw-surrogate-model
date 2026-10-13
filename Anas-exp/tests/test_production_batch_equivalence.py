"""
Unit Tests: test_production_batch_equivalence.py
Package: tests
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Addresses Mandates:
Section 1 — DCA Coefficient Attribution and Direct Matrix Prediction Invariant
Section 2 — Blocking Acceptance Test: Prove Exact Numerical Equivalence of
             Batch and Production Forecasting Across 100 Diverse Geological Vectors.
"""
import numpy as np
import pandas as pd
import pytest

from ensemble.data import load_uncertainty, resolve_shared_dir, TRAIN_CASES, YEAR_DAYS
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
)
from ensemble.generate_competition_deliverable import (
    load_locked_config,
    fit_locked_base_models,
    generate_scenario_predictions,
)
from geological_information_value.parameter_bounds import (
    PARAM_NAMES,
    LOWER_BOUNDS,
    UPPER_BOUNDS,
    PRIOR_MEANS,
)
from geological_information_value.forward_model_sensitivity import (
    evaluate_models_batch,
    inspect_fitted_representations,
)


@pytest.fixture(scope="module")
def fitted_environment():
    """Fits the exact frozen production models at January 2008."""
    shared_dir = resolve_shared_dir()
    unc_df = load_uncertainty(shared_dir)
    raw_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(raw_df)
    
    config = load_locked_config()
    weights = config["weights"]
    inference_origin = pd.Timestamp("2008-01-01")
    f_dates = pd.date_range("2008-04-01", "2028-01-01", freq="QS")
    
    # Historical anchor Case 0 at Jan 2008
    prod = pd.read_parquet(shared_dir / "production_timeseries.parquet")
    c0 = prod[prod["case_num"] == 0]
    c0_2008 = c0[c0["date"] == "2008-01-01"].set_index("metric")["value"]
    
    init_state = {
        "qo": float(c0_2008["oil_rate"]),
        "qw": float(c0_2008["water_rate"]),
        "qg": float(c0_2008["gas_rate"]),
        "Qo": float(c0_2008["oil_cum"]),
        "Qw": float(c0_2008["water_cum"]),
        "Qg": float(c0_2008["gas_cum"]),
        "wor": float(c0_2008["water_rate"] / max(c0_2008["oil_rate"], 1.0)),
        "res_age": 10.0,
        "origin_date": inference_origin,
    }
    
    models = fit_locked_base_models(case_dict, unc_df, TRAIN_CASES, inference_origin)
    return {
        "models": models,
        "weights": weights,
        "init_state": init_state,
        "f_dates": f_dates,
    }


def test_dca_coefficient_attribution_and_matrix_prediction(fitted_environment):
    """
    Section 1 Test:
    Verifies that:
    1. feat_cols is authoritatively ['fault', 'poro', 'perm', 'aq', 'qo_0', 'qw_0', 'wor_0', 'res_age'].
    2. Coefficients map to expected physical signs and magnitudes.
    3. Direct matrix product X @ beta + intercept equals model.predict(X) within 1e-12.
    4. Porosity range sensitivity dominates Fault, Permeability, and Aquifer PV.
    """
    models = fitted_environment["models"]
    m_exp = models["Learned_Exponential_DCA"]
    
    expected_feat_cols = ["fault", "poro", "perm", "aq", "qo_0", "qw_0", "wor_0", "res_age"]
    assert m_exp.feat_cols == expected_feat_cols
    
    rep = inspect_fitted_representations(models)
    coefs = rep["Learned_Exponential_DCA_coefs_D"]
    
    # Check coefficient values
    assert coefs["fault"] == pytest.approx(0.0051086, rel=1e-3)
    assert coefs["poro"] == pytest.approx(0.0440175, rel=1e-3)
    assert coefs["perm"] == pytest.approx(0.00012165, rel=1e-3)
    assert coefs["aq"] == pytest.approx(-6.33e-06, rel=1e-2)
    assert coefs["qo_0"] == pytest.approx(-2.249e-05, rel=1e-3)
    assert coefs["qw_0"] == pytest.approx(-1.134e-06, rel=1e-2)
    assert coefs["wor_0"] == pytest.approx(0.0175187, rel=1e-3)
    assert abs(coefs["res_age"]) < 1e-20
    
    # Check direct linear algebra evaluation X @ beta + intercept == predict(X)
    rng = np.random.RandomState(42)
    X_test = rng.uniform(0.1, 10.0, size=(25, 8))
    pred_sklearn = m_exp.model_D.predict(X_test)
    pred_direct = X_test @ m_exp.model_D.coef_ + m_exp.model_D.intercept_
    assert np.allclose(pred_sklearn, pred_direct, atol=1e-12)
    
    # Check range-normalized sensitivities
    phys_sens = rep["Learned_Exponential_DCA_physical_sensitivity_D"]
    assert phys_sens["poro"] > 5.0 * phys_sens["fault"]
    assert phys_sens["poro"] > 20.0 * phys_sens["perm"]
    assert phys_sens["poro"] > 15.0 * phys_sens["aq"]


def test_batch_vs_production_exact_equivalence_100_samples(fitted_environment):
    """
    Section 2 Blocking Acceptance Test:
    Compares scalar generate_scenario_predictions() with evaluate_models_batch()
    across 100 diverse geological vectors spanning:
    - Lower boundary corner
    - Upper boundary corner
    - Prior mean / center
    - Random uniform interior
    - Split-threshold points
    Asserts tight float64 equivalence (< 1e-10 on rates, < 1e-6 on cumulatives).
    """
    models = fitted_environment["models"]
    weights = fitted_environment["weights"]
    init_state = fitted_environment["init_state"]
    f_dates = fitted_environment["f_dates"]
    
    M = 100
    rng = np.random.RandomState(42)
    U_phys = np.column_stack([
        rng.uniform(LOWER_BOUNDS[j], UPPER_BOUNDS[j], size=M)
        for j in range(4)
    ])
    
    # Add boundary / corner cases
    U_phys[0] = LOWER_BOUNDS
    U_phys[1] = UPPER_BOUNDS
    U_phys[2] = 0.5 * (LOWER_BOUNDS + UPPER_BOUNDS)
    U_phys[3] = [0.01, 1.50, 0.10, 300.0]
    U_phys[4] = [1.00, 0.80, 10.00, 10.0]
    
    batch_res = evaluate_models_batch(models, weights, init_state, U_phys, f_dates)
    
    for m in range(M):
        u_series = pd.Series(U_phys[m], index=PARAM_NAMES)
        df_pred, aud = generate_scenario_predictions(models, weights, init_state, u_series, f_dates)
        
        # 1. Blend predictions
        b_qo = batch_res["Blend_Phase_Specific_Simplex"]["qo_rate"][m]
        b_qw = batch_res["Blend_Phase_Specific_Simplex"]["qw_rate"][m]
        b_co = batch_res["Blend_Phase_Specific_Simplex"]["cum_oil"][m]
        b_cw = batch_res["Blend_Phase_Specific_Simplex"]["cum_water"][m]
        
        assert np.max(np.abs(b_qo - aud["qo_rate"])) < 1e-10, f"Blend qo mismatch at sample {m}"
        assert np.max(np.abs(b_qw - aud["qw_rate"])) < 1e-10, f"Blend qw mismatch at sample {m}"
        assert np.max(np.abs(b_co - aud["cum_oil"])) < 1e-6, f"Blend cum_oil mismatch at sample {m}"
        assert np.max(np.abs(b_cw - aud["cum_water"])) < 1e-6, f"Blend cum_water mismatch at sample {m}"
        
        # 2. Individual component predictions
        r_et = models["Recursive_ExtraTrees"].predict_rates(init_state, u_series, f_dates)
        r_exp = models["Learned_Exponential_DCA"].predict_rates(init_state, u_series, f_dates)
        r_hyp = models["Learned_Hyperbolic_DCA"].predict_rates(init_state, u_series, f_dates)
        r_hyb = models["Hybrid_Dynamic"].predict_rates(init_state, u_series, f_dates)
        
        # ExtraTrees is bit-exact
        assert np.max(np.abs(batch_res["Recursive_ExtraTrees"]["qo_rate"][m] - r_et["oil_rate"])) < 1e-10
        assert np.max(np.abs(batch_res["Recursive_ExtraTrees"]["qw_rate"][m] - r_et["water_rate"])) < 1e-10
        
        # DCA components
        assert np.max(np.abs(batch_res["Learned_Exponential_DCA"]["qo_rate"][m] - r_exp["oil_rate"])) < 1e-10
        assert np.max(np.abs(batch_res["Learned_Exponential_DCA"]["qw_rate"][m] - r_exp["water_rate"])) < 1e-10
        assert np.max(np.abs(batch_res["Learned_Hyperbolic_DCA"]["qo_rate"][m] - r_hyp["oil_rate"])) < 1e-10
        assert np.max(np.abs(batch_res["Learned_Hyperbolic_DCA"]["qw_rate"][m] - r_hyp["water_rate"])) < 1e-10
        
        # Hybrid Dynamic matches handoff state and tail DCA
        assert np.max(np.abs(batch_res["Hybrid_Dynamic"]["qo_rate"][m] - r_hyb["oil_rate"])) < 1e-10
        assert np.max(np.abs(batch_res["Hybrid_Dynamic"]["qw_rate"][m] - r_hyb["water_rate"])) < 1e-10
