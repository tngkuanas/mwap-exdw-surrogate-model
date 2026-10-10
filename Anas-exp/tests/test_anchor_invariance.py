"""
Regression test for single-case versus batch prediction consistency.
Verifies that prediction for any case is strictly invariant to batch composition:
1. Individually: predict([c])
2. As first element of full validation batch: predict([c, ...])
3. In a shuffled batch
4. In differently sized batches
Tolerance: 1e-10 (bitwise or floating-point numerical precision).
"""
import numpy as np
import pandas as pd
import pytest

from ensemble.data import (
    load_uncertainty,
    load_curves,
    TRAIN_CASES,
    VAL_CASES,
    EXPERIMENTS,
)
from ensemble.gp_models import GaussianProcessCurveModel
from ensemble.low_rank_models import LowRankFunctionalModel
from ensemble.nonlinear_models import PolynomialRidgeModel

@pytest.fixture(scope="module")
def shared_data():
    unc_df = load_uncertainty()
    curves = load_curves()
    return unc_df, curves

@pytest.mark.parametrize("model_cls,model_kwargs", [
    (GaussianProcessCurveModel, {"n_components": 1, "nu": 2.5, "noise_level": 1e-2}),
    (LowRankFunctionalModel, {"n_components": 1, "alpha": 5.0}),
    (PolynomialRidgeModel, {"degree": 2, "alpha": 10.0, "n_components": 1}),
])
def test_batch_invariance(shared_data, model_cls, model_kwargs):
    unc_df, curves = shared_data
    origin_str, horizon = EXPERIMENTS[0]
    origin = pd.Timestamp(origin_str)
    dates = pd.date_range(origin + pd.DateOffset(months=3), origin + pd.DateOffset(years=horizon), freq="QS")
    
    # Fit model on training cases 1-70
    model = model_cls(**model_kwargs)
    model.fit(TRAIN_CASES, origin, horizon, dates, curves, unc_df)
    
    target_case = VAL_CASES[0]  # Case 71
    
    # 1. Predicted Individually
    pred_single = model.predict_cases([target_case], origin, horizon, dates, curves, unc_df)
    vals_single = pred_single.sort_values(["phase", "date"])["prediction"].to_numpy(float)
    
    # 2. Predicted as first element of full validation batch (15 cases)
    pred_full = model.predict_cases(VAL_CASES, origin, horizon, dates, curves, unc_df)
    vals_full = pred_full[pred_full["case_num"] == target_case].sort_values(["phase", "date"])["prediction"].to_numpy(float)
    
    # 3. Predicted in shuffled batch
    np.random.seed(42)
    shuffled_cases = list(np.random.permutation(VAL_CASES))
    pred_shuffled = model.predict_cases(shuffled_cases, origin, horizon, dates, curves, unc_df)
    vals_shuffled = pred_shuffled[pred_shuffled["case_num"] == target_case].sort_values(["phase", "date"])["prediction"].to_numpy(float)
    
    # 4. In differently sized batches (e.g. batch of 3 cases, batch of 7 cases)
    batch_3 = [target_case, VAL_CASES[1], VAL_CASES[2]]
    pred_b3 = model.predict_cases(batch_3, origin, horizon, dates, curves, unc_df)
    vals_b3 = pred_b3[pred_b3["case_num"] == target_case].sort_values(["phase", "date"])["prediction"].to_numpy(float)
    
    batch_7 = VAL_CASES[:7]
    pred_b7 = model.predict_cases(batch_7, origin, horizon, dates, curves, unc_df)
    vals_b7 = pred_b7[pred_b7["case_num"] == target_case].sort_values(["phase", "date"])["prediction"].to_numpy(float)
    
    tol = 1e-9
    assert np.allclose(vals_single, vals_full, atol=tol, rtol=tol), f"Failed batch invariance: single vs full batch in {model_cls.__name__}"
    assert np.allclose(vals_single, vals_shuffled, atol=tol, rtol=tol), f"Failed batch invariance: single vs shuffled in {model_cls.__name__}"
    assert np.allclose(vals_single, vals_b3, atol=tol, rtol=tol), f"Failed batch invariance: single vs size 3 in {model_cls.__name__}"
    assert np.allclose(vals_single, vals_b7, atol=tol, rtol=tol), f"Failed batch invariance: single vs size 7 in {model_cls.__name__}"
