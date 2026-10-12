"""
Unit Tests: test_authoritative_scorer.py
Package: geological_information_value
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Acceptance tests for Phase 1: Authoritative Scorer:
1. Perfect cumulative forecasts score exactly zero.
2. Symmetric positive and negative errors produce identical NRMSE and RMSE.
3. Doubling forecast error doubles NRMSE when denominator is fixed.
4. Arbitrarily inflating predicted rates makes the score worse (higher NRMSE), whereas
   the flawed implementation made it falsely better (~1.0 or lower).
5. Reproduces an independently calculated referee example.
6. Proves that the previous flawed implementation (compute_increment_nrmse(actual_rates, pred_rates))
   fails these tests.
"""
import numpy as np
import pandas as pd
import pytest

from ensemble.metrics import compute_increment_nrmse
from geological_information_value.authoritative_scorer import (
    score_cumulative_forecast,
    integrate_and_score_rates,
)


def flawed_scorer_from_previous_code(actual_rates: np.ndarray, pred_rates: np.ndarray) -> float:
    """The erroneous implementation used in previous sprint."""
    return compute_increment_nrmse(actual_rates, pred_rates)


def test_perfect_forecast_scores_zero():
    """A perfect forecast matching true cumulative production must score exactly 0.0 NRMSE."""
    q0 = 1e6
    true_cum = np.array([1.2e6, 1.5e6, 1.8e6, 2.1e6])
    pred_cum = true_cum.copy()
    
    res = score_cumulative_forecast(
        pred_cum_oil=pred_cum,
        pred_cum_wat=pred_cum,
        true_cum_oil=true_cum,
        true_cum_wat=true_cum,
        origin_cum_oil=q0,
        origin_cum_wat=q0,
    )
    
    assert res["macro_nrmse"] == 0.0
    assert res["oil_nrmse"] == 0.0
    assert res["water_nrmse"] == 0.0
    assert res["rmse_oil_stb"] == 0.0
    assert res["oil_bias"] == 0.0
    
    # Flawed scorer fails this test: passing identical rates gave ~1.0, not 0.0!
    flawed_val = flawed_scorer_from_previous_code(np.array([5000, 5000]), np.array([5000, 5000]))
    assert flawed_val == pytest.approx(1.0, abs=1e-5)
    assert flawed_val != 0.0


def test_symmetric_positive_and_negative_errors():
    """Positive and negative errors of equal magnitude must produce identical RMSE and NRMSE."""
    q0 = 1e6
    true_cum = np.array([1.2e6, 1.4e6, 1.6e6, 1.8e6])
    delta = 50000.0  # 50,000 STB offset
    
    pred_plus = true_cum + delta
    pred_minus = true_cum - delta
    
    res_plus = score_cumulative_forecast(
        pred_cum_oil=pred_plus,
        pred_cum_wat=pred_plus,
        true_cum_oil=true_cum,
        true_cum_wat=true_cum,
        origin_cum_oil=q0,
        origin_cum_wat=q0,
    )
    
    res_minus = score_cumulative_forecast(
        pred_cum_oil=pred_minus,
        pred_cum_wat=pred_minus,
        true_cum_oil=true_cum,
        true_cum_wat=true_cum,
        origin_cum_oil=q0,
        origin_cum_wat=q0,
    )
    
    assert res_plus["oil_nrmse"] == pytest.approx(res_minus["oil_nrmse"], rel=1e-9)
    assert res_plus["rmse_oil_stb"] == pytest.approx(res_minus["rmse_oil_stb"], rel=1e-9)
    # Bias is signed
    assert res_plus["oil_bias"] == pytest.approx(-res_minus["oil_bias"], rel=1e-9)


def test_doubling_error_doubles_nrmse():
    """Doubling the cumulative error vector must exactly double the NRMSE."""
    q0 = 1e6
    true_cum = np.array([1.2e6, 1.4e6, 1.6e6, 1.8e6])
    err1 = np.array([10000.0, 20000.0, -15000.0, 25000.0])
    err2 = 2.0 * err1
    
    pred1 = true_cum + err1
    pred2 = true_cum + err2
    
    res1 = score_cumulative_forecast(
        pred_cum_oil=pred1,
        pred_cum_wat=pred1,
        true_cum_oil=true_cum,
        true_cum_wat=true_cum,
        origin_cum_oil=q0,
        origin_cum_wat=q0,
    )
    
    res2 = score_cumulative_forecast(
        pred_cum_oil=pred2,
        pred_cum_wat=pred2,
        true_cum_oil=true_cum,
        true_cum_wat=true_cum,
        origin_cum_oil=q0,
        origin_cum_wat=q0,
    )
    
    assert res2["oil_nrmse"] == pytest.approx(2.0 * res1["oil_nrmse"], rel=1e-9)
    assert res2["macro_nrmse"] == pytest.approx(2.0 * res1["macro_nrmse"], rel=1e-9)


def test_inflating_predicted_rates_worsens_score():
    """Arbitrarily inflating predicted rates must worsen (increase) NRMSE, not make it smaller."""
    init_st = {"Qo": 1e6, "Qw": 5e5}
    dt = 91.3125
    n_steps = 4
    
    true_rates_o = np.array([5000.0, 4800.0, 4600.0, 4400.0])
    true_cum_o = init_st["Qo"] + np.cumsum(true_rates_o * dt)
    
    post_df = pd.DataFrame({
        "oil_cum_stb": true_cum_o,
        "water_cum_stb": init_st["Qw"] + np.cumsum(np.array([1000.0]*4) * dt),
        "oil_rate_stbd": true_rates_o,
        "water_rate_stbd": np.array([1000.0]*4),
    })
    
    # 1. Base forecast: close to truth (e.g. 5% error)
    base_rates = true_rates_o * 1.05
    res_base = integrate_and_score_rates(init_st, base_rates, np.array([1000.0]*4), post_df, dt)
    
    # 2. Inflated forecast: 10x higher rates
    inflated_rates = true_rates_o * 10.0
    res_inflated = integrate_and_score_rates(init_st, inflated_rates, np.array([1000.0]*4), post_df, dt)
    
    # Authoritative scorer: higher error -> much worse NRMSE
    assert res_inflated["oil_nrmse"] > res_base["oil_nrmse"] * 50.0
    
    # Flawed scorer behavior: compute_increment_nrmse(actual, inflated)
    # RMS(actual) / RMS(inflated) = 1/10 = 0.10!
    # Under the flawed scorer, inflating rates falsely made the score 10x "better" (0.10 vs 0.95)!
    flawed_base = flawed_scorer_from_previous_code(true_rates_o, base_rates)
    flawed_inflated = flawed_scorer_from_previous_code(true_rates_o, inflated_rates)
    assert flawed_inflated < flawed_base  # Demonstrates why the flawed scorer was catastrophically inverted


def test_referee_hand_calculated_example():
    """Independent hand-calculated referee test example."""
    # Origin cumulatives
    q0_oil = 1000000.0
    q0_wat = 500000.0
    
    # 3 quarterly periods
    true_cum_oil = np.array([1200000.0, 1500000.0, 1900000.0])
    true_cum_wat = np.array([550000.0, 620000.0, 710000.0])
    
    # Predicted cumulatives
    pred_cum_oil = np.array([1220000.0, 1480000.0, 1950000.0])  # errors: [+20000, -20000, +50000]
    pred_cum_wat = np.array([545000.0, 630000.0, 700000.0])    # errors: [-5000, +10000, -10000]
    
    # Hand calculations:
    # err_o = [20000, -20000, 50000] -> err_o^2 = [4e8, 4e8, 25e8] -> mean = 33e8/3 = 11e8 -> rms = sqrt(11e8) = 33166.2479
    # inc_o = [200000, 500000, 900000] -> inc_o^2 = [4e10, 25e10, 81e10] -> mean = 110e10/3 = 3.66667e11 -> norm = sqrt(1.1e12/3) = 605530.0708
    # nrmse_o = 33166.2479 / 605530.0708 = 0.05477225
    expected_rmse_o = np.sqrt((20000.0**2 + (-20000.0)**2 + 50000.0**2) / 3.0)
    expected_norm_inc_o = np.sqrt((200000.0**2 + 500000.0**2 + 900000.0**2) / 3.0)
    expected_nrmse_o = expected_rmse_o / expected_norm_inc_o
    
    expected_rmse_w = np.sqrt(((-5000.0)**2 + 10000.0**2 + (-10000.0)**2) / 3.0)
    expected_norm_inc_w = np.sqrt((50000.0**2 + 120000.0**2 + 210000.0**2) / 3.0)
    expected_nrmse_w = expected_rmse_w / expected_norm_inc_w
    expected_macro = 0.5 * (expected_nrmse_o + expected_nrmse_w)
    
    res = score_cumulative_forecast(
        pred_cum_oil=pred_cum_oil,
        pred_cum_wat=pred_cum_wat,
        true_cum_oil=true_cum_oil,
        true_cum_wat=true_cum_wat,
        origin_cum_oil=q0_oil,
        origin_cum_wat=q0_wat,
    )
    
    assert res["oil_nrmse"] == pytest.approx(expected_nrmse_o, rel=1e-6)
    assert res["water_nrmse"] == pytest.approx(expected_nrmse_w, rel=1e-6)
    assert res["macro_nrmse"] == pytest.approx(expected_macro, rel=1e-6)
    assert res["referee_convention_matched"] is True
