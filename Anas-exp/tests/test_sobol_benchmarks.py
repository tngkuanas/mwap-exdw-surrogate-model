"""
Unit Tests: test_sobol_benchmarks.py
Package: tests
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Addresses Section 3:
Validates Sobol estimators against 4 analytical benchmark functions:
1. Additive linear model with known Sobol indices (no interactions, S_i == S_Ti).
2. Multiplicative interaction model (S_1 = S_2 = 3/7, S_T1 = S_T2 = 4/7).
3. Ishigami function (known non-linear & interaction indices).
4. Model independent of one or more parameters (S_j == S_Tj == 0).
Tests both Jansen and Centered Saltelli estimators and verifies confidence intervals.
"""
import numpy as np
import pytest

from geological_information_value.sobol_estimators import (
    generate_scrambled_sobol_design,
    compute_centered_sobol_indices,
    bootstrap_sobol_confidence_intervals,
    analytical_linear_benchmark,
    analytical_multiplicative_benchmark,
    analytical_ishigami_benchmark,
)


def test_analytical_additive_linear_benchmark():
    """
    Test Benchmark 1: Additive Linear Model
    Y = 2*X0 + 3*X1 + 0*X2 + 1*X3
    Analytical: S = S_T = [4/14, 9/14, 0/14, 1/14] = [0.2857, 0.6429, 0.0, 0.0714]
    """
    N = 4096
    A, B, A_B = generate_scrambled_sobol_design(d=4, n_base=N, seed=101)
    
    yA = analytical_linear_benchmark(A)
    yB = analytical_linear_benchmark(B)
    yAB = [analytical_linear_benchmark(mat) for mat in A_B]
    
    # 1. Jansen estimator
    res_jansen = compute_centered_sobol_indices(yA, yB, yAB, estimator="jansen")
    S_jan = res_jansen["first_order"]
    ST_jan = res_jansen["total_order"]
    
    expected_S = np.array([4.0 / 14.0, 9.0 / 14.0, 0.0, 1.0 / 14.0])
    assert np.allclose(S_jan, expected_S, atol=0.03)
    assert np.allclose(ST_jan, expected_S, atol=0.03)
    # Check parameter 2 is exactly zero
    assert abs(S_jan[2]) < 0.015
    assert abs(ST_jan[2]) < 0.015
    
    # 2. Centered Saltelli estimator
    res_saltelli = compute_centered_sobol_indices(yA, yB, yAB, estimator="saltelli_centered")
    S_sal = res_saltelli["first_order"]
    assert np.allclose(S_sal, expected_S, atol=0.03)
    
    # 3. Invariance to large constant shift (e.g. historical anchor + 38.38e6)
    c_shift = 38.38e6
    yA_shifted = yA + c_shift
    yB_shifted = yB + c_shift
    yAB_shifted = [mat + c_shift for mat in yAB]
    
    res_shifted = compute_centered_sobol_indices(yA_shifted, yB_shifted, yAB_shifted, estimator="jansen")
    assert np.allclose(res_shifted["first_order"], S_jan, atol=1e-10)
    assert np.allclose(res_shifted["total_order"], ST_jan, atol=1e-10)


def test_analytical_multiplicative_benchmark():
    """
    Test Benchmark 2: Multiplicative Interaction Model
    Y = X0 * X1 (with 2 dummy inputs X2, X3)
    Analytical:
    S_0 = S_1 = 3/7 ~= 0.4286
    S_T0 = S_T1 = 4/7 ~= 0.5714
    S_2 = S_3 = S_T2 = S_T3 = 0.0
    """
    N = 4096
    A, B, A_B = generate_scrambled_sobol_design(d=4, n_base=N, seed=202)
    
    yA = analytical_multiplicative_benchmark(A)
    yB = analytical_multiplicative_benchmark(B)
    yAB = [analytical_multiplicative_benchmark(mat) for mat in A_B]
    
    res = compute_centered_sobol_indices(yA, yB, yAB, estimator="jansen")
    S = res["first_order"]
    ST = res["total_order"]
    
    assert S[0] == pytest.approx(3.0 / 7.0, abs=0.03)
    assert S[1] == pytest.approx(3.0 / 7.0, abs=0.03)
    assert ST[0] == pytest.approx(4.0 / 7.0, abs=0.03)
    assert ST[1] == pytest.approx(4.0 / 7.0, abs=0.03)
    
    # Dummy inputs are zero
    assert abs(S[2]) < 0.02
    assert abs(S[3]) < 0.02
    assert abs(ST[2]) < 0.02
    assert abs(ST[3]) < 0.02


def test_analytical_ishigami_benchmark():
    """
    Test Benchmark 3: Ishigami Non-Linear Function
    Analytical (Saltelli et al. 2008):
    S1 ~= 0.3139, S2 ~= 0.4424, S3 = 0
    ST1 ~= 0.5576, ST2 ~= 0.4424, ST3 ~= 0.2437
    """
    N = 8192
    A, B, A_B = generate_scrambled_sobol_design(d=3, n_base=N, seed=303)
    
    yA = analytical_ishigami_benchmark(A)
    yB = analytical_ishigami_benchmark(B)
    yAB = [analytical_ishigami_benchmark(mat) for mat in A_B]
    
    res = compute_centered_sobol_indices(yA, yB, yAB, estimator="jansen")
    S = res["first_order"]
    ST = res["total_order"]
    
    assert S[0] == pytest.approx(0.3139, abs=0.04)
    assert S[1] == pytest.approx(0.4424, abs=0.04)
    assert abs(S[2]) < 0.025  # True S3 is 0
    
    assert ST[0] == pytest.approx(0.5576, abs=0.04)
    assert ST[1] == pytest.approx(0.4424, abs=0.04)
    assert ST[2] == pytest.approx(0.2437, abs=0.04)


def test_independent_parameters_benchmark():
    """
    Test Benchmark 4: Model independent of one or more parameters
    Y = exp(X0) + sin(X1), independent of X2, X3, X4
    """
    N = 2048
    A, B, A_B = generate_scrambled_sobol_design(d=5, n_base=N, seed=404)
    
    def model(U):
        return np.exp(U[:, 0]) + np.sin(U[:, 1])
        
    yA = model(A)
    yB = model(B)
    yAB = [model(mat) for mat in A_B]
    
    res = compute_centered_sobol_indices(yA, yB, yAB, estimator="jansen")
    S = res["first_order"]
    ST = res["total_order"]
    
    # Active inputs have positive sensitivity
    assert S[0] > 0.3
    assert S[1] > 0.05
    
    # Inputs 2, 3, 4 are completely independent
    for idx in [2, 3, 4]:
        assert abs(S[idx]) < 0.025
        assert abs(ST[idx]) < 0.025


def test_sobol_bootstrap_confidence_intervals():
    """
    Verifies that bootstrap confidence intervals contain the true analytical indices
    and do not collapse or wildly blow up.
    """
    N = 2048
    A, B, A_B = generate_scrambled_sobol_design(d=4, n_base=N, seed=505)
    
    yA = analytical_linear_benchmark(A)
    yB = analytical_linear_benchmark(B)
    yAB = [analytical_linear_benchmark(mat) for mat in A_B]
    
    boot = bootstrap_sobol_confidence_intervals(yA, yB, yAB, n_bootstraps=200, confidence_level=0.95, seed=42)
    expected_S = np.array([4.0 / 14.0, 9.0 / 14.0, 0.0, 1.0 / 14.0])
    
    for i in range(4):
        # 95% CI should bracket true value
        assert boot["first_order_ci_low"][i] <= expected_S[i] + 0.02
        assert boot["first_order_ci_high"][i] >= expected_S[i] - 0.02
        # Standard error should be reasonable (e.g. < 0.05 for N=2048)
        assert boot["first_order_se"][i] < 0.05
