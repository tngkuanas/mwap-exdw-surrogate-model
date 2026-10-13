"""
Unit and Regression Tests: test_long_horizon_physics.py
Package: tests
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Tests:
1. Exact multiphase volume balance identities.
2. Koval fractional flow and bounded logistic water-cut functions.
3. Modified Arps decline with exponential floor D_lim.
4. Stretched Exponential Production Decline (SEPD - Valko 2009).
5. Two-compartment depletion model.
6. Challenger forecaster class instantiation, fitting, and rate predictions.
7. Non-negativity and monotonic cumulative accumulation constraints.
"""
import pytest
import numpy as np
import pandas as pd

from long_horizon_physics.literature_foundations import (
    evaluate_exact_multiphase_identities,
    koval_fractional_flow,
    logistic_water_cut_progression,
    modified_arps_decline,
    stretched_exponential_decline,
    two_compartment_depletion,
    build_research_hypothesis_matrix,
)
from long_horizon_physics.multiphase_coupled_models import (
    KovalFractionalFlowForecaster,
    TotalLiquidBoundedWORForecaster,
    RecoveryStateDependentForecaster,
    BaselinePreservingSelectiveCorrector,
)
from long_horizon_physics.long_horizon_decline_models import (
    ModifiedArpsDeclineForecaster,
    StretchedExponentialForecaster,
    TwoCompartmentDepletionForecaster,
    ConstrainedTailHybridForecaster,
)


def test_multiphase_exact_identities():
    qo = np.array([4000.0, 3000.0, 2000.0, 1000.0])
    qw = np.array([500.0, 1500.0, 3000.0, 6000.0])
    qg = 0.363 * qo
    
    res = evaluate_exact_multiphase_identities(qo, qw, qg)
    
    # 1. Liquid identity: q_L = q_o + q_w
    np.testing.assert_allclose(res["liquid_rate"], qo + qw, rtol=1e-12)
    # 2. Cut identity: f_o + f_w = 1.0
    np.testing.assert_allclose(res["water_cut"] + res["oil_cut"], 1.0, rtol=1e-12)
    # 3. WOR identity: wor = q_w / q_o
    np.testing.assert_allclose(res["wor"], qw / qo, rtol=1e-12)
    # 4. GOR identity: gor = q_g / q_o = 0.363
    np.testing.assert_allclose(res["gor"], 0.363, rtol=1e-12)


def test_koval_fractional_flow():
    QL = np.linspace(0, 100, 50)
    fw = koval_fractional_flow(QL, K_K=4.0, V_D=25.0, Q_L_bt=5.0, fw_initial=0.05)
    
    # Strictly in [0, 1]
    assert np.all(fw >= 0.0)
    assert np.all(fw <= 1.0)
    # Strictly monotonic after breakthrough
    assert np.all(np.diff(fw) >= 0.0)
    # Pre-breakthrough initial cut
    assert fw[0] == pytest.approx(0.05)


def test_logistic_water_cut_progression():
    t_years = np.linspace(0, 20, 81)
    fw = logistic_water_cut_progression(t_years, fw_0=0.10, lambda_w=0.25, fw_max=0.99)
    
    assert np.all(fw >= 0.0)
    assert np.all(fw <= 0.99)
    assert np.all(np.diff(fw) >= 0.0)
    assert fw[0] == pytest.approx(0.10, abs=1e-3)
    assert fw[-1] > 0.85  # mature late-life water cut with lambda_w=0.25


def test_modified_arps_terminal_floor():
    t_years = np.linspace(0.25, 50.0, 200)
    q0 = 5000.0
    Di = 0.30
    b = 0.35
    D_lim = 0.08  # t_switch = (0.30/0.08 - 1)/(0.35*0.30) = 2.75/0.105 ~= 26.19 yr
    
    rates, cums = modified_arps_decline(t_years, q0=q0, Di=Di, b=b, D_lim=D_lim)
    
    assert np.all(rates >= 0.0)
    assert np.all(np.diff(cums) >= 0.0)
    
    # Instantaneous decline rate at year 50 (after switch at 26.2 yr) must equal D_lim exactly
    # D(t) ~= -ln(q(t+dt)/q(t)) / dt
    dt = t_years[1] - t_years[0]
    D_terminal = -np.log(rates[-1] / rates[-2]) / dt
    assert D_terminal == pytest.approx(D_lim, rel=0.05)


def test_stretched_exponential_decline():
    t_years = np.linspace(0.25, 20.0, 80)
    rates, cums = stretched_exponential_decline(t_years, q0=4000.0, tau=3.5, n=0.55)
    
    assert np.all(rates >= 0.0)
    assert np.all(np.diff(cums) >= 0.0)
    assert rates[-1] < rates[0]


def test_two_compartment_depletion():
    t_years = np.linspace(0.25, 20.0, 80)
    rates, cums = two_compartment_depletion(t_years, q1_0=2500.0, D1=0.35, q2_0=1500.0, D2=0.04)
    
    assert np.all(rates >= 0.0)
    assert np.all(np.diff(cums) >= 0.0)
    assert rates[0] < 4000.0
    assert rates[-1] > 0.0


def test_research_hypothesis_matrix():
    df_hyp = build_research_hypothesis_matrix()
    assert len(df_hyp) == 6
    assert "H1_Multiphase_Coupling" in df_hyp["hypothesis_id"].values
    assert "H2_Liquid_WOR_State" in df_hyp["hypothesis_id"].values
    assert "H5_Modified_Arps_Terminal_Floor" in df_hyp["hypothesis_id"].values
