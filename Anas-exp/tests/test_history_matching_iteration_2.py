"""
Module: test_history_matching_iteration_2.py
Regression and unit test suite for Research Iteration 2:
1. Verifies time-aligned rate differentiation.
2. Verifies Cholesky positive definiteness and finite log-likelihood values.
3. Verifies Vehtari et al. (2021) MCMC convergence diagnostics (split-R-hat < 1.05).
4. Verifies parameter identifiability of porosity multiplier.
5. Verifies weak-case discrimination (Case 3 vs Case 8).
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def test_forward_rate_time_alignment():
    """Confirms forward production rates match discrete difference over calendar days."""
    dates = pd.date_range("1998-01-01", "2008-01-01", freq="QS")
    days = np.diff(dates).astype("timedelta64[D]").astype(float)
    assert len(days) == 40
    assert np.all((days >= 90) & (days <= 92))

    # Test differencing operator
    from history_matching_v2.emulator_revalidation_v2 import HistoricalSVDGPEmulatorV2
    emu = HistoricalSVDGPEmulatorV2()
    # Dummy fit to construct diff_matrix_fwd
    X_dummy = np.random.uniform(0.1, 0.9, size=(5, 4))
    Y_dummy = np.cumsum(np.random.uniform(1e5, 5e5, size=(5, 41)), axis=1)
    emu.fit(X_dummy, Y_dummy, Y_dummy, dates=dates)

    D = emu.diff_matrix_fwd
    assert D.shape == (40, 41)
    diff_rates = (D @ Y_dummy.T).T
    expected_rates = np.diff(Y_dummy, axis=1) / days
    np.testing.assert_allclose(diff_rates, expected_rates, rtol=1e-5)


def test_cholesky_log_likelihood_positive_definite():
    """Confirms Cholesky solves succeed and log-likelihood is finite and negative."""
    from history_matching_v2.emulator_revalidation_v2 import HistoricalSVDGPEmulatorV2
    from history_matching_v2.history_matching_likelihood_v2 import HistoryMatchingLikelihoodV2
    from ensemble.data import load_uncertainty, TRAIN_CASES, PARAMS
    from ensemble.run_horizon_adaptive_stacking_experiment import (
        load_clean_simulation_quarterly,
        build_case_quarterly_trajectories,
    )

    unc_df = load_uncertainty()
    q_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(q_df)
    fh = pd.read_parquet("../data/field_history.parquet")
    dates = pd.to_datetime(case_dict[1]["date"].to_numpy())

    X_train = unc_df.loc[TRAIN_CASES[:10], PARAMS].to_numpy()
    Y_oil_tr = np.array([case_dict[c]["oil_cum_stb"].to_numpy() for c in TRAIN_CASES[:10]])
    Y_wat_tr = np.array([case_dict[c]["water_cum_stb"].to_numpy() for c in TRAIN_CASES[:10]])

    emu = HistoricalSVDGPEmulatorV2(n_components=3, random_state=42).fit(X_train, Y_oil_tr, Y_wat_tr, dates=dates)
    lik = HistoryMatchingLikelihoodV2(emu, fh)

    th = np.array([0.10, 1.25, 4.0, 100.0])
    for model in ["L1", "L2", "L3"]:
        comp = lik.evaluate_log_likelihood(th, model=model, return_components=True)
        assert np.isfinite(comp["log_likelihood"])
        assert comp["log_likelihood"] < 0
        assert comp["quad_oil"] >= 0
        assert comp["quad_wat"] >= 0
        assert comp["logdet_cov"] > 0


def test_vehtari_diagnostics_convergence():
    """Verifies that MCMC diagnostic table satisfies Vehtari et al. (2021) criteria."""
    diag_path = Path("outputs/research/history_matching_iteration_2/tables/bayesian_posterior_diagnostics_v2.csv")
    if not diag_path.exists():
        pytest.skip("Diagnostic file not yet generated")

    df_diag = pd.read_csv(diag_path)
    # R-hat < 1.05
    assert (df_diag["rank_normalized_split_rhat"] < 1.05).all()
    assert (df_diag["folded_split_rhat"] < 1.05).all()
    # ESS > 400
    assert (df_diag["ess_bulk"] > 400).all()
    assert (df_diag["ess_tail"] > 400).all()


def test_porosity_identifiability():
    """Verifies that porosity multiplier exhibits significant variance reduction."""
    summary_path = Path("outputs/research/history_matching_iteration_2/tables/bayesian_posterior_summary_v2.csv")
    if not summary_path.exists():
        pytest.skip("Summary file not yet generated")

    df_summary = pd.read_csv(summary_path)
    poro_row = df_summary[df_summary["parameter"] == "Porosity Multiplier"].iloc[0]
    assert poro_row["identifiability_index"] > 0.50, "Porosity identifiability should exceed 50% variance reduction"


def test_weak_case_rejection():
    """Verifies that weak history match (Case 3) is separated from good match (Case 8)."""
    weak_path = Path("outputs/research/history_matching_iteration_2/tables/weak_case_diagnostic_comparison.csv")
    if not weak_path.exists():
        pytest.skip("Weak case diagnostic file not yet generated")

    df_weak = pd.read_csv(weak_path)
    c3 = df_weak[df_weak["Case_Sheet"] == "Case 3"].iloc[0]
    c8 = df_weak[df_weak["Case_Sheet"] == "Case 8"].iloc[0]

    # In Iteration 1, difference was tiny (< 0.6)
    assert abs(c3["Iter1_GLS_Misfit"] - c8["Iter1_GLS_Misfit"]) < 1.0
    # In Corrected L1, difference must be huge (> 20 log-likelihood units)
    log_diff = c8["Corrected_L1_LogLik"] - c3["Corrected_L1_LogLik"]
    assert log_diff > 20.0, f"Case 8 should have much higher log likelihood than Case 3, got diff {log_diff}"
