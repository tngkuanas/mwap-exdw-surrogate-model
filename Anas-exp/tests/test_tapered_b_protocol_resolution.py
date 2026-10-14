"""
Unit tests for Tapered-B Protocol Resolution, Deployment Tracing, and Deployment Reconciliation.
Project: ExxonMobil DataWorks Challenge 2026
Branch: research/tapered-b-protocol-resolution
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = REPO_ROOT / "outputs" / "research" / "phase_specific_research" / "tables"


def test_simulation_and_production_date_bounds():
    """Confirms all permitted data strictly terminates on or before 2008-01-01."""
    sim_path = REPO_ROOT.parent / "data" / "simulation_timeseries.parquet"
    if sim_path.exists():
        df_sim = pd.read_parquet(sim_path)
        assert df_sim["date"].max() == pd.Timestamp("2008-01-01")
        
    prod_path = REPO_ROOT.parent / "data" / "shared" / "production_timeseries.parquet"
    if prod_path.exists():
        df_prod = pd.read_parquet(prod_path)
        assert df_prod["date"].max() == pd.Timestamp("2008-01-01")


def test_deployment_fitting_path_is_strictly_protocol_a():
    """
    Confirms that at deployment origin 2008-01-01, post-origin training records are empty,
    forcing LearnedDCARateForecaster into historical pre-origin fitting (Protocol A equivalence).
    """
    from ensemble.data import load_uncertainty, resolve_shared_dir, TRAIN_CASES
    from ensemble.run_horizon_adaptive_stacking_experiment import (
        load_clean_simulation_quarterly,
        build_case_quarterly_trajectories,
        LearnedDCARateForecaster,
    )
    
    raw_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(raw_df)
    unc_df = load_uncertainty(resolve_shared_dir())
    
    origin = pd.Timestamp("2008-01-01")
    for c in TRAIN_CASES[:5]:
        df_c = case_dict[c]
        post = df_c[df_c["date"] > origin]
        assert len(post) == 0, f"Case {c} unexpectedly has post-2008 data!"


def test_tapered_b_head_to_head_reproducibility():
    """Confirms paired evaluation metrics across 15 held-out reservoirs x 17 conditions."""
    cases_csv = TABLE_DIR / "decisive_nested_cv_case_evaluations.csv"
    assert cases_csv.exists()
    
    df = pd.read_csv(cases_csv)
    sub = df[df["candidate_id"].isin(["Candidate_A_Frozen_Baseline", "Candidate_B_Tapered_Exp"])]
    piv = sub.pivot_table(index=["case_id", "origin", "horizon_quarters"], columns="candidate_id", values="macro_nrmse")
    
    assert len(piv) == 255
    diff = piv["Candidate_B_Tapered_Exp"] - piv["Candidate_A_Frozen_Baseline"]
    
    # Check overall difference
    assert np.isclose(diff.mean(), -0.01126365, atol=1e-5)
    
    # Check reservoir clustering
    res_diff = diff.groupby("case_id").mean()
    assert len(res_diff) == 15
    assert np.isclose(res_diff.mean(), -0.01126365, atol=1e-5)
    assert np.isclose(res_diff.std(ddof=1) / np.sqrt(15), 0.0030052, atol=1e-5)


def test_deployment_reconciliation_csv_values():
    """Confirms numerical reconciliation of deployment table with committed CSV."""
    dep_csv = TABLE_DIR / "deployment_2008_summary_comparison.csv"
    assert dep_csv.exists()
    
    df = pd.read_csv(dep_csv)
    b_taper = df[df["candidate_id"] == "Candidate_B_Tapered_Exp"].iloc[0]
    
    # Case 1 (40Q / 10-year)
    assert np.isclose(b_taper["case1_40q_eur_oil_stb"], 46436015.03, atol=1e-1)
    assert np.isclose(b_taper["case1_40q_inc_oil_stb"], 8057698.06, atol=1e-1)
    assert np.isclose(b_taper["case1_40q_terminal_oil_rate_stbd"], 729.21, atol=1e-2)
    assert np.isclose(b_taper["case1_40q_terminal_water_cut"], 0.94068, atol=1e-4)
    
    # Cases 2-10 (80Q / 20-year)
    assert np.isclose(b_taper["cases2_10_mean_20yr_eur_oil_stb"], 47584443.85, atol=1e-1)
    assert np.isclose(b_taper["cases2_10_mean_inc_oil_stb"], 9206126.87, atol=1e-1)
    assert np.isclose(b_taper["cases2_10_mean_terminal_oil_rate_stbd"], 104.99, atol=1e-2)
    assert np.isclose(b_taper["cases2_10_mean_terminal_water_cut"], 0.99249, atol=1e-4)
