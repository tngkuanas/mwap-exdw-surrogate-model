"""
Unit and Regression Tests for Research Iteration 1: Advanced History Matching & Case Selection.
ExxonMobil DataWorks Challenge 2026.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ensemble.submission_pipeline import validate_deliverable_workbook, RS_SOLUTION
from history_matching.historical_forward_emulator import HistoricalSVDGPEmulator
from history_matching.history_matching_objective import HistoryMatchingObjective
from ensemble.data import load_uncertainty, TRAIN_CASES, VAL_CASES, PARAMS
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
)


@pytest.fixture(scope="module")
def history_matching_data():
    unc_df = load_uncertainty()
    q_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(q_df)
    fh = pd.read_parquet("../data/field_history.parquet")

    X_train = unc_df.loc[TRAIN_CASES, PARAMS].to_numpy()
    Y_oil_tr = np.array([case_dict[c]["oil_cum_stb"].to_numpy() for c in TRAIN_CASES])
    Y_wat_tr = np.array([case_dict[c]["water_cum_stb"].to_numpy() for c in TRAIN_CASES])

    emulator = HistoricalSVDGPEmulator(n_components=4, random_state=42).fit(X_train, Y_oil_tr, Y_wat_tr)
    obj = HistoryMatchingObjective(emulator, fh)
    return unc_df, case_dict, fh, emulator, obj


def test_forward_emulator_accuracy(history_matching_data):
    unc_df, case_dict, _, emulator, _ = history_matching_data
    X_val = unc_df.loc[VAL_CASES, PARAMS].to_numpy()
    Y_oil_val = np.array([case_dict[c]["oil_cum_stb"].to_numpy() for c in VAL_CASES])
    Y_wat_val = np.array([case_dict[c]["water_cum_stb"].to_numpy() for c in VAL_CASES])

    pred = emulator.predict(X_val)

    norm_o = np.sqrt(np.mean(Y_oil_val ** 2, axis=1, keepdims=True))
    norm_w = np.sqrt(np.mean(Y_wat_val ** 2, axis=1, keepdims=True))

    nrmse_o = float(np.mean(np.sqrt(np.mean((pred["oil_cum"] - Y_oil_val) ** 2, axis=1, keepdims=True)) / norm_o))
    nrmse_w = float(np.mean(np.sqrt(np.mean((pred["water_cum"] - Y_wat_val) ** 2, axis=1, keepdims=True)) / norm_w))

    assert nrmse_o < 0.005, f"Oil emulator NRMSE exceeds 0.5%: {nrmse_o}"
    assert nrmse_w < 0.010, f"Water emulator NRMSE exceeds 1.0%: {nrmse_w}"


def test_history_matching_objective_gls(history_matching_data):
    unc_df, _, _, _, obj = history_matching_data
    theta_c6 = unc_df.loc[6, PARAMS].to_numpy()

    comp = obj.evaluate_misfit(theta_c6, objective_type="ar1_gls", return_components=True)
    assert comp["total_misfit"] > 0.0
    assert comp["gls_misfit"] > 0.0
    assert comp["nrmse_Qo"] < 0.05
    assert comp["nrmse_Qw"] < 0.05


def test_selected_10_cases_bounds_and_misfit():
    spec_path = Path("outputs/history_matching/tables/selected_10_cases_specification.csv")
    assert spec_path.exists(), "selected_10_cases_specification.csv missing!"
    df = pd.read_csv(spec_path)
    assert len(df) == 10

    # Physical bounds check
    assert (df["Fault_Transmissibility"] >= 0.04).all() and (df["Fault_Transmissibility"] <= 0.15).all()
    assert (df["Porosity_Multiplier"] >= 0.80).all() and (df["Porosity_Multiplier"] <= 1.50).all()
    assert (df["Permeability_Multiplier"] >= 0.50).all() and (df["Permeability_Multiplier"] <= 9.00).all()
    assert (df["Aquifer_Pore_Volume"] >= 50.0).all() and (df["Aquifer_Pore_Volume"] <= 200.0).all()

    # Misfit validity: all cases must have low historical misfit
    assert (df["GLS_Misfit"] < 45.0).all()
    assert (df["Oil_Cum_NRMSE"] < 0.08).all()


def test_history_matched_deliverable_workbook():
    wb_path = Path("outputs/history_matching/09_Template_Deliverable_History_Matched.xlsx")
    assert wb_path.exists(), "History matched deliverable workbook missing!"

    audit = validate_deliverable_workbook(wb_path)
    assert audit["all_checks_passed"] is True
    assert audit["total_sheets"] == 10
    assert audit["nan_count"] == 0
    assert audit["negative_values_count"] == 0
    assert audit["monotonicity_violations"] == 0
    assert audit["gor_discrepancies"] == 0

    assert audit["sheets"]["Case 1"]["num_quarters"] == 40
    for i in range(2, 11):
        assert audit["sheets"][f"Case {i}"]["num_quarters"] == 80
