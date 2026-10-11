"""
Module: reproduce_production_baseline.py
Package: conditional_challengers
Addresses: Section 1 — Correct the 20-Year Forecasting Experiment
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Reproduces the production audit's 2028 oil cumulatives for Cases 2-10 (approximately 51.17-51.19 MMSTB)
and Case 1 (47.65 MMSTB) using the identical Case 0 historical anchor and production scenario loader.
Treats failure to reproduce as a blocking defect.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import openpyxl

from ensemble.data import load_uncertainty, resolve_shared_dir, TRAIN_CASES
from ensemble.scenario_loader import load_production_scenarios
from ensemble.generate_competition_deliverable import (
    load_locked_config,
    fit_locked_base_models,
    generate_scenario_predictions,
    CONFIG_PATH_DEFAULT,
    OUTPUT_PATH_DEFAULT,
)
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
)

TABLE_DIR = Path("outputs/research/conditional_challengers/tables")
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def extract_field_case0_anchor() -> Dict[str, Any]:
    """Extracts the authoritative historical Field Case 0 anchor at 2008-01-01."""
    shared_dir = resolve_shared_dir()
    prod = pd.read_parquet(shared_dir / "production_timeseries.parquet")
    c0 = prod[prod["case_num"] == 0]
    c0_2008 = c0[c0["date"] == "2008-01-01"].set_index("metric")["value"]
    
    return {
        "qo": float(c0_2008["oil_rate"]),
        "qw": float(c0_2008["water_rate"]),
        "qg": float(c0_2008["gas_rate"]),
        "Qo": float(c0_2008["oil_cum"]),
        "Qw": float(c0_2008["water_cum"]),
        "Qg": float(c0_2008["gas_cum"]),
        "wor": float(c0_2008["water_rate"] / max(c0_2008["oil_rate"], 1.0)),
        "res_age": 10.0,
        "origin_date": pd.Timestamp("2008-01-01"),
    }


def verify_production_baseline_reproduction(
    workbook_path: Path = OUTPUT_PATH_DEFAULT,
) -> pd.DataFrame:
    """
    Executes identical production pipeline and verifies exact match against official deliverable.
    """
    print("\n" + "=" * 80)
    print("VERIFYING PRODUCTION BASELINE REPRODUCTION (CASE 0 ANCHOR)")
    print("=" * 80)
    
    field_anchor = extract_field_case0_anchor()
    print("Case 0 Field Anchor (2008-01-01):")
    print(f"  Qo: {field_anchor['Qo']:>16,.2f} STB | qw: {field_anchor['qw']:>10,.2f} STB/d")
    print(f"  Qw: {field_anchor['Qw']:>16,.2f} STB | qo: {field_anchor['qo']:>10,.2f} STB/d")
    print(f"  Qg: {field_anchor['Qg']:>16,.2f} MSCF | qg: {field_anchor['qg']:>10,.2f} MSCF/d")
    
    config = load_locked_config()
    weights = config["weights"]
    inference_origin = pd.Timestamp("2008-01-01")
    
    shared_dir = resolve_shared_dir()
    unc_df = load_uncertainty(shared_dir)
    raw_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(raw_df)
    
    # Fit locked base models on permitted training cases (Cases 1-70)
    models = fit_locked_base_models(case_dict, unc_df, TRAIN_CASES, inference_origin)
    
    dates_40q = pd.date_range("2008-04-01", "2018-01-01", freq="QS")
    dates_80q = pd.date_range("2008-04-01", "2028-01-01", freq="QS")
    
    scenarios = load_production_scenarios()
    
    # Load deliverable workbook for exact comparison
    assert workbook_path.exists(), f"Official deliverable not found at: {workbook_path}"
    wb = openpyxl.load_workbook(workbook_path, data_only=True)
    
    records = []
    max_abs_diff_o = 0.0
    max_abs_diff_w = 0.0
    
    print("\nChecking reproduction against official submission workbook:")
    print("-" * 90)
    print(f"{'Sheet':<8} | {'Qtrs':<5} | {'Repro Oil Cum':<14} | {'Target Oil Cum':<14} | {'Abs Diff':<10} | {'Status'}")
    print("-" * 90)
    
    for sc in scenarios:
        case_sheet = sc.case_id
        dates = dates_40q if sc.horizon_quarters == 40 else dates_80q
        
        df_pred, aud = generate_scenario_predictions(
            models=models,
            weights=weights,
            init_state=field_anchor,
            params=sc.params,
            forecast_dates=dates,
            decline_sensitivity=1.0,
        )
        
        # Read target sheet
        ws = wb[case_sheet]
        target_rows = list(ws.iter_rows(values_only=True))
        # Data starts at row 2
        target_data = [r for r in target_rows[1:] if r[0] is not None]
        assert len(target_data) == sc.horizon_quarters, (
            f"Sheet {case_sheet} has {len(target_data)} rows, expected {sc.horizon_quarters}"
        )
        
        target_oil_final = float(target_data[-1][2])
        target_wat_final = float(target_data[-1][3])
        target_gas_final = float(target_data[-1][1])
        
        repro_oil_final = float(df_pred["cum_oil_stb"].iloc[-1])
        repro_wat_final = float(df_pred["cum_water_stb"].iloc[-1])
        repro_gas_final = float(df_pred["cum_gas_mscf"].iloc[-1])
        
        diff_o = abs(repro_oil_final - target_oil_final)
        diff_w = abs(repro_wat_final - target_wat_final)
        diff_g = abs(repro_gas_final - target_gas_final)
        
        max_abs_diff_o = max(max_abs_diff_o, diff_o)
        max_abs_diff_w = max(max_abs_diff_w, diff_w)
        
        status = "EXACT MATCH" if diff_o < 0.05 and diff_w < 0.05 else "MISMATCH"
        
        records.append({
            "case_id": case_sheet,
            "candidate_id": sc.candidate_id,
            "horizon_quarters": sc.horizon_quarters,
            "repro_oil_cum_stb": repro_oil_final,
            "target_oil_cum_stb": target_oil_final,
            "abs_diff_oil_stb": diff_o,
            "repro_wat_cum_stb": repro_wat_final,
            "target_wat_cum_stb": target_wat_final,
            "abs_diff_wat_stb": diff_w,
            "repro_gas_cum_mscf": repro_gas_final,
            "target_gas_cum_mscf": target_gas_final,
            "status": status,
        })
        
        print(f"{case_sheet:<8} | {sc.horizon_quarters:<5} | {repro_oil_final:>14,.2f} | {target_oil_final:>14,.2f} | {diff_o:>10.4f} | {status}")
        
    print("-" * 90)
    print(f"Max Oil Cum Absolute Difference:   {max_abs_diff_o:.6f} STB")
    print(f"Max Water Cum Absolute Difference: {max_abs_diff_w:.6f} STB")
    
    assert max_abs_diff_o < 0.05, f"Blocking reproduction defect: Max oil difference {max_abs_diff_o} >= 0.05 STB"
    assert max_abs_diff_w < 0.05, f"Blocking reproduction defect: Max water difference {max_abs_diff_w} >= 0.05 STB"
    print("\n>>> PRODUCTION REPRODUCTION AUDIT: PASSED (ZERO DEFECTS) <<<")
    
    df_repro = pd.DataFrame(records)
    out_path = TABLE_DIR / "production_baseline_reproduction.csv"
    df_repro.to_csv(out_path, index=False)
    print(f"Saved Reproduction Table: {out_path}\n")
    return df_repro


if __name__ == "__main__":
    verify_production_baseline_reproduction()
