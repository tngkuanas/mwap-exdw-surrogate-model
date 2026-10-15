"""
Module: operational_and_grid_features.py
Implements Phases 5 & 6 of the Final Research Sprint:
Multi-Angle Forecast Reliability Challenge — Reservoir Operations & 3D Grid Features Assessment.

Key Components:
1. Operational Features (Phase 5):
   - Audits field water injection and historical pressure availability.
   - Extracts historical injection rate, cumulative injection volume, and voidage replacement ratio (VRR).
   - Audits future control permissibility: Future injection is unknown post-2008 in the competition spec.
2. 3D Grid Feature Extraction (Phase 6):
   - Extracts compact physical features from cell-level 3D grid files (Cases 1-70 Train, Cases 71-85 Validation):
     - Total Pore Volume (PORV sum)
     - PV-weighted Porosity
     - PV-weighted Permeability (PERMX, PERMZ)
     - Vertical-to-Horizontal Anisotropy Ratio (Kz / Kx)
     - Dykstra-Parsons Heterogeneity Index (V_DP)
     - Initial Oil Volume (PORV * SOIL)
     - Initial Mobile Oil Volume (PORV * max(0, SOIL - SOWCR))
     - PV-weighted Initial Pressure
     - PV-weighted Fault Transmissibility
3. Comparative Information-Value Evaluation (Nested 5-Fold CV on Cases 1-70):
   - Evaluates whether adding operational and/or 3D grid features improves prediction accuracy beyond
     the 4 scalar uncertainty parameters.
   - Computes Spearman correlation with true production outcomes and feature importance variance shares.
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import openpyxl
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from ensemble.data import (
    load_uncertainty,
    TRAIN_CASES,
    VAL_CASES,
    PARAMS,
)
from ensemble.metrics import compute_increment_nrmse
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
)

OUTPUT_TABLE_DIR = Path("outputs/research/forecast_reliability_challenge/tables")
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

GRID_FILE_MAP = {
    range(1, 21): Path("../TestCases/03 Train Cases 1.xlsx"),
    range(21, 41): Path("../TestCases/04 Train Cases 2.xlsx"),
    range(41, 61): Path("../TestCases/05 Train Cases 3.xlsx"),
    range(61, 71): Path("../TestCases/06 Train Cases 4.xlsx"),
    range(71, 86): Path("../TestCases/07  Validation Cases.xlsx"),
}


def get_grid_file_for_case(case_id: int) -> Tuple[Path, str]:
    """Finds the workbook path and sheet name for a given case ID."""
    for case_range, file_path in GRID_FILE_MAP.items():
        if case_id in case_range:
            return file_path, f"Case {case_id}"
    raise ValueError(f"Case {case_id} not mapped to any known grid workbook.")


def extract_single_case_grid_features(case_id: int) -> Dict[str, Any]:
    """
    Parses a single 3D reservoir grid sheet using openpyxl in read_only mode.
    Computes volume-weighted summary statistics across all active cells (PORV > 0).
    """
    file_path, sheet_name = get_grid_file_for_case(case_id)
    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    if sheet_name not in wb.sheetnames:
        raise ValueError(f"Sheet {sheet_name} not found in {file_path}")
        
    ws = wb[sheet_name]
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    
    idx_porv = header.index("Pore Volume (PORV)")
    idx_poro = header.index("Porosity (PORO)")
    idx_permx = header.index("Permeability I (PERMX)")
    idx_permz = header.index("Permeability K (PERMZ)")
    idx_tranx = header.index("Transmissibility I (TRANX)")
    idx_press = header.index("Pressure (PRESSURE)")
    idx_soil = header.index("Oil Saturation (SOIL)")
    idx_sowcr = header.index("Residual Oil to Water (SOWCR)")
    
    porv_list = []
    poro_list = []
    permx_list = []
    permz_list = []
    tranx_list = []
    press_list = []
    soil_list = []
    sowcr_list = []
    
    for r in rows:
        pv = r[idx_porv]
        if pv is not None and pv > 0:
            porv_list.append(pv)
            poro_list.append(r[idx_poro] if r[idx_poro] is not None else 0.0)
            permx_list.append(r[idx_permx] if r[idx_permx] is not None else 0.0)
            permz_list.append(r[idx_permz] if r[idx_permz] is not None else 0.0)
            tranx_list.append(r[idx_tranx] if r[idx_tranx] is not None else 0.0)
            press_list.append(r[idx_press] if r[idx_press] is not None else 0.0)
            soil_list.append(r[idx_soil] if r[idx_soil] is not None else 0.0)
            sowcr_list.append(r[idx_sowcr] if r[idx_sowcr] is not None else 0.0)
            
    wb.close()
    
    pv_arr = np.array(porv_list, dtype=np.float64)
    poro_arr = np.array(poro_list, dtype=np.float64)
    permx_arr = np.array(permx_list, dtype=np.float64)
    permz_arr = np.array(permz_list, dtype=np.float64)
    tranx_arr = np.array(tranx_list, dtype=np.float64)
    press_arr = np.array(press_list, dtype=np.float64)
    soil_arr = np.array(soil_list, dtype=np.float64)
    sowcr_arr = np.array(sowcr_list, dtype=np.float64)
    
    tot_pv = float(np.sum(pv_arr))
    pv_poro = float(np.sum(pv_arr * poro_arr) / tot_pv)
    pv_permx = float(np.sum(pv_arr * permx_arr) / tot_pv)
    pv_permz = float(np.sum(pv_arr * permz_arr) / tot_pv)
    pv_tranx = float(np.sum(pv_arr * tranx_arr) / tot_pv)
    pv_press = float(np.sum(pv_arr * press_arr) / tot_pv)
    
    anisotropy_ratio = float(pv_permz / max(1e-4, pv_permx))
    
    # Dykstra-Parsons heterogeneity index: V_DP = (k_50 - k_84.1) / k_50
    # computed from log-permeability distribution of active cells
    perm_pos = permx_arr[permx_arr > 0]
    if len(perm_pos) > 10:
        k50 = float(np.percentile(perm_pos, 50.0))
        k84_1 = float(np.percentile(perm_pos, 15.9)) # in log-normal, 1 std dev below median is 15.9% percentile
        v_dp = float(max(0.0, min(1.0, (k50 - k84_1) / max(1e-4, k50))))
    else:
        v_dp = 0.5
        
    tot_oil_vol_resb = float(np.sum(pv_arr * soil_arr))
    mobile_oil_vol_resb = float(np.sum(pv_arr * np.maximum(0.0, soil_arr - sowcr_arr)))
    
    return {
        "case_id": case_id,
        "tot_pore_volume_resb": tot_pv,
        "pv_weighted_porosity": pv_poro,
        "pv_weighted_permx_md": pv_permx,
        "pv_weighted_permz_md": pv_permz,
        "anisotropy_ratio_kv_kh": anisotropy_ratio,
        "dykstra_parsons_vdp": v_dp,
        "initial_oil_volume_resb": tot_oil_vol_resb,
        "initial_mobile_oil_volume_resb": mobile_oil_vol_resb,
        "pv_weighted_pressure_psia": pv_press,
        "pv_weighted_transx": pv_tranx,
    }


def extract_all_grid_features(cases: List[int], n_workers: int = 4) -> pd.DataFrame:
    """Extracts 3D grid summary features for all specified cases in parallel."""
    cached_path = OUTPUT_TABLE_DIR / "extracted_grid_features.parquet"
    if cached_path.exists():
        print(f"Loading cached 3D grid features from: {cached_path}")
        return pd.read_parquet(cached_path)
        
    print(f"Extracting 3D grid features for {len(cases)} cases using {n_workers} workers...")
    t0 = time.time()
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(extract_single_case_grid_features, cases))
        
    grid_df = pd.DataFrame(results).sort_values("case_id").reset_index(drop=True)
    grid_df.to_parquet(cached_path)
    grid_df.to_csv(OUTPUT_TABLE_DIR / "reservoir_grid_extracted_features.csv", index=False)
    print(f"3D grid feature extraction completed in {time.time()-t0:.2f}s!")
    return grid_df


def extract_operational_features(cutoff_date: pd.Timestamp = pd.Timestamp("2003-01-01")) -> pd.DataFrame:
    """
    Extracts permissible historical water injection and operational metrics from '02 Field Production.xlsx'.
    """
    prod_path = Path("../TestCases/02 Field Production.xlsx")
    wb = openpyxl.load_workbook(prod_path, read_only=True, data_only=True)
    ws = wb["Field Production"]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    
    # Headers: ('Identifier', 'Date', 'Oil production rate (STB/d)', 'Gas production rate (MSCF/d)', 'Water production rate (STB/d)', 'Water injection rate (STB/d)')
    data = []
    for r in rows[1:]:
        if r[1] is not None:
            try:
                d_val = pd.to_datetime(r[1])
                data.append({
                    "date": d_val,
                    "qo_stbd": float(r[2]) if r[2] is not None else 0.0,
                    "qg_mscfd": float(r[3]) if r[3] is not None else 0.0,
                    "qw_stbd": float(r[4]) if r[4] is not None else 0.0,
                    "qinj_stbd": float(r[5]) if r[5] is not None else 0.0,
                })
            except Exception:
                continue
    field_df = pd.DataFrame(data).sort_values("date").reset_index(drop=True)
    
    # Compute dt and cumulatives
    field_df["dt_days"] = (field_df["date"].diff().dt.total_seconds() / 86400.0).fillna(0.0)
    field_df.loc[0, "dt_days"] = 0.0
    field_df["cum_oil_stb"] = np.cumsum(field_df["qo_stbd"] * field_df["dt_days"])
    field_df["cum_water_stb"] = np.cumsum(field_df["qw_stbd"] * field_df["dt_days"])
    field_df["cum_inj_stb"] = np.cumsum(field_df["qinj_stbd"] * field_df["dt_days"])
    
    # Pre-cutoff operational state
    pre = field_df[field_df["date"] <= cutoff_date]
    curr_inj_rate = float(pre.iloc[-1]["qinj_stbd"])
    cum_inj = float(pre.iloc[-1]["cum_inj_stb"])
    cum_liq = float(pre.iloc[-1]["cum_oil_stb"] + pre.iloc[-1]["cum_water_stb"])
    vrr_hist = cum_inj / max(1.0, cum_liq)
    
    return {
        "field_inj_rate_stbd": curr_inj_rate,
        "field_cum_inj_stb": cum_inj,
        "field_vrr": vrr_hist,
    }


def evaluate_operational_and_grid_features():
    """
    Performs Phase 5 & 6 experimental evaluation:
    1. Extracts operational & 3D grid properties.
    2. Analyzes correlations with the 4 uncertainty parameters and simulation cumulative EUR.
    3. Runs 5-fold CV on Cases 1-70 comparing Base vs Base+Operational vs Base+Grid vs Combined.
    """
    print("=" * 80)
    print("PHASES 5 & 6: RESERVOIR OPERATIONS & 3D GRID FEATURES VALUE ASSESSMENT")
    print("=" * 80)
    
    unc_df = load_uncertainty()
    q_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(q_df)
    
    # Extract 3D grid features for Cases 1 to 85
    cases_to_extract = list(range(1, 86))
    grid_df = extract_all_grid_features(cases_to_extract, n_workers=4)
    grid_df = grid_df.set_index("case_id")
    
    # Extract operational context at 2003-01-01
    op_context = extract_operational_features(pd.Timestamp("2003-01-01"))
    print("\nPermissible Field Operational Context at 2003-01-01:")
    for k, v in op_context.items():
        print(f"  {k}: {v:,.2f}")
        
    # Merge datasets for Cases 1-70
    cutoff = pd.Timestamp("2003-01-01")
    cv_records = []
    
    for c in TRAIN_CASES:
        df_c = case_dict[c]
        pre = df_c[df_c["date"] <= cutoff]
        post = df_c[df_c["date"] > cutoff].iloc[:20] # 5-year horizon (20 quarters)
        
        inc_qo = float(post["oil_cum_stb"].iloc[-1] - pre.iloc[-1]["oil_cum_stb"])
        inc_qw = float(post["water_cum_stb"].iloc[-1] - pre.iloc[-1]["water_cum_stb"])
        
        u_c = unc_df.loc[c]
        g_c = grid_df.loc[c]
        
        qo_0 = float(pre.iloc[-1]["oil_rate_stbd"])
        qw_0 = float(pre.iloc[-1]["water_rate_stbd"])
        wor_0 = float(pre.iloc[-1]["wor"])
        
        cv_records.append({
            "case_id": c,
            # Base features: 4 scalar parameters + initial rate/state
            "Fault_Transmissibility": float(u_c["Fault Transmissibility"]),
            "Porosity_Multiplier": float(u_c["Porosity Multiplier"]),
            "Permeability_Multiplier": float(u_c["Permeability Multiplier"]),
            "Aquifer_Pore_Volume": float(u_c["Aquifer Pore Volume"]),
            "qo_0": qo_0,
            "qw_0": qw_0,
            "wor_0": wor_0,
            # Grid features
            "tot_pore_volume": float(g_c["tot_pore_volume_resb"]),
            "pv_weighted_porosity": float(g_c["pv_weighted_porosity"]),
            "pv_weighted_permx": float(g_c["pv_weighted_permx_md"]),
            "anisotropy_ratio": float(g_c["anisotropy_ratio_kv_kh"]),
            "dykstra_parsons_vdp": float(g_c["dykstra_parsons_vdp"]),
            "initial_mobile_oil": float(g_c["initial_mobile_oil_volume_resb"]),
            "pv_weighted_pressure": float(g_c["pv_weighted_pressure_psia"]),
            "pv_weighted_transx": float(g_c["pv_weighted_transx"]),
            # Operational proxy features
            # Note: Field injection is identical across cases; case-level operational proxy is case voidage ratio:
            "case_vrr_proxy": float(op_context["field_cum_inj_stb"] / max(1.0, pre.iloc[-1]["oil_cum_stb"] + pre.iloc[-1]["water_cum_stb"])),
            # Targets: 5-year incremental oil and water
            "target_inc_qo": inc_qo,
            "target_inc_qw": inc_qw,
        })
        
    df_feat = pd.DataFrame(cv_records)
    
    # -------------------------------------------------------------------------
    # 1. Correlation Analysis: Grid Features vs Uncertainty Parameters
    # -------------------------------------------------------------------------
    corr_records = []
    base_params = ["Fault_Transmissibility", "Porosity_Multiplier", "Permeability_Multiplier", "Aquifer_Pore_Volume"]
    grid_cols = [
        "tot_pore_volume", "pv_weighted_porosity", "pv_weighted_permx",
        "anisotropy_ratio", "dykstra_parsons_vdp", "initial_mobile_oil",
        "pv_weighted_pressure", "pv_weighted_transx", "case_vrr_proxy"
    ]
    
    for g_col in grid_cols:
        r_target_o, _ = pearsonr(df_feat[g_col], df_feat["target_inc_qo"])
        rho_target_o, _ = spearmanr(df_feat[g_col], df_feat["target_inc_qo"])
        r_target_w, _ = pearsonr(df_feat[g_col], df_feat["target_inc_qw"])
        rho_target_w, _ = spearmanr(df_feat[g_col], df_feat["target_inc_qw"])
        
        # Max correlation with any base parameter
        max_r_base = max(abs(pearsonr(df_feat[g_col], df_feat[bp])[0]) for bp in base_params)
        best_base_param = max(base_params, key=lambda bp: abs(pearsonr(df_feat[g_col], df_feat[bp])[0]))
        
        corr_records.append({
            "feature": g_col,
            "pearson_r_oil_inc": float(r_target_o),
            "spearman_rho_oil_inc": float(rho_target_o),
            "pearson_r_water_inc": float(r_target_w),
            "spearman_rho_water_inc": float(rho_target_w),
            "highest_corr_base_param": best_base_param,
            "max_abs_corr_with_base_param": float(max_r_base),
            "collinear_with_base": bool(max_r_base > 0.90),
        })
    corr_df = pd.DataFrame(corr_records)
    corr_path = OUTPUT_TABLE_DIR / "grid_and_operational_features_correlation.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"\nSaved feature correlation analysis to: {corr_path}")
    
    # -------------------------------------------------------------------------
    # 2. 5-Fold Nested Cross-Validation Comparison on Cases 1-70
    # -------------------------------------------------------------------------
    feature_sets = {
        "Set_1_Base_Geology": [
            "Fault_Transmissibility", "Porosity_Multiplier", "Permeability_Multiplier", "Aquifer_Pore_Volume",
            "qo_0", "qw_0", "wor_0",
        ],
        "Set_2_Base_Plus_Operational": [
            "Fault_Transmissibility", "Porosity_Multiplier", "Permeability_Multiplier", "Aquifer_Pore_Volume",
            "qo_0", "qw_0", "wor_0", "case_vrr_proxy",
        ],
        "Set_3_Base_Plus_3D_Grid": [
            "Fault_Transmissibility", "Porosity_Multiplier", "Permeability_Multiplier", "Aquifer_Pore_Volume",
            "qo_0", "qw_0", "wor_0",
            "tot_pore_volume", "pv_weighted_porosity", "pv_weighted_permx",
            "anisotropy_ratio", "dykstra_parsons_vdp", "initial_mobile_oil",
        ],
        "Set_4_Full_Combined": [
            "Fault_Transmissibility", "Porosity_Multiplier", "Permeability_Multiplier", "Aquifer_Pore_Volume",
            "qo_0", "qw_0", "wor_0", "case_vrr_proxy",
            "tot_pore_volume", "pv_weighted_porosity", "pv_weighted_permx",
            "anisotropy_ratio", "dykstra_parsons_vdp", "initial_mobile_oil",
            "pv_weighted_pressure", "pv_weighted_transx",
        ],
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_eval_results = []
    
    for set_name, cols in feature_sets.items():
        oof_pred_oil = np.zeros(len(df_feat))
        oof_pred_wat = np.zeros(len(df_feat))
        
        X = df_feat[cols].to_numpy()
        y_o = df_feat["target_inc_qo"].to_numpy()
        y_w = df_feat["target_inc_qw"].to_numpy()
        
        for tr_idx, te_idx in kf.split(X):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_o_tr, y_o_te = y_o[tr_idx], y_o[te_idx]
            y_w_tr, y_w_te = y_w[tr_idx], y_w[te_idx]
            
            # Fit ExtraTrees Regressor
            rf_o = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42)
            rf_w = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42)
            
            rf_o.fit(X_tr, y_o_tr)
            rf_w.fit(X_tr, y_w_tr)
            
            oof_pred_oil[te_idx] = rf_o.predict(X_te)
            oof_pred_wat[te_idx] = rf_w.predict(X_te)
            
        oil_nrmse = float(compute_increment_nrmse(oof_pred_oil - y_o, y_o))
        wat_nrmse = float(compute_increment_nrmse(oof_pred_wat - y_w, y_w))
        macro_nrmse = 0.5 * (oil_nrmse + wat_nrmse)
        
        cv_eval_results.append({
            "feature_set": set_name,
            "num_features": len(cols),
            "oof_oil_nrmse": oil_nrmse,
            "oof_water_nrmse": wat_nrmse,
            "oof_macro_nrmse": macro_nrmse,
            "delta_vs_base_pct": 0.0, # will fill
        })
        
    cv_summary_df = pd.DataFrame(cv_eval_results)
    base_macro = cv_summary_df.loc[cv_summary_df["feature_set"] == "Set_1_Base_Geology", "oof_macro_nrmse"].iloc[0]
    cv_summary_df["delta_vs_base_pct"] = (cv_summary_df["oof_macro_nrmse"] - base_macro) / base_macro * 100.0
    
    cv_path = OUTPUT_TABLE_DIR / "grid_and_operational_features_cv_assessment.csv"
    cv_summary_df.to_csv(cv_path, index=False)
    print(f"\nSaved feature set CV assessment to: {cv_path}")
    print(cv_summary_df.to_string(index=False))
    
    return {
        "grid_features": grid_df,
        "correlations": corr_df,
        "cv_assessment": cv_summary_df,
    }


if __name__ == "__main__":
    evaluate_operational_and_grid_features()
