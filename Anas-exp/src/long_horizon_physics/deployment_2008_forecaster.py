"""
Module: deployment_2008_forecaster.py
Package: long_horizon_physics
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

PHASE 6 — January 2008–2028 Forward Deployment Comparison
Evaluates:
1. Field Case 0 historical anchor at 2008-01-01.
2. 10 official selected geological scenarios (Strategy A: Ten Best Matches).
3. 20-year (80-quarter) forecast trajectories comparing protected baseline vs top challengers.
4. Generates comparison tables and trajectories without modifying the official workbook.
"""
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
import numpy as np
import pandas as pd

from ensemble.data import load_uncertainty, resolve_shared_dir, TRAIN_CASES, YEAR_DAYS
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
)
from ensemble.scenario_loader import load_production_scenarios
from ensemble.generate_competition_deliverable import (
    load_locked_config,
    generate_scenario_predictions,
)
from forecast_advancement.protocol_framework import (
    fit_base_models_under_protocol,
    build_asof_case_dict,
)
from long_horizon_physics.multiphase_coupled_models import (
    TotalLiquidBoundedWORForecaster,
    BaselinePreservingSelectiveCorrector,
)
from long_horizon_physics.long_horizon_decline_models import (
    ModifiedArpsDeclineForecaster,
    ConstrainedTailHybridForecaster,
)


def run_deployment_2008_comparison(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    inference_origin: pd.Timestamp = pd.Timestamp("2008-01-01"),
    horizon_quarters: int = 80,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Executes 20-year (2008-2028) forward deployment forecasts for Field Case 0
    across the 10 official selected scenarios, comparing the protected baseline
    with the top physics challengers.
    """
    shared_dir = resolve_shared_dir()
    prod = pd.read_parquet(shared_dir / "production_timeseries.parquet")
    c0 = prod[prod["case_num"] == 0]
    c0_2008 = c0[c0["date"] == "2008-01-01"].set_index("metric")["value"]
    
    case0_anchor = {
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
    
    config = load_locked_config()
    weights = config["weights"]
    dates_80q = pd.date_range("2008-04-01", periods=horizon_quarters, freq="QS")
    dt_days = 91.3125
    
    # Fit baseline and challenger models under strict Protocol A at 2008-01-01
    asof_pool = build_asof_case_dict(case_dict, inference_origin)
    base_models = fit_base_models_under_protocol(
        case_dict, unc_df, TRAIN_CASES, inference_origin, protocol="Protocol_A", horizon_quarters=horizon_quarters
    )
    if "Recursive_ExtraTrees" in base_models:
        base_models["Recursive_ExtraTrees"].model_qo.n_jobs = 1
        base_models["Recursive_ExtraTrees"].model_qw.n_jobs = 1
        
    m_selective_wat = BaselinePreservingSelectiveCorrector(
        base_models, weights, correction_mode="correct_water"
    ).fit(asof_pool, unc_df, TRAIN_CASES, inference_origin)
    
    m_liquid_wor = TotalLiquidBoundedWORForecaster().fit(
        asof_pool, unc_df, TRAIN_CASES, inference_origin
    )
    
    m_tail_hyb = ConstrainedTailHybridForecaster(
        near_model=base_models["Recursive_ExtraTrees"], switch_quarters=12, D_lim=0.06
    ).fit(asof_pool, unc_df, TRAIN_CASES, inference_origin)
    
    # Load 10 official selected scenarios
    scenarios_path = Path("outputs/research/history_matching_iteration_4/tables/selected_scenarios_v4.csv")
    df_v4 = pd.read_csv(scenarios_path)
    strat_a_df = df_v4[df_v4["strategy"] == "Strategy A: Ten Best Matches"].copy().reset_index(drop=True)
    
    case_records = []
    traj_records = []
    
    model_keys = [
        "Frozen_Ensemble_Baseline",
        "Challenger_Selective_Water_Correction",
        "Challenger_Total_Liquid_Bounded_WOR",
        "Challenger_Constrained_Tail_Hybrid",
    ]
    
    for s_idx, row in strat_a_df.iterrows():
        case_id = row["case_id"]
        u_ser = pd.Series({
            "Fault Transmissibility": float(row["Ft"]),
            "Porosity Multiplier": float(row["Poro"]),
            "Permeability Multiplier": float(row["Perm"]),
            "Aquifer Pore Volume": float(row["AqPV"]),
        })
        
        preds = {}
        # 1. Baseline
        _, aud_base = generate_scenario_predictions(
            base_models, weights, case0_anchor, u_ser, dates_80q, 1.0
        )
        preds["Frozen_Ensemble_Baseline"] = aud_base
        
        # 2. Selective Water Correction
        p_sel = m_selective_wat.predict_rates(case0_anchor, u_ser, dates_80q)
        cum_o_sel = case0_anchor["Qo"] + np.cumsum(p_sel["oil_rate"] * dt_days)
        cum_w_sel = case0_anchor["Qw"] + np.cumsum(p_sel["water_rate"] * dt_days)
        cum_g_sel = case0_anchor["Qg"] + np.cumsum(p_sel["gas_rate"] * dt_days)
        preds["Challenger_Selective_Water_Correction"] = {
            "qo_rate": p_sel["oil_rate"], "qw_rate": p_sel["water_rate"], "qg_rate": p_sel["gas_rate"],
            "cum_oil": cum_o_sel, "cum_water": cum_w_sel, "cum_gas": cum_g_sel,
        }
        
        # 3. Total Liquid & Bounded WOR
        p_liq = m_liquid_wor.predict_rates(case0_anchor, u_ser, dates_80q)
        cum_o_liq = case0_anchor["Qo"] + np.cumsum(p_liq["oil_rate"] * dt_days)
        cum_w_liq = case0_anchor["Qw"] + np.cumsum(p_liq["water_rate"] * dt_days)
        cum_g_liq = case0_anchor["Qg"] + np.cumsum(p_liq["gas_rate"] * dt_days)
        preds["Challenger_Total_Liquid_Bounded_WOR"] = {
            "qo_rate": p_liq["oil_rate"], "qw_rate": p_liq["water_rate"], "qg_rate": p_liq["gas_rate"],
            "cum_oil": cum_o_liq, "cum_water": cum_w_liq, "cum_gas": cum_g_liq,
        }
        
        # 4. Constrained Tail Hybrid
        p_hyb = m_tail_hyb.predict_rates(case0_anchor, u_ser, dates_80q, dt_days=dt_days)
        cum_o_hyb = case0_anchor["Qo"] + np.cumsum(p_hyb["oil_rate"] * dt_days)
        cum_w_hyb = case0_anchor["Qw"] + np.cumsum(p_hyb["water_rate"] * dt_days)
        cum_g_hyb = case0_anchor["Qg"] + np.cumsum(p_hyb["gas_rate"] * dt_days)
        preds["Challenger_Constrained_Tail_Hybrid"] = {
            "qo_rate": p_hyb["oil_rate"], "qw_rate": p_hyb["water_rate"], "qg_rate": p_hyb["gas_rate"],
            "cum_oil": cum_o_hyb, "cum_water": cum_w_hyb, "cum_gas": cum_g_hyb,
        }
        
        for m_name in model_keys:
            p_data = preds[m_name]
            case_records.append({
                "model_name": m_name,
                "scenario_case_id": case_id,
                "eur_oil_stb": float(p_data["cum_oil"][-1]),
                "eur_water_stb": float(p_data["cum_water"][-1]),
                "eur_gas_mscf": float(p_data["cum_gas"][-1]),
                "incremental_oil_stb": float(p_data["cum_oil"][-1] - case0_anchor["Qo"]),
                "terminal_oil_rate_stbd": float(p_data["qo_rate"][-1]),
                "terminal_water_rate_stbd": float(p_data["qw_rate"][-1]),
                "Ft": float(row["Ft"]),
                "Poro": float(row["Poro"]),
                "Perm": float(row["Perm"]),
                "AqPV": float(row["AqPV"]),
            })
            
    df_scenarios = pd.DataFrame(case_records)
    
    # Summary across the 10 scenarios
    summary_records = []
    for m_name, grp in df_scenarios.groupby("model_name"):
        eur_o = grp["eur_oil_stb"]
        eur_w = grp["eur_water_stb"]
        incr_o = grp["incremental_oil_stb"]
        term_o = grp["terminal_oil_rate_stbd"]
        term_w = grp["terminal_water_rate_stbd"]
        
        summary_records.append({
            "model_name": m_name,
            "mean_eur_oil_stb": float(eur_o.mean()),
            "spread_eur_oil_stb": float(eur_o.max() - eur_o.min()),
            "std_eur_oil_stb": float(eur_o.std()),
            "mean_incremental_oil_stb": float(incr_o.mean()),
            "spread_incremental_oil_stb": float(incr_o.max() - incr_o.min()),
            "mean_eur_water_stb": float(eur_w.mean()),
            "spread_eur_water_stb": float(eur_w.max() - eur_w.min()),
            "mean_terminal_oil_rate_stbd": float(term_o.mean()),
            "mean_terminal_water_rate_stbd": float(term_w.mean()),
        })
        
    df_summary = pd.DataFrame(summary_records).sort_values("mean_eur_oil_stb").reset_index(drop=True)
    
    comparison_findings = {
        "case0_anchor_cum_oil_stb": case0_anchor["Qo"],
        "baseline_mean_eur_oil_stb": float(df_summary.loc[df_summary["model_name"] == "Frozen_Ensemble_Baseline", "mean_eur_oil_stb"].iloc[0]),
        "baseline_spread_eur_oil_stb": float(df_summary.loc[df_summary["model_name"] == "Frozen_Ensemble_Baseline", "spread_eur_oil_stb"].iloc[0]),
        "selective_water_eur_oil_stb": float(df_summary.loc[df_summary["model_name"] == "Challenger_Selective_Water_Correction", "mean_eur_oil_stb"].iloc[0]),
        "tail_hybrid_eur_oil_stb": float(df_summary.loc[df_summary["model_name"] == "Challenger_Constrained_Tail_Hybrid", "mean_eur_oil_stb"].iloc[0]),
        "total_liquid_eur_oil_stb": float(df_summary.loc[df_summary["model_name"] == "Challenger_Total_Liquid_Bounded_WOR", "mean_eur_oil_stb"].iloc[0]),
    }
    
    return df_scenarios, df_summary, comparison_findings
