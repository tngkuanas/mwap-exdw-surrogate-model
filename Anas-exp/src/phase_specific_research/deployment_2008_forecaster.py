"""
Module: deployment_2008_forecaster.py
Package: phase_specific_research
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Section 8: January 2008 Research Deployment on Field Case 0 Across 10 Scenarios.
- Case 1: 40 quarters (10 years).
- Cases 2-10: 80 quarters (20 years).
- Original production units and integration convention.
- Direct comparison between Frozen Baseline and surviving challengers.
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

from ensemble.data import load_uncertainty, resolve_shared_dir, TRAIN_CASES
from ensemble.scenario_loader import load_production_scenarios, HORIZON_SPEC
from ensemble.run_horizon_adaptive_stacking_experiment import RS_SOLUTION
from ensemble.generate_competition_deliverable import (
    load_locked_config,
    fit_locked_base_models,
)
from phase_specific_research.phase_recombination import (
    predict_candidate_rates,
    ALL_CANDIDATE_IDS,
)

DT_DAYS = 91.3125


def run_deployment_2008_scenarios(
    models: Dict[str, Any],
    locked_weights: Dict[str, Dict[str, float]],
    candidates_to_deploy: List[str] = [
        "Candidate_A_Frozen_Baseline",
        "Candidate_C_FrozenOil_ExtraTreesWater",
        "Candidate_E_Horizon_Adaptive_Blending",
        "Candidate_H_Constrained_Long_Tail",
    ],
    origin: pd.Timestamp = pd.Timestamp("2008-01-01"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes January 2008 forward deployment on actual Field Case 0 history
    across all 10 official selected scenarios.
    """
    shared_dir = resolve_shared_dir()
    scenarios = load_production_scenarios()
    
    # Historical anchor Case 0 at Jan 2008
    prod = pd.read_parquet(shared_dir / "production_timeseries.parquet")
    c0 = prod[prod["case_num"] == 0]
    c0_2008 = c0[c0["date"] == "2008-01-01"].set_index("metric")["value"]
    
    init_state = {
        "qo": float(c0_2008["oil_rate"]),
        "qw": float(c0_2008["water_rate"]),
        "qg": float(c0_2008["gas_rate"]),
        "Qo": float(c0_2008["oil_cum"]),
        "Qw": float(c0_2008["water_cum"]),
        "Qg": float(c0_2008["gas_cum"]),
        "wor": float(c0_2008["water_rate"] / max(c0_2008["oil_rate"], 1.0)),
        "res_age": 10.0,
        "origin_date": origin,
    }
    
    case_records = []
    
    for sc in scenarios:
        case_name = sc.case_id
        n_q = sc.horizon_quarters
        params = sc.params
        f_dates = pd.date_range(origin + pd.DateOffset(months=3), periods=n_q, freq="QS")
        
        # Precompute base model predictions once per scenario
        base_rates = {
            "Recursive_ExtraTrees": models["Recursive_ExtraTrees"].predict_rates(init_state, params, f_dates),
            "Learned_Exponential_DCA": models["Learned_Exponential_DCA"].predict_rates(init_state, params, f_dates),
            "Learned_Hyperbolic_DCA": models["Learned_Hyperbolic_DCA"].predict_rates(init_state, params, f_dates),
            "Hybrid_Dynamic": models["Hybrid_Dynamic"].predict_rates(init_state, params, f_dates),
        }
        
        for cand_id in candidates_to_deploy:
            pred = predict_candidate_rates(
                candidate_id=cand_id,
                models=models,
                locked_weights=locked_weights,
                init_state=init_state,
                params=params,
                forecast_dates=f_dates,
                base_rates=base_rates,
            )
            
            qo = pred["oil_rate"]
            qw = pred["water_rate"]
            qg = pred["gas_rate"]
            cum_o = pred["cum_oil"]
            cum_w = pred["cum_water"]
            cum_g = pred["cum_gas"]
            
            # Milestone indices: 1 yr (4Q), 3 yr (12Q), 5 yr (20Q), 10 yr (40Q), 20 yr (80Q if available)
            rec = {
                "scenario_id": case_name,
                "candidate_id": cand_id,
                "horizon_quarters": n_q,
                "horizon_years": n_q / 4.0,
                "eur_oil_stb": float(cum_o[-1]),
                "incremental_oil_stb": float(cum_o[-1] - init_state["Qo"]),
                "eur_water_stb": float(cum_w[-1]),
                "incremental_water_stb": float(cum_w[-1] - init_state["Qw"]),
                "eur_gas_mscf": float(cum_g[-1]),
                "terminal_oil_rate_stbd": float(qo[-1]),
                "terminal_water_rate_stbd": float(qw[-1]),
                "terminal_gas_rate_mscfd": float(qg[-1]),
                "terminal_water_cut": float(qw[-1] / max(qo[-1] + qw[-1], 1e-6)),
                "cum_oil_1yr_stb": float(cum_o[3]),
                "cum_oil_3yr_stb": float(cum_o[11]),
                "cum_oil_5yr_stb": float(cum_o[19]),
                "cum_oil_10yr_stb": float(cum_o[39]),
                "cum_oil_20yr_stb": float(cum_o[79]) if n_q >= 80 else float("nan"),
            }
            case_records.append(rec)
            
    df_scenarios = pd.DataFrame(case_records)
    
    # Aggregate summary per candidate
    summary_records = []
    for cand_id in candidates_to_deploy:
        sub = df_scenarios[df_scenarios["candidate_id"] == cand_id]
        sub_80 = sub[sub["horizon_quarters"] == 80]
        
        summary_records.append({
            "candidate_id": cand_id,
            "mean_eur_oil_stb": float(sub["eur_oil_stb"].mean()),
            "spread_eur_oil_stb": float(sub["eur_oil_stb"].max() - sub["eur_oil_stb"].min()),
            "std_eur_oil_stb": float(sub["eur_oil_stb"].std()),
            "mean_incremental_oil_stb": float(sub["incremental_oil_stb"].mean()),
            "mean_eur_water_stb": float(sub["eur_water_stb"].mean()),
            "spread_eur_water_stb": float(sub["eur_water_stb"].max() - sub["eur_water_stb"].min()),
            "mean_terminal_oil_rate_stbd": float(sub["terminal_oil_rate_stbd"].mean()),
            "mean_terminal_water_rate_stbd": float(sub["terminal_water_rate_stbd"].mean()),
            "mean_terminal_water_cut": float(sub["terminal_water_cut"].mean()),
            "mean_20yr_eur_oil_stb": float(sub_80["eur_oil_stb"].mean()) if len(sub_80) > 0 else float("nan"),
        })
        
    df_summary = pd.DataFrame(summary_records)
    return df_scenarios, df_summary
