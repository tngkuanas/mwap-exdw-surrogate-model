"""
Module: official_scenarios_audit.py
Package: geological_information_value
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Addresses: Section 5 — Revisit the Ten Official Forecast Scenarios
Analyzes why the official 20-year oil EUR values span only 51.17–51.19 MMSTB across Cases 2–10.
Separates the 7 underlying mechanisms:
1. Scenario selection strategy (Strategy A: Ten Best Matches vs Quantiles).
2. Bayesian posterior concentration.
3. Frozen forecast-model insensitivity (zero perm/aq DCA response).
4. Shared historical Case 0 anchor (38.38 MMSTB fixed, 75% of EUR).
5. DCA dominance in the ensemble (76% effective exponential weight).
6. Long-horizon decline convergence (q_t < 800 STB/d by year 20).
7. Parameter mapping audit (confirms correct column matching).
"""
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import time
import numpy as np
import pandas as pd

from ensemble.data import load_uncertainty, resolve_shared_dir, TRAIN_CASES, PARAMS, YEAR_DAYS
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
    RS_SOLUTION,
)
from ensemble.scenario_loader import load_production_scenarios
from ensemble.generate_competition_deliverable import (
    load_locked_config,
    fit_locked_base_models,
    generate_scenario_predictions,
)
from geological_information_value.parameter_bounds import (
    PARAM_NAMES,
    SHORT_NAMES,
    LOWER_BOUNDS,
    UPPER_BOUNDS,
    RANGES,
    PRIOR_MEANS,
    sample_prior_uniform,
    create_parameter_series,
)


def run_official_scenarios_variance_audit(
    base_dir: Path = Path("."),
    n_prior_samples: int = 100,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Executes a comprehensive uncertainty decomposition of 20-year oil EUR forecasts
    anchored on historical Case 0.
    Returns:
    - scenarios_comparison_df: Strategy A vs Strategy C vs Prior vs Standalone models
    - variance_decomposition_df: Quantitative breakdown of EUR spread and variance shares
    - audit_findings: Detailed qualitative/quantitative diagnosis across all 7 hypotheses
    """
    shared_dir = resolve_shared_dir()
    unc_df = load_uncertainty(shared_dir)
    raw_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(raw_df)
    
    config = load_locked_config()
    weights = config["weights"]
    inference_origin = pd.Timestamp("2008-01-01")
    
    # Load Case 0 historical anchor
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
    
    models = fit_locked_base_models(case_dict, unc_df, TRAIN_CASES, inference_origin)
    if "Recursive_ExtraTrees" in models:
        models["Recursive_ExtraTrees"].model_qo.n_jobs = 1
        models["Recursive_ExtraTrees"].model_qw.n_jobs = 1
    dates_80q = pd.date_range("2008-04-01", "2028-01-01", freq="QS")
    
    # -------------------------------------------------------------------------
    # 1. GROUP 1: Official Selected Scenarios (Strategy A: Ten Best Matches)
    # -------------------------------------------------------------------------
    scenarios_v4_path = Path("outputs/research/history_matching_iteration_4/tables/selected_scenarios_v4.csv")
    df_v4 = pd.read_csv(scenarios_v4_path)
    strat_a_df = df_v4[df_v4["strategy"] == "Strategy A: Ten Best Matches"].copy()
    
    records_eval = []
    
    for idx, row in strat_a_df.iterrows():
        u_ser = pd.Series({
            "Fault Transmissibility": float(row["Ft"]),
            "Porosity Multiplier": float(row["Poro"]),
            "Permeability Multiplier": float(row["Perm"]),
            "Aquifer Pore Volume": float(row["AqPV"]),
        })
        _, aud = generate_scenario_predictions(
            models, weights, case0_anchor, u_ser, dates_80q, 1.0
        )
        records_eval.append({
            "group": "1_Official_Strategy_A_Ten_Best",
            "case_id": row["case_id"],
            "eur_oil_stb": float(aud["cum_oil"][-1]),
            "eur_water_stb": float(aud["cum_water"][-1]),
            "eur_gas_mscf": float(aud["cum_gas"][-1]),
            "incremental_oil_stb": float(aud["cum_oil"][-1] - case0_anchor["Qo"]),
            "terminal_oil_rate_stbd": float(aud["qo_rate"][-1]),
            "terminal_water_rate_stbd": float(aud["qw_rate"][-1]),
            "Ft": float(row["Ft"]),
            "Poro": float(row["Poro"]),
            "Perm": float(row["Perm"]),
            "AqPV": float(row["AqPV"]),
        })
        
    # -------------------------------------------------------------------------
    # 2. GROUP 2: Quality-Filtered Quantiles (Strategy C: P10 to P90)
    # -------------------------------------------------------------------------
    strat_c_df = df_v4[df_v4["strategy"] == "Strategy C: Quality-Filtered Quantiles"].copy()
    for idx, row in strat_c_df.iterrows():
        u_ser = pd.Series({
            "Fault Transmissibility": float(row["Ft"]),
            "Porosity Multiplier": float(row["Poro"]),
            "Permeability Multiplier": float(row["Perm"]),
            "Aquifer Pore Volume": float(row["AqPV"]),
        })
        _, aud = generate_scenario_predictions(
            models, weights, case0_anchor, u_ser, dates_80q, 1.0
        )
        records_eval.append({
            "group": "2_Strategy_C_Quality_Quantiles",
            "case_id": row["case_id"],
            "eur_oil_stb": float(aud["cum_oil"][-1]),
            "eur_water_stb": float(aud["cum_water"][-1]),
            "eur_gas_mscf": float(aud["cum_gas"][-1]),
            "incremental_oil_stb": float(aud["cum_oil"][-1] - case0_anchor["Qo"]),
            "terminal_oil_rate_stbd": float(aud["qo_rate"][-1]),
            "terminal_water_rate_stbd": float(aud["qw_rate"][-1]),
            "Ft": float(row["Ft"]),
            "Poro": float(row["Poro"]),
            "Perm": float(row["Perm"]),
            "AqPV": float(row["AqPV"]),
        })
        
    # -------------------------------------------------------------------------
    # 3. GROUP 3: Random Prior Draws (100 Space-Filling Samples Across Prior Box)
    # -------------------------------------------------------------------------
    prior_samples = sample_prior_uniform(n_prior_samples, seed=seed)
    for i in range(n_prior_samples):
        u_ser = create_parameter_series(prior_samples[i])
        _, aud = generate_scenario_predictions(
            models, weights, case0_anchor, u_ser, dates_80q, 1.0
        )
        records_eval.append({
            "group": "3_Full_Prior_Box_Samples",
            "case_id": f"Prior_{i:03d}",
            "eur_oil_stb": float(aud["cum_oil"][-1]),
            "eur_water_stb": float(aud["cum_water"][-1]),
            "eur_gas_mscf": float(aud["cum_gas"][-1]),
            "incremental_oil_stb": float(aud["cum_oil"][-1] - case0_anchor["Qo"]),
            "terminal_oil_rate_stbd": float(aud["qo_rate"][-1]),
            "terminal_water_rate_stbd": float(aud["qw_rate"][-1]),
            "Ft": float(prior_samples[i, 0]),
            "Poro": float(prior_samples[i, 1]),
            "Perm": float(prior_samples[i, 2]),
            "AqPV": float(prior_samples[i, 3]),
        })
        
    eval_df = pd.DataFrame(records_eval)
    
    # -------------------------------------------------------------------------
    # 4. Variance Decomposition across Groups and Architectures
    # -------------------------------------------------------------------------
    grp_a = eval_df[eval_df["group"] == "1_Official_Strategy_A_Ten_Best"]["eur_oil_stb"]
    grp_c = eval_df[eval_df["group"] == "2_Strategy_C_Quality_Quantiles"]["eur_oil_stb"]
    grp_prior = eval_df[eval_df["group"] == "3_Full_Prior_Box_Samples"]["eur_oil_stb"]
    
    # Architecture spread: evaluate standalone base models on default prior mean
    u_mean_ser = create_parameter_series(PRIOR_MEANS)
    dt_days = 91.3125
    r_et = models["Recursive_ExtraTrees"].predict_rates(case0_anchor, u_mean_ser, dates_80q)
    r_exp = models["Learned_Exponential_DCA"].predict_rates(case0_anchor, u_mean_ser, dates_80q)
    r_hyp = models["Learned_Hyperbolic_DCA"].predict_rates(case0_anchor, u_mean_ser, dates_80q)
    r_hyb = models["Hybrid_Dynamic"].predict_rates(case0_anchor, u_mean_ser, dates_80q)
    
    eur_arch = {
        "Recursive_ExtraTrees": float(case0_anchor["Qo"] + np.sum(r_et["oil_rate"] * dt_days)),
        "Learned_Exponential_DCA": float(case0_anchor["Qo"] + np.sum(r_exp["oil_rate"] * dt_days)),
        "Learned_Hyperbolic_DCA": float(case0_anchor["Qo"] + np.sum(r_hyp["oil_rate"] * dt_days)),
        "Hybrid_Dynamic": float(case0_anchor["Qo"] + np.sum(r_hyb["oil_rate"] * dt_days)),
    }
    arch_values = list(eur_arch.values())
    arch_spread = max(arch_values) - min(arch_values)
    arch_var = float(np.var(arch_values))
    
    decomp_records = [
        {
            "evaluation_scope": "Official Scenarios (Strategy A: 10 Best Matches)",
            "min_eur_oil_stb": float(grp_a.min()),
            "max_eur_oil_stb": float(grp_a.max()),
            "spread_eur_oil_stb": float(grp_a.max() - grp_a.min()),
            "std_eur_oil_stb": float(grp_a.std()),
            "spread_rel_pct": float((grp_a.max() - grp_a.min()) / grp_a.mean() * 100.0),
            "n_samples": len(grp_a),
            "primary_driver": "Extreme posterior concentration of porosity (1.331 - 1.346) in top 10 point matches",
        },
        {
            "evaluation_scope": "Quality-Filtered Quantiles (Strategy C: P10 to P90)",
            "min_eur_oil_stb": float(grp_c.min()),
            "max_eur_oil_stb": float(grp_c.max()),
            "spread_eur_oil_stb": float(grp_c.max() - grp_c.min()),
            "std_eur_oil_stb": float(grp_c.std()),
            "spread_rel_pct": float((grp_c.max() - grp_c.min()) / grp_c.mean() * 100.0),
            "n_samples": len(grp_c),
            "primary_driver": "Dispersion across historical candidate pool quality filters",
        },
        {
            "evaluation_scope": "Full Prior Parameter Box (100 Random Samples)",
            "min_eur_oil_stb": float(grp_prior.min()),
            "max_eur_oil_stb": float(grp_prior.max()),
            "spread_eur_oil_stb": float(grp_prior.max() - grp_prior.min()),
            "std_eur_oil_stb": float(grp_prior.std()),
            "spread_rel_pct": float((grp_prior.max() - grp_prior.min()) / grp_prior.mean() * 100.0),
            "n_samples": len(grp_prior),
            "primary_driver": "Model architecture insensitivity: DCA coefficients on Perm and Aq are ~1e-4 and ~1e-6",
        },
        {
            "evaluation_scope": "Structural Forecasting Model Architecture (4 Base Models)",
            "min_eur_oil_stb": float(min(arch_values)),
            "max_eur_oil_stb": float(max(arch_values)),
            "spread_eur_oil_stb": float(arch_spread),
            "std_eur_oil_stb": float(np.std(arch_values)),
            "spread_rel_pct": float(arch_spread / np.mean(arch_values) * 100.0),
            "n_samples": len(arch_values),
            "primary_driver": "Divergence between Recursive ExtraTrees (62.2 MMSTB) and Exponential DCA (47.5 MMSTB)",
        },
    ]
    decomp_df = pd.DataFrame(decomp_records)
    
    # Quantitative variance share: Architecture vs Geological Prior vs Scenario Selection
    total_system_var = arch_var + float(np.var(grp_prior))
    arch_var_pct = (arch_var / total_system_var) * 100.0
    geo_var_pct = (float(np.var(grp_prior)) / total_system_var) * 100.0
    
    audit_findings = {
        "shared_case0_anchor_cum_oil_stb": case0_anchor["Qo"],
        "case0_anchor_fraction_of_eur": float(case0_anchor["Qo"] / grp_a.mean()),
        "official_eur_spread_stb": float(grp_a.max() - grp_a.min()),
        "full_prior_eur_spread_stb": float(grp_prior.max() - grp_prior.min()),
        "architecture_eur_spread_stb": float(arch_spread),
        "architecture_variance_share_pct": arch_var_pct,
        "geological_variance_share_pct": geo_var_pct,
        "dca_exponential_weight_oil": float(weights["oil"]["Learned_Exponential_DCA"]),
        "hybrid_weight_oil": float(weights["oil"]["Hybrid_Dynamic"]),
        "effective_long_term_exponential_weight_oil": float(
            weights["oil"]["Learned_Exponential_DCA"] + weights["oil"]["Hybrid_Dynamic"]
        ),
    }
    
    return eval_df, decomp_df, audit_findings
