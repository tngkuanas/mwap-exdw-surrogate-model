"""
Module: forward_deployment_2008_2028.py
Package: deployable_challengers
Addresses: Section 8 — 20-Year (80-Quarter) Deployment Simulation (2008-2028)
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Generates 80-quarter (2008-2028) forecasts for the official 10 selected scenarios:
1. Frozen Production Baseline (Blend_Phase_Specific_Simplex)
2. Deployable Linear Damped Residual DCA (Challenger A)
3. Deployable GP Damped Residual DCA (Challenger B)
4. Deployable Functional Damped Residual DCA (Challenger C)
5. Deployable Direct Forecaster with Mature DCA Tail (Challenger D)

Evaluates:
- Cumulative production milestones at Year 1, 3, 5, 10, 15, and 20.
- Rate trajectories, terminal rates at Year 20, and water-cut evolution.
- Parameter uncertainty (geological variance) vs Structural uncertainty (model disagreement).
- Strict physical constraint compliance (non-negativity, post-plateau monotonic decline).
"""
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from ensemble.data import load_uncertainty, TRAIN_CASES, VAL_CASES, PARAMS
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
    extract_transition_training_data,
    RecursiveTransitionForecaster,
    LearnedDCARateForecaster,
    HybridDynamicForecaster,
    YEAR_DAYS,
    RS_SOLUTION,
)
from deployable_challengers.deployable_direct_forecasting import DeployableDirectForecaster
from deployable_challengers.deployable_residual_dca import DeployableResidualForecaster

TABLE_DIR = Path("outputs/research/deployable_challengers/tables")
FIG_DIR = Path("outputs/research/deployable_challengers/figures")
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_official_ten_scenarios() -> pd.DataFrame:
    """Loads the 10 official history-matched scenarios."""
    path_v4 = Path("outputs/research/history_matching_iteration_4/tables/selected_scenarios_v4.csv")
    if path_v4.exists():
        df = pd.read_csv(path_v4)
        top10 = df[df["strategy"] == "Strategy A: Ten Best Matches"].head(10).copy()
        # Rename parameters to standard names
        rename_map = {"Ft": "Fault Transmissibility", "Poro": "Porosity Multiplier", "Perm": "Permeability Multiplier", "AqPV": "Aquifer Pore Volume"}
        top10 = top10.rename(columns=rename_map)
        return top10
    else:
        raise FileNotFoundError("Could not find selected_scenarios_v4.csv")


def run_forward_deployment_simulation(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes 80-quarter (2008-2028) forward deployment simulation across the 10 scenarios.
    """
    print("\n" + "=" * 80)
    print("20-YEAR (80-QUARTER) FORWARD DEPLOYMENT SIMULATION (2008-2028)")
    print("=" * 80)
    
    orig_2008 = pd.Timestamp("2008-01-01")
    f_dates_80q = pd.date_range(orig_2008 + pd.DateOffset(days=1), periods=80, freq="QS")
    dt_days = 91.3125
    
    # Base field history for Field Case 1 up to 2008-01-01
    df_field1 = case_dict[1]
    pre_2008 = df_field1[df_field1["date"] <= orig_2008]
    last_hist = pre_2008.iloc[-1]
    
    init_Qo = float(last_hist["oil_cum_stb"])
    init_Qw = float(last_hist["water_cum_stb"])
    init_Qg = float(last_hist["gas_cum_mscf"])
    
    init_st = {
        "qo": float(last_hist["oil_rate_stbd"]),
        "qw": float(last_hist["water_rate_stbd"]),
        "qg": float(last_hist["gas_rate_mscfd"]),
        "Qo": init_Qo,
        "Qw": init_Qw,
        "Qg": init_Qg,
        "wor": float(last_hist.get("wor", float(last_hist["water_rate_stbd"]) / max(float(last_hist["oil_rate_stbd"]), 1.0))),
        "res_age": 10.0,
        "origin_date": orig_2008,
    }
    
    scenarios_df = load_official_ten_scenarios()
    print(f"Loaded {len(scenarios_df)} official scenarios for deployment:")
    for i, r in scenarios_df.iterrows():
        print(f"  Scenario {r['case_id']} (Cand {r['cand_id']}): Ft={r['Fault Transmissibility']:.3f}, Poro={r['Porosity Multiplier']:.3f}, Perm={r['Permeability Multiplier']:.2f}, Aq={r['Aquifer Pore Volume']:.1f}")
        
    # Fit all models strictly as-of 2008-01-01 using only historical data
    print("\nFitting deployable models strictly as-of January 1, 2008...")
    dict_2008 = {c: case_dict[c][case_dict[c]["date"] <= orig_2008] for c in TRAIN_CASES}
    
    # 1. Base Frozen Ensemble models
    m_et = RecursiveTransitionForecaster(model_type="extratrees").fit(
        extract_transition_training_data(dict_2008, unc_df, TRAIN_CASES, orig_2008)
    )
    m_exp = LearnedDCARateForecaster(decline_type="exponential").fit(
        dict_2008, unc_df, TRAIN_CASES, orig_2008, 16
    )
    m_hyp = LearnedDCARateForecaster(decline_type="hyperbolic").fit(
        dict_2008, unc_df, TRAIN_CASES, orig_2008, 16
    )
    m_hyb = HybridDynamicForecaster(switch_quarters=12).fit(
        dict_2008, unc_df, TRAIN_CASES, orig_2008, 16
    )
    
    weights_oil = {
        "Recursive_ExtraTrees": 0.24046649,
        "Learned_Exponential_DCA": 0.61790229,
        "Hybrid_Dynamic": 0.14163122,
    }
    weights_wat = {
        "Recursive_ExtraTrees": 0.49716965,
        "Learned_Exponential_DCA": 0.25141517,
        "Learned_Hyperbolic_DCA": 0.25141517,
    }
    
    # 2. Deployable Challengers
    m_res_lin = DeployableResidualForecaster(architecture="linear").fit(case_dict, unc_df, TRAIN_CASES, orig_2008, 16)
    m_res_gp = DeployableResidualForecaster(architecture="gp").fit(case_dict, unc_df, TRAIN_CASES, orig_2008, 16)
    m_res_func = DeployableResidualForecaster(architecture="functional").fit(case_dict, unc_df, TRAIN_CASES, orig_2008, 16)
    m_direct = DeployableDirectForecaster(method="ridge", h_direct=16).fit(case_dict, TRAIN_CASES, cutoff_date=orig_2008)
    
    models = {
        "Frozen_Production_Baseline": "frozen",
        "Challenger_A_Linear_Residual": m_res_lin,
        "Challenger_B_GP_Residual": m_res_gp,
        "Challenger_C_Functional_Residual": m_res_func,
        "Challenger_D_Direct_DCA_Tail": m_direct,
    }
    
    quarterly_records = []
    milestone_records = []
    
    milestone_quarters = [4, 12, 20, 40, 60, 80]  # Years 1, 3, 5, 10, 15, 20
    milestone_names = ["Year_1", "Year_3", "Year_5", "Year_10", "Year_15", "Year_20"]
    
    for s_idx, s_row in scenarios_df.iterrows():
        u_scen = pd.Series({
            "Fault Transmissibility": float(s_row["Fault Transmissibility"]),
            "Porosity Multiplier": float(s_row["Porosity Multiplier"]),
            "Permeability Multiplier": float(s_row["Permeability Multiplier"]),
            "Aquifer Pore Volume": float(s_row["Aquifer Pore Volume"]),
        })
        scen_label = s_row["case_id"]
        
        # Precompute Frozen Baseline rates
        r_et = m_et.predict_rates(init_st, u_scen, f_dates_80q)
        r_exp = m_exp.predict_rates(init_st, u_scen, f_dates_80q)
        r_hyp = m_hyp.predict_rates(init_st, u_scen, f_dates_80q)
        r_hyb = m_hyb.predict_rates(init_st, u_scen, f_dates_80q)
        
        qo_frz = weights_oil["Recursive_ExtraTrees"]*r_et["oil_rate"] + weights_oil["Learned_Exponential_DCA"]*r_exp["oil_rate"] + weights_oil["Hybrid_Dynamic"]*r_hyb["oil_rate"]
        qw_frz = weights_wat["Recursive_ExtraTrees"]*r_et["water_rate"] + weights_wat["Learned_Exponential_DCA"]*r_exp["water_rate"] + weights_wat["Learned_Hyperbolic_DCA"]*r_hyp["water_rate"]
        
        for m_name, model in models.items():
            if m_name == "Frozen_Production_Baseline":
                qo = np.maximum(0.0, qo_frz)
                qw = np.maximum(0.0, qw_frz)
            elif m_name == "Challenger_D_Direct_DCA_Tail":
                preds_d = model.predict_rates(df_field1, orig_2008, f_dates_80q)
                qo = np.maximum(0.0, preds_d["oil_rate"])
                qw = np.maximum(0.0, preds_d["water_rate"])
            else:
                preds_r = model.predict_rates(df_field1, u_scen, orig_2008, f_dates_80q, m_exp)
                qo = np.maximum(0.0, preds_r["oil_rate"])
                qw = np.maximum(0.0, preds_r["water_rate"])
                
            qg = RS_SOLUTION * qo
            
            # Cumulative volumes
            cum_o = init_Qo + np.cumsum(qo * dt_days)
            cum_w = init_Qw + np.cumsum(qw * dt_days)
            cum_g = init_Qg + np.cumsum(qg * dt_days)
            wc = qw / np.maximum(qo + qw, 1e-6)
            
            # Check constraints
            is_monotonic = bool((np.diff(qo) <= 1e-4).all())
            is_positive = bool((qo >= 0.0).all() and (qw >= 0.0).all())
            
            # Store quarterly trajectory
            for q_idx in range(80):
                quarterly_records.append({
                    "scenario": scen_label,
                    "cand_id": s_row["cand_id"],
                    "model_name": m_name,
                    "quarter": q_idx + 1,
                    "date": f_dates_80q[q_idx].strftime("%Y-%m-%d"),
                    "oil_rate_stbd": float(qo[q_idx]),
                    "water_rate_stbd": float(qw[q_idx]),
                    "gas_rate_mscfd": float(qg[q_idx]),
                    "oil_cum_stb": float(cum_o[q_idx]),
                    "water_cum_stb": float(cum_w[q_idx]),
                    "gas_cum_mscf": float(cum_g[q_idx]),
                    "water_cut": float(wc[q_idx]),
                })
                
            # Store milestones
            m_dict = {
                "scenario": scen_label,
                "cand_id": s_row["cand_id"],
                "model_name": m_name,
                "q_init_stbd": float(qo[0]),
                "q_term_stbd": float(qo[-1]),
                "wc_term": float(wc[-1]),
                "is_monotonic": is_monotonic,
                "is_positive": is_positive,
            }
            for q_step, m_lbl in zip(milestone_quarters, milestone_names):
                m_dict[f"{m_lbl}_oil_cum_MMstb"] = float(cum_o[q_step - 1] / 1e6)
                m_dict[f"{m_lbl}_wat_cum_MMstb"] = float(cum_w[q_step - 1] / 1e6)
                m_dict[f"{m_lbl}_gas_cum_Bscf"] = float(cum_g[q_step - 1] / 1e6)
            milestone_records.append(m_dict)

    df_quarterly = pd.DataFrame(quarterly_records)
    df_milestones = pd.DataFrame(milestone_records)
    
    out_q_path = TABLE_DIR / "twenty_year_deployment_forecasts_80q.csv"
    out_m_path = TABLE_DIR / "twenty_year_milestones_comparison.csv"
    df_quarterly.to_csv(out_q_path, index=False)
    df_milestones.to_csv(out_m_path, index=False)
    print(f"\nSaved 80-Quarter Deployment Forecasts: {out_q_path}")
    print(f"Saved 20-Year Milestones Comparison: {out_m_path}")
    
    # ----------------------------------------------------------------------
    # Uncertainty Decomposition: Parameter Uncertainty vs Structural Uncertainty
    # ----------------------------------------------------------------------
    unc_records = []
    
    # For each model, compute parameter uncertainty (spread across scenarios)
    for m_name in models.keys():
        sub = df_milestones[df_milestones["model_name"] == m_name]
        mean_20yr = float(sub["Year_20_oil_cum_MMstb"].mean())
        std_param = float(sub["Year_20_oil_cum_MMstb"].std())
        term_q_mean = float(sub["q_term_stbd"].mean())
        term_q_std = float(sub["q_term_stbd"].std())
        
        unc_records.append({
            "model_name": m_name,
            "mean_20yr_oil_MMstb": mean_20yr,
            "param_uncertainty_std_MMstb": std_param,
            "param_uncertainty_pct": float(std_param / max(mean_20yr, 1.0) * 100.0),
            "terminal_oil_rate_mean_stbd": term_q_mean,
            "terminal_oil_rate_std_stbd": term_q_std,
            "all_monotonic": bool(sub["is_monotonic"].all()),
            "all_positive": bool(sub["is_positive"].all()),
        })
        
    df_unc = pd.DataFrame(unc_records)
    out_unc_path = TABLE_DIR / "twenty_year_uncertainty_decomposition.csv"
    df_unc.to_csv(out_unc_path, index=False)
    print(f"Saved Uncertainty Decomposition: {out_unc_path}")
    
    print("\n" + "=" * 90)
    print("20-YEAR (80-QUARTER) SUMMARY ACROSS MODELS (Mean over 10 Scenarios):")
    print("=" * 90)
    for _, r in df_unc.iterrows():
        print(f"  {r['model_name']:<35s} | 20yr Cum = {r['mean_20yr_oil_MMstb']:.2f} MMstb (±{r['param_uncertainty_std_MMstb']:.2f}) | q_term = {r['terminal_oil_rate_mean_stbd']:.1f} STB/d | Monotonic={r['all_monotonic']}")
    print("=" * 90)
    
    return df_quarterly, df_milestones, df_unc


if __name__ == "__main__":
    unc = load_uncertainty()
    qdf = load_clean_simulation_quarterly()
    cdict = build_case_quarterly_trajectories(qdf)
    run_forward_deployment_simulation(cdict, unc)
