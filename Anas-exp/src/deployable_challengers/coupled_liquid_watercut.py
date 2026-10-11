"""
Module: coupled_liquid_watercut.py
Package: deployable_challengers
Addresses: Section 4 — Study of Physically Coupled Water Forecasting
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Physical Formulation:
    q_liquid = q_oil + q_water
    water_cut = q_water / q_liquid
    q_oil = q_liquid * (1 - water_cut)
    q_water = q_liquid * water_cut

Compares 5 Water Forecasting Paradigms on Cases 71-85:
1. Method 1: Independent Oil/Water Residual Correction (Independent DCA Residuals)
2. Method 2: Total-Liquid Forecasting + Bounded Logit Water-Cut Dynamics (Coupled Liquid/WC)
3. Method 3: Water-Oil Ratio (WOR) Semi-Log Forecasting (SPE Waterflood Fractional Flow: ln(WOR) = a + m*Q_o)
4. Method 4: Oil DCA with Separate Learned Water Relaxation (Separate Water Model)
5. Method 5: Existing Frozen Ensemble Water Forecasts (Production Benchmark)

Evaluates:
- Oil Increment NRMSE
- Water Increment NRMSE
- Water Cut RMSE
- Total Liquid Increment NRMSE
- Macro Increment NRMSE (0.5 * (Oil + Water))
"""
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from ensemble.data import load_uncertainty, TRAIN_CASES, VAL_CASES, PARAMS
from ensemble.metrics import compute_increment_nrmse, compute_bias
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
    extract_transition_training_data,
    RecursiveTransitionForecaster,
    LearnedDCARateForecaster,
    YEAR_DAYS,
)

TABLE_DIR = Path("outputs/research/deployable_challengers/tables")
FIG_DIR = Path("outputs/research/deployable_challengers/figures")
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


class CoupledLiquidWaterCutForecaster:
    """
    Method 2: Total-Liquid Forecasting + Bounded Logit Water-Cut Dynamics.
    """
    def __init__(self):
        self.fitted = False

    def predict_rates(
        self,
        df_case: pd.DataFrame,
        origin: pd.Timestamp,
        forecast_dates: pd.DatetimeIndex,
    ) -> Dict[str, np.ndarray]:
        pre = df_case[df_case["date"] <= origin]
        last = pre.iloc[-1]
        qo_0 = float(last["oil_rate_stbd"])
        qw_0 = float(last["water_rate_stbd"])
        ql_0 = qo_0 + qw_0
        wc_0 = np.clip(qw_0 / max(ql_0, 1e-6), 1e-4, 1.0 - 1e-4)
        
        days = (forecast_dates - origin).days.to_numpy(float)
        t_years = days / YEAR_DAYS
        
        # Estimate total liquid decline rate D_L from recent 1-year history
        if len(pre) >= 5:
            ql_4 = float(pre.iloc[-5]["oil_rate_stbd"]) + float(pre.iloc[-5]["water_rate_stbd"])
            d_l = max(0.005, -(np.log(max(ql_0, 10.0)) - np.log(max(ql_4, 10.0))))
            wc_4 = np.clip(float(pre.iloc[-5]["water_rate_stbd"]) / max(ql_4, 1e-6), 1e-4, 1.0 - 1e-4)
            # Water cut logit velocity
            beta = max(0.0, (logit(wc_0) - logit(wc_4)) / 1.0)
        else:
            d_l = 0.02
            beta = 0.05
            
        d_l = float(np.clip(d_l, 0.005, 0.15))
        beta = float(np.clip(beta, 0.0, 0.5))
        
        # Total liquid declines exponentially
        ql_fwd = ql_0 * np.exp(-d_l * t_years)
        
        # Water cut grows logistically
        wc_fwd = expit(logit(wc_0) + beta * np.sqrt(t_years + 0.1) - beta * np.sqrt(0.1))
        
        # Couple back to oil and water
        qo_fwd = ql_fwd * (1.0 - wc_fwd)
        qw_fwd = ql_fwd * wc_fwd
        
        return {
            "oil_rate": qo_fwd,
            "water_rate": qw_fwd,
            "liquid_rate": ql_fwd,
            "water_cut": wc_fwd,
        }


class WORSemiLogForecaster:
    """
    Method 3: SPE Waterflood Semi-Log WOR: ln(WOR) = ln(WOR_0) + m * (Q_o(t) - Q_o(0)).
    """
    def __init__(self):
        self.fitted = False

    def predict_rates(
        self,
        df_case: pd.DataFrame,
        u_c: pd.Series,
        origin: pd.Timestamp,
        forecast_dates: pd.DatetimeIndex,
        m_dca_o: LearnedDCARateForecaster,
    ) -> Dict[str, np.ndarray]:
        pre = df_case[df_case["date"] <= origin]
        last = pre.iloc[-1]
        qo_0 = float(last["oil_rate_stbd"])
        qw_0 = float(last["water_rate_stbd"])
        Qo_0 = float(last["oil_cum_stb"])
        wor_0 = max(qw_0 / max(qo_0, 1.0), 1e-4)
        
        init_st = {
            "qo": qo_0, "qw": qw_0, "qg": float(last["gas_rate_mscfd"]),
            "Qo": Qo_0, "Qw": float(last["water_cum_stb"]), "Qg": float(last["gas_cum_mscf"]),
            "wor": wor_0, "res_age": (origin - pre["date"].iloc[0]).days / YEAR_DAYS,
            "origin_date": origin,
        }
        
        # Oil rate forecasted via physical DCA
        dca_preds = m_dca_o.predict_rates(init_st, u_c, forecast_dates)
        qo_fwd = dca_preds["oil_rate"]
        
        # Estimate historical semi-log WOR slope m = d(ln WOR) / d(Qo)
        post_breakthrough = pre[pre["water_rate_stbd"] > 50.0]
        if len(post_breakthrough) >= 4:
            wor_hist = np.maximum(post_breakthrough["water_rate_stbd"] / np.maximum(post_breakthrough["oil_rate_stbd"], 1.0), 1e-4)
            ln_wor = np.log(wor_hist)
            qo_cum = post_breakthrough["oil_cum_stb"].values / 1e6
            slope, _ = np.polyfit(qo_cum, ln_wor, 1)
            m = max(0.01, slope)
        else:
            m = 0.1
            
        m = float(np.clip(m, 0.01, 1.0))
        
        # Cumulative oil increment
        dt_days = 91.3125
        delta_Qo_MM = np.cumsum(qo_fwd * dt_days) / 1e6
        
        # Forward WOR
        wor_fwd = np.clip(wor_0 * np.exp(m * delta_Qo_MM), 1e-4, 100.0)
        qw_fwd = qo_fwd * wor_fwd
        ql_fwd = qo_fwd + qw_fwd
        wc_fwd = qw_fwd / ql_fwd
        
        return {
            "oil_rate": qo_fwd,
            "water_rate": qw_fwd,
            "liquid_rate": ql_fwd,
            "water_cut": wc_fwd,
        }


def run_coupled_water_experiments(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    origins: List[Tuple[str, int]] = None,
) -> pd.DataFrame:
    """
    Compares the 5 water forecasting paradigms.
    """
    print("\n" + "=" * 80)
    print("PHYSICALLY COUPLED WATER FORECASTING BENCHMARK (Cases 71-85)")
    print("=" * 80)
    
    if origins is None:
        origins = [("2004-01-01", 16), ("2005-01-01", 12)]
        
    dt_days = 91.3125
    records = []
    
    weights_oil = {
        "Recursive_ExtraTrees": 0.240466,
        "Learned_Exponential_DCA": 0.617902,
        "Hybrid_Dynamic": 0.141631,
    }
    weights_wat = {
        "Recursive_ExtraTrees": 0.497170,
        "Learned_Exponential_DCA": 0.251415,
        "Learned_Hyperbolic_DCA": 0.251415,
    }
    
    for orig_str, hq in origins:
        orig = pd.Timestamp(orig_str)
        f_dates = pd.date_range(orig + pd.DateOffset(days=1), periods=hq, freq="QS")
        print(f"\n>>> Running Water Paradigm Comparison at Origin {orig_str} ({hq}Q) <<<")
        
        dict_asof = {c: case_dict[c][case_dict[c]["date"] <= orig] for c in TRAIN_CASES}
        m_dca_exp = LearnedDCARateForecaster(decline_type="exponential").fit(dict_asof, unc_df, TRAIN_CASES, orig, hq)
        m_dca_hyp = LearnedDCARateForecaster(decline_type="hyperbolic").fit(dict_asof, unc_df, TRAIN_CASES, orig, hq)
        m_et = RecursiveTransitionForecaster(model_type="extratrees").fit(
            extract_transition_training_data(dict_asof, unc_df, TRAIN_CASES, orig)
        )
        
        m_coupled = CoupledLiquidWaterCutForecaster()
        m_wor = WORSemiLogForecaster()
        
        for c in VAL_CASES:
            df_c = case_dict[c]
            pre = df_c[df_c["date"] <= orig]
            post = df_c[df_c["date"] > orig].iloc[:hq]
            last = pre.iloc[-1]
            u_c = unc_df.loc[c]
            
            init_Qo = float(last["oil_cum_stb"])
            init_Qw = float(last["water_cum_stb"])
            init_st = {
                "qo": float(last["oil_rate_stbd"]), "qw": float(last["water_rate_stbd"]), "qg": float(last["gas_rate_mscfd"]),
                "Qo": init_Qo, "Qw": init_Qw, "Qg": float(last["gas_cum_mscf"]),
                "wor": float(last.get("wor", float(last["water_rate_stbd"]) / max(float(last["oil_rate_stbd"]), 1.0))),
                "res_age": (orig - pre["date"].iloc[0]).days / YEAR_DAYS, "origin_date": orig,
            }
            
            y_true_o = post["oil_cum_stb"].values
            y_true_w = post["water_cum_stb"].values
            y_true_l = (post["oil_rate_stbd"] + post["water_rate_stbd"]).values
            y_true_wc = (post["water_rate_stbd"] / np.maximum(y_true_l, 1e-6)).values
            
            inc_o = y_true_o - init_Qo
            inc_w = y_true_w - init_Qw
            inc_l = np.cumsum(y_true_l * dt_days)
            
            # --- Model 1: Independent DCA ---
            r_dca = m_dca_exp.predict_rates(init_st, u_c, f_dates)
            qo_1 = r_dca["oil_rate"]
            qw_1 = r_dca["water_rate"]
            
            # --- Model 2: Coupled Liquid / Water Cut ---
            r_cpl = m_coupled.predict_rates(df_c, orig, f_dates)
            qo_2 = r_cpl["oil_rate"]
            qw_2 = r_cpl["water_rate"]
            
            # --- Model 3: Semi-Log WOR ---
            r_wor = m_wor.predict_rates(df_c, u_c, orig, f_dates, m_dca_exp)
            qo_3 = r_wor["oil_rate"]
            qw_3 = r_wor["water_rate"]
            
            # --- Model 4: Separate Water Model ---
            r_hyp = m_dca_hyp.predict_rates(init_st, u_c, f_dates)
            qo_4 = r_dca["oil_rate"]
            qw_4 = r_hyp["water_rate"]
            
            # --- Model 5: Frozen Ensemble ---
            r_et = m_et.predict_rates(init_st, u_c, f_dates)
            qo_5 = weights_oil["Recursive_ExtraTrees"] * r_et["oil_rate"] + weights_oil["Learned_Exponential_DCA"] * r_dca["oil_rate"]
            qw_5 = weights_wat["Recursive_ExtraTrees"] * r_et["water_rate"] + weights_wat["Learned_Exponential_DCA"] * r_dca["water_rate"] + weights_wat["Learned_Hyperbolic_DCA"] * r_hyp["water_rate"]
            
            paradigms = {
                "1_Independent_DCA": (qo_1, qw_1),
                "2_Coupled_Liquid_WaterCut": (qo_2, qw_2),
                "3_WOR_SemiLog_Waterflood": (qo_3, qw_3),
                "4_Separate_Water_Model": (qo_4, qw_4),
                "5_Frozen_Production_Ensemble": (qo_5, qw_5),
            }
            
            for p_name, (qo_p, qw_p) in paradigms.items():
                cum_o = init_Qo + np.cumsum(qo_p * dt_days)
                cum_w = init_Qw + np.cumsum(qw_p * dt_days)
                cum_l = np.cumsum((qo_p + qw_p) * dt_days)
                wc_p = qw_p / np.maximum(qo_p + qw_p, 1e-6)
                
                no = compute_increment_nrmse(cum_o - y_true_o, inc_o)
                nw = compute_increment_nrmse(cum_w - y_true_w, inc_w)
                nl = compute_increment_nrmse(cum_l - inc_l, inc_l)
                wc_rmse = float(np.sqrt(np.mean((wc_p - y_true_wc) ** 2)))
                mac = 0.5 * (no + nw)
                
                records.append({
                    "origin": orig_str,
                    "horizon_quarters": hq,
                    "case_id": c,
                    "paradigm": p_name,
                    "macro_nrmse": mac,
                    "oil_nrmse": no,
                    "water_nrmse": nw,
                    "liquid_nrmse": nl,
                    "water_cut_rmse": wc_rmse,
                })
                
    df_raw = pd.DataFrame(records)
    summary_records = []
    
    print("\n" + "-" * 85)
    print(f"{'Origin':<12} | {'Paradigm':<32} | {'Macro NRMSE':<11} | {'Oil NRMSE':<9} | {'Wat NRMSE':<9} | {'WC RMSE':<8}")
    print("-" * 85)
    for (orig_str, p_name), grp in df_raw.groupby(["origin", "paradigm"]):
        mac = float(grp["macro_nrmse"].mean())
        no = float(grp["oil_nrmse"].mean())
        nw = float(grp["water_nrmse"].mean())
        nl = float(grp["liquid_nrmse"].mean())
        wc_err = float(grp["water_cut_rmse"].mean())
        
        summary_records.append({
            "origin": orig_str,
            "paradigm": p_name,
            "macro_nrmse": mac,
            "oil_nrmse": no,
            "water_nrmse": nw,
            "liquid_nrmse": nl,
            "water_cut_rmse": wc_err,
        })
        print(f"{orig_str:<12} | {p_name:<32} | {mac:.4f}      | {no:.4f}    | {nw:.4f}    | {wc_err:.4f}")
    print("-" * 85)
    
    df_sum = pd.DataFrame(summary_records)
    out_path = TABLE_DIR / "coupled_water_paradigm_comparison.csv"
    df_sum.to_csv(out_path, index=False)
    print(f"\nSaved Coupled Water Paradigm Results: {out_path}")
    return df_sum


if __name__ == "__main__":
    unc = load_uncertainty()
    qdf = load_clean_simulation_quarterly()
    cdict = build_case_quarterly_trajectories(qdf)
    run_coupled_water_experiments(cdict, unc)
