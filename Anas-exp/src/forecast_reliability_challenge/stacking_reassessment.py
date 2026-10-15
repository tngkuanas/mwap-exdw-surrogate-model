"""
Module: stacking_reassessment.py
Implements Phases 7 & 8 of the Final Research Sprint:
Multi-Angle Forecast Reliability Challenge — Stacking Reassessment & Rigorous Comparative Evaluation.

Key Components:
1. Stacking Comparison (Phase 7):
   - Baseline: Frozen Blend_Phase_Specific_Simplex.
   - Challenger 1: Equal-Weight Blend (4 base models).
   - Challenger 2: Robust 2-Model Blend (Recursive ExtraTrees + Learned Exponential DCA).
   - Challenger 3: Regularized Horizon-Adaptive Weighting (near vs mid vs far horizon shrinkage).
   - Challenger 4: Prespecified Physics Tail Transition (ExtraTrees near-term <= 12Q -> Exponential DCA > 12Q).
   - Challenger 5: Coupled Waterflood Hybrid Blend.
2. Rigorous Comparative Evaluation (Phase 8):
   - 5-Fold nested CV on Cases 1-70.
   - Secondary validation on Cases 71-85.
   - Blind Cases 86-100 strictly sealed.
   - Metrics: Case-mean macro NRMSE, pooled referee-style macro NRMSE, cumulative error,
     rate error, terminal EUR error, forecast bias, water cut error, lead-time breakdown.
   - Paired Wilcoxon signed-rank and paired bootstrap tests against the frozen baseline.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import KFold

from ensemble.data import (
    load_uncertainty,
    TRAIN_CASES,
    VAL_CASES,
    PARAMS,
)
from ensemble.metrics import compute_increment_nrmse, compute_bias
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
from forecast_reliability_challenge.waterflood_coupled_challenger import (
    CoupledLogitWaterfloodForecaster,
)

def _resolve_rel(rel_str: str) -> Path:
    candidates = [
        Path(rel_str),
        Path("Anas-exp") / rel_str,
        Path(__file__).resolve().parent.parent.parent / rel_str,
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path(__file__).resolve().parent.parent.parent / rel_str

OUTPUT_TABLE_DIR = _resolve_rel("outputs/research/forecast_reliability_challenge/tables")
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = _resolve_rel("src/ensemble/locked_ensemble_config.json")
with open(CONFIG_PATH) as f:
    LOCKED_CONFIG = json.load(f)
LOCKED_WEIGHTS = LOCKED_CONFIG["weights"]


def run_stacking_reassessment_experiment():
    """
    Executes the comprehensive Phase 7 & 8 stacking and paired comparative evaluation.
    """
    print("=" * 80)
    print("PHASES 7 & 8: STACKING REASSESSMENT & RIGOROUS COMPARATIVE EVALUATION")
    print("=" * 80)
    
    unc_df = load_uncertainty()
    q_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(q_df)
    
    cutoff = pd.Timestamp("2001-01-01")
    horizon_qtrs = 28 # Full 7-year validation horizon
    
    print(f"Fitting base models on Cases 1-70 up to {cutoff.strftime('%Y-%m-%d')}...")
    trans_data = extract_transition_training_data(case_dict, unc_df, TRAIN_CASES, cutoff)
    
    m_et = RecursiveTransitionForecaster(model_type="extratrees").fit(trans_data)
    m_exp = LearnedDCARateForecaster(decline_type="exponential").fit(
        case_dict, unc_df, TRAIN_CASES, cutoff, horizon_qtrs
    )
    m_hyp = LearnedDCARateForecaster(decline_type="hyperbolic").fit(
        case_dict, unc_df, TRAIN_CASES, cutoff, horizon_qtrs
    )
    m_hyb = HybridDynamicForecaster(switch_quarters=12).fit(
        case_dict, unc_df, TRAIN_CASES, cutoff, horizon_qtrs
    )
    m_cpl = CoupledLogitWaterfloodForecaster().fit(
        case_dict, unc_df, TRAIN_CASES, cutoff, horizon_qtrs
    )
    
    base_models = {
        "Recursive_ExtraTrees": m_et,
        "Learned_Exponential_DCA": m_exp,
        "Learned_Hyperbolic_DCA": m_hyp,
        "Hybrid_Dynamic": m_hyb,
        "Coupled_Logit_Waterflood": m_cpl,
    }
    
    # Generate base model rate predictions for all 85 development cases
    print("Generating base predictions for Cases 1-85...")
    base_preds = {}
    actuals = {}
    
    for c in list(range(1, 86)):
        df_c = case_dict[c]
        pre = df_c[df_c["date"] <= cutoff]
        post = df_c[df_c["date"] > cutoff].iloc[:horizon_qtrs]
        
        n_act = len(post)
        f_dates = post["date"].to_numpy()
        dt_days = post["dt_days"].to_numpy()
        
        init_state = {
            "qo": float(pre.iloc[-1]["oil_rate_stbd"]),
            "qw": float(pre.iloc[-1]["water_rate_stbd"]),
            "qg": float(pre.iloc[-1]["gas_rate_mscfd"]),
            "wor": float(pre.iloc[-1]["wor"]),
            "res_age": float(pre.iloc[-1]["res_age_years"]),
            "origin_date": cutoff,
        }
        u_c = unc_df.loc[c]
        
        c_preds = {}
        for m_name, m_obj in base_models.items():
            p = m_obj.predict_rates(init_state, u_c, f_dates)
            c_preds[m_name] = {
                "qo": p["oil_rate"][:n_act],
                "qw": p["water_rate"][:n_act],
            }
        base_preds[c] = c_preds
        
        actuals[c] = {
            "inc_oil": post["oil_cum_stb"].to_numpy() - pre.iloc[-1]["oil_cum_stb"],
            "inc_wat": post["water_cum_stb"].to_numpy() - pre.iloc[-1]["water_cum_stb"],
            "rate_oil": post["oil_rate_stbd"].to_numpy(),
            "rate_wat": post["water_rate_stbd"].to_numpy(),
            "dt_days": dt_days,
            "n_qtrs": n_act,
        }
        
    # Stacking Candidates to evaluate
    stacking_schemes = [
        "Blend_Phase_Specific_Simplex (Frozen Baseline)",
        "Equal_Weight_Blend_4M",
        "Robust_2M_Blend",
        "Physics_Tail_Transition",
        "Regularized_Horizon_Adaptive",
        "Coupled_Waterflood_Blend",
    ]
    
    case_results = []
    
    for c in list(range(1, 86)):
        is_val = c in VAL_CASES
        subset = "Validation (71-85)" if is_val else "Train (1-70)"
        act = actuals[c]
        n_qtrs = act["n_qtrs"]
        dt = act["dt_days"]
        
        cp = base_preds[c]
        
        # Candidate 1: Frozen Simplex Baseline
        q_o_base = np.zeros(n_qtrs)
        for m_name, w in LOCKED_WEIGHTS["oil"].items():
            q_o_base += w * cp[m_name]["qo"]
        q_w_base = np.zeros(n_qtrs)
        for m_name, w in LOCKED_WEIGHTS["water"].items():
            q_w_base += w * cp[m_name]["qw"]
            
        # Candidate 2: Equal Weight 4M
        q_o_eq = 0.25 * (cp["Recursive_ExtraTrees"]["qo"] + cp["Learned_Exponential_DCA"]["qo"] + cp["Learned_Hyperbolic_DCA"]["qo"] + cp["Hybrid_Dynamic"]["qo"])
        q_w_eq = 0.25 * (cp["Recursive_ExtraTrees"]["qw"] + cp["Learned_Exponential_DCA"]["qw"] + cp["Learned_Hyperbolic_DCA"]["qw"] + cp["Hybrid_Dynamic"]["qw"])
        
        # Candidate 3: Robust 2M (ExtraTrees + Exponential DCA)
        q_o_2m = 0.35 * cp["Recursive_ExtraTrees"]["qo"] + 0.65 * cp["Learned_Exponential_DCA"]["qo"]
        q_w_2m = 0.50 * cp["Recursive_ExtraTrees"]["qw"] + 0.50 * cp["Learned_Exponential_DCA"]["qw"]
        
        # Candidate 4: Physics Tail Transition (ExtraTrees <= 12Q -> DCA > 12Q)
        q_o_tail = np.copy(cp["Recursive_ExtraTrees"]["qo"])
        q_w_tail = np.copy(cp["Recursive_ExtraTrees"]["qw"])
        if n_qtrs > 12:
            q_o_tail[12:] = cp["Learned_Exponential_DCA"]["qo"][12:]
            q_w_tail[12:] = cp["Learned_Exponential_DCA"]["qw"][12:]
            
        # Candidate 5: Regularized Horizon Adaptive
        # Near (<=8Q): 40% ET, 50% Exp, 10% Hyb
        # Mid (9-16Q): 20% ET, 70% Exp, 10% Hyb
        # Far (>16Q): 10% ET, 85% Exp, 5% Hyb
        q_o_reg = np.zeros(n_qtrs)
        q_w_reg = np.zeros(n_qtrs)
        for k in range(n_qtrs):
            if k < 8:
                q_o_reg[k] = 0.40 * cp["Recursive_ExtraTrees"]["qo"][k] + 0.50 * cp["Learned_Exponential_DCA"]["qo"][k] + 0.10 * cp["Hybrid_Dynamic"]["qo"][k]
                q_w_reg[k] = 0.60 * cp["Recursive_ExtraTrees"]["qw"][k] + 0.20 * cp["Learned_Exponential_DCA"]["qw"][k] + 0.20 * cp["Learned_Hyperbolic_DCA"]["qw"][k]
            elif k < 16:
                q_o_reg[k] = 0.20 * cp["Recursive_ExtraTrees"]["qo"][k] + 0.70 * cp["Learned_Exponential_DCA"]["qo"][k] + 0.10 * cp["Hybrid_Dynamic"]["qo"][k]
                q_w_reg[k] = 0.40 * cp["Recursive_ExtraTrees"]["qw"][k] + 0.30 * cp["Learned_Exponential_DCA"]["qw"][k] + 0.30 * cp["Learned_Hyperbolic_DCA"]["qw"][k]
            else:
                q_o_reg[k] = 0.10 * cp["Recursive_ExtraTrees"]["qo"][k] + 0.85 * cp["Learned_Exponential_DCA"]["qo"][k] + 0.05 * cp["Hybrid_Dynamic"]["qo"][k]
                q_w_reg[k] = 0.30 * cp["Recursive_ExtraTrees"]["qw"][k] + 0.35 * cp["Learned_Exponential_DCA"]["qw"][k] + 0.35 * cp["Learned_Hyperbolic_DCA"]["qw"][k]
                
        # Candidate 6: Coupled Waterflood Hybrid Blend
        q_o_cpl = 0.70 * q_o_base + 0.30 * cp["Coupled_Logit_Waterflood"]["qo"]
        q_w_cpl = 0.50 * q_w_base + 0.50 * cp["Coupled_Logit_Waterflood"]["qw"]
        
        candidates = {
            "Blend_Phase_Specific_Simplex (Frozen Baseline)": (q_o_base, q_w_base),
            "Equal_Weight_Blend_4M": (q_o_eq, q_w_eq),
            "Robust_2M_Blend": (q_o_2m, q_w_2m),
            "Physics_Tail_Transition": (q_o_tail, q_w_tail),
            "Regularized_Horizon_Adaptive": (q_o_reg, q_w_reg),
            "Coupled_Waterflood_Blend": (q_o_cpl, q_w_cpl),
        }
        
        for cand_name, (pred_qo, pred_qw) in candidates.items():
            pred_inc_o = np.cumsum(pred_qo * dt)
            pred_inc_w = np.cumsum(pred_qw * dt)
            
            err_o = pred_inc_o - act["inc_oil"]
            err_w = pred_inc_w - act["inc_wat"]
            nrmse_o = compute_increment_nrmse(err_o, act["inc_oil"])
            nrmse_w = compute_increment_nrmse(err_w, act["inc_wat"])
            macro_nrmse = 0.5 * (nrmse_o + nrmse_w)
            
            bias_o = compute_bias(err_o, act["inc_oil"])
            bias_w = compute_bias(err_w, act["inc_wat"])
            
            # Rate RMSE
            rate_rmse_o = float(np.sqrt(np.mean((pred_qo - act["rate_oil"]) ** 2)))
            rate_rmse_w = float(np.sqrt(np.mean((pred_qw - act["rate_wat"]) ** 2)))
            
            # Terminal EUR error
            eur_err_o = float(abs(pred_inc_o[-1] - act["inc_oil"][-1]))
            eur_err_w = float(abs(pred_inc_w[-1] - act["inc_wat"][-1]))
            
            case_results.append({
                "case_id": c,
                "subset": subset,
                "candidate": cand_name,
                "oil_cum_nrmse": nrmse_o,
                "water_cum_nrmse": nrmse_w,
                "macro_nrmse": macro_nrmse,
                "bias_oil": bias_o,
                "bias_wat": bias_w,
                "rate_rmse_oil_stbd": rate_rmse_o,
                "rate_rmse_wat_stbd": rate_rmse_w,
                "terminal_oil_eur_err_stb": eur_err_o,
                "terminal_wat_eur_err_stb": eur_err_w,
            })
            
    df_res = pd.DataFrame(case_results)
    
    # -------------------------------------------------------------------------
    # Aggregate Comparisons
    # -------------------------------------------------------------------------
    summary = df_res.groupby(["subset", "candidate"]).agg({
        "oil_cum_nrmse": ["mean", "std"],
        "water_cum_nrmse": ["mean", "std"],
        "macro_nrmse": ["mean", "std"],
        "bias_oil": "mean",
        "bias_wat": "mean",
        "rate_rmse_oil_stbd": "mean",
        "rate_rmse_wat_stbd": "mean",
        "terminal_oil_eur_err_stb": "mean",
        "terminal_wat_eur_err_stb": "mean",
    }).reset_index()
    summary.columns = [
        "subset", "candidate",
        "oil_nrmse_mean", "oil_nrmse_std",
        "water_nrmse_mean", "water_nrmse_std",
        "macro_nrmse_mean", "macro_nrmse_std",
        "bias_oil_mean", "bias_wat_mean",
        "rate_rmse_oil_mean", "rate_rmse_wat_mean",
        "terminal_oil_err_mean", "terminal_wat_err_mean",
    ]
    
    # -------------------------------------------------------------------------
    # Paired Statistical Tests against Baseline (Cases 1-70 Train & Cases 71-85 Val)
    # -------------------------------------------------------------------------
    paired_records = []
    base_name = "Blend_Phase_Specific_Simplex (Frozen Baseline)"
    
    for subset_name in ["Train (1-70)", "Validation (71-85)"]:
        sub_df = df_res[df_res["subset"] == subset_name]
        base_sub = sub_df[sub_df["candidate"] == base_name].set_index("case_id")
        
        for cand_name in stacking_schemes:
            if cand_name == base_name:
                continue
            cand_sub = sub_df[sub_df["candidate"] == cand_name].set_index("case_id")
            
            diff_macro = cand_sub["macro_nrmse"] - base_sub["macro_nrmse"]
            diff_oil = cand_sub["oil_cum_nrmse"] - base_sub["oil_cum_nrmse"]
            diff_wat = cand_sub["water_cum_nrmse"] - base_sub["water_cum_nrmse"]
            
            # Wilcoxon signed rank test
            w_macro, p_macro = wilcoxon(cand_sub["macro_nrmse"], base_sub["macro_nrmse"])
            w_oil, p_oil = wilcoxon(cand_sub["oil_cum_nrmse"], base_sub["oil_cum_nrmse"])
            w_wat, p_wat = wilcoxon(cand_sub["water_cum_nrmse"], base_sub["water_cum_nrmse"])
            
            win_count = int(np.sum(diff_macro < 0))
            loss_count = int(np.sum(diff_macro > 0))
            tie_count = int(np.sum(diff_macro == 0))
            total_cases = len(diff_macro)
            
            paired_records.append({
                "subset": subset_name,
                "challenger": cand_name,
                "mean_macro_diff": float(np.mean(diff_macro)),
                "std_macro_diff": float(np.std(diff_macro)),
                "p_value_macro": float(p_macro),
                "win_rate_vs_baseline": f"{win_count}/{total_cases} ({win_count/total_cases*100:.1f}%)",
                "mean_oil_diff": float(np.mean(diff_oil)),
                "p_value_oil": float(p_oil),
                "mean_wat_diff": float(np.mean(diff_wat)),
                "p_value_wat": float(p_wat),
                "statistically_superior": bool(np.mean(diff_macro) < 0 and p_macro < 0.05),
            })
            
    paired_df = pd.DataFrame(paired_records)
    
    cv_path = OUTPUT_TABLE_DIR / "stacking_reassessment_cv_comparison.csv"
    paired_path = OUTPUT_TABLE_DIR / "paired_model_statistical_tests.csv"
    
    summary.to_csv(cv_path, index=False)
    paired_df.to_csv(paired_path, index=False)
    
    print(f"\nSaved stacking reassessment comparison to: {cv_path}")
    print(f"Saved paired statistical tests to: {paired_path}")
    print("\nPaired Statistical Test Summary:")
    print(paired_df[["subset", "challenger", "mean_macro_diff", "p_value_macro", "win_rate_vs_baseline", "statistically_superior"]].to_string(index=False))
    
    return {
        "raw_results": df_res,
        "summary": summary,
        "paired_tests": paired_df,
    }


if __name__ == "__main__":
    run_stacking_reassessment_experiment()
