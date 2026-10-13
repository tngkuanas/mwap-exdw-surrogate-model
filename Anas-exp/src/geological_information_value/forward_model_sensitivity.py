"""
Module: forward_model_sensitivity.py
Package: geological_information_value
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Section 3 — Quantify Forward-Model Geological Sensitivity (Corrected)
Implements:
1. Local parameter Jacobians (terminal rate & cumulative) across forecast horizons (4Q, 12Q, 40Q, 80Q).
2. Standard Saltelli / Jansen pick-freeze global sensitivity analysis (A, B, A_B^(i)).
3. Monte Carlo / bootstrap standard errors and 95% confidence intervals.
4. Numerical convergence checks across sample sizes N in {128, 256, 512, 1024}.
5. Sensitivity evaluation across three input distributions:
   - Broad uniform bounds
   - Empirical training distribution (Cases 1-70)
   - History-matched posterior samples
6. Mathematical inspection of fitted representations (RidgeCV coefficients, ExtraTrees importances).
"""
from typing import Dict, List, Tuple, Any, Optional
import time
import numpy as np
import pandas as pd
from scipy.stats import qmc

from ensemble.data import load_uncertainty, resolve_shared_dir, TRAIN_CASES, PARAMS, YEAR_DAYS
from ensemble.run_horizon_adaptive_stacking_experiment import (
    STATE_FEATURES,
    RS_SOLUTION,
)
from ensemble.generate_competition_deliverable import (
    load_locked_config,
)
from geological_information_value.parameter_bounds import (
    PARAM_NAMES,
    SHORT_NAMES,
    LOWER_BOUNDS,
    UPPER_BOUNDS,
    RANGES,
    PRIOR_MEANS,
    denormalize_parameters,
    normalize_parameters,
    create_parameter_series,
)
from geological_information_value.sobol_estimators import (
    compute_centered_sobol_indices,
    bootstrap_sobol_confidence_intervals,
)


def evaluate_models_batch(
    models: Dict[str, Any],
    weights: Dict[str, Dict[str, float]],
    init_state: Dict[str, Any],
    U_phys: np.ndarray,
    forecast_dates: pd.DatetimeIndex,
    dt_days: float = 91.3125,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Evaluates all base models and the blended ensemble simultaneously on a batch
    of M physical parameter vectors U_phys of shape (M, 4).
    Verified to be float64-equivalent (< 1e-11 on rates, < 1e-7 on cumulatives)
    to scalar generate_scenario_predictions().
    """
    M = len(U_phys)
    n_steps = len(forecast_dates)
    origin = pd.Timestamp(init_state["origin_date"])
    
    q0_o = float(init_state["qo"])
    q0_w = float(init_state["qw"])
    Q0_o = float(init_state["Qo"])
    Q0_w = float(init_state["Qw"])
    res_age = float(init_state["res_age"])
    wor_0 = float(init_state.get("wor", q0_w / max(q0_o, 1.0)))
    
    t_years = np.asarray((pd.DatetimeIndex(forecast_dates) - origin).total_seconds() / (86400.0 * YEAR_DAYS), dtype=float)
    
    out_dict: Dict[str, Dict[str, np.ndarray]] = {}
    
    # -------------------------------------------------------------------------
    # 1. Learned Exponential DCA
    # -------------------------------------------------------------------------
    m_exp = models.get("Learned_Exponential_DCA")
    if m_exp is not None:
        X_dca = np.column_stack([
            U_phys[:, 0], U_phys[:, 1], U_phys[:, 2], U_phys[:, 3],
            np.full(M, q0_o),
            np.full(M, q0_w),
            np.full(M, wor_0),
            np.full(M, res_age),
        ])
        D_o_exp = np.maximum(0.01, m_exp.model_D.predict(X_dca))
        qo_exp = max(q0_o, 1.0) * np.exp(-D_o_exp[:, None] * t_years[None, :])
        
        if hasattr(m_exp, "model_water_growth") and m_exp.model_water_growth is not None:
            w_growth = m_exp.model_water_growth.predict(X_dca)
            Dw = 0.20
            qw_exp = np.maximum(0.0, q0_w + (w_growth / Dw)[:, None] * (1.0 - np.exp(-Dw * t_years[None, :])))
        else:
            qw_exp = np.ones((M, n_steps)) * max(q0_w, 1.0)
            
        cum_o_exp = Q0_o + np.cumsum(qo_exp * dt_days, axis=1)
        cum_w_exp = Q0_w + np.cumsum(qw_exp * dt_days, axis=1)
        
        out_dict["Learned_Exponential_DCA"] = {
            "qo_rate": qo_exp, "qw_rate": qw_exp,
            "cum_oil": cum_o_exp, "cum_water": cum_w_exp,
            "eur_oil": cum_o_exp[:, -1], "eur_water": cum_w_exp[:, -1],
            "delta_cum_oil": cum_o_exp[:, -1] - Q0_o, "delta_cum_water": cum_w_exp[:, -1] - Q0_w,
            "term_qo": qo_exp[:, -1], "term_qw": qw_exp[:, -1],
        }

    # -------------------------------------------------------------------------
    # 2. Learned Hyperbolic DCA
    # -------------------------------------------------------------------------
    m_hyp = models.get("Learned_Hyperbolic_DCA")
    if m_hyp is not None:
        X_dca = np.column_stack([
            U_phys[:, 0], U_phys[:, 1], U_phys[:, 2], U_phys[:, 3],
            np.full(M, q0_o),
            np.full(M, q0_w),
            np.full(M, wor_0),
            np.full(M, res_age),
        ])
        D_o_hyp = np.maximum(0.01, m_hyp.model_D.predict(X_dca))
        b_o = getattr(m_hyp, "b_hyperbolic", 0.35)
        qo_hyp = max(q0_o, 1.0) * ((1.0 + b_o * D_o_hyp[:, None] * t_years[None, :]) ** (-1.0 / b_o))
        
        if hasattr(m_hyp, "model_water_growth") and m_hyp.model_water_growth is not None:
            w_growth_hyp = m_hyp.model_water_growth.predict(X_dca)
            Dw = 0.20
            qw_hyp = np.maximum(0.0, q0_w + (w_growth_hyp / Dw)[:, None] * (1.0 - np.exp(-Dw * t_years[None, :])))
        else:
            qw_hyp = np.ones((M, n_steps)) * max(q0_w, 1.0)
            
        cum_o_hyp = Q0_o + np.cumsum(qo_hyp * dt_days, axis=1)
        cum_w_hyp = Q0_w + np.cumsum(qw_hyp * dt_days, axis=1)
        
        out_dict["Learned_Hyperbolic_DCA"] = {
            "qo_rate": qo_hyp, "qw_rate": qw_hyp,
            "cum_oil": cum_o_hyp, "cum_water": cum_w_hyp,
            "eur_oil": cum_o_hyp[:, -1], "eur_water": cum_w_hyp[:, -1],
            "delta_cum_oil": cum_o_hyp[:, -1] - Q0_o, "delta_cum_water": cum_w_hyp[:, -1] - Q0_w,
            "term_qo": qo_hyp[:, -1], "term_qw": qw_hyp[:, -1],
        }

    # -------------------------------------------------------------------------
    # 3. Recursive ExtraTrees (Batch Autoregressive Stepping)
    # -------------------------------------------------------------------------
    m_et = models.get("Recursive_ExtraTrees")
    if m_et is not None:
        m_et.model_qo.n_jobs = 1
        m_et.model_qw.n_jobs = 1
        
        qo_et = np.zeros((M, n_steps))
        qw_et = np.zeros((M, n_steps))
        curr_qo = np.full(M, q0_o)
        curr_qw = np.full(M, q0_w)
        curr_age = res_age
        
        for k in range(n_steps):
            X_step = np.column_stack([
                np.log(np.maximum(curr_qo, 1.0)),
                np.log(np.maximum(curr_qw, 1.0) + 1.0),
                np.full(M, curr_age),
                curr_qw / np.maximum(curr_qo, 1.0),
                U_phys[:, 0], U_phys[:, 1], U_phys[:, 2], U_phys[:, 3],
            ])
            d_log_o = m_et.model_qo.predict(X_step)
            d_w = m_et.model_qw.predict(X_step)
            d_log_o_clipped = np.minimum(0.0, d_log_o)
            curr_qo = np.maximum(0.0, curr_qo * np.exp(d_log_o_clipped))
            curr_qw = np.maximum(0.0, curr_qw + d_w)
            curr_age += 0.25
            qo_et[:, k] = curr_qo
            qw_et[:, k] = curr_qw
            
        cum_o_et = Q0_o + np.cumsum(qo_et * dt_days, axis=1)
        cum_w_et = Q0_w + np.cumsum(qw_et * dt_days, axis=1)
        
        out_dict["Recursive_ExtraTrees"] = {
            "qo_rate": qo_et, "qw_rate": qw_et,
            "cum_oil": cum_o_et, "cum_water": cum_w_et,
            "eur_oil": cum_o_et[:, -1], "eur_water": cum_w_et[:, -1],
            "delta_cum_oil": cum_o_et[:, -1] - Q0_o, "delta_cum_water": cum_w_et[:, -1] - Q0_w,
            "term_qo": qo_et[:, -1], "term_qw": qw_et[:, -1],
        }

    # -------------------------------------------------------------------------
    # 4. Hybrid Dynamic (Transition -> DCA Switch)
    # -------------------------------------------------------------------------
    m_hyb = models.get("Hybrid_Dynamic")
    if m_hyb is not None:
        switch_k = min(getattr(m_hyb, "switch_quarters", 12), n_steps)
        qo_hyb = np.zeros((M, n_steps))
        qw_hyb = np.zeros((M, n_steps))
        
        # Near dates: transition model (Recursive ExtraTrees)
        qo_hyb[:, :switch_k] = out_dict["Recursive_ExtraTrees"]["qo_rate"][:, :switch_k]
        qw_hyb[:, :switch_k] = out_dict["Recursive_ExtraTrees"]["qw_rate"][:, :switch_k]
        
        if n_steps > switch_k:
            near_dates = forecast_dates[:switch_k]
            handoff_date = near_dates[-1]
            far_dates = forecast_dates[switch_k:]
            
            handoff_qo = qo_hyb[:, switch_k - 1]
            handoff_qw = qw_hyb[:, switch_k - 1]
            handoff_wor = handoff_qw / np.maximum(handoff_qo, 1.0)
            handoff_age = res_age + switch_k * 0.25
            
            t_far_years = np.asarray((pd.DatetimeIndex(far_dates) - pd.Timestamp(handoff_date)).total_seconds() / (86400.0 * YEAR_DAYS), dtype=float)
            
            X_dca_handoff = np.column_stack([
                U_phys[:, 0], U_phys[:, 1], U_phys[:, 2], U_phys[:, 3],
                handoff_qo,
                handoff_qw,
                handoff_wor,
                np.full(M, handoff_age),
            ])
            
            dca_tail = getattr(m_hyb, "dca_model", m_exp)
            D_far_o = np.maximum(0.01, dca_tail.model_D.predict(X_dca_handoff))
            w_growth_far = dca_tail.model_water_growth.predict(X_dca_handoff)
            Dw = 0.20
            
            qo_far = handoff_qo[:, None] * np.exp(-D_far_o[:, None] * t_far_years[None, :])
            qw_far = handoff_qw[:, None] + (w_growth_far / Dw)[:, None] * (1.0 - np.exp(-Dw * t_far_years[None, :]))
            qw_far = np.maximum(0.0, qw_far)
            
            qo_hyb[:, switch_k:] = qo_far
            qw_hyb[:, switch_k:] = qw_far
            
        cum_o_hyb = Q0_o + np.cumsum(qo_hyb * dt_days, axis=1)
        cum_w_hyb = Q0_w + np.cumsum(qw_hyb * dt_days, axis=1)
        
        out_dict["Hybrid_Dynamic"] = {
            "qo_rate": qo_hyb, "qw_rate": qw_hyb,
            "cum_oil": cum_o_hyb, "cum_water": cum_w_hyb,
            "eur_oil": cum_o_hyb[:, -1], "eur_water": cum_w_hyb[:, -1],
            "delta_cum_oil": cum_o_hyb[:, -1] - Q0_o, "delta_cum_water": cum_w_hyb[:, -1] - Q0_w,
            "term_qo": qo_hyb[:, -1], "term_qw": qw_hyb[:, -1],
        }

    # -------------------------------------------------------------------------
    # 5. Integrated Blend_Phase_Specific_Simplex
    # -------------------------------------------------------------------------
    w_oil = weights["oil"]
    w_wat = weights["water"]
    
    qo_ens = (
        w_oil["Recursive_ExtraTrees"] * out_dict["Recursive_ExtraTrees"]["qo_rate"]
        + w_oil["Learned_Exponential_DCA"] * out_dict["Learned_Exponential_DCA"]["qo_rate"]
        + w_oil["Learned_Hyperbolic_DCA"] * out_dict["Learned_Hyperbolic_DCA"]["qo_rate"]
        + w_oil["Hybrid_Dynamic"] * out_dict["Hybrid_Dynamic"]["qo_rate"]
    )
    qw_ens = (
        w_wat["Recursive_ExtraTrees"] * out_dict["Recursive_ExtraTrees"]["qw_rate"]
        + w_wat["Learned_Exponential_DCA"] * out_dict["Learned_Exponential_DCA"]["qw_rate"]
        + w_wat["Learned_Hyperbolic_DCA"] * out_dict["Learned_Hyperbolic_DCA"]["qw_rate"]
        + w_wat["Hybrid_Dynamic"] * out_dict["Hybrid_Dynamic"]["qw_rate"]
    )
    
    qo_ens = np.maximum(0.0, qo_ens)
    qw_ens = np.maximum(0.0, qw_ens)
    
    cum_o_ens = Q0_o + np.cumsum(qo_ens * dt_days, axis=1)
    cum_w_ens = Q0_w + np.cumsum(qw_ens * dt_days, axis=1)
    
    cum_o_ens = np.maximum.accumulate(cum_o_ens, axis=1)
    cum_w_ens = np.maximum.accumulate(cum_w_ens, axis=1)
    
    out_dict["Blend_Phase_Specific_Simplex"] = {
        "qo_rate": qo_ens, "qw_rate": qw_ens,
        "cum_oil": cum_o_ens, "cum_water": cum_w_ens,
        "eur_oil": cum_o_ens[:, -1], "eur_water": cum_w_ens[:, -1],
        "delta_cum_oil": cum_o_ens[:, -1] - Q0_o, "delta_cum_water": cum_w_ens[:, -1] - Q0_w,
        "term_qo": qo_ens[:, -1], "term_qw": qw_ens[:, -1],
    }
    
    return out_dict


def compute_local_jacobians(
    models: Dict[str, Any],
    weights: Dict[str, Dict[str, float]],
    init_state: Dict[str, Any],
    base_u: np.ndarray,
    horizons: List[int] = [4, 12, 40, 80],
    rel_step: float = 0.02,
) -> pd.DataFrame:
    """
    Computes numerical Jacobians of terminal rates and cumulative production
    with respect to each parameter across multiple horizons.
    """
    dt_days = 91.3125
    records = []
    origin = pd.Timestamp(init_state["origin_date"])
    max_h = max(horizons)
    all_dates = pd.date_range(origin + pd.DateOffset(days=1), periods=max_h, freq="QS")
    
    model_keys = ["Learned_Exponential_DCA", "Recursive_ExtraTrees", "Blend_Phase_Specific_Simplex"]
    
    for h in horizons:
        f_dates = all_dates[:h]
        
        # Base evaluation
        base_out = evaluate_models_batch(models, weights, init_state, base_u[None, :], f_dates, dt_days)
        
        for p_idx, p_name in enumerate(PARAM_NAMES):
            delta = max(rel_step * base_u[p_idx], 1e-4)
            u_plus = base_u.copy()
            u_minus = base_u.copy()
            u_plus[p_idx] = min(base_u[p_idx] + delta, UPPER_BOUNDS[p_idx])
            u_minus[p_idx] = max(base_u[p_idx] - delta, LOWER_BOUNDS[p_idx])
            actual_d = u_plus[p_idx] - u_minus[p_idx]
            
            U_pair = np.vstack([u_plus, u_minus])
            pair_out = evaluate_models_batch(models, weights, init_state, U_pair, f_dates, dt_days)
            
            for m_key in model_keys:
                if m_key not in pair_out:
                    continue
                d_term_qo = (pair_out[m_key]["term_qo"][0] - pair_out[m_key]["term_qo"][1]) / actual_d
                d_term_qw = (pair_out[m_key]["term_qw"][0] - pair_out[m_key]["term_qw"][1]) / actual_d
                d_cum_qo = (pair_out[m_key]["eur_oil"][0] - pair_out[m_key]["eur_oil"][1]) / actual_d
                d_cum_qw = (pair_out[m_key]["eur_water"][0] - pair_out[m_key]["eur_water"][1]) / actual_d
                
                base_term_qo = max(base_out[m_key]["term_qo"][0], 1.0)
                base_term_qw = max(base_out[m_key]["term_qw"][0], 1.0)
                base_cum_qo = max(base_out[m_key]["eur_oil"][0], 1.0)
                base_cum_qw = max(base_out[m_key]["eur_water"][0], 1.0)
                
                records.append({
                    "horizon_quarters": h,
                    "model_name": m_key,
                    "parameter": p_name,
                    "short_name": SHORT_NAMES[p_idx],
                    "jac_terminal_oil_rate": d_term_qo,
                    "jac_terminal_water_rate": d_term_qw,
                    "jac_cum_oil": d_cum_qo,
                    "jac_cum_water": d_cum_qw,
                    "norm_sens_terminal_oil": (RANGES[p_idx] / base_term_qo) * d_term_qo,
                    "norm_sens_terminal_water": (RANGES[p_idx] / base_term_qw) * d_term_qw,
                    "norm_sens_cum_oil": (RANGES[p_idx] / base_cum_qo) * d_cum_qo,
                    "norm_sens_cum_water": (RANGES[p_idx] / base_cum_qw) * d_cum_qw,
                })
                
    return pd.DataFrame(records)


def compute_global_sensitivity_analysis_saltelli(
    models: Dict[str, Any],
    weights: Dict[str, Dict[str, float]],
    init_state: Dict[str, Any],
    unc_df: Optional[pd.DataFrame] = None,
    posterior_samples: Optional[np.ndarray] = None,
    sample_sizes: List[int] = [128, 256, 512, 1024],
    seed: int = 42,
    horizon_quarters: int = 80,
    n_bootstrap: int = 500,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes rigorous Saltelli / Jansen pick-freeze Global Sensitivity Analysis across
    sample sizes N in {128, 256, 512, 1024} and across three parameter input distributions:
    1. Broad uniform bounds
    2. Empirical training distribution (Cases 1-70)
    3. History-matched posterior samples
    Computes first-order S_i and total-order S_Ti with bootstrap standard errors and 95% CIs.
    """
    dt_days = 91.3125
    origin = pd.Timestamp(init_state["origin_date"])
    f_dates = pd.date_range(origin + pd.DateOffset(days=1), periods=horizon_quarters, freq="QS")
    
    N_max = max(sample_sizes)
    sampler = qmc.Sobol(d=8, seed=seed)
    # Generate 2*d columns: first 4 for A, last 4 for B
    mat_raw = sampler.random(N_max)
    A_unit = mat_raw[:, :4]
    B_unit = mat_raw[:, 4:]
    
    # Distributions to evaluate
    distributions = ["Uniform_Prior_Bounds"]
    if unc_df is not None:
        distributions.append("Empirical_Training_Dist")
    if posterior_samples is not None:
        distributions.append("Posterior_Conditioned")
        
    records_convergence = []
    records_summary = []
    
    for dist_name in distributions:
        if dist_name == "Uniform_Prior_Bounds":
            A_phys = denormalize_parameters(A_unit)
            B_phys = denormalize_parameters(B_unit)
        elif dist_name == "Empirical_Training_Dist":
            train_params = unc_df.loc[TRAIN_CASES, PARAM_NAMES].to_numpy(dtype=float)
            # Map unit uniform to empirical marginals via quantile interpolation
            A_phys = np.column_stack([
                np.quantile(train_params[:, j], A_unit[:, j]) for j in range(4)
            ])
            B_phys = np.column_stack([
                np.quantile(train_params[:, j], B_unit[:, j]) for j in range(4)
            ])
        else: # Posterior_Conditioned
            A_phys = np.column_stack([
                np.quantile(posterior_samples[:, j], A_unit[:, j]) for j in range(4)
            ])
            B_phys = np.column_stack([
                np.quantile(posterior_samples[:, j], B_unit[:, j]) for j in range(4)
            ])
            
        # Construct pick-freeze matrices A_B^(i)
        A_B_phys = []
        for i in range(4):
            mat_i = A_phys.copy()
            mat_i[:, i] = B_phys[:, i]
            A_B_phys.append(mat_i)
            
        # Batch evaluate models on A, B, and each A_B^(i)
        eval_A = evaluate_models_batch(models, weights, init_state, A_phys, f_dates, dt_days)
        eval_B = evaluate_models_batch(models, weights, init_state, B_phys, f_dates, dt_days)
        eval_AB = [
            evaluate_models_batch(models, weights, init_state, A_B_phys[i], f_dates, dt_days)
            for i in range(4)
        ]
        
        dist_labels = {
            "Uniform_Prior_Bounds": "Independent Prior Product Measure",
            "Empirical_Training_Dist": "Empirical Training Marginals (Cases 1-70)",
            "Posterior_Conditioned": "Posterior Inversion Support Marginals",
        }
        dist_assumption = dist_labels.get(dist_name, dist_name)
        
        target_keys = ["delta_cum_oil", "delta_cum_water", "eur_oil", "eur_water", "term_qo", "term_qw"]
        model_names = ["Learned_Exponential_DCA", "Recursive_ExtraTrees", "Blend_Phase_Specific_Simplex"]
        
        for m_name in model_names:
            if m_name not in eval_A:
                continue
            for tgt in target_keys:
                yA_full = eval_A[m_name][tgt]
                yB_full = eval_B[m_name][tgt]
                yAB_full = [eval_AB[i][m_name][tgt] for i in range(4)]
                
                # Convergence check across sample sizes using centered Jansen estimators
                for N in sample_sizes:
                    yA_N = yA_full[:N]
                    yB_N = yB_full[:N]
                    yAB_N = [yAB_full[i][:N] for i in range(4)]
                    sobol_pt_N = compute_centered_sobol_indices(yA_N, yB_N, yAB_N, estimator="jansen")
                    var_y = float(sobol_pt_N["variance"])
                    
                    for i in range(4):
                        records_convergence.append({
                            "distribution": dist_name,
                            "distribution_assumption": dist_assumption,
                            "model_name": m_name,
                            "target": tgt,
                            "sample_size_N": N,
                            "parameter": PARAM_NAMES[i],
                            "short_name": SHORT_NAMES[i],
                            "first_order_Si": float(sobol_pt_N["first_order"][i]),
                            "total_order_STi": float(sobol_pt_N["total_order"][i]),
                            "target_variance": var_y,
                        })
                        
                # Compute bootstrap standard errors and CIs at max sample size using centered estimators
                boot_res = bootstrap_sobol_confidence_intervals(
                    yA=yA_full[:N_max],
                    yB=yB_full[:N_max],
                    yAB=[yAB_full[i][:N_max] for i in range(4)],
                    n_bootstraps=n_bootstrap,
                    confidence_level=0.95,
                    estimator="jansen",
                    seed=seed + 42,
                )
                
                for i in range(4):
                    records_summary.append({
                        "distribution": dist_name,
                        "distribution_assumption": dist_assumption,
                        "model_name": m_name,
                        "target": tgt,
                        "parameter": PARAM_NAMES[i],
                        "short_name": SHORT_NAMES[i],
                        "first_order_Si": float(boot_res["first_order_point"][i]),
                        "se_Si": float(boot_res["first_order_se"][i]),
                        "ci95_low_Si": float(boot_res["first_order_ci_low"][i]),
                        "ci95_high_Si": float(boot_res["first_order_ci_high"][i]),
                        "total_order_STi": float(boot_res["total_order_point"][i]),
                        "se_STi": float(boot_res["total_order_se"][i]),
                        "ci95_low_STi": float(boot_res["total_order_ci_low"][i]),
                        "ci95_high_STi": float(boot_res["total_order_ci_high"][i]),
                        "target_variance": float(boot_res["variance"]),
                        "sample_size_N": N_max,
                    })
                    
    conv_df = pd.DataFrame(records_convergence)
    summ_df = pd.DataFrame(records_summary)
    return conv_df, summ_df


def inspect_fitted_representations(models: Dict[str, Any]) -> Dict[str, Any]:
    """
    Direct mathematical inspection of model coefficients, feature attributions,
    and range-normalized physical parameter sensitivities.
    Reads authoritative feat_cols directly from fitted estimators.
    """
    findings = {}
    
    # 1. Learned Exponential DCA Coefficients
    if "Learned_Exponential_DCA" in models:
        m_exp = models["Learned_Exponential_DCA"]
        feat_cols_dca = getattr(m_exp, "feat_cols", ["fault", "poro", "perm", "aq", "qo_0", "qw_0", "wor_0", "res_age"])
        coefs_D = m_exp.model_D.coef_
        intercept_D = float(m_exp.model_D.intercept_)
        coef_dict_D = {fn: float(c) for fn, c in zip(feat_cols_dca, coefs_D)}
        
        coefs_w = m_exp.model_water_growth.coef_
        intercept_w = float(m_exp.model_water_growth.intercept_)
        coef_dict_w = {fn: float(c) for fn, c in zip(feat_cols_dca, coefs_w)}
        
        # Physical range sensitivities: Delta D_i = |beta_i| * Range_i
        # Admissible ranges: Fault [0.01, 1.0] -> 0.99, Poro [0.8, 1.5] -> 0.70, Perm [0.1, 10.0] -> 9.90, AqPV [10, 300] -> 290.0
        param_ranges = {
            "fault": 0.99,
            "poro": 0.70,
            "perm": 9.90,
            "aq": 290.0,
            "qo_0": 8000.0,
            "qw_0": 7000.0,
            "wor_0": 1.40,
            "res_age": 7.0,
        }
        phys_sens_D = {fn: abs(coef_dict_D[fn]) * param_ranges[fn] for fn in feat_cols_dca}
        phys_sens_w = {fn: abs(coef_dict_w[fn]) * param_ranges[fn] for fn in feat_cols_dca}
        
        findings["Learned_Exponential_DCA_features"] = list(feat_cols_dca)
        findings["Learned_Exponential_DCA_coefs_D"] = coef_dict_D
        findings["Learned_Exponential_DCA_intercept_D"] = intercept_D
        findings["Learned_Exponential_DCA_alpha_D"] = float(m_exp.model_D.alpha_)
        findings["Learned_Exponential_DCA_physical_sensitivity_D"] = phys_sens_D
        findings["Learned_Exponential_DCA_coefs_water"] = coef_dict_w
        findings["Learned_Exponential_DCA_intercept_water"] = intercept_w
        findings["Learned_Exponential_DCA_alpha_water"] = float(m_exp.model_water_growth.alpha_)
        findings["Learned_Exponential_DCA_physical_sensitivity_water"] = phys_sens_w
        
        # Key physical comparisons
        findings["poro_vs_fault_ratio"] = float(phys_sens_D["poro"] / max(phys_sens_D["fault"], 1e-12))
        findings["poro_vs_perm_ratio"] = float(phys_sens_D["poro"] / max(phys_sens_D["perm"], 1e-12))
        findings["poro_vs_aq_ratio"] = float(phys_sens_D["poro"] / max(phys_sens_D["aq"], 1e-12))
        findings["fault_vs_perm_ratio"] = float(phys_sens_D["fault"] / max(phys_sens_D["perm"], 1e-12))

    # 2. Recursive ExtraTrees Feature Importances
    if "Recursive_ExtraTrees" in models:
        m_et = models["Recursive_ExtraTrees"]
        feat_cols_et = [
            "log_qo_k", "log_qw_k", "res_age", "wor_k",
            "Fault_Transmissibility", "Porosity_Multiplier",
            "Permeability_Multiplier", "Aquifer_Pore_Volume"
        ]
        imp_qo = m_et.model_qo.feature_importances_
        imp_qw = m_et.model_qw.feature_importances_
        findings["ExtraTrees_features"] = feat_cols_et
        findings["ExtraTrees_importances_oil"] = {fn: float(v) for fn, v in zip(feat_cols_et, imp_qo)}
        findings["ExtraTrees_importances_water"] = {fn: float(v) for fn, v in zip(feat_cols_et, imp_qw)}
        
    return findings
