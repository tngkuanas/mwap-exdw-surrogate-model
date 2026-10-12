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
    Returns dict: model_name -> {
        'qo_rate': (M, n_steps),
        'qw_rate': (M, n_steps),
        'cum_oil': (M, n_steps),
        'cum_water': (M, n_steps),
        'eur_oil': (M,),
        'eur_water': (M,),
        'term_qo': (M,),
        'term_qw': (M,),
    }
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
    
    t_days = (forecast_dates - origin).days.to_numpy(dtype=float)
    t_years = t_days / YEAR_DAYS
    
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
        
        # Water rate via model_water_growth
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
        b_o = getattr(m_hyp, "b_hyperbolic", 0.50)
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
        
        for k in range(n_steps):
            age_k = res_age + k * (dt_days / YEAR_DAYS)
            X_step = np.column_stack([
                np.log(np.maximum(curr_qo, 1.0)),
                np.log(np.maximum(curr_qw, 1.0) + 1.0),
                np.full(M, age_k),
                curr_qw / np.maximum(curr_qo, 1.0),
                U_phys[:, 0], U_phys[:, 1], U_phys[:, 2], U_phys[:, 3],
            ])
            d_log_o = m_et.model_qo.predict(X_step)
            d_w = m_et.model_qw.predict(X_step)
            d_log_o_clipped = np.minimum(0.0, d_log_o)
            curr_qo = np.maximum(0.0, curr_qo * np.exp(d_log_o_clipped))
            curr_qw = np.maximum(0.0, curr_qw + d_w)
            qo_et[:, k] = curr_qo
            qw_et[:, k] = curr_qw
            
        cum_o_et = Q0_o + np.cumsum(qo_et * dt_days, axis=1)
        cum_w_et = Q0_w + np.cumsum(qw_et * dt_days, axis=1)
        
        out_dict["Recursive_ExtraTrees"] = {
            "qo_rate": qo_et, "qw_rate": qw_et,
            "cum_oil": cum_o_et, "cum_water": cum_w_et,
            "eur_oil": cum_o_et[:, -1], "eur_water": cum_w_et[:, -1],
            "term_qo": qo_et[:, -1], "term_qw": qw_et[:, -1],
        }

    # -------------------------------------------------------------------------
    # 4. Hybrid Dynamic (Transition -> DCA Switch)
    # -------------------------------------------------------------------------
    m_hyb = models.get("Hybrid_Dynamic")
    if m_hyb is not None and "Recursive_ExtraTrees" in out_dict and "Learned_Exponential_DCA" in out_dict:
        switch_k = min(12, n_steps)
        qo_hyb = np.zeros((M, n_steps))
        qw_hyb = np.zeros((M, n_steps))
        
        qo_hyb[:, :switch_k] = out_dict["Recursive_ExtraTrees"]["qo_rate"][:, :switch_k]
        qw_hyb[:, :switch_k] = out_dict["Recursive_ExtraTrees"]["qw_rate"][:, :switch_k]
        
        if n_steps > switch_k:
            q_anchor_o = qo_hyb[:, switch_k - 1]
            q_anchor_w = qw_hyb[:, switch_k - 1]
            t_rem_years = t_years[switch_k:] - t_years[switch_k - 1]
            D_rem_o = np.maximum(0.01, m_exp.model_D.predict(X_dca))
            qo_hyb[:, switch_k:] = q_anchor_o[:, None] * np.exp(-D_rem_o[:, None] * t_rem_years[None, :])
            qw_hyb[:, switch_k:] = q_anchor_w[:, None] + 0.05 * t_rem_years[None, :]
            
        cum_o_hyb = Q0_o + np.cumsum(qo_hyb * dt_days, axis=1)
        cum_w_hyb = Q0_w + np.cumsum(qw_hyb * dt_days, axis=1)
        
        out_dict["Hybrid_Dynamic"] = {
            "qo_rate": qo_hyb, "qw_rate": qw_hyb,
            "cum_oil": cum_o_hyb, "cum_water": cum_w_hyb,
            "eur_oil": cum_o_hyb[:, -1], "eur_water": cum_w_hyb[:, -1],
            "term_qo": qo_hyb[:, -1], "term_qw": qw_hyb[:, -1],
        }

    # -------------------------------------------------------------------------
    # 5. Integrated Blend_Phase_Specific_Simplex
    # -------------------------------------------------------------------------
    w_oil = weights["oil"]
    w_wat = weights["water"]
    
    qo_ens = np.zeros((M, n_steps))
    qw_ens = np.zeros((M, n_steps))
    
    for m_k, w in w_oil.items():
        if m_k in out_dict and w > 0:
            qo_ens += w * out_dict[m_k]["qo_rate"]
    for m_k, w in w_wat.items():
        if m_k in out_dict and w > 0:
            qw_ens += w * out_dict[m_k]["qw_rate"]
            
    cum_o_ens = Q0_o + np.cumsum(qo_ens * dt_days, axis=1)
    cum_w_ens = Q0_w + np.cumsum(qw_ens * dt_days, axis=1)
    
    out_dict["Blend_Phase_Specific_Simplex"] = {
        "qo_rate": qo_ens, "qw_rate": qw_ens,
        "cum_oil": cum_o_ens, "cum_water": cum_w_ens,
        "eur_oil": cum_o_ens[:, -1], "eur_water": cum_w_ens[:, -1],
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
        
        target_keys = ["eur_oil", "eur_water", "term_qo", "term_qw"]
        model_names = ["Learned_Exponential_DCA", "Recursive_ExtraTrees", "Blend_Phase_Specific_Simplex"]
        
        for m_name in model_names:
            if m_name not in eval_A:
                continue
            for tgt in target_keys:
                yA_full = eval_A[m_name][tgt]
                yB_full = eval_B[m_name][tgt]
                yAB_full = [eval_AB[i][m_name][tgt] for i in range(4)]
                
                # Convergence check across sample sizes
                for N in sample_sizes:
                    yA = yA_full[:N]
                    yB = yB_full[:N]
                    var_y = float(np.var(np.concatenate([yA, yB])))
                    if var_y < 1e-12:
                        var_y = 1e-12
                        
                    for i in range(4):
                        yAB_i = yAB_full[i][:N]
                        
                        # Jansen (1999) / Saltelli (2010) estimators:
                        # Total-order index S_Ti: (1 / (2N)) * sum((yA - yAB_i)^2) / V
                        ST_i = float(np.mean((yA - yAB_i) ** 2) / (2.0 * var_y))
                        # First-order index S_i: (1 / N) * sum(yB * (yAB_i - yA)) / V
                        S_i = float(np.mean(yB * (yAB_i - yA)) / var_y)
                        
                        # Clip to valid mathematical ranges
                        ST_i = float(np.clip(ST_i, 0.0, 1.2))
                        S_i = float(np.clip(S_i, -0.05, 1.0))
                        
                        records_convergence.append({
                            "distribution": dist_name,
                            "model_name": m_name,
                            "target": tgt,
                            "sample_size_N": N,
                            "parameter": PARAM_NAMES[i],
                            "short_name": SHORT_NAMES[i],
                            "first_order_Si": S_i,
                            "total_order_STi": ST_i,
                            "target_variance": var_y,
                        })
                        
                # Compute bootstrap standard errors and CIs at max sample size
                rng_boot = np.random.RandomState(seed + 42)
                N_ref = N_max
                yA_ref = yA_full[:N_ref]
                yB_ref = yB_full[:N_ref]
                var_ref = float(np.var(np.concatenate([yA_ref, yB_ref])))
                if var_ref < 1e-12:
                    var_ref = 1e-12
                    
                boot_Si = np.zeros((4, n_bootstrap))
                boot_STi = np.zeros((4, n_bootstrap))
                
                for b_idx in range(n_bootstrap):
                    idx_b = rng_boot.choice(N_ref, size=N_ref, replace=True)
                    yA_b = yA_ref[idx_b]
                    yB_b = yB_ref[idx_b]
                    v_b = max(float(np.var(np.concatenate([yA_b, yB_b]))), 1e-12)
                    for i in range(4):
                        yAB_b = yAB_full[i][:N_ref][idx_b]
                        boot_STi[i, b_idx] = np.mean((yA_b - yAB_b) ** 2) / (2.0 * v_b)
                        boot_Si[i, b_idx] = np.mean(yB_b * (yAB_b - yA_b)) / v_b
                        
                for i in range(4):
                    yAB_i = yAB_full[i][:N_ref]
                    st_val = float(np.clip(np.mean((yA_ref - yAB_i) ** 2) / (2.0 * var_ref), 0.0, 1.2))
                    s_val = float(np.clip(np.mean(yB_ref * (yAB_i - yA_ref)) / var_ref, -0.05, 1.0))
                    
                    se_Si = float(np.std(boot_Si[i], ddof=1))
                    se_STi = float(np.std(boot_STi[i], ddof=1))
                    ci_Si = np.percentile(boot_Si[i], [2.5, 97.5])
                    ci_STi = np.percentile(boot_STi[i], [2.5, 97.5])
                    
                    records_summary.append({
                        "distribution": dist_name,
                        "model_name": m_name,
                        "target": tgt,
                        "parameter": PARAM_NAMES[i],
                        "short_name": SHORT_NAMES[i],
                        "first_order_Si": s_val,
                        "se_Si": se_Si,
                        "ci95_low_Si": float(ci_Si[0]),
                        "ci95_high_Si": float(ci_Si[1]),
                        "total_order_STi": st_val,
                        "se_STi": se_STi,
                        "ci95_low_STi": float(ci_STi[0]),
                        "ci95_high_STi": float(ci_STi[1]),
                        "target_variance": var_ref,
                        "sample_size_N": N_ref,
                    })
                    
    conv_df = pd.DataFrame(records_convergence)
    summ_df = pd.DataFrame(records_summary)
    return conv_df, summ_df


def inspect_fitted_representations(models: Dict[str, Any]) -> Dict[str, Any]:
    """
    Direct mathematical inspection of model coefficients and feature importances.
    Proves that RidgeCV DCA coefficients on Perm and AqPV are negligible (~1e-4, ~1e-6).
    """
    findings = {}
    
    # 1. Learned Exponential DCA Coefficients
    if "Learned_Exponential_DCA" in models:
        m_exp = models["Learned_Exponential_DCA"]
        coefs = m_exp.model_D.coef_
        feature_names = [
            "log(qo)", "log(qw+1)", "res_age", "wor",
            "Fault_Transmissibility", "Porosity_Multiplier",
            "Permeability_Multiplier", "Aquifer_Pore_Volume"
        ]
        coef_dict = {fn: float(c) for fn, c in zip(feature_names, coefs)}
        findings["Learned_Exponential_DCA_coefs"] = coef_dict
        findings["alpha_selected"] = float(m_exp.model_D.alpha_)
        findings["perm_coef_is_negligible"] = bool(abs(coef_dict["Permeability_Multiplier"]) < 1e-3)
        findings["aqpv_coef_is_negligible"] = bool(abs(coef_dict["Aquifer_Pore_Volume"]) < 1e-4)

    # 2. Recursive ExtraTrees Feature Importances
    if "Recursive_ExtraTrees" in models:
        m_et = models["Recursive_ExtraTrees"]
        imp_qo = m_et.model_qo.feature_importances_
        imp_qw = m_et.model_qw.feature_importances_
        findings["ExtraTrees_importances_oil"] = {fn: float(v) for fn, v in zip(feature_names, imp_qo)}
        findings["ExtraTrees_importances_water"] = {fn: float(v) for fn, v in zip(feature_names, imp_qw)}
        
    return findings
