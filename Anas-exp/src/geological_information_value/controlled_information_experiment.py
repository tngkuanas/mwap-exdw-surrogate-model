"""
Module: controlled_information_experiment.py
Package: geological_information_value
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Section 2 — Controlled Geological Information-Value Experiment (Corrected)
Isolates the predictive value of geological parameters u by holding the forecasting
architecture, weights, historical state, and referee scoring protocol 100% invariant,
while varying ONLY the geological input vector across controlled configurations:
  A. True geological parameters (u_true)
  B. Genuine point-inferred parameters (u_point from L2 likelihood minimization)
  C. Genuine Bayesian posterior mean (u_bayes from MCMC posterior draws)
  D. Out-of-fold population mean (strictly excluding held-out case c)
  E. Random prior-sampled geology (M=10 draws per case, reporting mean +- std)
  F. Adversarially mismatched / inverted geology (u_min + u_max - u_true)
  G1. Ablate Fault Transmissibility (Ft replaced with out-of-fold mean)
  G2. Ablate Porosity Multiplier (Poro replaced with out-of-fold mean)
  G3a. Ablate Permeability with Prior Reference default (k = 1.0)
  G3b. Ablate Permeability with Out-of-fold Population Mean (k ~= 5.293)
  G3c. Ablate Permeability with Out-of-fold Population Median (k ~= 5.340)
  G4. Ablate Aquifer Pore Volume (AqPV replaced with out-of-fold mean)

Evaluates all conditions using the authoritative referee scorer (authoritative_scorer.py).
"""
from typing import Dict, List, Tuple, Any, Optional
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t

from ensemble.data import load_uncertainty, resolve_shared_dir, TRAIN_CASES, PARAMS, YEAR_DAYS
from ensemble.generate_competition_deliverable import (
    load_locked_config,
    generate_scenario_predictions,
)
from forecast_advancement.protocol_framework import build_asof_case_dict, fit_base_models_under_protocol
from history_matching_v2.emulator_revalidation_v2 import HistoricalSVDGPEmulatorV2
from history_matching_v2.history_matching_likelihood_v2 import HistoryMatchingLikelihoodV2
from geological_information_value.authoritative_scorer import integrate_and_score_rates
from geological_information_value.parameter_bounds import (
    PARAM_NAMES,
    SHORT_NAMES,
    LOWER_BOUNDS,
    UPPER_BOUNDS,
    PRIOR_MEANS,
    normalize_parameters,
    denormalize_parameters,
    create_parameter_series,
    sample_prior_uniform,
    get_adversarial_mismatched_parameters,
)
from geological_information_value.bayesian_diagnostics import run_diagnostic_multichain_mcmc

EVAL_PSEUDO_FIELDS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
PARAM_BOUND_PAIRS = list(zip(LOWER_BOUNDS, UPPER_BOUNDS))


def fit_historical_emulator_and_likelihood(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    train_pool: List[int],
    held_out_pre_df: pd.DataFrame,
    origin: pd.Timestamp,
    random_state: int = 42,
) -> Tuple[HistoricalSVDGPEmulatorV2, HistoryMatchingLikelihoodV2]:
    """
    Fits HistoricalSVDGPEmulatorV2 strictly on pre-origin training data and
    constructs HistoryMatchingLikelihoodV2 for the held-out pre-origin history.
    """
    dates_pre = held_out_pre_df["date"].reset_index(drop=True)
    X_train = unc_df.loc[train_pool, PARAM_NAMES].to_numpy(dtype=float)
    Y_oil_tr = np.array([
        case_dict[c][case_dict[c]["date"] <= origin]["oil_cum_stb"].to_numpy(dtype=float)
        for c in train_pool
    ])
    Y_wat_tr = np.array([
        case_dict[c][case_dict[c]["date"] <= origin]["water_cum_stb"].to_numpy(dtype=float)
        for c in train_pool
    ])
    
    emu = HistoricalSVDGPEmulatorV2(n_components=4, random_state=random_state)
    emu.fit(X_train, Y_oil_tr, Y_wat_tr, dates=dates_pre)
    lik_obj = HistoryMatchingLikelihoodV2(
        emu, held_out_pre_df, rho_ar1=0.80, obs_noise_rel_oil=0.05, obs_noise_rel_wat=0.05
    )
    return emu, lik_obj


def run_point_inversion_l2(
    lik_obj: HistoryMatchingLikelihoodV2,
    u_init: np.ndarray,
) -> np.ndarray:
    """
    Minimizes negative L2 log-likelihood on pre-origin production rates using L-BFGS-B.
    """
    def obj(u: np.ndarray) -> float:
        return -float(lik_obj.evaluate_log_likelihood(u, model="L2"))
    
    res = minimize(
        obj,
        u_init,
        method="L-BFGS-B",
        bounds=PARAM_BOUND_PAIRS,
        options={"maxiter": 40, "ftol": 1e-4},
    )
    return np.asarray(res.x, dtype=float)


def run_bayesian_mcmc_posterior(
    lik_obj: HistoryMatchingLikelihoodV2,
    u_start: np.ndarray,
    n_samples: int = 500,
    burn_in: int = 200,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs diagnostic multi-chain Metropolis MCMC targeting L2 likelihood to obtain posterior draws.
    Returns:
        bayes_mean: (4,)
        bayes_std: (4,)
        ci_90_low: (4,)
        ci_90_high: (4,)
    """
    diag = run_diagnostic_multichain_mcmc(
        lik_obj,
        u_point=u_start,
        u_pop_mean=PRIOR_MEANS,
        n_chains=4,
        n_samples=n_samples,
        burn_in=burn_in,
        seed=seed,
    )
    return diag["bayes_mean"], diag["bayes_std"], diag["ci_90_low"], diag["ci_90_high"]


def run_controlled_geological_value_experiment(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    pseudo_fields: List[int] = EVAL_PSEUDO_FIELDS,
    origin: pd.Timestamp = pd.Timestamp("2005-01-01"),
    horizon_quarters: int = 12,
    seed: int = 42,
    n_random_draws: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes the controlled geological information value experiment across all 10+
    geological configurations with authoritative referee scoring.
    """
    config = load_locked_config()
    weights = config["weights"]
    f_dates = pd.date_range(origin + pd.DateOffset(days=1), periods=horizon_quarters, freq="QS")
    
    case_records = []
    
    for c_idx, c in enumerate(pseudo_fields):
        # Strictly exclude held-out field c from training pool
        train_pool = [x for x in TRAIN_CASES if x != c]
        
        # Fit base models once on permitted training pool under Protocol A (strict as-of)
        m_fit = fit_base_models_under_protocol(case_dict, unc_df, train_pool, origin, "Protocol_A")
        
        df_c = case_dict[c]
        pre = df_c[df_c["date"] <= origin].reset_index(drop=True)
        post = df_c[df_c["date"] > origin].iloc[:horizon_quarters].reset_index(drop=True)
        assert len(post) == horizon_quarters, f"Case {c} has insufficient post-origin observations"
        
        last = pre.iloc[-1]
        u_true = unc_df.loc[c, PARAM_NAMES].to_numpy(dtype=float)
        
        init_st = {
            "qo": float(last["oil_rate_stbd"]),
            "qw": float(last["water_rate_stbd"]),
            "qg": float(last["gas_rate_mscfd"]),
            "Qo": float(last["oil_cum_stb"]),
            "Qw": float(last["water_cum_stb"]),
            "Qg": float(last["gas_cum_mscf"]),
            "wor": float(last.get("wor", float(last["water_rate_stbd"]) / max(float(last["oil_rate_stbd"]), 1.0))),
            "res_age": (origin - pre["date"].iloc[0]).days / YEAR_DAYS,
            "origin_date": origin,
        }
        
        # 1. Out-of-fold training population statistics (STRICTLY excluding held-out case c)
        pool_df = unc_df.loc[train_pool, PARAM_NAMES]
        u_pop_mean = pool_df.mean().to_numpy(dtype=float)
        u_pop_median = pool_df.median().to_numpy(dtype=float)
        
        # 2. Genuine Inversion via Historical SVD/GP Emulator & L2 Likelihood
        emu, lik_obj = fit_historical_emulator_and_likelihood(
            case_dict=case_dict,
            unc_df=unc_df,
            train_pool=train_pool,
            held_out_pre_df=pre,
            origin=origin,
            random_state=seed + c,
        )
        u_point = run_point_inversion_l2(lik_obj, u_init=u_pop_mean)
        diag_mcmc = run_diagnostic_multichain_mcmc(
            lik_obj,
            u_point=u_point,
            u_pop_mean=u_pop_mean,
            n_chains=4,
            n_samples=500,
            burn_in=200,
            seed=seed + 100 * c,
        )
        u_bayes = diag_mcmc["bayes_mean"]
        u_bayes_sd = diag_mcmc["bayes_std"]
        ci_low = diag_mcmc["ci_90_low"]
        ci_high = diag_mcmc["ci_90_high"]
        
        # Shrinkage between estimated geology and population mean
        u_shrink = 0.5 * u_bayes + 0.5 * u_pop_mean
        
        # 3. Adversarially Mismatched / Inverted Geology
        u_mismatch = get_adversarial_mismatched_parameters(u_true)
        
        # 4. Geological Ablations (using strictly out-of-fold values)
        u_abl_ft = np.array([u_pop_mean[0], u_true[1], u_true[2], u_true[3]])
        u_abl_poro = np.array([u_true[0], u_pop_mean[1], u_true[2], u_true[3]])
        u_abl_perm_prior = np.array([u_true[0], u_true[1], 1.000, u_true[3]])
        u_abl_perm_mean = np.array([u_true[0], u_true[1], u_pop_mean[2], u_true[3]])
        u_abl_perm_med = np.array([u_true[0], u_true[1], u_pop_median[2], u_true[3]])
        u_abl_aq = np.array([u_true[0], u_true[1], u_true[2], u_pop_mean[3]])
        
        conditions_single: Dict[str, np.ndarray] = {
            "A_True_Geology": u_true,
            "B_Point_Inferred": u_point,
            "C_Bayes_Posterior_Mean": u_bayes,
            "C2_Shrinkage_50_50": u_shrink,
            "D_Population_Mean": u_pop_mean,
            "F_Adversarial_Mismatch": u_mismatch,
            "G1_Ablate_FaultTrans": u_abl_ft,
            "G2_Ablate_Porosity": u_abl_poro,
            "G3a_Ablate_Perm_PriorRef": u_abl_perm_prior,
            "G3b_Ablate_Perm_PopMean": u_abl_perm_mean,
            "G3c_Ablate_Perm_PopMedian": u_abl_perm_med,
            "G4_Ablate_AquiferVolume": u_abl_aq,
        }
        
        # Helper to generate and authoritative-score a forecast
        def evaluate_u_vector(u_vec: np.ndarray, cond_label: str, draw_idx: Optional[int] = None) -> Dict[str, Any]:
            u_ser = create_parameter_series(u_vec)
            _, aud = generate_scenario_predictions(
                models=m_fit,
                weights=weights,
                init_state=init_st,
                params=u_ser,
                forecast_dates=f_dates,
                decline_sensitivity=1.0,
            )
            score_dict = integrate_and_score_rates(
                init_state=init_st,
                pred_rate_oil=aud["qo_rate"],
                pred_rate_wat=aud["qw_rate"],
                post_obs_df=post,
                dt_days=91.3125,
            )
            rec = {
                "case_id": c,
                "condition": cond_label,
                "draw_index": draw_idx if draw_idx is not None else 0,
                "macro_nrmse": score_dict["macro_nrmse"],
                "oil_nrmse": score_dict["oil_nrmse"],
                "water_nrmse": score_dict["water_nrmse"],
                "oil_bias": score_dict["oil_bias"],
                "water_bias": score_dict["water_bias"],
                "rmse_oil_stb": score_dict["rmse_oil_stb"],
                "rmse_wat_stb": score_dict["rmse_wat_stb"],
                "terminal_oil_rate": score_dict["pred_terminal_oil_rate"],
                "terminal_water_rate": score_dict["pred_terminal_water_rate"],
                "terminal_oil_cum": score_dict["pred_terminal_oil_cum"],
                "terminal_water_cum": score_dict["pred_terminal_water_cum"],
                "terminal_oil_rate_err": score_dict["terminal_rate_err_oil_stbd"],
                "terminal_wat_rate_err": score_dict["terminal_rate_err_wat_stbd"],
                "terminal_oil_cum_err": score_dict["terminal_cum_err_oil_stb"],
                "terminal_wat_cum_err": score_dict["terminal_cum_err_wat_stb"],
                "param_Ft": u_vec[0],
                "param_Poro": u_vec[1],
                "param_Perm": u_vec[2],
                "param_AqPV": u_vec[3],
                "true_Ft": u_true[0],
                "true_Poro": u_true[1],
                "true_Perm": u_true[2],
                "true_AqPV": u_true[3],
            }
            return rec
        
        # Evaluate deterministic conditions
        for cond_name, u_vec in conditions_single.items():
            case_records.append(evaluate_u_vector(u_vec, cond_name))
            
        # Condition C3: Full Posterior Predictive Mean Forecast E[f(u)]
        n_p_draws = len(diag_mcmc["pooled_draws"])
        idx_p_draws = np.random.RandomState(seed + 999 * c).choice(n_p_draws, size=min(20, n_p_draws), replace=False)
        sub_p_draws = diag_mcmc["pooled_draws"][idx_p_draws]
        p_qos = []
        p_qws = []
        for u_pd in sub_p_draws:
            _, aud_pd = generate_scenario_predictions(
                models=m_fit,
                weights=weights,
                init_state=init_st,
                params=create_parameter_series(u_pd),
                forecast_dates=f_dates,
                decline_sensitivity=1.0,
            )
            p_qos.append(aud_pd["qo_rate"])
            p_qws.append(aud_pd["qw_rate"])
        qo_post_mean = np.mean(p_qos, axis=0)
        qw_post_mean = np.mean(p_qws, axis=0)
        sc_post_pred = integrate_and_score_rates(
            init_state=init_st,
            pred_rate_oil=qo_post_mean,
            pred_rate_wat=qw_post_mean,
            post_obs_df=post,
            dt_days=91.3125,
        )
        case_records.append({
            "case_id": c,
            "condition": "C3_Post_Predictive_Mean",
            "draw_index": 0,
            "macro_nrmse": sc_post_pred["macro_nrmse"],
            "oil_nrmse": sc_post_pred["oil_nrmse"],
            "water_nrmse": sc_post_pred["water_nrmse"],
            "oil_bias": sc_post_pred["oil_bias"],
            "water_bias": sc_post_pred["water_bias"],
            "rmse_oil_stb": sc_post_pred["rmse_oil_stb"],
            "rmse_wat_stb": sc_post_pred["rmse_wat_stb"],
            "terminal_oil_rate": sc_post_pred["pred_terminal_oil_rate"],
            "terminal_water_rate": sc_post_pred["pred_terminal_water_rate"],
            "terminal_oil_cum": sc_post_pred["pred_terminal_oil_cum"],
            "terminal_water_cum": sc_post_pred["pred_terminal_water_cum"],
            "terminal_oil_rate_err": sc_post_pred["terminal_rate_err_oil_stbd"],
            "terminal_wat_rate_err": sc_post_pred["terminal_rate_err_wat_stbd"],
            "terminal_oil_cum_err": sc_post_pred["terminal_cum_err_oil_stb"],
            "terminal_wat_cum_err": sc_post_pred["terminal_cum_err_wat_stb"],
            "param_Ft": u_bayes[0],
            "param_Poro": u_bayes[1],
            "param_Perm": u_bayes[2],
            "param_AqPV": u_bayes[3],
            "true_Ft": u_true[0],
            "true_Poro": u_true[1],
            "true_Perm": u_true[2],
            "true_AqPV": u_true[3],
        })

        # Condition E: Multiple Random Prior Draws (M=10)
        rng_prior = np.random.RandomState(seed + 1000 * c)
        rand_recs = []
        for m_idx in range(n_random_draws):
            u_rand = denormalize_parameters(rng_prior.uniform(0.0, 1.0, 4))
            r = evaluate_u_vector(u_rand, "E_Random_Prior_Sample", draw_idx=m_idx)
            rand_recs.append(r)
            
        case_records.extend(rand_recs)
        
    raw_case_df = pd.DataFrame(case_records)
    
    # Create consolidated case-level DataFrame averaging across random draws for Condition E
    cond_e_avg = (
        raw_case_df[raw_case_df["condition"] == "E_Random_Prior_Sample"]
        .groupby(["case_id", "condition"], as_index=False)
        .agg({
            "macro_nrmse": "mean",
            "oil_nrmse": "mean",
            "water_nrmse": "mean",
            "oil_bias": "mean",
            "water_bias": "mean",
            "rmse_oil_stb": "mean",
            "rmse_wat_stb": "mean",
            "terminal_oil_rate": "mean",
            "terminal_water_rate": "mean",
            "terminal_oil_cum": "mean",
            "terminal_water_cum": "mean",
            "terminal_oil_rate_err": "mean",
            "terminal_wat_rate_err": "mean",
            "terminal_oil_cum_err": "mean",
            "terminal_wat_cum_err": "mean",
            "param_Ft": "mean",
            "param_Poro": "mean",
            "param_Perm": "mean",
            "param_AqPV": "mean",
            "true_Ft": "first",
            "true_Poro": "first",
            "true_Perm": "first",
            "true_AqPV": "first",
        })
    )
    cond_e_avg["draw_index"] = -1
    
    non_e_df = raw_case_df[raw_case_df["condition"] != "E_Random_Prior_Sample"]
    case_df = pd.concat([non_e_df, cond_e_avg], ignore_index=True)
    
    # Compute paired differences against Condition A (True Geology)
    base_case_df = case_df[case_df["condition"] == "A_True_Geology"].set_index("case_id")
    case_df["diff_vs_true_macro"] = case_df.apply(
        lambda row: row["macro_nrmse"] - base_case_df.loc[row["case_id"], "macro_nrmse"], axis=1
    )
    case_df["diff_vs_true_oil"] = case_df.apply(
        lambda row: row["oil_nrmse"] - base_case_df.loc[row["case_id"], "oil_nrmse"], axis=1
    )
    case_df["diff_vs_true_water"] = case_df.apply(
        lambda row: row["water_nrmse"] - base_case_df.loc[row["case_id"], "water_nrmse"], axis=1
    )
    
    # Aggregate Summary
    summary_list = []
    base_mean = case_df[case_df["condition"] == "A_True_Geology"]["macro_nrmse"].mean()
    
    for cond, grp in case_df.groupby("condition"):
        m_macro = grp["macro_nrmse"].mean()
        s_macro = grp["macro_nrmse"].std(ddof=1) if len(grp) > 1 else 0.0
        m_oil = grp["oil_nrmse"].mean()
        m_wat = grp["water_nrmse"].mean()
        
        diff_arr = grp["diff_vs_true_macro"].to_numpy()
        m_diff_macro = float(np.mean(diff_arr))
        s_diff_macro = float(np.std(diff_arr, ddof=1)) if len(diff_arr) > 1 else 0.0
        n_c = len(grp)
        
        # Paired standard error and 95% Confidence Interval
        se_diff = s_diff_macro / np.sqrt(max(n_c, 1))
        t_crit = student_t.ppf(0.975, df=max(n_c - 1, 1))
        ci_95_low = m_diff_macro - t_crit * se_diff
        ci_95_high = m_diff_macro + t_crit * se_diff
        
        # Paired t-test p-value
        if se_diff > 1e-12:
            t_stat = m_diff_macro / se_diff
            p_val = float(2.0 * (1.0 - student_t.cdf(abs(t_stat), df=n_c - 1)))
        else:
            p_val = 1.0 if cond == "A_True_Geology" else 0.0
            
        win_rate = float((grp["diff_vs_true_macro"] < -1e-6).mean())
        rel_deg_pct = ((m_macro - base_mean) / base_mean) * 100.0
        
        summary_list.append({
            "condition": cond,
            "macro_nrmse_mean": m_macro,
            "macro_nrmse_std": s_macro,
            "oil_nrmse_mean": m_oil,
            "water_nrmse_mean": m_wat,
            "diff_vs_true_macro": m_diff_macro,
            "diff_se": se_diff,
            "diff_ci95_low": ci_95_low,
            "diff_ci95_high": ci_95_high,
            "rel_degradation_pct": rel_deg_pct,
            "p_value_vs_true": p_val,
            "win_rate_vs_true": win_rate,
            "n_cases": n_c,
        })
        
    summary_df = pd.DataFrame(summary_list).sort_values("condition").reset_index(drop=True)
    
    # Holm-Bonferroni step-down multiple comparison correction
    non_base_mask = summary_df["condition"] != "A_True_Geology"
    sub_p = summary_df.loc[non_base_mask, "p_value_vs_true"].values
    m_tests = len(sub_p)
    sort_idx = np.argsort(sub_p)
    adj_p = np.zeros(m_tests)
    running_max = 0.0
    for rank, idx in enumerate(sort_idx):
        mult = m_tests - rank
        raw_p = sub_p[idx]
        p_cand = min(1.0, mult * raw_p)
        running_max = max(running_max, p_cand)
        adj_p[idx] = min(1.0, running_max)
        
    full_adj = np.ones(len(summary_df))
    full_adj[non_base_mask.values] = adj_p
    summary_df["p_value_holm_bonferroni"] = full_adj
    
    return case_df, summary_df


def run_multi_origin_geological_value_study(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    pseudo_fields: List[int] = EVAL_PSEUDO_FIELDS,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Evaluates True Geology vs Out-of-fold Population Mean across multiple origins:
    2003-01-01 (4Q, 12Q, 20Q), 2004-01-01 (4Q, 12Q), 2005-01-01 (4Q, 12Q).
    Quantifies whether the apparent population-mean advantage survives multiple horizons
    and forecast origins, or is an artifact of the specific 2005 12Q sample.
    """
    config = load_locked_config()
    weights = config["weights"]
    
    eval_grid = [
        ("2003-01-01", 4),
        ("2003-01-01", 12),
        ("2003-01-01", 20),
        ("2004-01-01", 4),
        ("2004-01-01", 12),
        ("2005-01-01", 4),
        ("2005-01-01", 12),
    ]
    
    study_records = []
    
    for orig_str, h_qtrs in eval_grid:
        origin = pd.Timestamp(orig_str)
        f_dates = pd.date_range(origin + pd.DateOffset(days=1), periods=h_qtrs, freq="QS")
        
        for c in pseudo_fields:
            train_pool = [x for x in TRAIN_CASES if x != c]
            m_fit = fit_base_models_under_protocol(case_dict, unc_df, train_pool, origin, "Protocol_A")
            
            df_c = case_dict[c]
            pre = df_c[df_c["date"] <= origin].reset_index(drop=True)
            post = df_c[df_c["date"] > origin].iloc[:h_qtrs].reset_index(drop=True)
            if len(post) < h_qtrs:
                continue
                
            last = pre.iloc[-1]
            u_true = unc_df.loc[c, PARAM_NAMES].to_numpy(dtype=float)
            u_pop = unc_df.loc[train_pool, PARAM_NAMES].mean().to_numpy(dtype=float)
            
            init_st = {
                "qo": float(last["oil_rate_stbd"]),
                "qw": float(last["water_rate_stbd"]),
                "qg": float(last["gas_rate_mscfd"]),
                "Qo": float(last["oil_cum_stb"]),
                "Qw": float(last["water_cum_stb"]),
                "Qg": float(last["gas_cum_mscf"]),
                "wor": float(last.get("wor", float(last["water_rate_stbd"]) / max(float(last["oil_rate_stbd"]), 1.0))),
                "res_age": (origin - pre["date"].iloc[0]).days / YEAR_DAYS,
                "origin_date": origin,
            }
            
            # Forecast True Geology
            _, aud_true = generate_scenario_predictions(
                m_fit, weights, init_st, create_parameter_series(u_true), f_dates, 1.0
            )
            sc_true = integrate_and_score_rates(
                init_state=init_st,
                pred_rate_oil=aud_true["qo_rate"],
                pred_rate_wat=aud_true["qw_rate"],
                post_obs_df=post,
                dt_days=91.3125,
            )
            
            # Forecast Population Mean Geology
            _, aud_pop = generate_scenario_predictions(
                m_fit, weights, init_st, create_parameter_series(u_pop), f_dates, 1.0
            )
            sc_pop = integrate_and_score_rates(
                init_state=init_st,
                pred_rate_oil=aud_pop["qo_rate"],
                pred_rate_wat=aud_pop["qw_rate"],
                post_obs_df=post,
                dt_days=91.3125,
            )
            
            study_records.append({
                "origin": orig_str,
                "horizon_quarters": h_qtrs,
                "case_id": c,
                "macro_true": sc_true["macro_nrmse"],
                "macro_pop": sc_pop["macro_nrmse"],
                "macro_diff": sc_pop["macro_nrmse"] - sc_true["macro_nrmse"],
                "oil_true": sc_true["oil_nrmse"],
                "oil_pop": sc_pop["oil_nrmse"],
                "oil_diff": sc_pop["oil_nrmse"] - sc_true["oil_nrmse"],
                "wat_true": sc_true["water_nrmse"],
                "wat_pop": sc_pop["water_nrmse"],
                "wat_diff": sc_pop["water_nrmse"] - sc_true["water_nrmse"],
            })
            
    return pd.DataFrame(study_records)
