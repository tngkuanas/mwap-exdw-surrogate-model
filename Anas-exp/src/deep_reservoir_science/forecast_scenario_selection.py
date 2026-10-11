"""
Module: forecast_scenario_selection.py
Addresses: Research 5 — Forecast-Oriented Geological Scenario Selection (Oliver 2026)
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Methodology based on:
Oliver (2026), "Improved forecasting under imperfect posterior sampling"
Computational Geosciences. DOI: 10.1007/s10596-026-10461-2

Compares scenario selection and weighting rules on leave-reservoir-out pseudo-fields:
1. Policy 1: Ten Best Historical Matches (equal weights)
2. Policy 2: Posterior L2 Probabilistic Weighting (Gaussian likelihood weights)
3. Policy 3: Forecast-Validated Weighting (Oliver 2026: weights conditioned on historical validation stability)
4. Policy 4: Representative Trajectory Clustering (K-means medoids over top matches)

Investigates:
- Does alternative scenario weighting materially change mean forecast accuracy?
- Or does it primarily change forecast dispersion / spread?
- Ensures forecast weights are learned ONLY from training fields, never touching target forecasts.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.cluster import KMeans

from ensemble.data import load_uncertainty, TRAIN_CASES, VAL_CASES, PARAMS
from ensemble.metrics import compute_increment_nrmse, compute_bias
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
    LearnedDCARateForecaster,
    YEAR_DAYS,
)

TABLE_DIR = Path("outputs/research/deep_reservoir_science/tables")
FIG_DIR = Path("outputs/research/deep_reservoir_science/figures")
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def compute_historical_misfits(
    obs_qo: np.ndarray,
    obs_qw: np.ndarray,
    cand_qo_dict: Dict[int, np.ndarray],
    cand_qw_dict: Dict[int, np.ndarray],
    candidates: List[int],
) -> pd.DataFrame:
    """Computes normalized L2 historical misfit for each training candidate."""
    norm_o = np.std(obs_qo) + 1e-6
    norm_w = np.std(obs_qw) + 1e-6
    
    records = []
    for c in candidates:
        qo_c = cand_qo_dict[c]
        qw_c = cand_qw_dict[c]
        
        misfit_o = np.mean(((qo_c - obs_qo) / norm_o) ** 2)
        misfit_w = np.mean(((qw_c - obs_qw) / norm_w) ** 2)
        total_misfit = 0.5 * (misfit_o + misfit_w)
        
        records.append({
            "candidate_id": c,
            "misfit_oil": misfit_o,
            "misfit_water": misfit_w,
            "total_misfit": total_misfit,
        })
        
    df_m = pd.DataFrame(records).sort_values("total_misfit").reset_index(drop=True)
    return df_m


def select_scenarios_by_policy(
    df_misfits: pd.DataFrame,
    cand_qo_dict: Dict[int, np.ndarray],
    policy: str,
    n_scenarios: int = 10,
) -> Tuple[List[int], np.ndarray]:
    """
    Returns selected candidate IDs and their normalized weights under the given policy.
    """
    if policy == "ten_best_historical":
        # Policy 1: 10 lowest historical misfits, equal weights
        top = df_misfits.iloc[:n_scenarios]
        cands = top["candidate_id"].tolist()
        weights = np.ones(n_scenarios) / n_scenarios
        return cands, weights

    elif policy == "posterior_l2_weighting":
        # Policy 2: 10 lowest misfits, softmax weights proportional to negative L2
        top = df_misfits.iloc[:n_scenarios]
        cands = top["candidate_id"].tolist()
        misfits = top["total_misfit"].values
        # Temperature scaling
        tau = np.std(misfits) + 1e-4
        weights = softmax(-misfits / tau)
        return cands, weights

    elif policy == "forecast_validated_weighting":
        # Policy 3 (Oliver 2026): Weight penalized by candidate late-history divergence
        # Emphasizes candidates whose late-time trajectory matches the trend
        top = df_misfits.iloc[:n_scenarios]
        cands = top["candidate_id"].tolist()
        misfits = top["total_misfit"].values
        # Penalize candidate curvature/instability in the last 4 quarters of history
        curv_penalties = []
        for c in cands:
            qo_c = cand_qo_dict[c]
            # Second difference as stability proxy
            curv = np.mean(np.diff(qo_c[-4:], n=2) ** 2) if len(qo_c) >= 4 else 0.0
            curv_penalties.append(curv / 1e6)
        scores = -misfits - 0.2 * np.array(curv_penalties)
        weights = softmax(scores)
        return cands, weights

    elif policy == "representative_clustering":
        # Policy 4: Top 25 candidates clustered into 10 groups, pick closest to cluster center
        pool_size = min(25, len(df_misfits))
        top_pool = df_misfits.iloc[:pool_size]
        pool_ids = top_pool["candidate_id"].tolist()
        
        # Trajectory matrix
        trajs = np.array([cand_qo_dict[c] for c in pool_ids])
        kmeans = KMeans(n_clusters=n_scenarios, random_state=42, n_init=5).fit(trajs)
        
        selected_cands = []
        for k in range(n_scenarios):
            cluster_indices = np.where(kmeans.labels_ == k)[0]
            if len(cluster_indices) == 0:
                continue
            center = kmeans.cluster_centers_[k]
            dists = np.linalg.norm(trajs[cluster_indices] - center, axis=1)
            best_in_cluster = cluster_indices[np.argmin(dists)]
            selected_cands.append(pool_ids[best_in_cluster])
            
        # Ensure exactly n_scenarios
        while len(selected_cands) < n_scenarios:
            for c in pool_ids:
                if c not in selected_cands:
                    selected_cands.append(c)
                    break
                    
        weights = np.ones(len(selected_cands)) / len(selected_cands)
        return selected_cands, weights

    else:
        raise ValueError(f"Unknown policy {policy}")


def run_scenario_selection_experiments(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    test_fields: List[int] = None,
    origin: str = "2005-01-01",
    horizon_quarters: int = 12,
) -> pd.DataFrame:
    """
    Evaluates scenario selection policies on leave-reservoir-out pseudo-fields.
    """
    print("\n" + "=" * 80)
    print(f"RESEARCH 5: FORECAST-ORIENTED SCENARIO SELECTION (Oliver 2026)")
    print(f"Origin: {origin}, Horizon: {horizon_quarters}Q")
    print("=" * 80)
    
    orig = pd.Timestamp(origin)
    f_dates = pd.date_range(orig + pd.DateOffset(days=1), periods=horizon_quarters, freq="QS")
    dt_days = 91.3125
    
    if test_fields is None:
        test_fields = [3, 8, 14, 19, 25, 31, 38, 44, 50, 56, 62, 65, 67, 69, 70]
        
    policies = [
        "ten_best_historical",
        "posterior_l2_weighting",
        "forecast_validated_weighting",
        "representative_clustering",
    ]
    
    records = []
    
    for idx, f_id in enumerate(test_fields):
        train_pool = [c for c in TRAIN_CASES if c != f_id]
        df_target = case_dict[f_id]
        pre_target = df_target[df_target["date"] <= orig]
        post_target = df_target[df_target["date"] > orig].iloc[:horizon_quarters]
        
        last = pre_target.iloc[-1]
        init_st = {
            "qo": float(last["oil_rate_stbd"]),
            "qw": float(last["water_rate_stbd"]),
            "qg": float(last["gas_rate_mscfd"]),
            "Qo": float(last["oil_cum_stb"]),
            "Qw": float(last["water_cum_stb"]),
            "Qg": float(last["gas_cum_mscf"]),
            "wor": float(last["wor"]),
            "res_age": float(last["res_age_years"]),
            "origin_date": orig,
        }
        
        y_true_o = post_target["oil_cum_stb"].values
        y_true_w = post_target["water_cum_stb"].values
        inc_o = y_true_o - init_st["Qo"]
        inc_w = y_true_w - init_st["Qw"]
        
        # Candidate historical trajectories
        cand_qo = {c: case_dict[c][case_dict[c]["date"] <= orig]["oil_rate_stbd"].values for c in train_pool}
        cand_qw = {c: case_dict[c][case_dict[c]["date"] <= orig]["water_rate_stbd"].values for c in train_pool}
        
        # Misfits against target
        obs_qo = pre_target["oil_rate_stbd"].values
        obs_qw = pre_target["water_rate_stbd"].values
        df_m = compute_historical_misfits(obs_qo, obs_qw, cand_qo, cand_qw, train_pool)
        
        # Fit DCA forecaster on train_pool
        case_dict_asof = {c: case_dict[c][case_dict[c]["date"] <= orig] for c in train_pool}
        m_dca = LearnedDCARateForecaster(decline_type="exponential").fit(
            case_dict_asof, unc_df, train_pool, orig, horizon_quarters
        )
        
        for pol in policies:
            cands, weights = select_scenarios_by_policy(df_m, cand_qo, pol, n_scenarios=10)
            
            # Predict forecasts for each of the 10 selected scenarios using candidate geology
            scen_oil_cums = []
            scen_wat_cums = []
            
            for c_cand in cands:
                u_cand = unc_df.loc[c_cand]
                preds = m_dca.predict_rates(init_st, u_cand, f_dates)
                cum_o = init_st["Qo"] + np.cumsum(preds["oil_rate"] * dt_days)
                cum_w = init_st["Qw"] + np.cumsum(preds["water_rate"] * dt_days)
                scen_oil_cums.append(cum_o)
                scen_wat_cums.append(cum_w)
                
            scen_oil_arr = np.array(scen_oil_cums)  # (10, H)
            scen_wat_arr = np.array(scen_wat_cums)  # (10, H)
            
            # Weighted ensemble forecast
            ens_cum_o = np.sum(scen_oil_arr * weights[:, None], axis=0)
            ens_cum_w = np.sum(scen_wat_arr * weights[:, None], axis=0)
            
            # Dispersion: standard deviation across 10 scenarios at terminal horizon
            disp_o = float(np.std(scen_oil_arr[:, -1]))
            disp_w = float(np.std(scen_wat_arr[:, -1]))
            
            no = compute_increment_nrmse(ens_cum_o - y_true_o, inc_o)
            nw = compute_increment_nrmse(ens_cum_w - y_true_w, inc_w)
            mac = 0.5 * (no + nw)
            
            records.append({
                "pseudo_field_id": f_id,
                "policy": pol,
                "macro_nrmse": mac,
                "oil_nrmse": no,
                "water_nrmse": nw,
                "oil_dispersion_stb": disp_o,
                "water_dispersion_stb": disp_w,
            })
            
    df_raw = pd.DataFrame(records)
    summary_records = []
    
    print("\n" + "-" * 85)
    print(f"{'Policy':<32} | {'Macro NRMSE':<11} | {'Oil NRMSE':<9} | {'Wat NRMSE':<9} | {'Oil Dispersion (STB)':<20}")
    print("-" * 85)
    for pol, grp in df_raw.groupby("policy"):
        mac = float(grp["macro_nrmse"].mean())
        no = float(grp["oil_nrmse"].mean())
        nw = float(grp["water_nrmse"].mean())
        disp = float(grp["oil_dispersion_stb"].mean())
        
        summary_records.append({
            "policy": pol,
            "macro_nrmse": mac,
            "oil_nrmse": no,
            "water_nrmse": nw,
            "mean_oil_dispersion_stb": disp,
        })
        print(f"{pol:<32} | {mac:.4f}      | {no:.4f}    | {nw:.4f}    | {disp:,.0f}")
    print("-" * 85)
    
    df_summary = pd.DataFrame(summary_records)
    out_path = TABLE_DIR / "scenario_selection_comparison.csv"
    df_summary.to_csv(out_path, index=False)
    print(f"\nSaved Scenario Selection Comparison: {out_path}")
    return df_summary


if __name__ == "__main__":
    unc = load_uncertainty()
    qdf = load_clean_simulation_quarterly()
    cdict = build_case_quarterly_trajectories(qdf)
    run_scenario_selection_experiments(cdict, unc)
