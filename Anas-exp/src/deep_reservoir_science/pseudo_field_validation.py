"""
Module: pseudo_field_validation.py
Addresses: Research 1 — End-to-End Pseudo-Field Validation
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Implements a leave-reservoir-out evaluation framework across synthetic fields (Cases 1-70).
For every held-out synthetic field:
1. Excludes the field from emulator fitting and model training.
2. Exposes only historical production up to origin T_0.
3. Estimates geological parameters via history matching (inversion).
4. Runs forecasting models under three information regimes:
   - Approach A (True Geology Oracle): Actual 4 geological parameters supplied.
   - Approach B (Inferred Geology): Inverted/history-matched geological parameters supplied.
   - Approach C (No Geology Inversion): Pure historical observables and state variables directly.
5. Decomposes total forecast error into:
   Total Error = Forecasting Model Error (Oracle) + Parameter Inversion Penalty
"""
from pathlib import Path
from typing import Dict, List, Tuple, Any
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import TruncatedSVD
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

from ensemble.data import load_uncertainty, TRAIN_CASES, VAL_CASES, PARAMS
from ensemble.metrics import compute_increment_nrmse, compute_bias
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
    extract_transition_training_data,
    RecursiveTransitionForecaster,
    LearnedDCARateForecaster,
    HybridDynamicForecaster,
    YEAR_DAYS,
)

TABLE_DIR = Path("outputs/research/deep_reservoir_science/tables")
FIG_DIR = Path("outputs/research/deep_reservoir_science/figures")
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

PARAM_BOUNDS = [
    (0.04, 0.15),   # Fault Transmissibility
    (0.80, 1.50),   # Porosity Multiplier
    (0.50, 9.00),   # Permeability Multiplier
    (50.0, 200.0),  # Aquifer Pore Volume
]
BOUNDS_ARR = np.array(PARAM_BOUNDS)


class FastLeaveOutEmulator:
    """Fast historical trajectory emulator for leave-one-reservoir-out inversion."""
    def __init__(self, n_components: int = 3, random_state: int = 42):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.svd_o = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.svd_w = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.gps_o = []
        self.gps_w = []

    def fit(self, X: np.ndarray, Yo: np.ndarray, Yw: np.ndarray):
        Xs = self.scaler.fit_transform(X)
        Co = self.svd_o.fit_transform(Yo)
        Cw = self.svd_w.fit_transform(Yw)
        
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2))
            * Matern(length_scale=np.ones(4), length_scale_bounds=(1e-1, 10.0), nu=2.5)
            + WhiteKernel(1e-3, (1e-5, 1e-1))
        )
        self.gps_o = [GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, random_state=42+k).fit(Xs, Co[:, k]) for k in range(self.n_components)]
        self.gps_w = [GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, random_state=52+k).fit(Xs, Cw[:, k]) for k in range(self.n_components)]
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xs = self.scaler.transform(X)
        co = np.column_stack([gp.predict(Xs) for gp in self.gps_o])
        cw = np.column_stack([gp.predict(Xs) for gp in self.gps_w])
        yo = self.svd_o.inverse_transform(co)
        yw = self.svd_w.inverse_transform(cw)
        return yo, yw


def invert_geology_for_field(
    emulator: FastLeaveOutEmulator,
    obs_qo: np.ndarray,
    obs_qw: np.ndarray,
    train_params: np.ndarray,
) -> np.ndarray:
    """Finds best matching geological parameter vector using emulator misfit minimization."""
    # 1. Best match from existing training cases
    best_dist = np.inf
    best_theta = train_params[0].copy()
    
    # Evaluate across a grid of training candidates
    yo_cand, yw_cand = emulator.predict(train_params)
    norm_o = np.std(obs_qo) + 1e-6
    norm_w = np.std(obs_qw) + 1e-6
    
    dist_cand = np.mean(((yo_cand - obs_qo) / norm_o) ** 2, axis=1) + np.mean(((yw_cand - obs_qw) / norm_w) ** 2, axis=1)
    best_idx = np.argmin(dist_cand)
    best_theta = train_params[best_idx].copy()
    
    # 2. Local L-BFGS refinement from best training candidate
    def obj_fn(th):
        yo, yw = emulator.predict(th.reshape(1, -1))
        return float(np.mean(((yo[0] - obs_qo) / norm_o) ** 2) + np.mean(((yw[0] - obs_qw) / norm_w) ** 2))
        
    res = minimize(obj_fn, best_theta, method="L-BFGS-B", bounds=PARAM_BOUNDS, options={"maxiter": 20})
    if res.success and np.isfinite(res.fun):
        return res.x
    return best_theta


def run_pseudo_field_validation(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    test_fields: List[int] = None,
    origin: str = "2005-01-01",
    horizon_quarters: int = 12,
) -> pd.DataFrame:
    """
    Executes end-to-end leave-reservoir-out pseudo-field validation on Cases 1-70.
    """
    print("\n" + "=" * 80)
    print(f"RESEARCH 1: END-TO-END PSEUDO-FIELD VALIDATION (Origin: {origin}, {horizon_quarters}Q)")
    print("=" * 80)
    
    orig = pd.Timestamp(origin)
    f_dates = pd.date_range(orig + pd.DateOffset(days=1), periods=horizon_quarters, freq="QS")
    dt_days = 91.3125
    
    if test_fields is None:
        # Select 15 diverse pseudo-fields across Cases 1-70 spanning parameter bounds
        test_fields = [3, 8, 14, 19, 25, 31, 38, 44, 50, 56, 62, 65, 67, 69, 70]
        
    print(f"Selected {len(test_fields)} pseudo-fields for leave-reservoir-out evaluation: {test_fields}")
    
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
    
    records = []
    
    for idx, f_id in enumerate(test_fields):
        t0 = time.time()
        train_pool = [c for c in TRAIN_CASES if c != f_id]
        
        # Historical production up to origin
        df_target = case_dict[f_id]
        pre_target = df_target[df_target["date"] <= orig]
        post_target = df_target[df_target["date"] > orig].iloc[:horizon_quarters]
        
        last_t = pre_target.iloc[-1]
        init_st = {
            "qo": float(last_t["oil_rate_stbd"]),
            "qw": float(last_t["water_rate_stbd"]),
            "qg": float(last_t["gas_rate_mscfd"]),
            "Qo": float(last_t["oil_cum_stb"]),
            "Qw": float(last_t["water_cum_stb"]),
            "Qg": float(last_t["gas_cum_mscf"]),
            "wor": float(last_t["wor"]),
            "res_age": float(last_t["res_age_years"]),
            "origin_date": orig,
        }
        
        y_true_o = post_target["oil_cum_stb"].values
        y_true_w = post_target["water_cum_stb"].values
        inc_o = y_true_o - init_st["Qo"]
        inc_w = y_true_w - init_st["Qw"]
        
        # 1. Fit historical emulator on train_pool up to orig
        X_train = unc_df.loc[train_pool][PARAMS].values
        Yo_train = np.array([case_dict[c][case_dict[c]["date"] <= orig]["oil_rate_stbd"].values for c in train_pool])
        Yw_train = np.array([case_dict[c][case_dict[c]["date"] <= orig]["water_rate_stbd"].values for c in train_pool])
        
        emulator = FastLeaveOutEmulator(n_components=3, random_state=42).fit(X_train, Yo_train, Yw_train)
        
        # 2. Invert geology for pseudo-field f_id
        obs_qo = pre_target["oil_rate_stbd"].values
        obs_qw = pre_target["water_rate_stbd"].values
        inferred_theta = invert_geology_for_field(emulator, obs_qo, obs_qw, X_train)
        
        true_theta = unc_df.loc[f_id][PARAMS].values
        u_true = pd.Series(true_theta, index=PARAMS)
        u_inferred = pd.Series(inferred_theta, index=PARAMS)
        
        param_errors = np.abs(inferred_theta - true_theta) / np.ptp(BOUNDS_ARR, axis=1)
        mean_param_error = float(np.mean(param_errors))
        
        # 3. Fit forecasting models on train_pool (strict as-of)
        case_dict_asof = {c: case_dict[c][case_dict[c]["date"] <= orig] for c in train_pool}
        
        m_et = RecursiveTransitionForecaster(model_type="extratrees").fit(
            extract_transition_training_data(case_dict_asof, unc_df, train_pool, orig)
        )
        m_exp = LearnedDCARateForecaster(decline_type="exponential").fit(
            case_dict_asof, unc_df, train_pool, orig, horizon_quarters
        )
        m_hyp = LearnedDCARateForecaster(decline_type="hyperbolic").fit(
            case_dict_asof, unc_df, train_pool, orig, horizon_quarters
        )
        m_hyb = HybridDynamicForecaster(switch_quarters=12).fit(
            case_dict_asof, unc_df, train_pool, orig, horizon_quarters
        )
        
        # ----------------------------------------------------------------------
        # APPROACH A: True Geology (Oracle Benchmark)
        # ----------------------------------------------------------------------
        r_et_A = m_et.predict_rates(init_st, u_true, f_dates)
        r_exp_A = m_exp.predict_rates(init_st, u_true, f_dates)
        r_hyp_A = m_hyp.predict_rates(init_st, u_true, f_dates)
        r_hyb_A = m_hyb.predict_rates(init_st, u_true, f_dates)
        
        qo_A = weights_oil["Recursive_ExtraTrees"]*r_et_A["oil_rate"] + weights_oil["Learned_Exponential_DCA"]*r_exp_A["oil_rate"] + weights_oil["Hybrid_Dynamic"]*r_hyb_A["oil_rate"]
        qw_A = weights_wat["Recursive_ExtraTrees"]*r_et_A["water_rate"] + weights_wat["Learned_Exponential_DCA"]*r_exp_A["water_rate"] + weights_wat["Learned_Hyperbolic_DCA"]*r_hyp_A["water_rate"]
        
        cum_o_A = init_st["Qo"] + np.cumsum(qo_A * dt_days)
        cum_w_A = init_st["Qw"] + np.cumsum(qw_A * dt_days)
        no_A = compute_increment_nrmse(cum_o_A - y_true_o, inc_o)
        nw_A = compute_increment_nrmse(cum_w_A - y_true_w, inc_w)
        mac_A = 0.5 * (no_A + nw_A)
        
        # ----------------------------------------------------------------------
        # APPROACH B: Inferred Geology (HM Inversion)
        # ----------------------------------------------------------------------
        r_et_B = m_et.predict_rates(init_st, u_inferred, f_dates)
        r_exp_B = m_exp.predict_rates(init_st, u_inferred, f_dates)
        r_hyp_B = m_hyp.predict_rates(init_st, u_inferred, f_dates)
        r_hyb_B = m_hyb.predict_rates(init_st, u_inferred, f_dates)
        
        qo_B = weights_oil["Recursive_ExtraTrees"]*r_et_B["oil_rate"] + weights_oil["Learned_Exponential_DCA"]*r_exp_B["oil_rate"] + weights_oil["Hybrid_Dynamic"]*r_hyb_B["oil_rate"]
        qw_B = weights_wat["Recursive_ExtraTrees"]*r_et_B["water_rate"] + weights_wat["Learned_Exponential_DCA"]*r_exp_B["water_rate"] + weights_wat["Learned_Hyperbolic_DCA"]*r_hyp_B["water_rate"]
        
        cum_o_B = init_st["Qo"] + np.cumsum(qo_B * dt_days)
        cum_w_B = init_st["Qw"] + np.cumsum(qw_B * dt_days)
        no_B = compute_increment_nrmse(cum_o_B - y_true_o, inc_o)
        nw_B = compute_increment_nrmse(cum_w_B - y_true_w, inc_w)
        mac_B = 0.5 * (no_B + nw_B)
        
        # ----------------------------------------------------------------------
        # APPROACH C: No Geology Inversion (Historical Production Observables Only)
        # Uses unconditioned observable historical DCA + ET state transition directly
        # with average geology prior [midpoint of parameter bounds]
        # ----------------------------------------------------------------------
        u_nogeo = pd.Series(np.mean(BOUNDS_ARR, axis=1), index=PARAMS)
        r_et_C = m_et.predict_rates(init_st, u_nogeo, f_dates)
        r_exp_C = m_exp.predict_rates(init_st, u_nogeo, f_dates)
        r_hyp_C = m_hyp.predict_rates(init_st, u_nogeo, f_dates)
        r_hyb_C = m_hyb.predict_rates(init_st, u_nogeo, f_dates)
        
        qo_C = weights_oil["Recursive_ExtraTrees"]*r_et_C["oil_rate"] + weights_oil["Learned_Exponential_DCA"]*r_exp_C["oil_rate"] + weights_oil["Hybrid_Dynamic"]*r_hyb_C["oil_rate"]
        qw_C = weights_wat["Recursive_ExtraTrees"]*r_et_C["water_rate"] + weights_wat["Learned_Exponential_DCA"]*r_exp_C["water_rate"] + weights_wat["Learned_Hyperbolic_DCA"]*r_hyp_C["water_rate"]
        
        cum_o_C = init_st["Qo"] + np.cumsum(qo_C * dt_days)
        cum_w_C = init_st["Qw"] + np.cumsum(qw_C * dt_days)
        no_C = compute_increment_nrmse(cum_o_C - y_true_o, inc_o)
        nw_C = compute_increment_nrmse(cum_w_C - y_true_w, inc_w)
        mac_C = 0.5 * (no_C + nw_C)
        
        # Error decomposition
        inv_penalty = mac_B - mac_A
        inv_penalty_pct = (inv_penalty / max(mac_B, 1e-4)) * 100.0
        
        records.append({
            "pseudo_field_id": f_id,
            "origin": origin,
            "horizon_quarters": horizon_quarters,
            "macro_A_true_oracle": mac_A,
            "oil_A_true_oracle": no_A,
            "wat_A_true_oracle": nw_A,
            "macro_B_inferred": mac_B,
            "oil_B_inferred": no_B,
            "wat_B_inferred": nw_B,
            "macro_C_nogeo": mac_C,
            "oil_C_nogeo": no_C,
            "wat_C_nogeo": nw_C,
            "inversion_penalty": inv_penalty,
            "inversion_penalty_pct": inv_penalty_pct,
            "mean_geology_inversion_error": mean_param_error,
            "runtime_sec": time.time() - t0,
        })
        print(f"  Field {f_id:2d} ({idx+1}/{len(test_fields)}): Oracle A = {mac_A:.4f} | Inferred B = {mac_B:.4f} | NoGeol C = {mac_C:.4f} | Penalty = {inv_penalty:+.4f} ({time.time()-t0:.1f}s)")

    df_res = pd.DataFrame(records)
    out_path = TABLE_DIR / "pseudo_field_inversion_vs_forecasting_audit.csv"
    df_res.to_csv(out_path, index=False)
    print(f"\nSaved Pseudo-Field Validation Audit: {out_path}")
    
    # Print Executive Decomposition Summary
    mean_A = df_res["macro_A_true_oracle"].mean()
    mean_B = df_res["macro_B_inferred"].mean()
    mean_C = df_res["macro_C_nogeo"].mean()
    mean_pen = df_res["inversion_penalty"].mean()
    mean_pen_pct = df_res["inversion_penalty_pct"].mean()
    
    print("\n" + "=" * 60)
    print("RESEARCH 1 ERROR DECOMPOSITION SUMMARY:")
    print("=" * 60)
    print(f"  Approach A (True Geology Oracle Benchmark): {mean_A:.4f} Macro NRMSE")
    print(f"  Approach B (Inferred Geology via Inversion): {mean_B:.4f} Macro NRMSE")
    print(f"  Approach C (No Geology Inversion / Observables): {mean_C:.4f} Macro NRMSE")
    print(f"  Net Inversion Penalty (B - A):              {mean_pen:+.4f} NRMSE ({mean_pen_pct:.1f}% of total error)")
    print(f"  Forecasting Model Share of Error (A / B):    {(mean_A / mean_B)*100.0:.1f}%")
    print(f"  Parameter Estimation Share of Error:         {(mean_pen / mean_B)*100.0:.1f}%")
    print("=" * 60)
    
    return df_res


if __name__ == "__main__":
    unc = load_uncertainty()
    qdf = load_clean_simulation_quarterly()
    cdict = build_case_quarterly_trajectories(qdf)
    run_pseudo_field_validation(cdict, unc)
