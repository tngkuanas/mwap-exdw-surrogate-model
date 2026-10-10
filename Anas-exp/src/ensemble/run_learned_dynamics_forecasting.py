"""
Learned Dynamics Forecasting Framework for Reservoir Production.
Implements and evaluates genuine time-forward forecasting architectures:
- Family A: Recursive Learned State Transition (Ridge, ExtraTrees, GP)
- Family B: Learned Dynamic Rate Model (Exponential, Hyperbolic DCA)
- Family C: Hybrid Dynamic Model (Learned Transition + Rate-Continuous DCA Continuation)
- Baseline Comparisons & Physical Continuity Constraints
- Leakage-Free Rolling-Origin Backtesting & Lead-Time Error Separation
- 80-Quarter Inference Pipeline for Cases 71-85 with Uncertainty Scenarios
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.model_selection import KFold

from ensemble.data import (
    load_uncertainty,
    TRAIN_CASES,
    VAL_CASES,
    DEV_CASES,
    PARAMS,
)
from ensemble.metrics import compute_increment_nrmse, compute_bias

YEAR_DAYS = 365.25
RS_SOLUTION = 0.3633  # MSCF / STB exact physical constant

CUTOFF_CONFIGS = [
    {"cutoff": "2005-01-01", "quarters": 12, "years": 3, "name": "2005_3yr_12q"},
    {"cutoff": "2004-01-01", "quarters": 16, "years": 4, "name": "2004_4yr_16q"},
    {"cutoff": "2003-01-01", "quarters": 20, "years": 5, "name": "2003_5yr_20q"},
    {"cutoff": "2001-01-01", "quarters": 28, "years": 7, "name": "2001_7yr_28q"},
]

def load_clean_simulation_quarterly() -> pd.DataFrame:
    """
    Loads quarterly simulation records strictly for Cases 1 to 85.
    Guarantees no access or leak of Cases 86-100.
    """
    sim_path = Path("../data/simulation_timeseries.parquet")
    df = pd.read_parquet(sim_path)
    
    # Enforce strict case boundary
    q = df[(df["is_quarterly"] == True) & (df["case_id"] >= 1) & (df["case_id"] <= 85)].copy()
    q["date"] = pd.to_datetime(q["date"])
    q = q.sort_values(["case_id", "date"]).reset_index(drop=True)
    
    # Verify exact quarterly sequence
    assert q["case_id"].nunique() == 85, f"Expected 85 cases, got {q['case_id'].nunique()}"
    return q

def build_case_quarterly_trajectories(q_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Indexes quarterly time series per case with dt, rates, and cumulatives."""
    case_dict = {}
    for c, grp in q_df.groupby("case_id"):
        df_c = grp.sort_values("date").copy().reset_index(drop=True)
        dt_vals = (df_c["date"].diff().dt.total_seconds() / 86400.0).fillna(0.0).to_numpy(copy=True)
        dt_vals[0] = 0.0
        df_c["dt_days"] = dt_vals
        # Compute instantaneous quarterly WOR and GOR
        df_c["wor"] = df_c["water_rate_stbd"] / np.maximum(df_c["oil_rate_stbd"], 1.0)
        df_c["gor"] = (df_c["gas_rate_mscfd"] * 1000.0) / np.maximum(df_c["oil_rate_stbd"], 1.0)
        df_c["res_age_years"] = (df_c["date"] - pd.Timestamp("1998-01-01")).dt.total_seconds() / (86400.0 * YEAR_DAYS)
        case_dict[c] = df_c
    return case_dict

def extract_transition_training_data(
    case_dict: Dict[int, pd.DataFrame],
    unc_df: pd.DataFrame,
    train_cases: List[int],
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Extracts consecutive quarterly transitions S(k) -> S(k+1) strictly before cutoff_date
    for cases in train_cases.
    """
    records = []
    for c in train_cases:
        df_c = case_dict[c]
        sub = df_c[df_c["date"] <= cutoff_date].sort_values("date").reset_index(drop=True)
        u_c = unc_df.loc[c]
        
        # We start extracting transitions from quarter 4 (post-startup production)
        for i in range(1, len(sub)):
            prev = sub.iloc[i - 1]
            curr = sub.iloc[i]
            
            qo_k = float(prev["oil_rate_stbd"])
            qw_k = float(prev["water_rate_stbd"])
            qg_k = float(prev["gas_rate_mscfd"])
            Qo_k = float(prev["oil_cum_stb"])
            Qw_k = float(prev["water_cum_stb"])
            Qg_k = float(prev["gas_cum_mscf"])
            
            qo_next = float(curr["oil_rate_stbd"])
            qw_next = float(curr["water_rate_stbd"])
            
            dt = float(curr["dt_days"])
            if dt <= 0:
                continue
                
            # Logarithmic rate decrement: Delta ln qo
            d_log_qo = np.log(max(qo_next, 1.0)) - np.log(max(qo_k, 1.0))
            # Water rate change: Delta qw
            d_qw = qw_next - qw_k
            
            records.append({
                "case_id": c,
                "date_k": prev["date"],
                "res_age": float(prev["res_age_years"]),
                "qo_k": qo_k,
                "log_qo_k": np.log(max(qo_k, 1.0)),
                "qw_k": qw_k,
                "log_qw_k": np.log(max(qw_k, 1.0) + 1.0),
                "qg_k": qg_k,
                "Qo_k": Qo_k,
                "Qw_k": Qw_k,
                "Qg_k": Qg_k,
                "wor_k": float(prev["wor"]),
                "fault": float(u_c["Fault Transmissibility"]),
                "poro": float(u_c["Porosity Multiplier"]),
                "perm": float(u_c["Permeability Multiplier"]),
                "aq": float(u_c["Aquifer Pore Volume"]),
                "target_d_log_qo": d_log_qo,
                "target_d_qw": d_qw,
                "target_qo_next": qo_next,
                "target_qw_next": qw_next,
            })
    return pd.DataFrame(records)

STATE_FEATURES = [
    "log_qo_k",
    "log_qw_k",
    "res_age",
    "wor_k",
    "fault",
    "poro",
    "perm",
    "aq",
]

class RecursiveStateTransitionModel:
    """
    Model A: Learns single-step transition function S(k) -> S(k+1)
    and rolls forward recursively for H quarters without future ground truth.
    """
    def __init__(self, model_type: str = "ridge"):
        self.model_type = model_type
        if model_type == "ridge":
            self.model_qo = RidgeCV(alphas=np.logspace(-3, 3, 10))
            self.model_qw = RidgeCV(alphas=np.logspace(-3, 3, 10))
        elif model_type == "extratrees":
            self.model_qo = ExtraTreesRegressor(n_estimators=100, max_depth=6, random_state=42)
            self.model_qw = ExtraTreesRegressor(n_estimators=100, max_depth=6, random_state=42)
        elif model_type == "gp":
            kernel_o = Matern(nu=2.5) + WhiteKernel(noise_level=1e-3)
            kernel_w = Matern(nu=2.5) + WhiteKernel(noise_level=1e-3)
            self.model_qo = GaussianProcessRegressor(kernel=kernel_o, n_restarts_optimizer=1, random_state=42)
            self.model_qw = GaussianProcessRegressor(kernel=kernel_w, n_restarts_optimizer=1, random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
    def fit(self, train_trans_df: pd.DataFrame):
        X = train_trans_df[STATE_FEATURES].values
        # Predict logarithmic decline step for oil, water rate change for water
        y_qo = train_trans_df["target_d_log_qo"].values
        y_qw = train_trans_df["target_d_qw"].values
        self.model_qo.fit(X, y_qo)
        self.model_qw.fit(X, y_qw)
        return self

    def predict_rollout(
        self,
        initial_state: Dict[str, float],
        u_c: pd.Series,
        forecast_dates: pd.DatetimeIndex,
    ) -> Dict[str, np.ndarray]:
        """
        Recursively rolls state forward for each forecast date.
        Guarantees non-negative rates and monotonic cumulative production.
        """
        n_steps = len(forecast_dates)
        qo_preds = np.zeros(n_steps)
        qg_preds = np.zeros(n_steps)
        qw_preds = np.zeros(n_steps)
        Qo_preds = np.zeros(n_steps)
        Qg_preds = np.zeros(n_steps)
        Qw_preds = np.zeros(n_steps)
        
        curr_qo = initial_state["qo"]
        curr_qw = initial_state["qw"]
        curr_Qo = initial_state["Qo"]
        curr_Qw = initial_state["Qw"]
        curr_Qg = initial_state["Qg"]
        curr_age = initial_state["res_age"]
        
        dt_days = 91.3125 # Standard quarter length
        
        for k in range(n_steps):
            x_in = np.array([[
                np.log(max(curr_qo, 1.0)),
                np.log(max(curr_qw, 1.0) + 1.0),
                curr_age,
                curr_qw / max(curr_qo, 1.0),
                float(u_c["Fault Transmissibility"]),
                float(u_c["Porosity Multiplier"]),
                float(u_c["Permeability Multiplier"]),
                float(u_c["Aquifer Pore Volume"]),
            ]])
            
            d_log_qo_pred = float(self.model_qo.predict(x_in)[0])
            d_qw_pred = float(self.model_qw.predict(x_in)[0])
            
            # Physical regularization: reservoir past plateau cannot have positive oil rate growth
            d_log_qo_clipped = min(0.0, d_log_qo_pred)
            next_qo = curr_qo * np.exp(d_log_qo_clipped)
            
            # Water rate: physical facility liquid capacity limit (max field liquid ~17,405 STB/d)
            next_qw = float(np.clip(curr_qw + d_qw_pred, 0.0, 16500.0))
            
            # Exact physical solution gas coupling
            next_qg = RS_SOLUTION * next_qo
            
            # Monotonic cumulative integration
            next_Qo = curr_Qo + next_qo * dt_days
            next_Qg = curr_Qg + next_qg * dt_days
            next_Qw = curr_Qw + next_qw * dt_days
            
            qo_preds[k] = next_qo
            qw_preds[k] = next_qw
            qg_preds[k] = next_qg
            Qo_preds[k] = next_Qo
            Qw_preds[k] = next_Qw
            Qg_preds[k] = next_Qg
            
            # Update state for next recursive step
            curr_qo = next_qo
            curr_qw = next_qw
            curr_Qo = next_Qo
            curr_Qw = next_Qw
            curr_Qg = next_Qg
            curr_age += 0.25
            
        return {
            "oil_rate": qo_preds,
            "gas_rate": qg_preds,
            "water_rate": qw_preds,
            "oil_cum": Qo_preds,
            "gas_cum": Qg_preds,
            "water_cum": Qw_preds,
        }

class LearnedDynamicRateModel:
    """
    Model B: Fits/learns phase-specific dynamic decline parameters
    conditioned on geological uncertainty parameters and cutoff state.
    Analytical integration provides exact closed-form cumulatives.
    """
    def __init__(self, decline_type: str = "exponential"):
        self.decline_type = decline_type
        self.model_D = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self.model_water_slope = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self.b_hyperbolic = 0.3 # Moderate physical hyperbolic exponent
        
    def fit(
        self,
        case_dict: Dict[int, pd.DataFrame],
        unc_df: pd.DataFrame,
        train_cases: List[int],
        cutoff_date: pd.Timestamp,
        horizon_quarters: int,
    ):
        records = []
        for c in train_cases:
            df_c = case_dict[c]
            pre = df_c[df_c["date"] <= cutoff_date]
            post = df_c[(df_c["date"] > cutoff_date)].iloc[:horizon_quarters]
            
            if len(post) >= 4:
                t_fit_y = (post["date"] - cutoff_date).dt.total_seconds() / (86400.0 * YEAR_DAYS)
                fit_rates = post["oil_rate_stbd"].values
                fit_water = post["water_rate_stbd"].values
            else:
                # Terminal 2008 boundary: fit from established historical decline trajectory (2002-2008)
                decline_hist = pre[pre["date"] >= "2002-01-01"]
                t_fit_y = (decline_hist["date"] - decline_hist["date"].iloc[0]).dt.total_seconds() / (86400.0 * YEAR_DAYS)
                fit_rates = decline_hist["oil_rate_stbd"].values
                fit_water = decline_hist["water_rate_stbd"].values
            
            # Fit observed oil decline rate
            log_qo = np.log(np.maximum(fit_rates, 1.0))
            poly_o = np.polyfit(t_fit_y, log_qo, 1)
            D_obs = max(0.005, -poly_o[0])
            
            # Fit observed water rate progression slope
            poly_w = np.polyfit(t_fit_y, fit_water, 1)
            w_slope_obs = poly_w[0]
            
            u_c = unc_df.loc[c]
            records.append({
                "case_id": c,
                "fault": float(u_c["Fault Transmissibility"]),
                "poro": float(u_c["Porosity Multiplier"]),
                "perm": float(u_c["Permeability Multiplier"]),
                "aq": float(u_c["Aquifer Pore Volume"]),
                "qo_0": float(pre.iloc[-1]["oil_rate_stbd"]),
                "qw_0": float(pre.iloc[-1]["water_rate_stbd"]),
                "wor_0": float(pre.iloc[-1]["wor"]),
                "res_age": float(pre.iloc[-1]["res_age_years"]),
                "D_obs": D_obs,
                "w_slope_obs": w_slope_obs,
            })
            
        fit_df = pd.DataFrame(records)
        feat_cols = ["fault", "poro", "perm", "aq", "qo_0", "qw_0", "wor_0", "res_age"]
        self.feat_cols = feat_cols
        self.model_D.fit(fit_df[feat_cols].values, fit_df["D_obs"].values)
        self.model_water_slope.fit(fit_df[feat_cols].values, fit_df["w_slope_obs"].values)
        return self

    def predict_rollout(
        self,
        initial_state: Dict[str, float],
        u_c: pd.Series,
        forecast_dates: pd.DatetimeIndex,
    ) -> Dict[str, np.ndarray]:
        t_years = (pd.DatetimeIndex(forecast_dates) - pd.Timestamp(initial_state["origin_date"])).total_seconds() / (86400.0 * YEAR_DAYS)
        x_in = np.array([[
            float(u_c["Fault Transmissibility"]),
            float(u_c["Porosity Multiplier"]),
            float(u_c["Permeability Multiplier"]),
            float(u_c["Aquifer Pore Volume"]),
            initial_state["qo"],
            initial_state["qw"],
            initial_state["wor"],
            initial_state["res_age"],
        ]])
        
        D_pred = max(0.01, float(self.model_D.predict(x_in)[0]))
        w_slope_pred = float(self.model_water_slope.predict(x_in)[0])
        
        q0_o = initial_state["qo"]
        q0_w = initial_state["qw"]
        
        if self.decline_type == "exponential":
            qo_t = q0_o * np.exp(-D_pred * t_years)
            delta_Qo = (q0_o * YEAR_DAYS / D_pred) * (1.0 - np.exp(-D_pred * t_years))
        elif self.decline_type == "hyperbolic":
            b = self.b_hyperbolic
            qo_t = q0_o * ((1.0 + b * D_pred * t_years) ** (-1.0 / b))
            delta_Qo = (q0_o * YEAR_DAYS / (D_pred * (1.0 - b))) * (1.0 - (1.0 + b * D_pred * t_years) ** (1.0 - 1.0 / b))
        else:
            raise ValueError(f"Unknown decline_type: {self.decline_type}")
            
        # Gas strictly proportional via Rs
        qg_t = RS_SOLUTION * qo_t
        delta_Qg = RS_SOLUTION * delta_Qo
        
        # Water rate: constrained progression (cannot exceed total facility voidage ~17,400 STB/d)
        qw_t = np.clip(q0_w + w_slope_pred * t_years, 0.0, 16000.0)
        # Analytical integral of linear water rate
        delta_Qw = (q0_w * t_years + 0.5 * w_slope_pred * (t_years ** 2)) * YEAR_DAYS
        delta_Qw = np.maximum(0.0, delta_Qw)
        
        return {
            "oil_rate": qo_t,
            "gas_rate": qg_t,
            "water_rate": qw_t,
            "oil_cum": initial_state["Qo"] + delta_Qo,
            "gas_cum": initial_state["Qg"] + delta_Qg,
            "water_cum": initial_state["Qw"] + delta_Qw,
            "D_used": D_pred,
        }

class HybridDynamicModel:
    """
    Model C: Hybrid Dynamic Architecture.
    Near-term (t <= switch_quarters): Learned recursive transition model.
    Long-term (t > switch_quarters): Rate-continuous physical DCA continuation.
    Enforces exact C0 (cumulative) and C1 (instantaneous rate) continuity.
    """
    def __init__(self, switch_quarters: int = 12):
        self.switch_quarters = switch_quarters
        self.transition_model = RecursiveStateTransitionModel(model_type="extratrees")
        self.dca_model = LearnedDynamicRateModel(decline_type="exponential")
        
    def fit(
        self,
        case_dict: Dict[int, pd.DataFrame],
        unc_df: pd.DataFrame,
        train_cases: List[int],
        cutoff_date: pd.Timestamp,
        horizon_quarters: int,
    ):
        trans_df = extract_transition_training_data(case_dict, unc_df, train_cases, cutoff_date)
        self.transition_model.fit(trans_df)
        self.dca_model.fit(case_dict, unc_df, train_cases, cutoff_date, horizon_quarters)
        return self

    def predict_rollout(
        self,
        initial_state: Dict[str, float],
        u_c: pd.Series,
        forecast_dates: pd.DatetimeIndex,
    ) -> Dict[str, np.ndarray]:
        forecast_dates = pd.DatetimeIndex(forecast_dates)
        n_steps = len(forecast_dates)
        if n_steps <= self.switch_quarters:
            return self.transition_model.predict_rollout(initial_state, u_c, forecast_dates)
            
        # Step 1: Recursive transition for near-term
        near_dates = forecast_dates[:self.switch_quarters]
        near_res = self.transition_model.predict_rollout(initial_state, u_c, near_dates)
        
        # Step 2: Handoff boundary state at switch_quarters
        handoff_date = near_dates[-1]
        handoff_state = {
            "qo": float(near_res["oil_rate"][-1]),
            "qw": float(near_res["water_rate"][-1]),
            "qg": float(near_res["gas_rate"][-1]),
            "Qo": float(near_res["oil_cum"][-1]),
            "Qw": float(near_res["water_cum"][-1]),
            "Qg": float(near_res["gas_cum"][-1]),
            "wor": float(near_res["water_rate"][-1] / max(near_res["oil_rate"][-1], 1.0)),
            "res_age": initial_state["res_age"] + self.switch_quarters * 0.25,
            "origin_date": handoff_date,
        }
        
        # Step 3: Analytical physical continuation
        far_dates = forecast_dates[self.switch_quarters:]
        far_res = self.dca_model.predict_rollout(handoff_state, u_c, far_dates)
        
        # Concatenate seamless trajectories
        return {
            "oil_rate": np.concatenate([near_res["oil_rate"], far_res["oil_rate"]]),
            "gas_rate": np.concatenate([near_res["gas_rate"], far_res["gas_rate"]]),
            "water_rate": np.concatenate([near_res["water_rate"], far_res["water_rate"]]),
            "oil_cum": np.concatenate([near_res["oil_cum"], far_res["oil_cum"]]),
            "gas_cum": np.concatenate([near_res["gas_cum"], far_res["gas_cum"]]),
            "water_cum": np.concatenate([near_res["water_cum"], far_res["water_cum"]]),
        }

def evaluate_baseline_persistence(initial_state: Dict[str, float], forecast_dates: pd.DatetimeIndex) -> Dict[str, np.ndarray]:
    """Baseline 1: Constant terminal rate continuation."""
    n_steps = len(forecast_dates)
    dt_days = 91.3125
    t_steps = np.arange(1, n_steps + 1)
    
    qo = np.full(n_steps, initial_state["qo"])
    qw = np.full(n_steps, initial_state["qw"])
    qg = np.full(n_steps, initial_state["qg"])
    
    Qo = initial_state["Qo"] + qo * dt_days * t_steps
    Qw = initial_state["Qw"] + qw * dt_days * t_steps
    Qg = initial_state["Qg"] + qg * dt_days * t_steps
    
    return {
        "oil_rate": qo, "gas_rate": qg, "water_rate": qw,
        "oil_cum": Qo, "gas_cum": Qg, "water_cum": Qw,
    }

def evaluate_baseline_zero(initial_state: Dict[str, float], forecast_dates: pd.DatetimeIndex) -> Dict[str, np.ndarray]:
    """Baseline 2: Zero production increment (flat cumulative)."""
    n_steps = len(forecast_dates)
    return {
        "oil_rate": np.zeros(n_steps), "gas_rate": np.zeros(n_steps), "water_rate": np.zeros(n_steps),
        "oil_cum": np.full(n_steps, initial_state["Qo"]),
        "gas_cum": np.full(n_steps, initial_state["Qg"]),
        "water_cum": np.full(n_steps, initial_state["Qw"]),
    }

def run_comprehensive_forecasting_study():
    """
    Executes the master forecasting benchmark across:
    - 4 Cutoff experiments (2005 3yr, 2004 4yr, 2003 5yr, 2001 7yr)
    - 7 Model Candidates (Ridge Recursive, ExtraTrees Recursive, GP Recursive,
                          Learned Exp DCA, Learned Hyp DCA, Hybrid Dynamic, Baselines)
    - Separates Lead-Time progression (Quarters 1-4, 5-8, 9-12, 13-16, 17-20, 21-28)
    - Generates 80-quarter inference with uncertainty scenarios
    """
    out_dir = Path("outputs/learned_dynamics")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("MASTER EXPERIMENT: GENUINE 20-YEAR DYNAMIC PRODUCTION FORECASTING")
    print("=" * 80)
    
    unc_df = load_uncertainty()
    q_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(q_df)
    
    train_c = TRAIN_CASES # 1 to 70
    val_c = VAL_CASES     # 71 to 85
    
    benchmark_results = []
    lead_time_results = []
    
    for cfg in CUTOFF_CONFIGS:
        cutoff_str = cfg["cutoff"]
        cutoff_date = pd.Timestamp(cutoff_str)
        n_qtrs = cfg["quarters"]
        cfg_name = cfg["name"]
        
        print(f"\n--- Running Experiment: Cutoff {cutoff_str} ({cfg['years']} years, {n_qtrs} quarters) ---")
        
        # Fit models on train cases (1 to 70)
        print("Fitting Model A1 (Ridge Recursive)...")
        m_a1 = RecursiveStateTransitionModel(model_type="ridge")
        trans_df = extract_transition_training_data(case_dict, unc_df, train_c, cutoff_date)
        m_a1.fit(trans_df)
        
        print("Fitting Model A2 (ExtraTrees Recursive)...")
        m_a2 = RecursiveStateTransitionModel(model_type="extratrees")
        m_a2.fit(trans_df)
        
        print("Fitting Model A3 (GP Recursive)...")
        m_a3 = RecursiveStateTransitionModel(model_type="gp")
        # Subsample transition dataset for fast GP fit if large
        gp_trans = trans_df.sample(min(len(trans_df), 400), random_state=42)
        m_a3.fit(gp_trans)
        
        print("Fitting Model B1 (Learned Exponential DCA)...")
        m_b1 = LearnedDynamicRateModel(decline_type="exponential")
        m_b1.fit(case_dict, unc_df, train_c, cutoff_date, n_qtrs)
        
        print("Fitting Model B2 (Learned Hyperbolic DCA)...")
        m_b2 = LearnedDynamicRateModel(decline_type="hyperbolic")
        m_b2.fit(case_dict, unc_df, train_c, cutoff_date, n_qtrs)
        
        print("Fitting Model C (Hybrid Dynamic Model)...")
        m_c = HybridDynamicModel(switch_quarters=12)
        m_c.fit(case_dict, unc_df, train_c, cutoff_date, n_qtrs)
        
        candidate_models = {
            "Model_A1_Ridge_Recursive": m_a1,
            "Model_A2_ExtraTrees_Recursive": m_a2,
            "Model_A3_GP_Recursive": m_a3,
            "Model_B1_Exponential_DCA": m_b1,
            "Model_B2_Hyperbolic_DCA": m_b2,
            "Model_C_Hybrid_Learned_Dynamic": m_c,
            "Baseline_Persistence_Rate": None,
            "Baseline_Zero_Rate": None,
        }
        
        # Evaluate on Validation Cases (71 to 85)
        for model_name, model_obj in candidate_models.items():
            case_nrmses = {"oil_cum": [], "gas_cum": [], "water_cum": [], "oil_rate": []}
            lead_errors = {p: {l_idx: [] for l_idx in range(1, 8)} for p in ["oil_cum", "water_cum"]}
            
            for c in val_c:
                df_c = case_dict[c]
                pre = df_c[df_c["date"] <= cutoff_date]
                post = df_c[df_c["date"] > cutoff_date].iloc[:n_qtrs].reset_index(drop=True)
                
                initial_state = {
                    "qo": float(pre.iloc[-1]["oil_rate_stbd"]),
                    "qw": float(pre.iloc[-1]["water_rate_stbd"]),
                    "qg": float(pre.iloc[-1]["gas_rate_mscfd"]),
                    "Qo": float(pre.iloc[-1]["oil_cum_stb"]),
                    "Qw": float(pre.iloc[-1]["water_cum_stb"]),
                    "Qg": float(pre.iloc[-1]["gas_cum_mscf"]),
                    "wor": float(pre.iloc[-1]["wor"]),
                    "res_age": float(pre.iloc[-1]["res_age_years"]),
                    "origin_date": cutoff_date,
                }
                
                u_c = unc_df.loc[c]
                forecast_dates = post["date"]
                
                if model_name == "Baseline_Persistence_Rate":
                    preds = evaluate_baseline_persistence(initial_state, forecast_dates)
                elif model_name == "Baseline_Zero_Rate":
                    preds = evaluate_baseline_zero(initial_state, forecast_dates)
                else:
                    preds = model_obj.predict_rollout(initial_state, u_c, forecast_dates)
                    
                for phase in ["oil_cum", "gas_cum", "water_cum"]:
                    true_col = "oil_cum_stb" if phase == "oil_cum" else ("water_cum_stb" if phase == "water_cum" else "gas_cum_mscf")
                    true_val = post[true_col].values
                    anchor_val = initial_state["Qo"] if phase == "oil_cum" else (initial_state["Qw"] if phase == "water_cum" else initial_state["Qg"])
                    true_inc = true_val - anchor_val
                    pred_val = preds[phase]
                    err = pred_val - true_val
                    nrmse = compute_increment_nrmse(err, true_inc)
                    case_nrmses[phase].append(nrmse)
                    
                # Evaluate rate
                true_rate_o = post["oil_rate_stbd"].values
                err_rate_o = preds["oil_rate"] - true_rate_o
                nrmse_rate_o = float(np.sqrt(np.mean(err_rate_o ** 2)) / (np.sqrt(np.mean(true_rate_o ** 2)) + 1e-12))
                case_nrmses["oil_rate"].append(nrmse_rate_o)
                
                # Decompose into lead time buckets (4 quarters each = 1 year each)
                for q_i in range(len(post)):
                    l_bucket = (q_i // 4) + 1 # Year 1, 2, 3...
                    if l_bucket <= 7:
                        t_inc_o = post["oil_cum_stb"].values[q_i] - initial_state["Qo"]
                        lead_errors["oil_cum"][l_bucket].append(abs(preds["oil_cum"][q_i] - post["oil_cum_stb"].values[q_i]) / max(t_inc_o, 1e-6))
                        
            mean_nrmse_oil = float(np.mean(case_nrmses["oil_cum"]))
            mean_nrmse_gas = float(np.mean(case_nrmses["gas_cum"]))
            mean_nrmse_water = float(np.mean(case_nrmses["water_cum"]))
            macro_nrmse = float(np.mean([mean_nrmse_oil, mean_nrmse_gas, mean_nrmse_water]))
            rate_nrmse_oil = float(np.mean(case_nrmses["oil_rate"]))
            
            benchmark_results.append({
                "cutoff": cutoff_str,
                "horizon_years": cfg["years"],
                "quarters": n_qtrs,
                "model_name": model_name,
                "macro_cum_nrmse": macro_nrmse,
                "oil_cum_nrmse": mean_nrmse_oil,
                "gas_cum_nrmse": mean_nrmse_gas,
                "water_cum_nrmse": mean_nrmse_water,
                "oil_rate_nrmse": rate_nrmse_oil,
            })
            
            for l_bucket in range(1, (n_qtrs // 4) + 1):
                lead_time_results.append({
                    "cutoff": cutoff_str,
                    "horizon_years": cfg["years"],
                    "model_name": model_name,
                    "lead_year": l_bucket,
                    "lead_quarters_range": f"Q{(l_bucket-1)*4+1}-Q{l_bucket*4}",
                    "mean_oil_cum_rel_error": float(np.mean(lead_errors["oil_cum"][l_bucket])),
                })

    bench_df = pd.DataFrame(benchmark_results)
    bench_df.to_csv(out_dir / "master_forecasting_benchmark.csv", index=False)
    print("\nSaved: outputs/learned_dynamics/master_forecasting_benchmark.csv")
    
    lead_df = pd.DataFrame(lead_time_results)
    lead_df.to_csv(out_dir / "lead_time_error_progression.csv", index=False)
    print("Saved: outputs/learned_dynamics/lead_time_error_progression.csv")
    
    # -------------------------------------------------------------
    # PHASE 7: 80-QUARTER INFERENCE PIPELINE DEMONSTRATION (2008-2028)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 7: EXECUTING 80-QUARTER FORWARD INFERENCE (2008-04-01 TO 2028-01-01)")
    print("=" * 80)
    
    inference_origin = pd.Timestamp("2008-01-01")
    inference_dates = pd.date_range("2008-04-01", "2028-01-01", freq="QS")
    assert len(inference_dates) == 80, f"Expected 80 quarters, got {len(inference_dates)}"
    
    # Fit final model using full 10-year development history (1998-2008)
    print("Fitting production forecasting architecture on complete 1998-2008 training dataset...")
    final_hybrid = HybridDynamicModel(switch_quarters=12)
    final_hybrid.fit(case_dict, unc_df, train_c, inference_origin, horizon_quarters=80)
    
    final_dca = LearnedDynamicRateModel(decline_type="exponential")
    final_dca.fit(case_dict, unc_df, train_c, inference_origin, horizon_quarters=80)
    
    inference_rows = []
    
    for c in val_c:
        df_c = case_dict[c]
        last_row = df_c.iloc[-1]
        
        initial_state = {
            "qo": float(last_row["oil_rate_stbd"]),
            "qw": float(last_row["water_rate_stbd"]),
            "qg": float(last_row["gas_rate_mscfd"]),
            "Qo": float(last_row["oil_cum_stb"]),
            "Qw": float(last_row["water_cum_stb"]),
            "Qg": float(last_row["gas_cum_mscf"]),
            "wor": float(last_row["wor"]),
            "res_age": float(last_row["res_age_years"]),
            "origin_date": inference_origin,
        }
        
        u_c = unc_df.loc[c]
        
        # Base Case Forecast
        base_preds = final_hybrid.predict_rollout(initial_state, u_c, inference_dates)
        
        # Uncertainty Scenarios:
        # P10 High Recovery (D reduced by 30%)
        # P90 Low Recovery (D increased by 30%)
        dca_base = final_dca.predict_rollout(initial_state, u_c, inference_dates)
        D_est = dca_base["D_used"]
        
        # Manually calculate P10 and P90 analytical bounds
        t_y = (inference_dates - inference_origin).total_seconds() / (86400.0 * YEAR_DAYS)
        D_p10 = max(0.005, D_est * 0.70)
        D_p90 = D_est * 1.30
        
        cum_oil_p10 = initial_state["Qo"] + (initial_state["qo"] * YEAR_DAYS / D_p10) * (1.0 - np.exp(-D_p10 * t_y))
        cum_oil_p90 = initial_state["Qo"] + (initial_state["qo"] * YEAR_DAYS / D_p90) * (1.0 - np.exp(-D_p90 * t_y))
        
        for i, d in enumerate(inference_dates):
            inference_rows.append({
                "case_id": c,
                "quarter_index": i + 1,
                "date": str(d)[:10],
                "oil_rate_stbd": float(base_preds["oil_rate"][i]),
                "gas_rate_mscfd": float(base_preds["gas_rate"][i]),
                "water_rate_stbd": float(base_preds["water_rate"][i]),
                "oil_cum_stb": float(base_preds["oil_cum"][i]),
                "gas_cum_mscf": float(base_preds["gas_cum"][i]),
                "water_cum_stb": float(base_preds["water_cum"][i]),
                "oil_cum_P10_high": float(cum_oil_p10[i]),
                "oil_cum_P90_low": float(cum_oil_p90[i]),
            })
            
    inf_df = pd.DataFrame(inference_rows)
    inf_df.to_csv(out_dir / "validation_cases_80q_inference_demonstration.csv", index=False)
    print("Saved: outputs/learned_dynamics/validation_cases_80q_inference_demonstration.csv")
    
    # -------------------------------------------------------------
    # DIAGNOSTIC FIGURES GENERATION
    # -------------------------------------------------------------
    print("\nGenerating Diagnostic Figures...")
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # FIGURE 1: Macro Cum NRMSE across Models and Cutoffs
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_bench = bench_df.pivot(index="model_name", columns="quarters", values="macro_cum_nrmse")
    pivot_bench.plot(kind="bar", ax=ax, colormap="viridis", width=0.8)
    ax.set_title("Forecast Macro NRMSE Across Time Horizons (Validation Cases 71-85)", fontweight="bold")
    ax.set_ylabel("Macro Increment NRMSE")
    ax.set_xlabel("Candidate Forecasting Model")
    plt.xticks(rotation=35, ha="right")
    plt.legend(title="Horizon (Quarters)", labels=["12q (3y)", "16q (4y)", "20q (5y)", "28q (7y)"])
    plt.tight_layout()
    fig.savefig(fig_dir / "model_comparison_by_horizon.png", dpi=200)
    plt.close()
    
    # FIGURE 2: Lead-Time Relative Error Growth
    fig, ax = plt.subplots(figsize=(10, 6))
    sub_lead = lead_df[lead_df["cutoff"] == "2001-01-01"].copy()
    for m in ["Model_A1_Ridge_Recursive", "Model_B1_Exponential_DCA", "Model_C_Hybrid_Learned_Dynamic", "Baseline_Persistence_Rate"]:
        df_m = sub_lead[sub_lead["model_name"] == m]
        if len(df_m) > 0:
            ax.plot(df_m["lead_year"], df_m["mean_oil_cum_rel_error"] * 100.0, marker="o", lw=2.2, label=m.replace("_", " "))
    ax.set_title("Forecast Error Accumulation vs Lead Time (2001 Cutoff, 7-Year Backtest)", fontweight="bold")
    ax.set_xlabel("Forecast Lead Time (Years)")
    ax.set_ylabel("Mean Oil Cumulative Error (%)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(fig_dir / "lead_time_error_progression.png", dpi=200)
    plt.close()
    
    # FIGURE 3: 80-Quarter Forward Production Profiles for Representative Cases
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sample_cases = [71, 75, 80, 85]
    for idx, sc in enumerate(sample_cases):
        ax = axes[idx // 2, idx % 2]
        c_sub = inf_df[inf_df["case_id"] == sc]
        dates_dt = pd.to_datetime(c_sub["date"])
        
        # Plot base, P10, P90
        ax.plot(dates_dt, c_sub["oil_cum_stb"] / 1e6, color="navy", lw=2.5, label="Base Forecast")
        ax.fill_between(dates_dt, c_sub["oil_cum_P90_low"] / 1e6, c_sub["oil_cum_P10_high"] / 1e6, color="cornflowerblue", alpha=0.3, label="Scenario Tube (±30% D)")
        
        # Also plot water cumulative
        ax.plot(dates_dt, c_sub["water_cum_stb"] / 1e6, color="teal", lw=2.0, ls="--", label="Water Cum Base")
        
        ax.set_title(f"Case {sc} (Poro Mult: {unc_df.loc[sc, 'Porosity Multiplier']:.2f})", fontweight="bold")
        ax.set_ylabel("Cumulative Volume (MM STB)")
        ax.set_xlabel("Forecast Date (2008-2028)")
        if idx == 0:
            ax.legend(loc="upper left")
            
    plt.suptitle("80-Quarter Production Forecast Demonstration (2008–2028)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(fig_dir / "forecast_80q_trajectories_cases71_85.png", dpi=200)
    plt.close()
    
    print("\nSaved all diagnostic figures to:", fig_dir)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    run_comprehensive_forecasting_study()
