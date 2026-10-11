"""
Module: cross_fitted_maturity_gates.py
Package: causal_maturity
Addresses: Sections 3, 4, 6 — Cross-Fitted Gating Architectures & Real Metric Optimization
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Implementations:
1. FixedWeightGate (constant w in [0, 1])
2. CalendarAgeGate (logistic gate on elapsed calendar time)
3. PhysicalMaturityGate (logistic gate on physical decline depth and duration)
4. MultivariateStateGate (logistic gate on full physical reservoir state)
5. UncertaintyAwareGate (variance-attenuated gate using GP predictive uncertainty)
6. OilOnlyMaturityGate (physical maturity for oil, strictly frozen for water)
7. HorizonMaturityGate (time-decaying maturity gate)

All gates are optimized directly on the REAL prespecified competition evaluation metric:
Macro increment NRMSE of cumulative production.
"""
from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize
from ensemble.metrics import compute_increment_nrmse


def compute_case_real_macro_nrmse(
    qo_pred: np.ndarray,
    qw_pred: np.ndarray,
    df_post: pd.DataFrame,
    init_state: Dict[str, float],
    dt_days: float = 91.3125,
) -> float:
    """Computes exact macro increment NRMSE for a single case trajectory."""
    qo_pred = np.asarray(qo_pred, dtype=float)
    qw_pred = np.asarray(qw_pred, dtype=float)
    
    # Monotonic clamping safeguard for oil
    qo_clean = qo_pred.copy()
    for t in range(1, len(qo_clean)):
        if qo_clean[t] > qo_clean[t - 1]:
            qo_clean[t] = qo_clean[t - 1]
    qo_clean = np.maximum(qo_clean, 0.0)
    qw_clean = np.maximum(qw_pred, 0.0)
    
    cum_o_pred = init_state["Qo"] + np.cumsum(qo_clean * dt_days)
    cum_w_pred = init_state["Qw"] + np.cumsum(qw_clean * dt_days)
    
    true_o = df_post["oil_cum_stb"].values if "oil_cum_stb" in df_post.columns else df_post["oil_cum"].values
    true_w = df_post["water_cum_stb"].values if "water_cum_stb" in df_post.columns else df_post["water_cum"].values
    
    inc_o = true_o - init_state["Qo"]
    inc_w = true_w - init_state["Qw"]
    
    err_o = cum_o_pred - true_o
    err_w = cum_w_pred - true_w
    
    no = compute_increment_nrmse(err_o, inc_o)
    nw = compute_increment_nrmse(err_w, inc_w)
    return 0.5 * (no + nw)


class BaseGatingModel:
    """Abstract base class for all gating models."""
    def __init__(self, name: str):
        self.name = name
        self.params = np.array([])
        self.fitted = False

    def get_gate_weights(
        self,
        maturity: Dict[str, Any],
        n_steps: int,
        gp_variance: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (weight_oil_array, weight_water_array) for the forecast horizon."""
        raise NotImplementedError

    def fit_on_real_loss(
        self,
        train_cases: List[int],
        res_preds: Dict[int, Dict[str, np.ndarray]],
        frz_preds: Dict[int, Dict[str, np.ndarray]],
        post_dfs: Dict[int, pd.DataFrame],
        init_states: Dict[int, Dict[str, float]],
        maturities: Dict[int, Dict[str, Any]],
        gp_variances: Optional[Dict[int, np.ndarray]] = None,
    ) -> "BaseGatingModel":
        raise NotImplementedError


class FixedWeightGate(BaseGatingModel):
    """Constant scalar weight w in [0, 1] optimized over training folds."""
    def __init__(self):
        super().__init__("Fixed_Weight_Gate")
        self.w = 0.5

    def get_gate_weights(self, maturity, n_steps, gp_variance=None):
        w_arr = np.full(n_steps, self.w)
        return w_arr, w_arr

    def fit_on_real_loss(self, train_cases, res_preds, frz_preds, post_dfs, init_states, maturities, gp_variances=None):
        def obj(w_val):
            w = float(np.clip(w_val[0], 0.0, 1.0))
            losses = []
            for c in train_cases:
                qo = w * res_preds[c]["oil_rate"] + (1.0 - w) * frz_preds[c]["oil_rate"]
                qw = w * res_preds[c]["water_rate"] + (1.0 - w) * frz_preds[c]["water_rate"]
                loss_c = compute_case_real_macro_nrmse(qo, qw, post_dfs[c], init_states[c])
                losses.append(loss_c)
            return np.mean(losses)

        res = minimize(obj, [0.5], method="Nelder-Mead", bounds=[(0.0, 1.0)])
        self.w = float(np.clip(res.x[0], 0.0, 1.0))
        self.fitted = True
        return self


class CalendarAgeGate(BaseGatingModel):
    """Logistic gate based strictly on elapsed calendar field age (field_age_years)."""
    def __init__(self):
        super().__init__("Calendar_Age_Gate")
        self.params = np.array([0.0, 0.0])

    def get_gate_weights(self, maturity, n_steps, gp_variance=None):
        age = maturity.get("field_age_years", 6.0)
        # Normalize: 6 years = 1.0
        val = float(expit(self.params[0] + self.params[1] * (age / 6.0)))
        w_arr = np.full(n_steps, val)
        return w_arr, w_arr

    def fit_on_real_loss(self, train_cases, res_preds, frz_preds, post_dfs, init_states, maturities, gp_variances=None):
        def obj(p):
            losses = []
            for c in train_cases:
                age = maturities[c].get("field_age_years", 6.0)
                w = float(expit(p[0] + p[1] * (age / 6.0)))
                qo = w * res_preds[c]["oil_rate"] + (1.0 - w) * frz_preds[c]["oil_rate"]
                qw = w * res_preds[c]["water_rate"] + (1.0 - w) * frz_preds[c]["water_rate"]
                losses.append(compute_case_real_macro_nrmse(qo, qw, post_dfs[c], init_states[c]))
            return np.mean(losses) + 0.01 * (p[0]**2 + p[1]**2)

        res = minimize(obj, [0.0, -0.5], method="BFGS")
        self.params = res.x
        self.fitted = True
        return self


class PhysicalMaturityGate(BaseGatingModel):
    """Logistic gate based on physical decline depth and quarters in sustained decline."""
    def __init__(self):
        super().__init__("Physical_Maturity_Gate")
        self.params = np.array([0.0, 0.0, 0.0])

    def get_gate_weights(self, maturity, n_steps, gp_variance=None):
        depth = maturity.get("decline_depth", 0.0)
        q_decl = maturity.get("quarters_in_decline", 0) / 4.0 # years
        val = float(expit(self.params[0] + self.params[1] * depth + self.params[2] * q_decl))
        w_arr = np.full(n_steps, val)
        return w_arr, w_arr

    def fit_on_real_loss(self, train_cases, res_preds, frz_preds, post_dfs, init_states, maturities, gp_variances=None):
        def obj(p):
            losses = []
            for c in train_cases:
                depth = maturities[c].get("decline_depth", 0.0)
                q_decl = maturities[c].get("quarters_in_decline", 0) / 4.0
                w = float(expit(p[0] + p[1] * depth + p[2] * q_decl))
                qo = w * res_preds[c]["oil_rate"] + (1.0 - w) * frz_preds[c]["oil_rate"]
                qw = w * res_preds[c]["water_rate"] + (1.0 - w) * frz_preds[c]["water_rate"]
                losses.append(compute_case_real_macro_nrmse(qo, qw, post_dfs[c], init_states[c]))
            return np.mean(losses) + 0.01 * np.sum(p**2)

        res = minimize(obj, [0.0, -1.0, -0.5], method="BFGS")
        self.params = res.x
        self.fitted = True
        return self


class MultivariateStateGate(BaseGatingModel):
    """Logistic gate based on physical decline depth, water cut, WOR, and instantaneous decline rate."""
    def __init__(self):
        super().__init__("Multivariate_State_Gate")
        self.params = np.zeros(5)

    def get_gate_weights(self, maturity, n_steps, gp_variance=None):
        depth = maturity.get("decline_depth", 0.0)
        wc = maturity.get("water_cut", 0.0)
        d_inst = maturity.get("instantaneous_decline_rate", 0.0)
        cv = maturity.get("rate_variability_cv", 0.0)
        z = self.params[0] + self.params[1] * depth + self.params[2] * wc + self.params[3] * d_inst + self.params[4] * cv
        val = float(expit(z))
        w_arr = np.full(n_steps, val)
        return w_arr, w_arr

    def fit_on_real_loss(self, train_cases, res_preds, frz_preds, post_dfs, init_states, maturities, gp_variances=None):
        def obj(p):
            losses = []
            for c in train_cases:
                depth = maturities[c].get("decline_depth", 0.0)
                wc = maturities[c].get("water_cut", 0.0)
                d_inst = maturities[c].get("instantaneous_decline_rate", 0.0)
                cv = maturities[c].get("rate_variability_cv", 0.0)
                z = p[0] + p[1] * depth + p[2] * wc + p[3] * d_inst + p[4] * cv
                w = float(expit(z))
                qo = w * res_preds[c]["oil_rate"] + (1.0 - w) * frz_preds[c]["oil_rate"]
                qw = w * res_preds[c]["water_rate"] + (1.0 - w) * frz_preds[c]["water_rate"]
                losses.append(compute_case_real_macro_nrmse(qo, qw, post_dfs[c], init_states[c]))
            return np.mean(losses) + 0.01 * np.sum(p**2)

        res = minimize(obj, np.zeros(5), method="BFGS")
        self.params = res.x
        self.fitted = True
        return self


class OilOnlyMaturityGate(BaseGatingModel):
    """
    Physical maturity gate applied strictly to oil production rate.
    Water production rate is kept 100% frozen (weight_water = 0).
    """
    def __init__(self):
        super().__init__("Oil_Only_Maturity_Gate")
        self.params = np.array([0.0, 0.0, 0.0])

    def get_gate_weights(self, maturity, n_steps, gp_variance=None):
        depth = maturity.get("decline_depth", 0.0)
        q_decl = maturity.get("quarters_in_decline", 0) / 4.0
        val = float(expit(self.params[0] + self.params[1] * depth + self.params[2] * q_decl))
        w_oil = np.full(n_steps, val)
        w_wat = np.zeros(n_steps) # strictly frozen water
        return w_oil, w_wat

    def fit_on_real_loss(self, train_cases, res_preds, frz_preds, post_dfs, init_states, maturities, gp_variances=None):
        def obj(p):
            losses = []
            for c in train_cases:
                depth = maturities[c].get("decline_depth", 0.0)
                q_decl = maturities[c].get("quarters_in_decline", 0) / 4.0
                w = float(expit(p[0] + p[1] * depth + p[2] * q_decl))
                qo = w * res_preds[c]["oil_rate"] + (1.0 - w) * frz_preds[c]["oil_rate"]
                qw = frz_preds[c]["water_rate"] # strictly frozen
                losses.append(compute_case_real_macro_nrmse(qo, qw, post_dfs[c], init_states[c]))
            return np.mean(losses) + 0.01 * np.sum(p**2)

        res = minimize(obj, [0.0, -1.0, -0.5], method="BFGS")
        self.params = res.x
        self.fitted = True
        return self


class UncertaintyAwareGate(BaseGatingModel):
    """
    Attenuates residual model weight by the predictive variance of the GP.
    w(t) = w_base * exp( - sigma^2(t) / (2 * sigma_0^2) ).
    """
    def __init__(self):
        super().__init__("Uncertainty_Aware_Gate")
        self.params = np.array([0.5, 0.5]) # [w_base, sigma_0]

    def get_gate_weights(self, maturity, n_steps, gp_variance=None):
        w_base = float(np.clip(self.params[0], 0.0, 1.0))
        sigma_0 = max(float(self.params[1]), 1e-3)
        if gp_variance is None:
            w_arr = np.full(n_steps, w_base)
            return w_arr, w_arr
            
        var_arr = np.asarray(gp_variance, dtype=float)[:n_steps]
        attenuation = np.exp(-var_arr / (2.0 * sigma_0**2))
        w_arr = np.clip(w_base * attenuation, 0.0, 1.0)
        return w_arr, w_arr

    def fit_on_real_loss(self, train_cases, res_preds, frz_preds, post_dfs, init_states, maturities, gp_variances=None):
        def obj(p):
            w_base = float(np.clip(p[0], 0.0, 1.0))
            s0 = max(float(p[1]), 1e-3)
            losses = []
            for c in train_cases:
                var = gp_variances[c] if (gp_variances and c in gp_variances) else np.zeros(len(res_preds[c]["oil_rate"]))
                att = np.exp(-var / (2.0 * s0**2))
                w = np.clip(w_base * att, 0.0, 1.0)
                qo = w * res_preds[c]["oil_rate"] + (1.0 - w) * frz_preds[c]["oil_rate"]
                qw = w * res_preds[c]["water_rate"] + (1.0 - w) * frz_preds[c]["water_rate"]
                losses.append(compute_case_real_macro_nrmse(qo, qw, post_dfs[c], init_states[c]))
            return np.mean(losses) + 0.01 * (p[0]**2 + (p[1] - 0.5)**2)

        res = minimize(obj, [0.5, 0.5], method="Nelder-Mead")
        self.params = res.x
        self.fitted = True
        return self


class HorizonMaturityGate(BaseGatingModel):
    """
    Decays residual weight across forecast quarters:
    w(t_k) = expit( b0 + b1 * maturity - b2 * (k / 4.0) )
    """
    def __init__(self):
        super().__init__("Horizon_Maturity_Gate")
        self.params = np.array([0.0, 0.0, 0.5])

    def get_gate_weights(self, maturity, n_steps, gp_variance=None):
        depth = maturity.get("decline_depth", 0.0)
        k_steps = np.arange(n_steps) / 4.0 # years ahead
        z = self.params[0] + self.params[1] * depth - self.params[2] * k_steps
        w_arr = np.clip(expit(z), 0.0, 1.0)
        return w_arr, w_arr

    def fit_on_real_loss(self, train_cases, res_preds, frz_preds, post_dfs, init_states, maturities, gp_variances=None):
        def obj(p):
            losses = []
            for c in train_cases:
                depth = maturities[c].get("decline_depth", 0.0)
                n_s = len(res_preds[c]["oil_rate"])
                k_steps = np.arange(n_s) / 4.0
                z = p[0] + p[1] * depth - p[2] * k_steps
                w = np.clip(expit(z), 0.0, 1.0)
                qo = w * res_preds[c]["oil_rate"] + (1.0 - w) * frz_preds[c]["oil_rate"]
                qw = w * res_preds[c]["water_rate"] + (1.0 - w) * frz_preds[c]["water_rate"]
                losses.append(compute_case_real_macro_nrmse(qo, qw, post_dfs[c], init_states[c]))
            return np.mean(losses) + 0.01 * np.sum(p**2)

        res = minimize(obj, [0.0, -0.5, 0.5], method="BFGS")
        self.params = res.x
        self.fitted = True
        return self
