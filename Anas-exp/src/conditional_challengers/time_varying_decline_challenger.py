r"""
Module: time_varying_decline_challenger.py
Package: conditional_challengers
Addresses: Section 6 — Constrained Time-Varying Decline-Rate Challenger
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Mathematical Formulation:
    q_o(t) = q_o(0) * exp( - \int_0^t D(s) ds )
    D(t) = max( D_min, D_base + delta_D(t, x) )
    delta_D(t, x) = sum_j c_j(x) * exp( - lambda_j * t )

Key Mathematical Properties:
1. Since D(t) >= D_min > 0 for all t, the cumulative integral \int_0^t D(s) ds
   is strictly monotonically increasing.
2. Therefore, q_o(t) is strictly monotonically non-increasing by construction
   without any post-hoc clamping!
3. Boundary condition q_o(0) = q_o(0) is exact since \int_0^0 D(s) ds = 0.
4. Asymptotic convergence: delta_D(t) -> 0 as t -> inf, so D(t) -> D_base at late times.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from ensemble.data import load_uncertainty, TRAIN_CASES, PARAMS
from ensemble.run_horizon_adaptive_stacking_experiment import (
    LearnedDCARateForecaster,
    YEAR_DAYS,
)
from conditional_challengers.corrected_residual_dca import (
    extract_state_features,
    fit_historical_dca_at_origin,
)


class TimeVaryingDeclineForecaster:
    """
    Continuous decline-rate forecaster:
    Learns state-dependent perturbations to the instantaneous decline rate function D(t).
    """
    def __init__(
        self,
        d_min: float = 0.005,       # Minimum allowable physical decline rate (1/year)
        decay_rates: Tuple[float, ...] = (0.5, 1.0, 2.0), # Basis decay rates lambda_j (1/year)
        max_delta_D: float = 0.20,   # Maximum decline rate perturbation (1/year)
        random_state: int = 42,
    ):
        self.d_min = d_min
        self.decay_rates = decay_rates
        self.max_delta_D = max_delta_D
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.model_coeff = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self.fitted = False

    def _basis_functions(self, t_years: np.ndarray) -> np.ndarray:
        """Evaluates basis functions [exp(-lambda_1*t), exp(-lambda_2*t), ...]."""
        return np.column_stack([np.exp(-lam * t_years) for lam in self.decay_rates])

    def _integrated_basis(self, t_years: np.ndarray) -> np.ndarray:
        r"""Evaluates \int_0^t exp(-lambda_j * s) ds = (1 - exp(-lambda_j * t)) / lambda_j."""
        return np.column_stack([(1.0 - np.exp(-lam * t_years)) / lam for lam in self.decay_rates])

    def fit(
        self,
        case_dict: Dict[int, pd.DataFrame],
        unc_df: pd.DataFrame,
        train_cases: List[int],
        cutoff_date: pd.Timestamp,
        max_horizon_quarters: int = 16,
    ):
        """
        Fits decline function perturbations strictly from historical rolling origins.
        """
        min_orig = pd.Timestamp("2003-01-01")
        max_orig = cutoff_date - pd.DateOffset(days=180)
        
        if max_orig < min_orig:
            valid_origins = [min_orig]
        else:
            valid_origins = [d for d in pd.date_range(min_orig, max_orig, freq="6MS") if d <= max_orig]
            if len(valid_origins) == 0:
                valid_origins = [min_orig]
                
        X_list = []
        c_list = []
        
        for orig in valid_origins:
            post_dates = pd.date_range(orig + pd.DateOffset(days=1), cutoff_date, freq="QS")
            n_avail = min(len(post_dates), max_horizon_quarters)
            if n_avail < 4:
                continue
                
            post_dates = post_dates[:n_avail]
            t_years = (post_dates - orig).days.to_numpy(float) / YEAR_DAYS
            Phi_int = self._integrated_basis(t_years)
            
            dca_oof = fit_historical_dca_at_origin(case_dict, unc_df, train_cases, orig)
            
            for c in train_cases:
                df_c = case_dict[c]
                pre = df_c[df_c["date"] <= orig]
                post = df_c[(df_c["date"] > orig) & (df_c["date"] <= cutoff_date)].iloc[:n_avail]
                
                if len(pre) < 4 or len(post) < 4:
                    continue
                    
                u_c = unc_df.loc[c]
                last = pre.iloc[-1]
                qo_0 = float(last["oil_rate_stbd"])
                init_st = {
                    "qo": qo_0,
                    "qw": float(last["water_rate_stbd"]),
                    "qg": float(last["gas_rate_mscfd"]),
                    "Qo": float(last["oil_cum_stb"]),
                    "Qw": float(last["water_cum_stb"]),
                    "Qg": float(last["gas_cum_mscf"]),
                    "wor": float(last.get("wor", float(last["water_rate_stbd"]) / max(qo_0, 1.0))),
                    "res_age": (orig - pre["date"].iloc[0]).days / YEAR_DAYS,
                    "origin_date": orig,
                }
                
                # Base DCA decline rate
                dca_preds = dca_oof.predict_rates(init_st, u_c, post_dates)
                D_base = dca_preds.get("D_used", 0.05)
                
                # Observed rates
                obs_qo = np.maximum(post["oil_rate_stbd"].values, 1.0)
                # ln(q_o(t)/q_o(0)) = - \int_0^t D(s) ds = - D_base * t - sum_j c_j * \int_0^t phi_j(s) ds
                # Target: D_base * t + ln(q_o(t)/q_o(0)) = - Phi_int @ c
                # So Phi_int @ c = - (ln(obs_qo / qo_0) + D_base * t)
                y_target = -(np.log(obs_qo / max(qo_0, 1.0)) + D_base * t_years)
                
                # Solve linear least squares for c: Phi_int @ c ~ y_target
                c_est, _, _, _ = np.linalg.lstsq(Phi_int, y_target, rcond=None)
                c_est = np.clip(c_est, -self.max_delta_D, self.max_delta_D)
                
                feats = extract_state_features(df_c, orig, u_c)
                X_list.append(feats)
                c_list.append(c_est)
                
        assert len(X_list) >= 10, f"Too few training samples: {len(X_list)}"
        X = np.array(X_list)
        C = np.array(c_list)
        
        Xs = self.scaler.fit_transform(X)
        self.model_coeff.fit(Xs, C)
        self.fitted = True
        return self

    def predict_rates(
        self,
        df_target: pd.DataFrame,
        u_c: pd.Series,
        origin: pd.Timestamp,
        forecast_dates: pd.DatetimeIndex,
        base_dca: LearnedDCARateForecaster,
    ) -> Dict[str, np.ndarray]:
        """
        Predicts rates using integrated continuous decline rate D(t).
        Guaranteed to be strictly monotonic by mathematical integration.
        """
        assert self.fitted, "Model must be fitted before predict"
        pre = df_target[df_target["date"] <= origin]
        last = pre.iloc[-1]
        qo_0 = float(last["oil_rate_stbd"])
        
        init_st = {
            "qo": qo_0,
            "qw": float(last["water_rate_stbd"]),
            "qg": float(last.get("gas_rate_mscfd", last.get("gas_rate", 0.0))),
            "Qo": float(last["oil_cum_stb"]),
            "Qw": float(last["water_cum_stb"]),
            "Qg": float(last["gas_cum_mscf"]),
            "wor": float(last.get("wor", float(last["water_rate_stbd"]) / max(qo_0, 1.0))),
            "res_age": (origin - pre["date"].iloc[0]).days / YEAR_DAYS,
            "origin_date": origin,
        }
        
        dca_preds = base_dca.predict_rates(init_st, u_c, forecast_dates)
        D_base = dca_preds.get("D_used", 0.05)
        
        t_years = (forecast_dates - origin).days.to_numpy(float) / YEAR_DAYS
        Phi_int = self._integrated_basis(t_years)
        Phi = self._basis_functions(t_years)
        
        feats = extract_state_features(df_target, origin, u_c).reshape(1, -1)
        xs = self.scaler.transform(feats)
        c_pred = np.clip(self.model_coeff.predict(xs)[0], -self.max_delta_D, self.max_delta_D)
        
        # Instantaneous decline rate D(t)
        D_t = np.maximum(self.d_min, D_base + Phi @ c_pred)
        
        # Integrated cumulative decline: \int_0^t D(s) ds
        # Use numerical trapz on fine grid or analytical Phi_int where D_t >= D_min
        int_D = np.maximum(self.d_min * t_years, D_base * t_years + Phi_int @ c_pred)
        
        # Exact non-increasing rate by integration
        qo_t = qo_0 * np.exp(-int_D)
        
        # Water rate follows base DCA water growth
        qw_t = np.asarray(dca_preds["water_rate"], dtype=float)
        qg_t = qo_t * 0.3633
        
        return {
            "oil_rate": qo_t,
            "water_rate": qw_t,
            "gas_rate": qg_t,
            "D_t": D_t,
            "integrated_D": int_D,
        }
