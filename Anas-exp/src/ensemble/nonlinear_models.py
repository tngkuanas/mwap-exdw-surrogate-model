"""
Family 4: Regularized Nonlinear Regression Models.
Implements Polynomial Ridge regression and compact regularized MLP.
"""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.linalg import svd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

PARAMS = [
    "Fault Transmissibility",
    "Porosity Multiplier",
    "Permeability Multiplier",
    "Aquifer Pore Volume",
]

class PolynomialRidgeModel:
    """
    Degree-2 Polynomial Feature Expansion + L2 Ridge Regularization.
    Captures nonlinear parameter interactions (e.g. Porosity * Aquifer Volume).
    """
    def __init__(self, degree: int = 2, alpha: float = 10.0, n_components: int = 1):
        self.degree = degree
        self.alpha = alpha
        self.n_components = n_components
        self.model_id = f"PolynomialRidge_deg{degree}_a{int(alpha)}"
        
        self.pipelines = {}
        self.mean_increments = {}
        self.basis_vectors = {}
        self.train_anchor_means = {}
        
    def fit(
        self,
        train_cases: List[int],
        origin: pd.Timestamp,
        horizon_years: int,
        forecast_dates: pd.DatetimeIndex,
        curves: Dict[Tuple[int, str], pd.Series],
        unc_df: pd.DataFrame,
    ):
        origin = pd.Timestamp(origin)
        forecast_dates = pd.DatetimeIndex(forecast_dates)
        train_cases = sorted(train_cases)
        
        for phase in ["oil_cum", "gas_cum", "water_cum"]:
            # 1. Trajectory increments
            inc_matrix = []
            for c in train_cases:
                series = curves[(c, phase)]
                anchor = float(series.loc[:origin].iloc[-1])
                future_vals = np.interp(
                    forecast_dates.asi8.astype(float),
                    series.index.asi8.astype(float),
                    series.to_numpy(float),
                )
                inc_matrix.append(future_vals - anchor)
            inc_matrix = np.array(inc_matrix)
            
            # 2. SVD
            mean_inc = inc_matrix.mean(axis=0, keepdims=True)
            centered = inc_matrix - mean_inc
            U, S, Vt = svd(centered, full_matrices=False)
            K = min(self.n_components, len(S))
            basis = Vt[:K, :]
            coeffs = centered @ basis.T  # (N_train, K)
            
            # 3. Features
            X = unc_df.loc[train_cases, PARAMS].to_numpy(float)
            anchors = np.array([float(curves[(c, phase)].loc[:origin].iloc[-1]) for c in train_cases])[:, None]
            anchor_mean = float(anchors.mean())
            anchor_scaled = anchors / (anchor_mean + 1e-6)
            X_feat = np.hstack([X, anchor_scaled])
            
            pipe = Pipeline([
                ("poly", PolynomialFeatures(degree=self.degree, include_bias=False)),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha, random_state=42)),
            ])
            pipe.fit(X_feat, coeffs)
            
            key = (str(origin)[:10], int(horizon_years), phase)
            self.pipelines[key] = pipe
            self.mean_increments[key] = mean_inc
            self.basis_vectors[key] = basis
            self.train_anchor_means[key] = anchor_mean
            
    def predict_cases(
        self,
        cases: List[int],
        origin: pd.Timestamp,
        horizon_years: int,
        forecast_dates: pd.DatetimeIndex,
        curves: Dict[Tuple[int, str], pd.Series],
        unc_df: pd.DataFrame,
    ) -> pd.DataFrame:
        origin = pd.Timestamp(origin)
        forecast_dates = pd.DatetimeIndex(forecast_dates)
        rows = []
        
        for phase in ["oil_cum", "gas_cum", "water_cum"]:
            key = (str(origin)[:10], int(horizon_years), phase)
            pipe = self.pipelines[key]
            mean_inc = self.mean_increments[key]
            basis = self.basis_vectors[key]
            anchor_mean = self.train_anchor_means[key]
            
            X = unc_df.loc[cases, PARAMS].to_numpy(float)
            anchors = np.array([float(curves[(c, phase)].loc[:origin].iloc[-1]) for c in cases])[:, None]
            anchor_scaled = anchors / (anchor_mean + 1e-6)
            X_feat = np.hstack([X, anchor_scaled])
            
            pred_coeffs = pipe.predict(X_feat)
            if pred_coeffs.ndim == 1 and basis.shape[0] == 1:
                pred_coeffs = pred_coeffs[:, None]
                
            pred_increments = mean_inc + pred_coeffs @ basis
            
            for i, c in enumerate(cases):
                anchor_val = float(anchors[i, 0])
                c_cum = anchor_val + pred_increments[i, :]
                for t_idx, d in enumerate(forecast_dates):
                    rows.append({
                        "model_id": self.model_id,
                        "case_num": c,
                        "origin": origin,
                        "cutoff": origin,
                        "horizon_years": horizon_years,
                        "date": d,
                        "phase": phase,
                        "prediction": float(c_cum[t_idx]),
                    })
        return pd.DataFrame(rows)


class CompactMLPModel:
    """
    Compact regularized Multi-Layer Perceptron (4 -> 8 -> K).
    Uses heavy L2 regularization (weight decay) to prevent overfitting on N=70 cases.
    """
    def __init__(self, hidden_dim: int = 8, alpha: float = 1e-1, n_components: int = 1):
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.n_components = n_components
        self.model_id = f"CompactMLP_{hidden_dim}h_a{self.alpha}"
        
        self.pipelines = {}
        self.mean_increments = {}
        self.basis_vectors = {}
        self.train_anchor_means = {}
        
    def fit(
        self,
        train_cases: List[int],
        origin: pd.Timestamp,
        horizon_years: int,
        forecast_dates: pd.DatetimeIndex,
        curves: Dict[Tuple[int, str], pd.Series],
        unc_df: pd.DataFrame,
    ):
        origin = pd.Timestamp(origin)
        forecast_dates = pd.DatetimeIndex(forecast_dates)
        train_cases = sorted(train_cases)
        
        for phase in ["oil_cum", "gas_cum", "water_cum"]:
            inc_matrix = []
            for c in train_cases:
                series = curves[(c, phase)]
                anchor = float(series.loc[:origin].iloc[-1])
                future_vals = np.interp(
                    forecast_dates.asi8.astype(float),
                    series.index.asi8.astype(float),
                    series.to_numpy(float),
                )
                inc_matrix.append(future_vals - anchor)
            inc_matrix = np.array(inc_matrix)
            
            mean_inc = inc_matrix.mean(axis=0, keepdims=True)
            centered = inc_matrix - mean_inc
            U, S, Vt = svd(centered, full_matrices=False)
            K = min(self.n_components, len(S))
            basis = Vt[:K, :]
            coeffs = centered @ basis.T
            
            X = unc_df.loc[train_cases, PARAMS].to_numpy(float)
            anchors = np.array([float(curves[(c, phase)].loc[:origin].iloc[-1]) for c in train_cases])[:, None]
            anchor_mean = float(anchors.mean())
            anchor_scaled = anchors / (anchor_mean + 1e-6)
            X_feat = np.hstack([X, anchor_scaled])
            
            mlp = MLPRegressor(
                hidden_layer_sizes=(self.hidden_dim,),
                activation="tanh",
                alpha=self.alpha,
                max_iter=600,
                random_state=42,
            )
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", mlp),
            ])
            pipe.fit(X_feat, coeffs)
            
            key = (str(origin)[:10], int(horizon_years), phase)
            self.pipelines[key] = pipe
            self.mean_increments[key] = mean_inc
            self.basis_vectors[key] = basis
            self.train_anchor_means[key] = anchor_mean
            
    def predict_cases(
        self,
        cases: List[int],
        origin: pd.Timestamp,
        horizon_years: int,
        forecast_dates: pd.DatetimeIndex,
        curves: Dict[Tuple[int, str], pd.Series],
        unc_df: pd.DataFrame,
    ) -> pd.DataFrame:
        origin = pd.Timestamp(origin)
        forecast_dates = pd.DatetimeIndex(forecast_dates)
        rows = []
        
        for phase in ["oil_cum", "gas_cum", "water_cum"]:
            key = (str(origin)[:10], int(horizon_years), phase)
            pipe = self.pipelines[key]
            mean_inc = self.mean_increments[key]
            basis = self.basis_vectors[key]
            anchor_mean = self.train_anchor_means[key]
            
            X = unc_df.loc[cases, PARAMS].to_numpy(float)
            anchors = np.array([float(curves[(c, phase)].loc[:origin].iloc[-1]) for c in cases])[:, None]
            anchor_scaled = anchors / (anchor_mean + 1e-6)
            X_feat = np.hstack([X, anchor_scaled])
            
            pred_coeffs = pipe.predict(X_feat)
            if pred_coeffs.ndim == 1 and basis.shape[0] == 1:
                pred_coeffs = pred_coeffs[:, None]
                
            pred_increments = mean_inc + pred_coeffs @ basis
            
            for i, c in enumerate(cases):
                anchor_val = float(anchors[i, 0])
                c_cum = anchor_val + pred_increments[i, :]
                for t_idx, d in enumerate(forecast_dates):
                    rows.append({
                        "model_id": self.model_id,
                        "case_num": c,
                        "origin": origin,
                        "cutoff": origin,
                        "horizon_years": horizon_years,
                        "date": d,
                        "phase": phase,
                        "prediction": float(c_cum[t_idx]),
                    })
        return pd.DataFrame(rows)
