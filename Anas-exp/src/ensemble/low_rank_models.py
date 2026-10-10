"""
Family 2: Low-Rank Functional Regression Models.
Decomposes trajectory increments into SVD/PCA basis fitted inside training folds,
predicting basis coefficients via Ridge regression.
"""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.linalg import svd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler

PARAMS = [
    "Fault Transmissibility",
    "Porosity Multiplier",
    "Permeability Multiplier",
    "Aquifer Pore Volume",
]

class LowRankFunctionalModel:
    """
    SVD/PCA Basis Decomposition with Regularized Coefficient Regression.
    Fitted strictly on training cases in each fold.
    """
    def __init__(self, n_components: int = 1, alpha: float = 1.0, use_elastic: bool = False):
        self.n_components = n_components
        self.alpha = alpha
        self.use_elastic = use_elastic
        self.model_id = f"LowRank_SVD_{n_components}comp_Ridge"
        
        # Fitted state per (origin, horizon, phase)
        self.scalers = {}
        self.regressors = {}
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
            # 1. Build matrix of true forward increments on train cases
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
                
            inc_matrix = np.array(inc_matrix)  # Shape: (N_train, T_forecast)
            
            # 2. SVD on centered increments
            mean_inc = inc_matrix.mean(axis=0, keepdims=True)
            centered = inc_matrix - mean_inc
            U, S, Vt = svd(centered, full_matrices=False)
            
            K = min(self.n_components, len(S))
            basis = Vt[:K, :]  # Shape: (K, T_forecast)
            coeffs = centered @ basis.T  # Shape: (N_train, K)
            
            # 3. Features: 4 uncertainty params + historical anchor
            X = unc_df.loc[train_cases, PARAMS].to_numpy(float)
            anchors = np.array([float(curves[(c, phase)].loc[:origin].iloc[-1]) for c in train_cases])[:, None]
            # Normalize anchor to prevent scale distortion
            anchor_mean = float(anchors.mean())
            anchor_scaled = anchors / (anchor_mean + 1e-6)
            X_feat = np.hstack([X, anchor_scaled])
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_feat)
            
            if self.use_elastic:
                reg = ElasticNet(alpha=self.alpha, l1_ratio=0.5, random_state=42)
            else:
                reg = Ridge(alpha=self.alpha, random_state=42)
                
            reg.fit(X_scaled, coeffs)
            
            key = (str(origin)[:10], int(horizon_years), phase)
            self.scalers[key] = scaler
            self.regressors[key] = reg
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
            scaler = self.scalers[key]
            reg = self.regressors[key]
            mean_inc = self.mean_increments[key]
            basis = self.basis_vectors[key]
            anchor_mean = self.train_anchor_means[key]
            
            X = unc_df.loc[cases, PARAMS].to_numpy(float)
            anchors = np.array([float(curves[(c, phase)].loc[:origin].iloc[-1]) for c in cases])[:, None]
            anchor_scaled = anchors / (anchor_mean + 1e-6)
            X_feat = np.hstack([X, anchor_scaled])
            X_scaled = scaler.transform(X_feat)
            
            pred_coeffs = reg.predict(X_scaled)  # (N_cases, K)
            if pred_coeffs.ndim == 1 and basis.shape[0] == 1:
                pred_coeffs = pred_coeffs[:, None]
                
            pred_increments = mean_inc + pred_coeffs @ basis  # (N_cases, T_forecast)
            
            # Cumulative prediction = anchor + predicted forward increment
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
