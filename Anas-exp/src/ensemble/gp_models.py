"""
Family 3: Gaussian Process Regression Models.
Maps reservoir uncertainty parameters and historical anchor to low-rank curve coefficients
with ARD Matérn kernels and analytical uncertainty calibration.
"""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.linalg import svd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

PARAMS = [
    "Fault Transmissibility",
    "Porosity Multiplier",
    "Permeability Multiplier",
    "Aquifer Pore Volume",
]

class GaussianProcessCurveModel:
    """
    Gaussian Process Surrogate with ARD Matérn 5/2 Kernel.
    Predicts low-rank basis coefficients with analytical uncertainty.
    """
    def __init__(self, n_components: int = 1, nu: float = 2.5, noise_level: float = 1e-3):
        self.n_components = n_components
        self.nu = nu
        self.noise_level = noise_level
        self.model_id = f"GaussianProcess_Matern_{n_components}comp"
        
        self.scalers = {}
        self.gp_models = {}
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
            # 1. Trajectory increment matrix
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
            
            # 3. Features: 4 uncertainty params + anchor
            X = unc_df.loc[train_cases, PARAMS].to_numpy(float)
            anchors = np.array([float(curves[(c, phase)].loc[:origin].iloc[-1]) for c in train_cases])[:, None]
            anchor_mean = float(anchors.mean())
            anchor_scaled = anchors / (anchor_mean + 1e-6)
            X_feat = np.hstack([X, anchor_scaled])
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_feat)
            
            # ARD Matern kernel with length scale per feature
            n_features = X_scaled.shape[1]
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=np.ones(n_features),
                length_scale_bounds=(1e-2, 1e2),
                nu=self.nu,
            ) + WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-5, 1e1))
            
            # Fit GP per component
            gp_list = []
            for k_idx in range(K):
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-6,
                    n_restarts_optimizer=2,
                    random_state=42 + k_idx,
                    normalize_y=True,
                )
                gp.fit(X_scaled, coeffs[:, k_idx])
                gp_list.append(gp)
                
            key = (str(origin)[:10], int(horizon_years), phase)
            self.scalers[key] = scaler
            self.gp_models[key] = gp_list
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
            gp_list = self.gp_models[key]
            mean_inc = self.mean_increments[key]
            basis = self.basis_vectors[key]
            anchor_mean = self.train_anchor_means[key]
            
            X = unc_df.loc[cases, PARAMS].to_numpy(float)
            anchors = np.array([float(curves[(c, phase)].loc[:origin].iloc[-1]) for c in cases])[:, None]
            anchor_scaled = anchors / (anchor_mean + 1e-6)
            X_feat = np.hstack([X, anchor_scaled])
            X_scaled = scaler.transform(X_feat)
            
            K = len(gp_list)
            pred_coeffs = np.zeros((len(cases), K))
            pred_stds = np.zeros((len(cases), K))
            for k_idx, gp in enumerate(gp_list):
                m, s = gp.predict(X_scaled, return_std=True)
                pred_coeffs[:, k_idx] = m
                pred_stds[:, k_idx] = s
                
            pred_increments = mean_inc + pred_coeffs @ basis
            
            # Also calculate approximate trajectory uncertainty via basis projection
            var_traj = (pred_stds ** 2) @ (basis ** 2)
            std_traj = np.sqrt(np.maximum(1e-12, var_traj))
            
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
                        "uncertainty_std": float(std_traj[i, t_idx]),
                    })
                    
        return pd.DataFrame(rows)
