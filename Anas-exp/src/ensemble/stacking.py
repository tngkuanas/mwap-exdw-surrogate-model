"""
Stacking and Meta-Learner Module.
Assembles case-level out-of-fold meta-features, evaluates residual diversity,
and trains leakage-free meta-models (Ridge, Simple Average, Non-Negative Weights).
"""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

class SimpleAverageEnsemble:
    """Unweighted arithmetic mean of base model predictions."""
    def __init__(self, model_id: str = "Stack_SimpleAverage"):
        self.model_id = model_id
        
    def predict(self, base_preds_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        model_names = list(base_preds_dict.keys())
        first_df = base_preds_dict[model_names[0]].copy()
        
        pred_cols = []
        for name in model_names:
            pred_cols.append(base_preds_dict[name]["prediction"].to_numpy(float))
            
        avg_pred = np.mean(pred_cols, axis=0)
        first_df["model_id"] = self.model_id
        first_df["prediction"] = avg_pred
        return first_df

class NonNegativeWeightedEnsemble:
    """
    Constrained non-negative convex combination of base models:
    min || y - sum(w_i * y_i) ||_2^2  s.t. w_i >= 0, sum(w_i) = 1
    """
    def __init__(self, model_id: str = "Stack_NonNegativeWeighted"):
        self.model_id = model_id
        self.weights = {}  # Per phase weights
        
    def fit(self, meta_df: pd.DataFrame, base_model_cols: List[str], truth_col: str = "truth"):
        self.weights = {}
        for phase in ["oil_cum", "gas_cum", "water_cum"]:
            sub = meta_df[meta_df["phase"] == phase]
            y = sub[truth_col].to_numpy(float)
            X = sub[base_model_cols].to_numpy(float)
            
            K = len(base_model_cols)
            # Objective: 0.5 * || y - X w ||^2
            def loss(w):
                return 0.5 * np.mean((y - X @ w) ** 2)
            
            # Constraints: sum(w) = 1, w >= 0
            w0 = np.full(K, 1.0 / K)
            bounds = [(0.0, 1.0) for _ in range(K)]
            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
            
            res = minimize(loss, w0, bounds=bounds, constraints=constraints, method="SLSQP")
            self.weights[phase] = res.x if res.success else w0
            
    def predict(self, base_preds_dict: Dict[str, pd.DataFrame], base_model_cols: List[str]) -> pd.DataFrame:
        first_df = base_preds_dict[base_model_cols[0]].copy()
        pred_rows = []
        
        # Merge base predictions
        merged = first_df[["case_num", "cutoff", "horizon_years", "date", "phase"]].copy()
        for name in base_model_cols:
            merged[name] = base_preds_dict[name]["prediction"].to_numpy(float)
            
        final_preds = np.zeros(len(merged))
        for phase, w in self.weights.items():
            mask = merged["phase"] == phase
            if mask.sum() > 0:
                X_phase = merged.loc[mask, base_model_cols].to_numpy(float)
                final_preds[mask] = X_phase @ w
                
        first_df["model_id"] = self.model_id
        first_df["prediction"] = final_preds
        return first_df

class RidgeMetaEnsemble:
    """L2 Regularized Ridge Meta-Learner fitted on out-of-fold predictions."""
    def __init__(self, alpha: float = 10.0, model_id: str = "Stack_RidgeMeta"):
        self.alpha = alpha
        self.model_id = model_id
        self.models = {}
        
    def fit(self, meta_df: pd.DataFrame, base_model_cols: List[str], truth_col: str = "truth"):
        self.models = {}
        for phase in ["oil_cum", "gas_cum", "water_cum"]:
            sub = meta_df[meta_df["phase"] == phase]
            y = sub[truth_col].to_numpy(float)
            X = sub[base_model_cols].to_numpy(float)
            
            # Fit Ridge with positive=False (or True if supported)
            ridge = Ridge(alpha=self.alpha, fit_intercept=True, random_state=42)
            ridge.fit(X, y)
            self.models[phase] = ridge
            
    def predict(self, base_preds_dict: Dict[str, pd.DataFrame], base_model_cols: List[str]) -> pd.DataFrame:
        first_df = base_preds_dict[base_model_cols[0]].copy()
        merged = first_df[["case_num", "cutoff", "horizon_years", "date", "phase"]].copy()
        for name in base_model_cols:
            merged[name] = base_preds_dict[name]["prediction"].to_numpy(float)
            
        final_preds = np.zeros(len(merged))
        for phase, model in self.models.items():
            mask = merged["phase"] == phase
            if mask.sum() > 0:
                X_phase = merged.loc[mask, base_model_cols].to_numpy(float)
                final_preds[mask] = model.predict(X_phase)
                
        first_df["model_id"] = self.model_id
        first_df["prediction"] = final_preds
        return first_df

def compute_residual_diversity(meta_df: pd.DataFrame, base_model_cols: List[str], truth_col: str = "truth") -> pd.DataFrame:
    """
    Computes pairwise residual correlations between candidate base models.
    Lower correlation indicates higher complementary diversity.
    """
    residuals = {}
    for col in base_model_cols:
        residuals[col] = meta_df[col] - meta_df[truth_col]
        
    res_df = pd.DataFrame(residuals)
    corr_matrix = res_df.corr()
    return corr_matrix
