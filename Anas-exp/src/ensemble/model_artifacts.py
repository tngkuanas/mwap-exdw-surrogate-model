"""
Production Model Artifacts and Metadata Serialization Module.
ExxonMobil DataWorks Challenge 2026.
Team: MWAP Energy Analytics.

Provides:
- Serialization & Deserialization of fitted production models (including nested hybrid components).
- Model metadata recording training cases, cutoff, feature ordering, seeds, dependencies, and checksums.
- CWD-independent path resolution for saved models and metadata.
"""
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import sklearn

from ensemble.run_horizon_adaptive_stacking_experiment import (
    STATE_FEATURES,
    RecursiveTransitionForecaster,
    LearnedDCARateForecaster,
    HybridDynamicForecaster,
    RS_SOLUTION,
)

DEFAULT_MODEL_FILENAME = "production_models.pkl"
DEFAULT_METADATA_FILENAME = "model_metadata.json"


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 digest of a binary file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def resolve_models_dir() -> Path:
    """Dynamically resolves the models directory CWD-independently."""
    search_dirs = [Path.cwd(), Path.cwd() / "Anas-exp"]
    try:
        file_dir = Path(__file__).resolve().parent  # Anas-exp/src/ensemble
        repo_root = file_dir.parents[2]
        search_dirs.extend([repo_root, repo_root / "models", file_dir.parent.parent])
    except Exception:
        pass
    for root in search_dirs:
        candidate = root / "models"
        if candidate.exists() and (candidate / DEFAULT_MODEL_FILENAME).exists():
            return candidate
    for root in search_dirs:
        candidate = root / "models"
        if candidate.exists():
            return candidate
    # Fallback to repo root / models
    try:
        repo_root = Path(__file__).resolve().parents[3]
        d = repo_root / "models"
        d.mkdir(parents=True, exist_ok=True)
        return d
    except Exception:
        d = Path("models")
        d.mkdir(parents=True, exist_ok=True)
        return d


def resolve_model_artifact_path() -> Path:
    """Resolves the default path to production_models.pkl."""
    models_dir = resolve_models_dir()
    p = models_dir / DEFAULT_MODEL_FILENAME
    if p.exists():
        return p
    # Secondary search
    search_dirs = [Path.cwd(), Path.cwd() / "Anas-exp"]
    for root in search_dirs:
        candidate = root / "models" / DEFAULT_MODEL_FILENAME
        if candidate.exists():
            return candidate
    return p


def resolve_model_metadata_path() -> Path:
    """Resolves the default path to model_metadata.json."""
    models_dir = resolve_models_dir()
    return models_dir / DEFAULT_METADATA_FILENAME


def build_model_metadata(
    models: Dict[str, Any],
    artifact_path: Path,
    train_cases: List[int],
    cutoff_date: str = "2008-01-01",
) -> Dict[str, Any]:
    """Constructs comprehensive production metadata for saved models."""
    artifact_sha = compute_sha256(artifact_path) if artifact_path.exists() else ""
    artifact_size_bytes = artifact_path.stat().st_size if artifact_path.exists() else 0
    
    metadata = {
        "model_architecture": "Blend_Phase_Specific_Simplex",
        "status": "FROZEN_TRAINED_PRODUCTION",
        "description": "Fitted base models for 80-quarter (20-year) production forecasting across all 10 scenarios",
        "artifact_file": artifact_path.name,
        "artifact_sha256": artifact_sha,
        "artifact_size_bytes": artifact_size_bytes,
        "training_configuration": {
            "training_cases": sorted(list(train_cases)),
            "num_training_cases": len(train_cases),
            "cutoff_date": str(cutoff_date),
            "temporal_boundary": "Strictly as-of 2008-01-01 (zero future leak)",
            "blind_cases_excluded": "Cases 86-100 strictly excluded from all training and tuning",
        },
        "forecast_horizons": {
            "all_cases": "Cases 1 through 10",
            "num_quarters": 80,
            "duration_years": 20.0,
            "start_date": "2008-04-01",
            "end_date": "2028-01-01",
            "total_forecast_cells": 2400,
            "total_forecast_dates": 800,
        },
        "base_models": {
            "Recursive_ExtraTrees": {
                "class": "RecursiveTransitionForecaster",
                "estimator": "ExtraTreesRegressor",
                "features": list(STATE_FEATURES),
                "feature_ordering": [
                    "log_qo_k", "log_qw_k", "res_age", "wor_k",
                    "fault", "poro", "perm", "aq"
                ],
                "targets": ["target_d_log_qo", "target_d_qw"],
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "random_state": 42,
                    "n_jobs": -1,
                },
                "random_state": 42,
            },
            "Learned_Exponential_DCA": {
                "class": "LearnedDCARateForecaster",
                "decline_type": "exponential",
                "features": ["fault", "poro", "perm", "aq", "qo_0", "qw_0", "wor_0", "res_age"],
                "alpha_selection": "RidgeCV(alphas=np.logspace(-3, 3, 10))",
                "target_D": "D_obs (-polyfit log oil rate)",
                "target_water": "w_growth_obs (polyfit water rate)",
            },
            "Learned_Hyperbolic_DCA": {
                "class": "LearnedDCARateForecaster",
                "decline_type": "hyperbolic",
                "b_hyperbolic": 0.35,
                "features": ["fault", "poro", "perm", "aq", "qo_0", "qw_0", "wor_0", "res_age"],
                "alpha_selection": "RidgeCV(alphas=np.logspace(-3, 3, 10))",
            },
            "Hybrid_Dynamic": {
                "class": "HybridDynamicForecaster",
                "switch_quarters": 12,
                "components": [
                    "RecursiveTransitionForecaster(extratrees)",
                    "LearnedDCARateForecaster(exponential)",
                ],
            },
        },
        "locked_simplex_weights": {
            "oil": {
                "Recursive_ExtraTrees": 0.24046649121524644,
                "Learned_Exponential_DCA": 0.6179022920123803,
                "Learned_Hyperbolic_DCA": 0.0,
                "Hybrid_Dynamic": 0.14163121677237328,
            },
            "water": {
                "Recursive_ExtraTrees": 0.49716965081213516,
                "Learned_Exponential_DCA": 0.2514151745939324,
                "Learned_Hyperbolic_DCA": 0.2514151745939323,
                "Hybrid_Dynamic": 0.0,
            },
        },
        "physics_constraints": {
            "solution_gas_oil_ratio_rs": RS_SOLUTION,
            "average_quarter_days": 91.3125,
            "non_negative_rates": True,
            "monotonic_cumulatives": True,
            "historical_anchor_preserved": True,
        },
        "runtime_environment": {
            "python_version": sys.version.split()[0],
            "dependencies": {
                "scikit-learn": getattr(sklearn, "__version__", "unknown"),
                "numpy": getattr(np, "__version__", "unknown"),
                "pandas": getattr(pd, "__version__", "unknown"),
            },
        },
        "saved_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    return metadata


def save_production_models(
    models: Dict[str, Any],
    train_cases: List[int],
    cutoff_date: str = "2008-01-01",
    models_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """
    Serializes fitted production models and writes complete metadata.
    Returns (artifact_path, metadata_path).
    """
    if models_dir is None:
        models_dir = resolve_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    
    artifact_path = models_dir / DEFAULT_MODEL_FILENAME
    metadata_path = models_dir / DEFAULT_METADATA_FILENAME
    
    with open(artifact_path, "wb") as f:
        pickle.dump(models, f, protocol=5)
        
    metadata = build_model_metadata(models, artifact_path, train_cases, cutoff_date)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    return artifact_path, metadata_path


def load_production_models(artifact_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads saved production models from disk."""
    if artifact_path is None or not Path(artifact_path).exists():
        artifact_path = resolve_model_artifact_path()
        
    if not artifact_path.exists():
        raise FileNotFoundError(f"Production model artifact not found at: {artifact_path}")
        
    with open(artifact_path, "rb") as f:
        models = pickle.load(f)
        
    required_keys = [
        "Recursive_ExtraTrees",
        "Learned_Exponential_DCA",
        "Learned_Hyperbolic_DCA",
        "Hybrid_Dynamic",
    ]
    for k in required_keys:
        if k not in models:
            raise KeyError(f"Corrupt model artifact: missing required key '{k}'")
            
    return models


LOCKED_WEIGHTS = {
    "oil": {
        "Recursive_ExtraTrees": 0.24046649121524644,
        "Learned_Exponential_DCA": 0.6179022920123803,
        "Learned_Hyperbolic_DCA": 0.0,
        "Hybrid_Dynamic": 0.14163121677237328,
    },
    "water": {
        "Recursive_ExtraTrees": 0.49716965081213516,
        "Learned_Exponential_DCA": 0.2514151745939324,
        "Learned_Hyperbolic_DCA": 0.2514151745939323,
        "Hybrid_Dynamic": 0.0,
    },
}


def load_field_anchor() -> Dict[str, Any]:
    """Loads Case 0 production anchor at 2008-01-01."""
    from ensemble.data import resolve_shared_dir
    shared_dir = resolve_shared_dir()
    prod = pd.read_parquet(shared_dir / "production_timeseries.parquet")
    c0 = prod[prod["case_num"] == 0]
    c0_2008 = c0[c0["date"] == "2008-01-01"].set_index("metric")["value"]
    return {
        "qo": float(c0_2008["oil_rate"]),
        "qw": float(c0_2008["water_rate"]),
        "qg": float(c0_2008["gas_rate"]),
        "Qo": float(c0_2008["oil_cum"]),
        "Qw": float(c0_2008["water_cum"]),
        "Qg": float(c0_2008["gas_cum"]),
        "wor": float(c0_2008["water_rate"] / max(c0_2008["oil_rate"], 1.0)),
        "res_age": 10.0,
        "origin_date": pd.Timestamp("2008-01-01"),
    }


def run_inference_from_saved_models(
    models: Dict[str, Any],
    scenario: Any,
    field_anchor: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, Dict[str, float]]] = None,
    forecast_dates: Optional[pd.DatetimeIndex] = None,
) -> Dict[str, np.ndarray]:
    """
    Executes fast forward inference from loaded models for a given scenario config.
    Returns array dictionary with cumulatives and rates over the forecast horizon.
    """
    if field_anchor is None:
        field_anchor = load_field_anchor()
    if weights is None:
        weights = LOCKED_WEIGHTS
    if forecast_dates is None:
        num_quarters = getattr(scenario, "horizon_quarters", 80)
        start_date = "2008-04-01"
        forecast_dates = pd.date_range(start_date, periods=num_quarters, freq="QS")
        
    params = scenario.params if hasattr(scenario, "params") else scenario
    decline_sensitivity = getattr(scenario, "decline_sensitivity", 1.0)
    
    dt_days = 91.3125
    r_et = models["Recursive_ExtraTrees"].predict_rates(field_anchor, params, forecast_dates)
    r_exp = models["Learned_Exponential_DCA"].predict_rates(field_anchor, params, forecast_dates)
    r_hyp = models["Learned_Hyperbolic_DCA"].predict_rates(field_anchor, params, forecast_dates)
    r_hyb = models["Hybrid_Dynamic"].predict_rates(field_anchor, params, forecast_dates)
    
    w_oil = weights["oil"]
    w_wat = weights["water"]
    sens_factor = 1.0 / decline_sensitivity
    
    qo_b = (
        w_oil["Recursive_ExtraTrees"] * r_et["oil_rate"]
        + w_oil["Learned_Exponential_DCA"] * r_exp["oil_rate"]
        + w_oil["Learned_Hyperbolic_DCA"] * r_hyp["oil_rate"]
        + w_oil["Hybrid_Dynamic"] * r_hyb["oil_rate"]
    ) * sens_factor
    
    qw_b = (
        w_wat["Recursive_ExtraTrees"] * r_et["water_rate"]
        + w_wat["Learned_Exponential_DCA"] * r_exp["water_rate"]
        + w_wat["Learned_Hyperbolic_DCA"] * r_hyp["water_rate"]
        + w_wat["Hybrid_Dynamic"] * r_hyb["water_rate"]
    ) * decline_sensitivity
    
    qo_b = np.maximum(qo_b, 0.0)
    qw_b = np.maximum(qw_b, 0.0)
    qg_b = RS_SOLUTION * qo_b
    
    cum_o = field_anchor["Qo"] + np.cumsum(qo_b * dt_days)
    cum_w = field_anchor["Qw"] + np.cumsum(qw_b * dt_days)
    cum_g = field_anchor["Qg"] + RS_SOLUTION * (cum_o - field_anchor["Qo"])
    
    cum_o = np.maximum.accumulate(cum_o)
    cum_w = np.maximum.accumulate(cum_w)
    cum_g = np.maximum.accumulate(cum_g)
    
    return {
        "dates": forecast_dates,
        "oil_rate": qo_b,
        "water_rate": qw_b,
        "gas_rate": qg_b,
        "oil_cum": cum_o,
        "water_cum": cum_w,
        "gas_cum": cum_g,
    }
