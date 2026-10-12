"""
Module: parameter_bounds.py
Package: geological_information_value
Project: ExxonMobil DataWorks Challenge 2026
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Provides authoritative geological parameter bounds, prior sampling distributions,
scaling transformations, and perturbation utilities for the 4 subsurface parameters:
1. Fault Transmissibility (Ft)
2. Porosity Multiplier (Poro)
3. Permeability Multiplier (Perm)
4. Aquifer Pore Volume (AqPV)
"""
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import pandas as pd

# Authoritative Parameter Names and Admissible Ranges
PARAM_SPECS: Dict[str, Dict[str, Any]] = {
    "Fault Transmissibility": {
        "short_name": "Ft",
        "min": 0.01,
        "max": 1.00,
        "scale": "log",
        "default_prior_mean": 0.10,
    },
    "Porosity Multiplier": {
        "short_name": "Poro",
        "min": 0.80,
        "max": 1.50,
        "scale": "linear",
        "default_prior_mean": 1.15,
    },
    "Permeability Multiplier": {
        "short_name": "Perm",
        "min": 0.10,
        "max": 10.00,
        "scale": "log",
        "default_prior_mean": 1.00,
    },
    "Aquifer Pore Volume": {
        "short_name": "AqPV",
        "min": 10.00,
        "max": 300.00,
        "scale": "linear",
        "default_prior_mean": 155.00,
    },
}

PARAM_NAMES: List[str] = list(PARAM_SPECS.keys())
SHORT_NAMES: List[str] = [PARAM_SPECS[k]["short_name"] for k in PARAM_NAMES]

LOWER_BOUNDS = np.array([PARAM_SPECS[k]["min"] for k in PARAM_NAMES], dtype=float)
UPPER_BOUNDS = np.array([PARAM_SPECS[k]["max"] for k in PARAM_NAMES], dtype=float)
RANGES = UPPER_BOUNDS - LOWER_BOUNDS
PRIOR_MEANS = np.array([PARAM_SPECS[k]["default_prior_mean"] for k in PARAM_NAMES], dtype=float)


def normalize_parameters(u_vec: np.ndarray) -> np.ndarray:
    """Normalizes geological parameter vector u to unit hypercube [0, 1]^4."""
    u = np.asarray(u_vec, dtype=float)
    return np.clip((u - LOWER_BOUNDS) / RANGES, 0.0, 1.0)


def denormalize_parameters(u_norm: np.ndarray) -> np.ndarray:
    """Maps unit hypercube [0, 1]^4 back to physical parameter ranges."""
    u = np.asarray(u_norm, dtype=float)
    return LOWER_BOUNDS + np.clip(u, 0.0, 1.0) * RANGES


def sample_prior_uniform(n_samples: int, seed: Optional[int] = 42) -> np.ndarray:
    """Samples n_samples uniformly from the admissible parameter box."""
    rng = np.random.RandomState(seed)
    u_unit = rng.uniform(0.0, 1.0, size=(n_samples, 4))
    return denormalize_parameters(u_unit)


def sample_prior_log_uniform(n_samples: int, seed: Optional[int] = 42) -> np.ndarray:
    """
    Samples with log-uniform priors on Ft and Perm, and uniform on Poro and AqPV.
    Reflects standard petroleum engineering prior beliefs.
    """
    rng = np.random.RandomState(seed)
    samples = np.zeros((n_samples, 4))
    
    # 0: Ft (log-uniform over [0.01, 1.00])
    log_ft = rng.uniform(np.log10(0.01), np.log10(1.00), size=n_samples)
    samples[:, 0] = 10.0 ** log_ft
    
    # 1: Poro (uniform over [0.80, 1.50])
    samples[:, 1] = rng.uniform(0.80, 1.50, size=n_samples)
    
    # 2: Perm (log-uniform over [0.10, 10.00])
    log_perm = rng.uniform(np.log10(0.10), np.log10(10.00), size=n_samples)
    samples[:, 2] = 10.0 ** log_perm
    
    # 3: AqPV (uniform over [10.00, 300.00])
    samples[:, 3] = rng.uniform(10.00, 300.00, size=n_samples)
    
    return samples


def create_parameter_series(u_vec: np.ndarray) -> pd.Series:
    """Converts a 4-element vector into a pandas Series keyed by parameter names."""
    return pd.Series(u_vec, index=PARAM_NAMES)


def get_adversarial_mismatched_parameters(u_true: np.ndarray) -> np.ndarray:
    """
    Constructs a deliberately mismatched geological parameter vector:
    inverts each parameter across its midpoint (u_max + u_min - u).
    """
    u = np.asarray(u_true, dtype=float)
    return LOWER_BOUNDS + UPPER_BOUNDS - u
