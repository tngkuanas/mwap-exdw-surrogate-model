"""
Data loader and case-level fold splitting module for the ensemble framework.
Enforces strict case-level cross-validation without temporal leakage.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

PARAMS = [
    "Fault Transmissibility",
    "Porosity Multiplier",
    "Permeability Multiplier",
    "Aquifer Pore Volume",
]

PHASES = ["oil_cum", "gas_cum", "water_cum"]
RATE_METRICS = ["oil_rate", "gas_rate", "water_rate"]
ALL_METRICS = PHASES + RATE_METRICS
YEAR_DAYS = 365.25

EXPERIMENTS = [
    ("2003-01-01", 3),
    ("2003-01-01", 5),
    ("2004-01-01", 3),
    ("2005-01-01", 3),
]

TRAIN_CASES = list(range(1, 71))
VAL_CASES = list(range(71, 86))
DEV_CASES = list(range(1, 86))
TEST_CASES = list(range(86, 101))

def resolve_shared_dir() -> Path:
    """Finds the shared data directory dynamically."""
    curr = Path.cwd()
    for parent in [curr] + list(curr.parents):
        p = parent / "data" / "shared"
        if p.exists() and (p / "uncertainty_params.parquet").exists():
            return p
    raise FileNotFoundError("Could not resolve data/shared directory.")

def load_uncertainty(shared_dir: Path = None) -> pd.DataFrame:
    """Loads and standardizes the uncertainty parameters DataFrame."""
    if shared_dir is None:
        shared_dir = resolve_shared_dir()
    unc = pd.read_parquet(shared_dir / "uncertainty_params.parquet")
    unc["case_num"] = pd.to_numeric(unc["case_num"], errors="raise").astype(int)
    unc = unc.sort_values("case_num").set_index("case_num", drop=False)
    return unc

def get_case_folds(unc_df: pd.DataFrame = None, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Constructs a 5-fold case-level split for Cases 1-70.
    Returns DataFrame with columns ['case_num', 'fold'].
    """
    if unc_df is None:
        unc_df = load_uncertainty()
        
    train_cases = np.array(TRAIN_CASES)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    fold_assignments = np.full(len(train_cases), -1, dtype=int)
    for fold_idx, (_, test_indices) in enumerate(kf.split(train_cases)):
        fold_assignments[test_indices] = fold_idx
        
    folds_df = pd.DataFrame({
        "case_num": train_cases,
        "fold": fold_assignments,
    })
    return folds_df

def load_curves(shared_dir: Path = None) -> Dict[Tuple[int, str], pd.Series]:
    """
    Loads all quarterly and cumulative production curves for development cases (0 to 85).
    Returns dict mapping (case_num, metric) -> pd.Series indexed by DatetimeIndex.
    """
    if shared_dir is None:
        shared_dir = resolve_shared_dir()
        
    prod = pd.read_parquet(
        shared_dir / "production_timeseries.parquet",
        columns=["case_num", "date", "metric", "value"],
    )
    prod["case_num"] = pd.to_numeric(prod["case_num"], errors="coerce")
    prod = prod[(prod["case_num"] >= 0) & (prod["case_num"] <= 85)].copy()
    prod["date"] = pd.to_datetime(prod["date"], errors="coerce")
    prod["value"] = pd.to_numeric(prod["value"], errors="coerce")
    prod = prod[prod["metric"].isin(ALL_METRICS)].dropna(subset=["date", "value"])
    
    curves = {}
    for (case, metric), grp in prod.groupby(["case_num", "metric"], observed=True):
        # Ensure unique timestamps per series
        series = grp.groupby("date")["value"].first().sort_index().astype(float)
        curves[(int(case), str(metric))] = series
        
    return curves

def load_forecast_truth() -> pd.DataFrame:
    """Loads the official forecast truth dataset for the 4 backtesting experiments."""
    curr = Path.cwd()
    for parent in [curr] + list(curr.parents):
        for candidate in [parent / "Anas" / "data" / "forecast_truth.parquet", parent / "data" / "forecast_truth.parquet"]:
            if candidate.exists():
                df = pd.read_parquet(candidate)
                df["cutoff"] = pd.to_datetime(df["cutoff"])
                df["date"] = pd.to_datetime(df["date"])
                df["phase"] = df["metric"]
                return df
    raise FileNotFoundError("Could not find forecast_truth.parquet")
