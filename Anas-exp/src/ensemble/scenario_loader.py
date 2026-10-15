"""
Module: scenario_loader.py
Implements the versioned scenario-loading interface for production forecast generation.
Replaces legacy hardcoded parameter vectors and artificial sensitivity multipliers.
Source of Truth: Research Iteration 4 Acceptance Gate Artifact (selected_scenarios_v4.csv).
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from ensemble.data import PARAMS

DEFAULT_RESEARCH_SCENARIO_PATH = Path(
    "outputs/research/history_matching_iteration_4/tables/selected_scenarios_v4.csv"
)
RESEARCH_PROVENANCE_COMMIT = "3dc43f3"

# Authoritative geological parameter bounds
PARAMETER_BOUNDS = {
    "Fault Transmissibility": (0.01, 1.00),
    "Porosity Multiplier": (0.80, 1.50),
    "Permeability Multiplier": (0.10, 10.00),
    "Aquifer Pore Volume": (10.00, 300.00),
}

# Submission sheet requirements
HORIZON_SPEC = {
    "Case 1": 40,   # 10 years (through 2018-01-01)
    "Case 2": 80,   # 20 years (through 2028-01-01)
    "Case 3": 80,
    "Case 4": 80,
    "Case 5": 80,
    "Case 6": 80,
    "Case 7": 80,
    "Case 8": 80,
    "Case 9": 80,
    "Case 10": 80,
}


@dataclass(frozen=True)
class ProductionScenarioConfig:
    """Robust, validated contract for a single production forecast scenario."""
    case_id: str
    candidate_id: int
    params: pd.Series
    horizon_quarters: int
    selection_strategy: str
    target_role: str
    history_match_metrics: Dict[str, float]
    research_source_artifact: str
    provenance_commit: str
    decline_sensitivity: float = 1.0

    def __post_init__(self):
        # Enforce central un-skewed sensitivity
        if abs(self.decline_sensitivity - 1.0) > 1e-6:
            raise ValueError(
                f"Production central forecast must have decline_sensitivity = 1.0, got {self.decline_sensitivity}"
            )
        # Validate parameter names
        missing = set(PARAMS) - set(self.params.index)
        if missing:
            raise ValueError(f"Missing required geological parameters: {missing}")
        # Validate parameter bounds
        for p_name, (low, high) in PARAMETER_BOUNDS.items():
            val = float(self.params[p_name])
            if not (low <= val <= high):
                raise ValueError(
                    f"Parameter '{p_name}' value {val} out of authoritative bounds [{low}, {high}]"
                )
        # Validate horizon
        expected_h = HORIZON_SPEC.get(self.case_id)
        if expected_h is not None and self.horizon_quarters != expected_h:
            raise ValueError(
                f"{self.case_id} requires {expected_h} quarters, got {self.horizon_quarters}"
            )


def resolve_default_research_scenario_path() -> Path:
    """Dynamically resolves the Iteration 4 scenario artifact independently of CWD."""
    search_dirs = [Path.cwd(), Path.cwd() / "Anas-exp"]
    try:
        file_dir = Path(__file__).resolve().parents[2]  # Anas-exp root
        search_dirs.extend([file_dir, file_dir.parent / "Anas-exp"])
    except Exception:
        pass
    for root in search_dirs:
        candidate = root / "outputs" / "research" / "history_matching_iteration_4" / "tables" / "selected_scenarios_v4.csv"
        if candidate.exists():
            return candidate
    return DEFAULT_RESEARCH_SCENARIO_PATH


def load_production_scenarios(
    source_path: Optional[Path] = None,
    strategy: str = "Strategy A: Ten Best Matches",
) -> List[ProductionScenarioConfig]:
    """
    Loads and validates the 10 production forecast scenarios directly from the latest
    verified research scenario artifact.
    """
    if source_path is None:
        source_path = resolve_default_research_scenario_path()
        
    if not source_path.exists():
        raise FileNotFoundError(f"Research scenario artifact not found at: {source_path}")
        
    df = pd.read_csv(source_path)
    if "strategy" not in df.columns:
        raise ValueError(f"Artifact {source_path} missing 'strategy' column.")
        
    sub_df = df[df["strategy"] == strategy].copy()
    if len(sub_df) != 10:
        raise ValueError(
            f"Strategy '{strategy}' must contain exactly 10 scenarios, got {len(sub_df)}"
        )
        
    # Sort by case_id order (Case 1 .. Case 10)
    case_order = [f"Case {i}" for i in range(1, 11)]
    sub_df["case_order_idx"] = sub_df["case_id"].apply(lambda c: case_order.index(c) if c in case_order else 999)
    sub_df = sub_df.sort_values("case_order_idx").reset_index(drop=True)
    
    configs = []
    for _, row in sub_df.iterrows():
        case_id = str(row["case_id"])
        cand_id = int(row["cand_id"])
        
        # Canonical geological parameters
        params = pd.Series({
            "Fault Transmissibility": float(row["Ft"]),
            "Porosity Multiplier": float(row["Poro"]),
            "Permeability Multiplier": float(row["Perm"]),
            "Aquifer Pore Volume": float(row["AqPV"]),
        })
        
        horizon = HORIZON_SPEC[case_id]
        
        metrics = {
            "oil_cum_nrmse": float(row.get("oil_cum_nrmse", np.nan)),
            "water_cum_nrmse": float(row.get("water_cum_nrmse", np.nan)),
            "oil_rate_nrmse": float(row.get("oil_rate_nrmse", np.nan)),
            "water_rate_nrmse": float(row.get("water_rate_nrmse", np.nan)),
            "log_lik_l1": float(row.get("log_lik_l1", np.nan)),
            "log_lik_l2": float(row.get("log_lik_l2", np.nan)),
            "terminal_oil_err": float(row.get("terminal_oil_err", np.nan)),
            "terminal_wat_err": float(row.get("terminal_wat_err", np.nan)),
        }
        
        sc_cfg = ProductionScenarioConfig(
            case_id=case_id,
            candidate_id=cand_id,
            params=params,
            horizon_quarters=horizon,
            selection_strategy=strategy,
            target_role=str(row.get("target_role", "Best Match")),
            history_match_metrics=metrics,
            research_source_artifact=str(source_path),
            provenance_commit=RESEARCH_PROVENANCE_COMMIT,
            decline_sensitivity=1.0,  # Un-skewed central physics
        )
        configs.append(sc_cfg)
        
    return configs


def load_research_sensitivity_scenarios(
    source_path: Optional[Path] = None,
    strategy: str = "Strategy A: Ten Best Matches",
    sensitivity_map: Optional[Dict[str, float]] = None,
) -> List[ProductionScenarioConfig]:
    """
    Research-only interface for evaluating sensitivity envelopes.
    Explicitly labeled as research-only sensitivity analysis; never used in central production.
    """
    configs = load_production_scenarios(source_path, strategy)
    if sensitivity_map is None:
        return configs
        
    sens_configs = []
    for sc in configs:
        sens = sensitivity_map.get(sc.case_id, 1.0)
        # Bypasses __post_init__ strict check via object.__setattr__ for research study only
        sc_dict = {
            "case_id": sc.case_id,
            "candidate_id": sc.candidate_id,
            "params": sc.params,
            "horizon_quarters": sc.horizon_quarters,
            "selection_strategy": f"{sc.selection_strategy} (Sensitivity Tube)",
            "target_role": sc.target_role,
            "history_match_metrics": sc.history_match_metrics,
            "research_source_artifact": sc.research_source_artifact,
            "provenance_commit": sc.provenance_commit,
            "decline_sensitivity": sens,
        }
        sens_configs.append(sc_dict)
    return sens_configs
