"""
Unit and regression tests for Fundamental Reservoir Physics Research.
ExxonMobil DataWorks Challenge 2026.
Branch: research/physics-first-reservoir-forecasting
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest


def test_dca_target_mismatch_outputs():
    target_path = Path("outputs/research/physics_first/tables/dca_target_comparison_hist_vs_future.csv")
    assert target_path.exists(), "DCA target comparison table missing!"
    df = pd.read_csv(target_path)
    assert len(df) == 85, f"Expected 85 cases, got {len(df)}"
    
    # Check that historical D has positive bias relative to future D
    assert df["D_hist"].mean() > df["D_future"].mean()
    
    # Check feature scaling output
    feat_path = Path("outputs/research/physics_first/tables/dca_feature_scaling_impact.csv")
    assert feat_path.exists(), "Feature scaling impact table missing!"
    df_feat = pd.read_csv(feat_path)
    assert "feature" in df_feat.columns
    assert len(df_feat) == 7


def test_identifiability_and_fisher_modes():
    fisher_path = Path("outputs/research/physics_first/tables/fisher_information_matrix.csv")
    assert fisher_path.exists(), "Fisher information matrix table missing!"
    F = pd.read_csv(fisher_path, index_col=0).values
    assert F.shape == (4, 4)
    # Symmetry
    assert np.allclose(F, F.T, atol=1e-5)
    
    modes_path = Path("outputs/research/physics_first/tables/fisher_modes_and_gradient_projections.csv")
    assert modes_path.exists(), "Fisher modes table missing!"
    df_modes = pd.read_csv(modes_path)
    assert len(df_modes) == 4
    # Mode 1 must dominate Fisher information (> 90%)
    assert df_modes.iloc[0]["info_share_pct"] > 90.0
    # Mode 1 must capture majority of forecast gradient energy (> 80%)
    assert df_modes.iloc[0]["energy_share_pct"] > 80.0


def test_crm_vs_baseline_validation():
    crm_path = Path("outputs/research/physics_first/tables/crm_vs_frozen_ensemble_validation.csv")
    assert crm_path.exists(), "CRM validation table missing!"
    df_crm = pd.read_csv(crm_path)
    assert len(df_crm) == 15, f"Expected 15 validation cases, got {len(df_crm)}"
    
    mean_ens_macro = df_crm["ens_macro_nrmse"].mean()
    mean_crm_macro = df_crm["crm_macro_nrmse"].mean()
    
    # Frozen ensemble must outperform CRM by a wide margin
    assert mean_ens_macro < 0.06
    assert mean_crm_macro > mean_ens_macro * 3.0


def test_structural_divergence_decomposition():
    div_path = Path("outputs/research/physics_first/tables/structural_vs_parameter_variance_decomposition.csv")
    assert div_path.exists(), "Structural divergence table missing!"
    df_div = pd.read_csv(div_path)
    assert len(df_div) == 80, f"Expected 80 quarters, got {len(df_div)}"
    
    q80 = df_div.iloc[-1]
    assert q80["quarter"] == 80
    # Model structural variance must dominate parameter variance at 80Q (> 1000x)
    assert q80["ratio_model_to_param"] > 1000.0
    assert q80["model_variance_share_pct"] > 99.0


def test_reservoir_volumetrics_material_balance():
    vol_path = Path("outputs/research/physics_first/tables/reservoir_volumetrics_and_material_balance.json")
    assert vol_path.exists(), "Volumetrics summary missing!"
    with open(vol_path) as f:
        vol = json.load(f)
        
    assert vol["material_balance_valid"] is True
    assert vol["recovery_factor_2028_pct"] < vol["theoretical_max_rf_pct"]
    assert 30.0 < vol["recovery_factor_2028_pct"] < 45.0
    assert 40.0 < vol["recovery_factor_mobile_pct"] < 65.0
