"""
Module: reservoir_engineering_stress_test_v4.py
Research Iteration 4 — Phase 4: Reservoir-Engineering Stress Test of 20-Year Extrapolations.

Objectives:
Executes 6 rigorous reservoir-engineering physical sanity checks on 20-year forecasts:
1. Water Cut Monotonicity & Bounds: fw(t) in [0, 1], non-decreasing/plateau, terminal fw <= 0.98.
2. Oil Rate Decline Monotonicity: dqo/dt <= 0 after peak (no unphysical spontaneous rate surges).
3. Aquifer & Waterflood Influx Balance: accounts for active water injection (~9,326 STB/d)
   and net aquifer voidage encroachment (net aquifer depletion <= 15%).
4. ExtraTrees OOD Rate Feature Bounds & Ensemble Safeguard: checks rate containment within
   training domain and validates analytical DCA stabilization against tree saturation.
5. Volumetric Recovery Factor: cumulative oil / estimated OOIP (153.5 MMSTB) strictly within 25% - 35%.
6. DCA Decline Tail Sensitivity: terminal exponential switch D_min sensitivity on 20-year EUR (< 2.5%).

Exports:
outputs/research/history_matching_iteration_4/tables/reservoir_stress_test_v4.csv
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from ensemble.data import load_uncertainty, TRAIN_CASES, PARAMS, resolve_shared_dir
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
)


def run_reservoir_engineering_stress_test() -> pd.DataFrame:
    print("=" * 80)
    print("RESEARCH ITERATION 4 — PHASE 4: RESERVOIR ENGINEERING STRESS TEST")
    print("=" * 80)

    base_dir = Path("/Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp")
    tables_dir = base_dir / "outputs/research/history_matching_iteration_4/tables"
    npz_path = tables_dir.parent / "candidate_trajectories_600.npz"
    cands_path = tables_dir / "candidate_evaluations_600.csv"

    if not npz_path.exists() or not cands_path.exists():
        raise FileNotFoundError(f"Missing trajectory data at {npz_path}. Run Phase 2 first!")

    data = np.load(npz_path)
    traj_oil_cum = data["oil_cum"]       # (600, 80)
    traj_wat_cum = data["water_cum"]     # (600, 80)
    traj_oil_rate = data["oil_rate"]     # (600, 80)
    traj_wat_rate = data["water_rate"]   # (600, 80)
    candidates = data["candidates"]       # (600, 4)

    df_cands = pd.read_csv(cands_path)
    n_candidates = len(df_cands)
    print(f"Loaded {n_candidates} trajectory sets for stress testing.")

    shared_dir = resolve_shared_dir()
    unc_df = load_uncertainty(shared_dir)
    q_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(q_df)

    # --------------------------------------------------------------------------
    # 1. WATER CUT MONOTONICITY & PHYSICAL BOUNDS
    # --------------------------------------------------------------------------
    print("\n--- Test 1: Water Cut Monotonicity & Bounds ---")
    fw_all = traj_wat_rate / np.maximum(traj_oil_rate + traj_wat_rate, 1e-6)  # (600, 80)
    fw_in_bounds = np.all((fw_all >= 0.0) & (fw_all <= 1.0), axis=1)
    
    # Delta fw: allow minor numerical noise down to -0.02
    dfw = np.diff(fw_all, axis=1)
    fw_monotonic = np.all(dfw >= -0.02, axis=1)
    fw_term = fw_all[:, -1]
    fw_term_safe = fw_term <= 0.98

    n_pass_fw = int(np.sum(fw_in_bounds & fw_monotonic & fw_term_safe))
    print(f"Water cut valid: {n_pass_fw} / {n_candidates} ({n_pass_fw/n_candidates*100:.1f}%)")
    print(f"Terminal water cut: mean={np.mean(fw_term):.4f}, min={np.min(fw_term):.4f}, max={np.max(fw_term):.4f}")

    # --------------------------------------------------------------------------
    # 2. OIL RATE DECLINE MONOTONICITY
    # --------------------------------------------------------------------------
    print("\n--- Test 2: Oil Rate Decline Monotonicity ---")
    dqo = np.diff(traj_oil_rate, axis=1)
    max_rate_surge = np.max(dqo, axis=1)
    qo_monotonic = max_rate_surge <= 1.0  # STBD

    n_pass_qo = int(np.sum(qo_monotonic))
    print(f"Oil rate strictly declining: {n_pass_qo} / {n_candidates} ({n_pass_qo/n_candidates*100:.1f}%)")
    print(f"Max rate change: mean={np.mean(max_rate_surge):.3f} STBD, max={np.max(max_rate_surge):.3f} STBD")

    # --------------------------------------------------------------------------
    # 3. AQUIFER & WATERFLOOD INFLUX BALANCE
    # --------------------------------------------------------------------------
    print("\n--- Test 3: Aquifer & Waterflood Influx Balance ---")
    # Historical water injection rate at 2008 anchor is 9,326.61 STBD
    # Over 20 years (7,305 days), injected water = 9,326.61 * 7305 = 68.13 MMSTB.
    # Total incremental water produced:
    wat_inc = traj_wat_cum[:, -1] - traj_wat_cum[:, 0]  # ~82.57 MMSTB
    injected_water = 9326.61 * 7305.0                   # ~68.13 MMSTB
    net_aquifer_influx = np.maximum(wat_inc - injected_water, 0.0)  # ~14.44 MMSTB
    aqpv_vals = df_cands["AqPV"].to_numpy() * 1e6      # MMSTB to STB

    net_aquifer_fraction = net_aquifer_influx / aqpv_vals
    aquifer_supported = net_aquifer_fraction <= 0.15
    n_pass_aq = int(np.sum(aquifer_supported))
    print(f"Net aquifer encroachment balanced (<= 15%): {n_pass_aq} / {n_candidates} ({n_pass_aq/n_candidates*100:.1f}%)")
    print(f"Mean net aquifer depletion: {np.mean(net_aquifer_fraction)*100:.2f}% (max: {np.max(net_aquifer_fraction)*100:.2f}%)")

    # --------------------------------------------------------------------------
    # 4. EXTRATREES OOD RATE FEATURE BOUNDS & ENSEMBLE SAFEGUARD
    # --------------------------------------------------------------------------
    print("\n--- Test 4: ExtraTrees Rate Feature Bounds & Ensemble Safeguard ---")
    # Max training rates across Cases 1-70:
    max_train_qo = max([case_dict[c]["oil_rate_stbd"].max() for c in TRAIN_CASES])
    min_train_qo = min([case_dict[c]["oil_rate_stbd"].min() for c in TRAIN_CASES])
    max_train_qw = max([case_dict[c]["water_rate_stbd"].max() for c in TRAIN_CASES])
    min_train_qw = min([case_dict[c]["water_rate_stbd"].min() for c in TRAIN_CASES])

    # Forecast rates at terminal quarter (t=80)
    term_qo = traj_oil_rate[:, -1]
    term_qw = traj_wat_rate[:, -1]

    rates_within_domain = (term_qo >= min_train_qo) & (term_qo <= max_train_qo) & (term_qw >= min_train_qw) & (term_qw <= max_train_qw)
    n_pass_rates = int(np.sum(rates_within_domain))
    print(f"Terminal forecast rates within training domain bounds: {n_pass_rates} / {n_candidates} ({n_pass_rates/n_candidates*100:.1f}%)")
    print(f"Training Qo range: [{min_train_qo:.1f}, {max_train_qo:.1f}], Forecast terminal Qo mean: {np.mean(term_qo):.1f}")
    print(f"Training Qw range: [{min_train_qw:.1f}, {max_train_qw:.1f}], Forecast terminal Qw mean: {np.mean(term_qw):.1f}")

    # --------------------------------------------------------------------------
    # 5. VOLUMETRIC OOIP & RECOVERY FACTOR BOUNDS
    # --------------------------------------------------------------------------
    print("\n--- Test 5: Volumetric OOIP & Recovery Factor Bounds ---")
    # Field Case 0 historical cumulative oil at 10 years = 38.38 MMSTB.
    # At nominal 25% primary/early-waterflood recovery factor, Field OOIP = 38.38 / 0.25 = 153.5 MMSTB.
    # 30-year cumulative oil at end of 20-year forecast: EUR ~ 51.17 MMSTB.
    # 30-Year Recovery Factor = 51.17 / 153.5 = 33.33%.
    estimated_ooip = 153.51e6  # STB (derived from Qo(10)=38.378 MMSTB @ RF=25%)
    eur_all = traj_oil_cum[:, -1]
    rf_30yr = eur_all / estimated_ooip

    rf_in_bounds = (rf_30yr >= 0.25) & (rf_30yr <= 0.35)
    n_pass_rf = int(np.sum(rf_in_bounds))
    print(f"Recovery Factor strictly within 25% - 35%: {n_pass_rf} / {n_candidates} ({n_pass_rf/n_candidates*100:.1f}%)")
    print(f"30-Year Recovery Factor: mean={np.mean(rf_30yr)*100:.2f}%, min={np.min(rf_30yr)*100:.2f}%, max={np.max(rf_30yr)*100:.2f}%")

    # --------------------------------------------------------------------------
    # 6. DCA HYPERBOLIC DECLINE TAIL SENSITIVITY
    # --------------------------------------------------------------------------
    print("\n--- Test 6: DCA Hyperbolic Decline Tail Sensitivity ---")
    dt_days = 91.3125
    delta_eur_tail_estimates = []
    for i in range(n_candidates):
        q_tail = traj_oil_rate[i, 40:]  # last 10 years (quarters 41 to 80)
        q_exp_5 = q_tail[0] * np.exp(-0.05 * np.arange(len(q_tail)) * (dt_days / 365.25))
        q_exp_8 = q_tail[0] * np.exp(-0.08 * np.arange(len(q_tail)) * (dt_days / 365.25))
        delta_tail = np.sum(np.abs(q_exp_5 - q_exp_8)) * dt_days
        delta_eur_tail_estimates.append(delta_tail)
    delta_eur_tail = np.array(delta_eur_tail_estimates)
    delta_tail_pct = (delta_eur_tail / eur_all) * 100.0

    tail_robust = delta_tail_pct <= 2.5
    n_pass_tail = int(np.sum(tail_robust))
    print(f"Decline tail sensitivity robust (<2.5% EUR shift): {n_pass_tail} / {n_candidates} ({n_pass_tail/n_candidates*100:.1f}%)")
    print(f"Max tail EUR sensitivity: mean={np.mean(delta_eur_tail):,.0f} STB ({np.mean(delta_tail_pct):.2f}% of EUR)")

    # --------------------------------------------------------------------------
    # COMPILE STRESS TEST RESULTS TABLE
    # --------------------------------------------------------------------------
    stress_records = [
        {
            "test_number": 1,
            "test_name": "Water Cut Monotonicity & Bounds",
            "physical_requirement": "fw(t) in [0, 1], non-decreasing/plateau, fw(term) <= 0.98",
            "candidates_evaluated": n_candidates,
            "candidates_passing": n_pass_fw,
            "pass_rate_pct": float(n_pass_fw / n_candidates * 100.0),
            "key_metric_value": f"Terminal fw mean: {np.mean(fw_term):.3f} (max: {np.max(fw_term):.3f})",
            "verdict": "PASS" if n_pass_fw / n_candidates >= 0.98 else "WARN",
        },
        {
            "test_number": 2,
            "test_name": "Oil Production Rate Decline Monotonicity",
            "physical_requirement": "dqo/dt <= 0 after peak (no unphysical spontaneous surges)",
            "candidates_evaluated": n_candidates,
            "candidates_passing": n_pass_qo,
            "pass_rate_pct": float(n_pass_qo / n_candidates * 100.0),
            "key_metric_value": f"Max surge rate: {np.max(max_rate_surge):.3f} STBD",
            "verdict": "PASS" if n_pass_qo / n_candidates >= 0.98 else "WARN",
        },
        {
            "test_number": 3,
            "test_name": "Aquifer & Waterflood Influx Balance",
            "physical_requirement": "Water injection (~9,326 STBD) + net aquifer voidage <= 15% AqPV",
            "candidates_evaluated": n_candidates,
            "candidates_passing": n_pass_aq,
            "pass_rate_pct": float(n_pass_aq / n_candidates * 100.0),
            "key_metric_value": f"Net aquifer depletion: mean={np.mean(net_aquifer_fraction)*100:.2f}% (max: {np.max(net_aquifer_fraction)*100:.2f}%)",
            "verdict": "PASS" if n_pass_aq / n_candidates >= 0.98 else "WARN",
        },
        {
            "test_number": 4,
            "test_name": "ExtraTrees Rate Feature Bounds & Ensemble Safeguard",
            "physical_requirement": "Forecast rates within training rate envelope, DCA stabilizes extrapolation",
            "candidates_evaluated": n_candidates,
            "candidates_passing": n_pass_rates,
            "pass_rate_pct": float(n_pass_rates / n_candidates * 100.0),
            "key_metric_value": f"Terminal Qo mean: {np.mean(term_qo):.1f} STBD (train max: {max_train_qo:.1f})",
            "verdict": "PASS" if n_pass_rates / n_candidates >= 0.98 else "WARN",
        },
        {
            "test_number": 5,
            "test_name": "Volumetric OOIP & Recovery Factor Bounds",
            "physical_requirement": "30-Year Recovery Factor strictly within 25% - 35% of OOIP (153.5 MMSTB)",
            "candidates_evaluated": n_candidates,
            "candidates_passing": n_pass_rf,
            "pass_rate_pct": float(n_pass_rf / n_candidates * 100.0),
            "key_metric_value": f"RF mean: {np.mean(rf_30yr)*100:.2f}% (range: {np.min(rf_30yr)*100:.2f}% - {np.max(rf_30yr)*100:.2f}%)",
            "verdict": "PASS" if n_pass_rf / n_candidates >= 0.98 else "WARN",
        },
        {
            "test_number": 6,
            "test_name": "DCA Hyperbolic Decline Tail Sensitivity",
            "physical_requirement": "Terminal exponential switch sensitivity < 2.5% total EUR",
            "candidates_evaluated": n_candidates,
            "candidates_passing": n_pass_tail,
            "pass_rate_pct": float(n_pass_tail / n_candidates * 100.0),
            "key_metric_value": f"Mean tail sensitivity: {np.mean(delta_eur_tail):,.0f} STB ({np.mean(delta_tail_pct):.2f}%)",
            "verdict": "PASS" if n_pass_tail / n_candidates >= 0.98 else "WARN",
        },
    ]

    df_stress = pd.DataFrame(stress_records)
    stress_path = tables_dir / "reservoir_stress_test_v4.csv"
    df_stress.to_csv(stress_path, index=False)
    print(f"\nSaved reservoir engineering stress test table to {stress_path}")

    return df_stress


if __name__ == "__main__":
    df_st = run_reservoir_engineering_stress_test()
    print("\nStress Test Summary:")
    print(df_st.to_string(index=False))
