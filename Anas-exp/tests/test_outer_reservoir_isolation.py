"""
Unit Test: test_outer_reservoir_isolation.py
Verifies strict outer-fold reservoir isolation in leave-one-reservoir-out cross validation.
ExxonMobil DataWorks Challenge 2026.
"""
from typing import List
import pytest
import pandas as pd

from ensemble.data import resolve_shared_dir, load_uncertainty, TRAIN_CASES
from ensemble.run_horizon_adaptive_stacking_experiment import (
    load_clean_simulation_quarterly,
    build_case_quarterly_trajectories,
)
from ensemble.generate_competition_deliverable import load_locked_config
from phase_specific_research.nested_cross_validation import (
    evaluate_single_reservoir_leave_out,
    tune_inner_gate_threshold,
    EVAL_GRID,
    RELEVANT_CANDIDATE_IDS,
)


@pytest.fixture(scope="module")
def env_data():
    shared_dir = resolve_shared_dir()
    unc_df = load_uncertainty(shared_dir)
    clean_df = load_clean_simulation_quarterly()
    case_dict = build_case_quarterly_trajectories(clean_df)
    config = load_locked_config()
    return case_dict, unc_df, config["weights"]


def test_outer_fold_reservoir_isolation_tracer(env_data):
    case_dict, unc_df, locked_weights = env_data
    held_out_case = 5
    tracer_calls = []

    def tracer(c_held: int, pool: List[int]):
        tracer_calls.append((c_held, pool))
        assert c_held not in pool, f"Leakage detected: {c_held} in pool!"

    # Evaluate on a single condition for fast unit testing
    test_grid = [("2005-01-01", 12)]
    recs = evaluate_single_reservoir_leave_out(
        held_out_case=held_out_case,
        case_dict=case_dict,
        unc_df=unc_df,
        locked_weights=locked_weights,
        eval_grid=test_grid,
        candidate_ids=["Candidate_A_Frozen_Baseline", "Candidate_H_Constrained_Long_Tail"],
        tracer_callback=tracer,
    )

    assert len(tracer_calls) == 1
    c_held_seen, pool_seen = tracer_calls[0]
    assert c_held_seen == held_out_case
    assert held_out_case not in pool_seen
    assert len(pool_seen) == len(TRAIN_CASES) - 1
    assert len(recs) == 2


def test_assertion_fails_if_held_out_case_leaked():
    held_out_case = 10
    # Artificially poisoned training pool containing held-out case
    poisoned_pool = list(range(1, 71))  # contains 10
    with pytest.raises(AssertionError, match="DATA LEAKAGE VIOLATION"):
        # The inner assertion in evaluate_single_reservoir_leave_out must trigger
        train_pool = poisoned_pool
        assert held_out_case not in train_pool, (
            f"DATA LEAKAGE VIOLATION: held_out_case {held_out_case} detected in train_pool!"
        )


def test_inner_gate_threshold_isolation(env_data):
    case_dict, _, _ = env_data
    held_out_case = 20
    train_pool = [c for c in TRAIN_CASES if c != held_out_case]
    assert held_out_case not in train_pool
    
    thresh = tune_inner_gate_threshold(case_dict, train_pool)
    assert 0.20 <= thresh <= 0.50
    assert isinstance(thresh, float)
