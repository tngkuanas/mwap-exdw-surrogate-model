"""backtest_engine.py — W4 Referee Scoring Engine

Owner: Anas (Workstream 4)
Purpose: Standardized evaluation of ANY team member's forecast predictions.
Every model must pass this engine before submission.

Experiments:
    2003-01-01 → 3yr  (12 forecast quarters)
    2003-01-01 → 5yr  (20 forecast quarters)
    2004-01-01 → 3yr  (12 forecast quarters)
    2005-01-01 → 3yr  (12 forecast quarters)

Usage:
    from backtest_engine import BacktestEngine
    engine = BacktestEngine(shared_data_path, curves)
    report = engine.score_predictions(predictions_df)
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────
EXPERIMENTS = [
    ("2003-01-01", 3),
    ("2003-01-01", 5),
    ("2004-01-01", 3),
    ("2005-01-01", 3),
]

PHASES = ["oil_cum", "gas_cum", "water_cum"]
TRAIN_CASES = list(range(1, 71))
VAL_CASES = list(range(71, 86))
TEST_CASES = list(range(86, 101))
YEAR_DAYS = 365.25

# Required columns in prediction submissions
REQUIRED_COLUMNS = [
    "model_id", "case_num", "origin", "horizon_years",
    "date", "phase", "prediction",
]


class BacktestEngine:
    """Standardized forecast evaluation engine for the MWAP team."""

    def __init__(self, curves: dict, allow_test: bool = False):
        """
        Args:
            curves: dict of (case_num, metric) -> pd.Series from temporal_utils.load_shared
            allow_test: if False, rejects predictions for test cases 86-100
        """
        self.curves = curves
        self.allow_test = allow_test

    def interpolate_truth(self, case: int, phase: str, dates) -> np.ndarray:
        """Get ground-truth cumulative values at requested dates."""
        dates = pd.DatetimeIndex(dates)
        series = self.curves.get((int(case), phase))
        if series is None or len(series) < 2:
            raise ValueError(f"Missing history: Case {case}, {phase}")
        if dates.min() < series.index.min() or dates.max() > series.index.max():
            raise ValueError(
                f"Requested dates outside history: Case {case}, {phase}, "
                f"requested {dates.min()} to {dates.max()}, "
                f"available {series.index.min()} to {series.index.max()}"
            )
        return np.interp(
            dates.asi8.astype(float),
            series.index.asi8.astype(float),
            series.to_numpy(float),
        )

    def validate_submission(self, preds: pd.DataFrame) -> list[str]:
        """Validate a prediction DataFrame. Returns list of errors (empty = valid)."""
        errors = []

        # Check required columns
        missing = set(REQUIRED_COLUMNS) - set(preds.columns)
        if missing:
            errors.append(f"Missing required columns: {sorted(missing)}")
            return errors  # Can't proceed without columns

        # Check for duplicates
        key_cols = ["model_id", "case_num", "origin", "horizon_years", "date", "phase"]
        if preds.duplicated(key_cols).any():
            n_dup = preds.duplicated(key_cols).sum()
            errors.append(f"Duplicate prediction keys: {n_dup} rows")

        # Check phases
        bad_phases = set(preds["phase"]) - set(PHASES)
        if bad_phases:
            errors.append(f"Unknown phases: {bad_phases}. Expected: {PHASES}")

        # Check test set isolation
        if not self.allow_test:
            test_cases = preds[preds["case_num"].isin(TEST_CASES)]
            if len(test_cases) > 0:
                errors.append(
                    f"Test cases 86-100 are FORBIDDEN in backtests. "
                    f"Found {len(test_cases)} rows."
                )

        # Check predictions are finite
        non_finite = (~np.isfinite(preds["prediction"])).sum()
        if non_finite > 0:
            errors.append(f"Non-finite predictions: {non_finite} rows")

        # Check experiment validity
        for _, row in preds[["origin", "horizon_years"]].drop_duplicates().iterrows():
            origin = pd.Timestamp(row["origin"])
            horizon = int(row["horizon_years"])
            if (origin.strftime("%Y-%m-%d"), horizon) not in [
                (o, h) for o, h in EXPERIMENTS
            ]:
                errors.append(
                    f"Unknown experiment: {origin.date()} → {horizon}yr. "
                    f"Allowed: {EXPERIMENTS}"
                )

        return errors

    def score_predictions(self, preds: pd.DataFrame) -> dict:
        """Score a validated prediction DataFrame.

        Returns a dict with:
            - 'validation_errors': list of errors (empty = passed)
            - 'phase_scores': per-experiment, per-phase scores
            - 'macro_scores': aggregated model-level scores
            - 'quarter_scores': quarter-by-quarter error trajectory
            - 'physical_violations': physical validity report
            - 'summary': one-line verdict per model
        """
        preds = preds.copy()
        preds["origin"] = pd.to_datetime(preds["origin"])
        preds["date"] = pd.to_datetime(preds["date"])

        # Validate first
        validation_errors = self.validate_submission(preds)
        if validation_errors:
            return {"validation_errors": validation_errors}

        # Assign split
        preds["split"] = np.select(
            [preds["case_num"].between(1, 70), preds["case_num"].between(71, 85)],
            ["train", "validation"], default="test",
        )

        # Compute truth, anchor, increments
        truth_rows = []
        for (case, phase, origin), idx in preds.groupby(
            ["case_num", "phase", "origin"]
        ).groups.items():
            dates = preds.loc[idx, "date"]
            truth = self.interpolate_truth(case, phase, dates)
            anchor = self.interpolate_truth(case, phase, [origin])[0]
            for i, (dt, t) in enumerate(zip(dates, truth)):
                truth_rows.append({
                    "_idx": idx[i] if hasattr(idx, '__getitem__') else idx,
                    "truth": t, "anchor": anchor,
                })

        truth_df = pd.DataFrame(truth_rows).set_index("_idx")
        preds["truth"] = truth_df["truth"]
        preds["anchor"] = truth_df["anchor"]
        preds["true_increment"] = preds["truth"] - preds["anchor"]
        preds["pred_increment"] = preds["prediction"] - preds["anchor"]
        preds["error"] = preds["prediction"] - preds["truth"]

        # Lead quarter
        preds["lead_quarter"] = (
            (preds["date"].dt.year - preds["origin"].dt.year) * 12
            + preds["date"].dt.month - preds["origin"].dt.month
        ) // 3

        # ── Phase Scores ──────────────────────────────────────
        phase_rows = []
        for keys, group in preds.groupby(
            ["model_id", "split", "origin", "horizon_years", "phase"]
        ):
            model, split, origin, horizon, phase = keys
            errors = group["error"].to_numpy()
            true_inc = group["true_increment"].to_numpy()
            norm = np.sqrt(np.mean(true_inc ** 2))
            phase_rows.append({
                "model_id": model, "split": split,
                "origin": origin, "horizon_years": horizon, "phase": phase,
                "cases": group["case_num"].nunique(),
                "cumulative_RMSE": np.sqrt(np.mean(errors ** 2)),
                "increment_NRMSE": np.sqrt(np.mean(errors ** 2)) / norm if norm > 0 else np.nan,
                "bias": np.mean(errors) / norm if norm > 0 else np.nan,
            })
        phase_scores = pd.DataFrame(phase_rows)

        # ── Macro Scores ──────────────────────────────────────
        macro_rows = []
        for model, group in phase_scores.groupby("model_id"):
            dev = group[group["split"].isin(["train", "validation"])]
            by_exp = dev.groupby(["origin", "horizon_years"])["increment_NRMSE"].mean()
            macro_rows.append({
                "model_id": model,
                "mean_dev_NRMSE": by_exp.mean(),
                "worst_exp_NRMSE": by_exp.max(),
                "median_dev_NRMSE": by_exp.median(),
                "experiments_scored": len(by_exp),
            })
        macro_scores = pd.DataFrame(macro_rows).sort_values("mean_dev_NRMSE")

        # ── Quarter Scores ────────────────────────────────────
        quarter_rows = []
        for keys, group in preds.groupby(
            ["model_id", "split", "origin", "horizon_years", "phase", "lead_quarter"]
        ):
            model, split, origin, horizon, phase, lq = keys
            errors = group["error"].to_numpy()
            norm = np.sqrt(np.mean(group["true_increment"].to_numpy() ** 2))
            quarter_rows.append({
                "model_id": model, "split": split,
                "origin": origin, "horizon_years": horizon,
                "phase": phase, "lead_quarter": lq,
                "cases": len(group),
                "increment_NRMSE": np.sqrt(np.mean(errors ** 2)) / norm if norm > 0 else np.nan,
                "cumulative_RMSE": np.sqrt(np.mean(errors ** 2)),
            })
        quarter_scores = pd.DataFrame(quarter_rows)

        # ── Physical Violations ───────────────────────────────
        phys_rows = []
        for (model, split), group in preds.groupby(["model_id", "split"]):
            n = len(group)
            neg_cum = (group["prediction"] < -1e-6).sum()

            # Check decreasing cumulative within case/phase trajectories
            decreasing = 0
            for _, traj in group.sort_values("date").groupby(
                ["case_num", "phase", "origin", "horizon_years"]
            ):
                diffs = traj["prediction"].diff().dropna()
                decreasing += (diffs < -1e-6).sum()

            phys_rows.append({
                "model_id": model, "split": split,
                "total_predictions": n,
                "negative_cumulative": int(neg_cum),
                "negative_cum_pct": 100 * neg_cum / n,
                "decreasing_steps": int(decreasing),
                "decreasing_pct": 100 * decreasing / max(n - 1, 1),
            })
        physical_violations = pd.DataFrame(phys_rows)

        # ── Summary ───────────────────────────────────────────
        summary = {}
        for _, row in macro_scores.iterrows():
            model = row["model_id"]
            phys = physical_violations[physical_violations["model_id"] == model]
            total_violations = phys["negative_cumulative"].sum() + phys["decreasing_steps"].sum()
            verdict = "PASS" if row["mean_dev_NRMSE"] < 0.30 and total_violations == 0 else "REVIEW"
            if row["mean_dev_NRMSE"] >= 0.50 or total_violations > 100:
                verdict = "FAIL"
            summary[model] = (
                f"{verdict} | mean_dev={row['mean_dev_NRMSE']:.4f} | "
                f"worst_exp={row['worst_exp_NRMSE']:.4f} | "
                f"violations={total_violations}"
            )

        return {
            "validation_errors": [],
            "phase_scores": phase_scores,
            "macro_scores": macro_scores,
            "quarter_scores": quarter_scores,
            "physical_violations": physical_violations,
            "summary": summary,
        }

    def export_report(self, report: dict, output_dir: Path) -> None:
        """Save all report DataFrames to CSVs."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for key in ["phase_scores", "macro_scores", "quarter_scores", "physical_violations"]:
            if key in report and isinstance(report[key], pd.DataFrame):
                report[key].to_csv(output_dir / f"referee_{key}.csv", index=False)

        # Summary text
        with open(output_dir / "referee_summary.txt", "w") as f:
            for model, verdict in report.get("summary", {}).items():
                f.write(f"{model}: {verdict}\n")

        print(f"Report exported to {output_dir}")
