# Anas Temporal Validation Handoff

## Scope
- Training: Cases 1-70, fixed five-fold held-case OOF predictions.
- Validation: Cases 71-85, parameter maps fitted on Cases 1-70.
- Test production labels: not loaded.
- Origins: 2003, 2004, 2005.
- Horizons: fixed three-year comparisons plus a five-year 2003-origin comparison.
- Default target extraction: three-year pre-cutoff fitting window.

## Pipeline
- Four uncertainty inputs.
- One-PC historical cumulative/rate heads with Polynomial Ridge.
- Transformed temporal-parameter heads with Polynomial Ridge.
- Positive-rate decoder and cumulative integration from predicted anchors.

## Development ranking
Best mean input-only training-OOF decoder: Separate

This is a development ranking, not proof of superiority at 2028.
Check phase scores, paired case comparisons, late-horizon errors,
parameter-map errors, and fitting-window sensitivity before locking a decoder.

## Essential Distinctions
1. History-conditioned forecasts isolate temporal extrapolation error.
2. Input-only forecasts include geological mapping error.
3. Cumulative error includes anchor error.
4. Increment error isolates predicted future growth.
5. Twenty-year scenario spread is structural disagreement, not a calibrated interval.

## Constraints and Assumptions
- Nonnegative production rates.
- Nondecreasing cumulative production.
- Liquid exponential candidates assume nonincreasing liquid production.
- Logistic water-cut candidates assume nondecreasing water cut.
- No arbitrary 96% water-cut cap.
- GOR remains fixed at its cutoff-history estimate.
- Future injection and operating controls are unknown.

## Required Review
- tables/06_crossed_backtest_macro_scores.csv
- tables/06_crossed_backtest_phase_scores.csv
- tables/06_paired_decoder_comparison.csv
- tables/07_early_vs_late_error.csv
- tables/02_window_sensitivity_summary.csv
- tables/05_identifiability_summary.csv
- tables/10_forecast_review_flags.csv

## Limits
No future simulator labels exist beyond 2008 in the supplied project.
A successful 3-5-year backtest does not validate a 20-year forecast.
The 0.20 warning threshold is a team review trigger, not an official criterion.
Do not report scenario envelopes as P10/P50/P90 calibrated forecast uncertainty.

## Next Step
Lock the temporal specification, transformations, fitting window, and model
hyperparameters before the team's one-time Cases 86-100 evaluation.
If tuning is added, use inner case folds and retain outer case folds for
honest development evaluation.
