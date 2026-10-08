# Forecast Modeling Handoff — Anas Workstream 4

## Locked Specification
- **Model**: GP regressor + Separate temporal decoder
- **Training**: Cases 1-70, 5-fold CV, 2008 cutoff, 3y window
- **Validation**: Cases 71-85 (input-only, no fitting)
- **Test**: Cases 86-100 (RESERVED, not touched)

## Key Results
- Historical surrogate: see tables/historical_surrogate_scores.csv
- Forecast backtests: see tables/macro_scores.csv
- 2028 endpoints: see tables/validation_2028_endpoints.csv
- Physical validity: see tables/physical_validity.csv

## Exported Artifacts
- `data/forecast_modeling_backtests.parquet` — crossed case×time backtest predictions
- `data/forecast_modeling_2028.parquet` — 2028 conditional forecasts (val cases)
- `outputs/forecast_modeling/models/` — fitted model objects
- `outputs/forecast_modeling/locked_specification.json` — full spec
- `outputs/forecast_modeling/tables/` — all score tables
- `outputs/forecast_modeling/figures/` — all diagnostic figures

## Limitations
- 2028 forecasts are CONDITIONAL on the Separate decoder assumptions
- 2003 cutoff forecasts remain at ~0.20 NRMSE (3yr) / ~0.29 NRMSE (5yr)
- ~1% physical violations corrected by clipping
- Validation has been used for decoder selection; not an untouched final test

## Integration
To use Anas predictions in team stacking:
1. Load `forecast_modeling_2028.parquet`
2. Filter by `model='GP'`, `decoder='Separate'`
3. Join on `case_num`, `date`, `metric`
