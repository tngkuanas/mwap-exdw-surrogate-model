# W4 Handoff — Strict Timeline & Referee Validation

Owner: Anas | Generated: 2026-10-08 23:33

## Challenger Model: Strict_b0_Anas
- Oil: strict b=0 exponential decline (D fitted by anchored log-regression)
- Gas: constant GOR from historical median
- Water: relaxation dynamics capped at 15853 STB/day (2.0x observed 2008)
- Integration: daily trapezoid from cutoff cumulative anchor

## Key Results
- Verdict: PASS | mean_dev=0.1525 | worst_exp=0.3006 | violations=0
- Backtest predictions: 14280 rows across 4 experiments
- 2028 forecasts: 15 validation cases, 80 quarterly dates
- Zero physical violations (by construction: nonneg rates + monotonic integration)

## Files
- `backtest_engine.py` — Referee scoring engine (use for ALL team models)
- `strict_challenger/data/strict_challenger_backtests.parquet` — Backtest predictions
- `strict_challenger/data/strict_challenger_2028.parquet` — 2028 forecasts
- `strict_challenger/tables/referee_*.csv` — Scoring report
- `strict_challenger/tables/strict_challenger_fit_parameters.csv` — Fit params per case/origin
- `strict_challenger/w4_manifest.json` — Full configuration

## Prediction Contract
Any team member's predictions must have these columns:
`model_id, case_num, origin, horizon_years, date, phase, prediction`

Phases: oil_cum, gas_cum, water_cum
Origins: 2003-01-01, 2004-01-01, 2005-01-01 (backtests); 2008-01-01 (deployment)
Test cases 86-100: FORBIDDEN in backtests

## For Nazrul (W1), Afiq (W2), Hazeem (W3)
To score your model through the referee engine:
```python
from backtest_engine import BacktestEngine
import temporal_utils as tu
unc, curves = tu.load_shared('path/to/shared/data')
engine = BacktestEngine(curves)
report = engine.score_predictions(your_predictions_df)
print(report['summary'])
```

## Hash Provenance
- temporal_utils.py: 3ba3b98c
- backtest_engine.py: 13d59280
