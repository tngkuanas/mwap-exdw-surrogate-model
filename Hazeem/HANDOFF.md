# Hazeem Handoff: Geology Features + Hyperbolic Decline

Owner: Hazeem | Track: Feature Engineering & Geology

## Result
Scored with `backtest_engine.py` on the same 4 experiments, cases and folds as W4:

| Model | Mean NRMSE | Worst exp | Oil | Water | Validation only |
|---|---|---|---|---|---|
| Baseline (Anas, history-fit exponential) | 0.153 | 0.301 | 0.153 | 0.151 | 0.154 |
| **SVR + grid-normalised state → hyperbolic** | **0.083** | **0.142** | **0.049** | 0.151 | **0.080** |
| Ridge + grid-normalised state → hyperbolic | 0.110 | 0.176 | 0.089 | 0.151 | 0.112 |

Zero physical violations. Full table: `outputs/model_comparison.csv`.

## What the grid files actually contain
- **Porosity mult.** scales PORO and PORV in all 42,512 cells.
- **Permeability mult.** scales PERMX/TRANX/TRANY only in the **tight facies**: 11,575 cells (~0.5 mD) in K13–31 holding ~19% of the oil. Good rock, Zone 1 (K1–5), PERMZ and TRANZ never change.
- **Aquifer PV** scales PORV of 2,286 east-edge cells (I = 51–53), which are ~84% of total pore volume.
- **Fault transmissibility is in no exported column.** The simulator applies it on the fault faces. That's why every grid summary failed to correlate with it.
- SOIL, SWAT and PRESSURE are identical across cases (initial state).
- The main fault = the FIPNUM 1/2 boundary. It splits the field into a west block (19% of oil) and an east block (81%) with the aquifer on its east side.
- There are 3 oil zones (K1–5, K13–19, K21–31), separated by barrier layers K6–12 and K20.

## Method
- **Arps decline:** D = K·q^b, so ln D = ln K + b·ln q. That's linear, so Ridge/SVR fit it.
- **Label:** realised 1-year-ahead oil decline, taken at every quarter of every training case.
- **Inputs:** ln q, water cut, recent decline and water-cut trend, **recovery factor (Np / grid OOIP)** and **water index (Wp / grid aquifer PV)**.
- **No time leakage:** only labels that end before the cutoff are used. Cases 1–70 are scored out-of-fold (KFold 5, seed 42, same as W4); cases 71–85 use a model trained on 1–70.
- **b:** the within-case slope of ln D on ln q after each case's peak decline, clipped to [0, 1]. Values: 0 at 2003 (too little data), 0.43 at 2004, 0.82 at 2005, **1.0 at 2008** (raw 1.42).
- **Gas** = GOR × oil (GOR is constant at 0.3633). **Water** = W4's history-fitted relaxation.

## Key findings for the team
1. **Grid features help by normalising the production state.** Recovery factor and water index take SVR from 0.113 to 0.083.
2. **Feeding more geology features or the raw 4 parameters into the regressor makes it worse** (overfits, because there are few post-plateau rows before each cutoff).
3. **Water (0.15) is now the weakest phase.** My learned water models were unstable and blew up.
4. **The 2028 decline shape is the biggest open decision.** Median extra oil 2008→2028:
   - baseline: 12.7 M STB
   - SVR with b = 0: 16.3 M STB
   - SVR with b = 1: 20.9 M STB

   The backtests (≤5 years) can't confirm 20-year curvature.

## Deployment (2008 → 2028)
- The SVR is trained on cases 1–70 using all quarters up to 2008.
- The decline at 2008 is clipped to 0.5–1.5× the observed 2007 decline. This costs nothing in the 2004/2005 backtests, and it adjusted 6 cases that sit outside the training range: 10, 12, 16, 59, 71, 73.
- Outputs:
  - `outputs/forecast_2008_2028.parquet` in the W4 engine format (3 model_ids)
  - `outputs/preds_Hazeem.parquet` in the DATA_CONTRACT format

## Files
- `01 Geology Features + Hyperbolic Decline.ipynb`: end-to-end, about 1.5 minutes on a laptop.
- `geo_features.py`: grid features. `hz_model.py`: panel, regressors, decoder. `hz_plots.py`: figures.
- `00_prepare_data.py`: rebuilds the 3 shared parquet files from the raw Excel. It reproduces the team's files exactly.
- `temporal_utils.py` and `backtest_engine.py`: unchanged copies of Anas's files.

## Notes for Anas
- `BacktestEngine.score_predictions` needs a **unique row index**. Concatenating prediction frames without `ignore_index=True` crashes it with an IndexError.
- Test cases 86–100 were never loaded.
