# Hazeem Handoff: Geology Features + Hyperbolic Decline (v2)

Owner: Hazeem | Track: Feature Engineering & Geology

## Result
Scored with `backtest_engine.py` on the same 4 experiments, cases and folds as W4:

| Model | Mean NRMSE | Worst exp | Oil | Water | Validation only |
|---|---|---|---|---|---|
| Baseline (Anas, history-fit exponential) | 0.153 | 0.301 | 0.153 | 0.151 | 0.154 |
| v1: SVR + grid-normalised state → hyperbolic | 0.083 | 0.142 | 0.049 | 0.151 | 0.080 |
| **v2: + liquid-capped water** | **0.077** | **0.126** | **0.049** | **0.132** | **0.073** |

Zero physical violations. Full tables: `outputs/model_comparison.csv` and `outputs/water_model_comparison.csv`.

## What changed in v2
### 1. Water fixed
The v1 water (W4's relaxation curve) had two problems:
- **Backtest:** its error sat almost entirely at the 2003 cutoff (0.20 / 0.28), with a positive bias. Water growth dropped from +32%/yr (2002) to +8%/yr (2003+), and the curve kept the fast early rise.
- **2028:** water grew 1.7× (up to 2.1×), so **total liquid grew 20% (up to 51%)**. In the simulator, liquid has been flat (−0.7%/yr over 2006–08) and injection is falling, so that isn't physical.

**Fix:** keep the relaxation shape, but cap water so total liquid can't exceed its level at the cutoff: `qw ≤ (qo+qw)_T − qo(t)`, with water cut ≤ 97%.
- **Water error 0.151 → 0.132.** Better at 2003 (0.201 → 0.173, 0.284 → 0.238), unchanged at 2004 and 2005. It is never worse than v1 in any experiment.
- **In 2028,** liquid stays at its 2008 level (ratio 1.00) instead of up to 1.51.

Tested and rejected:

| Water model | Water NRMSE | Why rejected |
|---|---|---|
| WOR-Np, history slope | 0.71 | Fits 2002–08 well in hindsight (R² 0.99), but the 2003 slope comes from early breakthrough and blows up |
| WOR-Np, learned slope | 1.44 | Same problem |
| Liquid trend − oil | 0.22 | Small liquid errors become large water errors |
| Learned liquid − oil | 0.20 | Same |
| Relaxation + last-year liquid cap | 0.128 | Slightly better mean, but doubles the 2004 water error (0.053 → 0.117) |

**Note:** with the cap, water accuracy depends on the oil forecast.

### 2. Oil: hyperbolic vs exponential compared side by side
| Oil NRMSE | 2003-3y | 2003-5y | 2004-3y | 2005-3y | Mean |
|---|---|---|---|---|---|
| Baseline | 0.214 | 0.309 | 0.058 | 0.033 | 0.153 |
| SVR exponential (b = 0) | 0.059 | 0.071 | 0.039 | 0.029 | 0.0494 |
| SVR hyperbolic (b from data) | 0.059 | 0.071 | 0.034 | 0.033 | 0.0493 |

- **The two shapes tie over 3–5 years.** The b estimated at each cutoff: 0 (2003), 0.43, 0.82, 0.44, 1.0, 1.0 (2008).
- **The Arps plot** (decline rate vs rate, figure 04a) shows the decline slowing as rate falls, which points to b ≈ 1.
- **Over 20 years they differ a lot.** Median extra oil 2008→2028:
  - hyperbolic: **20.9 M STB**
  - exponential: **16.3 M STB**
  - baseline: 12.7 M STB

  **The team should pick one or report both as a range.**

## What the grid files contain (unchanged from v1)
- **Porosity mult.** scales PORO and PORV in every cell.
- **Permeability mult.** scales PERMX/TRANX/TRANY only in the tight facies (11,575 cells, about 0.5 mD, in K13–31, about 19% of the oil).
- **Aquifer PV** scales PORV of the 2,286 east-edge cells (I = 51–53).
- **Fault transmissibility is in no exported column.**
- **The main fault** is the FIPNUM 1/2 boundary: west block 19% of oil, east block 81%.

## Method (unchanged)
- **Arps decline:** D = K·q^b, so ln D = ln K + b·ln q (a linear model).
- **SVR** predicts the 1-year-ahead decline from: ln q, water cut, recent trends, **recovery factor (Np / grid OOIP)** and **water index (Wp / grid aquifer PV)**.
- **No time leakage:** only labels that end before the cutoff are used. Cases 1–70 are scored out-of-fold (KFold 5, seed 42); cases 71–85 use a model trained on 1–70.
- **Gas** = GOR × oil.

## Deployment (2008 → 2028), `outputs/forecast_2008_2028.parquet`
- **Model IDs:**
  - `Hazeem_SVR_hyperbolic` (b = 1.0)
  - `Hazeem_SVR_exponential` (b = 0)
  - `*_uncappedwater` (v1 water, for comparison)
  - `Baseline_Anas_Separate_hist`
- **Decline guardrail:** the 2008 decline is clipped to 0.5–1.5× the observed 2007 decline. It adjusted cases 10, 12, 16, 59, 71, 73.
- **Also exported** as `outputs/preds_Hazeem.parquet` in the DATA_CONTRACT format.

## Files
- `01 Geology Features + Hyperbolic Decline.ipynb`: end to end, about 2.5 minutes.
- `geo_features.py`, `hz_model.py`, `hz_plots.py`.
- `00_prepare_data.py`: rebuilds the parquet files from the raw Excel.
- `temporal_utils.py`, `backtest_engine.py`: unchanged copies of W4.

## Notes for Anas
- `BacktestEngine.score_predictions` needs a unique row index. Use `pd.concat(..., ignore_index=True)`.
- The W4 strict-b0 water (relaxation with a 2× rate cap) can still let total liquid grow. The liquid cap here may be a useful drop-in.
- Test cases 86–100 were never loaded.
