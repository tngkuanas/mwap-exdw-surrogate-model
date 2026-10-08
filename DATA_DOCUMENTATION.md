# MWAP Dataset Documentation
## ExxonMobil DataWorks Challenge 2026

> **Topic:** Machine Learning-Assisted History Matching and the Challenge of Production Forecasting

---

## 1. Problem Statement

### What is the problem?
Build a **machine learning surrogate model** that replaces a physics-based reservoir simulator. Given a set of subsurface **uncertainty parameters** that control rock and fluid properties across a 3D reservoir grid, the model must predict **cumulative oil, gas, and water production** over time.

The two-part task:
1. **History Matching** — reproduce the observed 10-year field production (1998–2008) by finding which simulation cases best fit reality
2. **Production Forecasting** — predict cumulative production beyond 2008 (10–20 years ahead) for 10 selected cases

### What is a Reservoir Simulation Model?
A mathematical representation of an underground oil/gas reservoir, discretized into thousands of 3D grid cells. Each cell is assigned **rock properties** (porosity, permeability, transmissibility) and **fluid properties** (saturations, pressure, PVT data). The simulator solves partial differential equations (mass balance + Darcy flow) to predict how fluids move and how much oil/gas/water is produced.

Running the full physics simulator is expensive. The challenge is to build a **fast ML proxy** (surrogate) that maps uncertainty inputs directly to production curves.

### The 4 Uncertainty Parameters
Each of the 100 simulation cases uses a different combination of these multipliers, which globally scale the reservoir properties:

| Parameter | Physical Meaning | Range | Units |
| --- | --- | --- | --- |
| Fault Transmissibility | Controls flow across geological faults — higher = more cross-fault flow | 0.050 – 0.150 | dimensionless multiplier |
| Porosity Multiplier | Scales all cell porosities — higher = more storage capacity | 0.801 – 1.494 | multiplier on base porosity |
| Permeability Multiplier | Scales all cell permeabilities — higher = easier fluid flow | 0.521 – 9.785 | multiplier on base perm |
| Aquifer Pore Volume | Size of the connected aquifer — larger = stronger water drive | 53.95 – 199.84 | relative PV units |

Statistics (across 100 cases):

| Parameter | Mean | Std | Min | Max |
| --- | --- | --- | --- | --- |
| Fault Transmissibility | 0.0996 | 0.0301 | 0.0501 | 0.1500 |
| Porosity Multiplier | 1.1485 | 0.2041 | 0.8012 | 1.4940 |
| Permeability Multiplier | 5.1998 | 2.7267 | 0.5205 | 9.7845 |
| Aquifer Pore Volume | 125.17 | 42.37 | 53.95 | 199.84 |

### What is History Matching?
In reservoir engineering, **history matching** is the process of adjusting uncertain model parameters until the simulation output matches observed field data (oil/gas/water production rates). A well-matched model is trusted for future forecasting. Here, ML replaces the manual iterative process.

---

## 2. Dataset Inventory

### File Summary

| File | Sheets | Description |
| --- | --- | --- |
| `01 Introduction.xlsx` | 2 | Property nomenclature + 100-case uncertainty parameters |
| `02 Field Production.xlsx` | 7 | Observed history + simulation time series for all 100 cases |
| `03 Train Cases 1.xlsx` | 20 | Grid-cell data for Cases 1–20 |
| `04 Train Cases 2.xlsx` | 20 | Grid-cell data for Cases 21–40 |
| `05 Train Cases 3.xlsx` | 20 | Grid-cell data for Cases 41–60 |
| `06 Train Cases 4.xlsx` | 10 | Grid-cell data for Cases 61–70 |
| `07 Validation Cases.xlsx` | 15 | Grid-cell data for Cases 71–85 |
| `08 Test Cases.xlsx` | 15 | Grid-cell data for Cases 86–100 |
| `09 Template Deliverable.xlsx` | 10 | Empty output template (to populate with predictions) |

---

### 01 Introduction.xlsx

#### Sheet: Nomenclature (32 rows × 4 cols)
Defines all 29 grid-cell properties. See Section 4 for full details.

#### Sheet: Uncertainty Values (100 rows × 5 cols)
One row per simulation case. Columns:
- `Item` — Case identifier ("Case 1" through "Case 100")
- `Fault Transmissibility` — float
- `Porosity Multiplier` — float
- `Permeability Multiplier` — float
- `Aquifer Pore Volume` — float

---

### 02 Field Production.xlsx

#### Sheet: Field Production (41 rows × 6 cols)
**The observed field history** — the ground truth for history matching.

| Column | Unit | Min | Max | Last Value (2008-01-01) |
| --- | --- | --- | --- | --- |
| Identifier | — | "Field" | "Field" | "Field" |
| Date | — | 1998-01-01 | 2008-01-01 | 2008-01-01 |
| Oil production rate | STB/d | 0.00 | 15,507.77 | 5,066.68 |
| Gas production rate | MSCF/d | 0.00 | 5,633.97 | 1,838.96 |
| Water production rate | STB/d | 0.00 | 7,926.36 | 7,926.36 |
| Water injection rate | STB/d | 0.00 | 14,748.87 | 9,326.61 |

Quarterly resolution (every 3 months). Production begins 1998-10-01. First 3 timesteps are zero (pre-production).

**Key observations:**
- Oil rate declines from peak ~15,500 STB/d (1998) to ~5,067 STB/d (2008) — **67% decline**
- Water rate rises continuously from 0 to ~7,926 STB/d — typical waterflood behavior
- Gas rate declines proportionally with oil (constant GOR ≈ 0.363 MSCF/STB)
- Water injection rate decreases from ~14,749 to ~9,327 STB/d over time

#### Sheets: Production Time Series (6 sheets, each 11,098 rows × 102 cols)
**Simulation outputs** for all 100 cases + observed history (RAMP_History).

Sheets: `Gas production cumulative`, `Gas production rate`, `Oil production cumulative`, `Oil production rate`, `Water production cumulative`, `Water production rate`

Structure:
- **Row 0**: Case names ("Case_1" through "Case_100" + "RAMP_History")
- **Row 1**: Metric labels (e.g., "Oil production cumulative [STB]")
- **Rows 2–11,097**: Daily timestep data (~11,096 data rows)
- **Column X**: Date (1998-01-01 to 2008-01-01)
- **Columns Y1–Y100**: Simulation output per case
- **Column Y101**: RAMP_History (the reference/observed history at daily resolution)

Final cumulative values at 2008-01-01:

| Metric | Min across cases | Max across cases | Mean | RAMP_History |
| --- | --- | --- | --- | --- |
| Oil cumulative (STB) | ~35.2M | ~39.5M | ~37.5M | 38.4M |
| Gas cumulative (MSCF) | ~10.9M | ~14.6M | ~13.1M | — |
| Water cumulative (STB) | ~11.0M | ~19.1M | ~14.7M | — |

---

### 03–08 Train/Validation/Test Cases

Each case = 1 Excel sheet with **90,365 rows × 29 columns** of grid-cell-level data.

| File | Cases | Purpose | Count |
| --- | --- | --- | --- |
| 03 Train Cases 1 | Case 1 – Case 20 | Training | 20 |
| 04 Train Cases 2 | Case 21 – Case 40 | Training | 20 |
| 05 Train Cases 3 | Case 41 – Case 60 | Training | 20 |
| 06 Train Cases 4 | Case 61 – Case 70 | Training | 10 |
| 07 Validation Cases | Case 71 – Case 85 | Validation | 15 |
| 08 Test Cases | Case 86 – Case 100 | Test | 15 |
| **Total** | | | **100** |

---

### 09 Template Deliverable.xlsx
**The output you must submit.** 10 sheets (Case 1 through Case 10) — these are your "best 10 forecast cases."

| Column | Type |
| --- | --- |
| Date | Quarterly dates (MM/DD/YYYY) |
| Gas production cumulative [MSCF] | float — YOUR PREDICTION |
| Oil production cumulative [STB] | float — YOUR PREDICTION |
| Water production cumulative [STB] | float — YOUR PREDICTION |

| Sheet | Rows | Forecast Period |
| --- | --- | --- |
| Case 1 | 40 | 04/01/2008 → 01/01/2018 (10 years) |
| Cases 2–10 | 80 each | 04/01/2008 → 01/01/2028 (20 years) |

All values are NaN — must be populated with your model's predictions.

---

## 3. Data Architecture

### Case Split

```
Train:      Cases 1–70   (70 cases)  — learn the mapping
Validation: Cases 71–85  (15 cases)  — tune hyperparameters
Test:       Cases 86–100 (15 cases)  — final evaluation
```

### 3D Grid Structure

```
I dimension: 1 – 53  (53 cells in x-direction)
J dimension: 1 – 55  (55 cells in y-direction)
K dimension: 1 – 31  (31 layers in z-direction, depth)

Total grid cells per case: 53 × 55 × 31 = 90,365
Active cells:   42,512 (47.0%)
Inactive cells: 47,853 (53.0%)  — marked with sentinel -999.25
```

The grid is **identical across all 100 cases** (same I, J, K structure, same active cell count). What changes between cases are the **property values** due to different uncertainty multipliers.

Coordinate system:
- X: 550,348 – 560,443 (meters, ~10 km field extent E-W)
- Y: 6,798,565 – 6,809,365 (meters, ~11 km field extent N-S)
- Z: -9,156 – -7,572 (meters below sea level, ~1,584m reservoir thickness)

### Time Axes

| Context | Resolution | Period | Timesteps |
| --- | --- | --- | --- |
| Field Production (observed) | Quarterly | 1998-01-01 → 2008-01-01 | 41 |
| Simulation time series (02 sheets) | ~Daily | 1998-01-01 → 2008-01-01 | 11,096 |
| Template output (forecast) | Quarterly | 04/01/2008 → 01/01/2018 or 2028 | 40 or 80 |

**The simulation runs only cover the 10-year history period.** Your ML model must extrapolate beyond 2008.

### Data Flow

```
4 Uncertainty Parameters  ──►  Grid Properties (90K cells × 29 features)  ──►  Production Curves
(per case)                     (per case, from 03-08 files)                    (per case, from 02 file)
```

The uncertainty parameters act as **global multipliers/scalars** that modify the base reservoir properties. Different parameter combinations produce different grid property distributions, which produce different production behavior when simulated.

---

## 4. Input Features Deep Dive

### Grid Cell Properties (29 columns per cell)

#### Spatial Indices (not features, used for cell location)

| Column | Description |
| --- | --- |
| I Index | Grid block index in x-direction (1–53) |
| J Index | Grid block index in y-direction (1–55) |
| K Index | Grid block index in z-direction/depth (1–31) |
| X Coordinate | Easting in meters |
| Y Coordinate | Northing in meters |
| Z Coordinate | Depth below sea level (negative meters) |

#### Static Rock Properties (do NOT change with time)

| Column | Abbrev | Unit | Description | Active Cell Range |
| --- | --- | --- | --- | --- |
| Pore Volume | PORV | reservoir barrels | Volume of pore space at reference P&T | 0.63 – 2,426,922 |
| Porosity | PORO | fraction (0–1) | Fraction of rock volume that is pore space | 0.039 – 0.199 |
| Permeability I | PERMX | millidarcies | How easily fluid flows in x-direction | 0.15 – 433.07 |
| Permeability J | PERMY | millidarcies | How easily fluid flows in y-direction | 0.15 – 433.07 |
| Permeability K | PERMZ | millidarcies | How easily fluid flows vertically | 0.05 – 216.54 |
| Net-to-Gross | NTG | fraction (0–1) | Fraction of gross rock that is net reservoir | 1.00 (constant) |
| Transmissibility I | TRANX | cP·rb/d/psi | Flow capacity between cells in x | 0.00 – 24.66 |
| Transmissibility J | TRANY | cP·rb/d/psi | Flow capacity between cells in y | 0.00 – 13.19 |
| Transmissibility K | TRANZ | cP·rb/d/psi | Flow capacity between cells in z | 0.00 – 1,008.11 |

#### Region & Endpoint Properties (static, categorical/bounded)

| Column | Abbrev | Description | Values |
| --- | --- | --- | --- |
| Fluid In Place Region | FIPNUM | Region with distinct fluid volume accounting | 1 or 2 |
| PVT Region | PVTNUM | Region with distinct pressure-volume-temperature fluid behavior | 1 or 2 |
| Connate Water Saturation | SWL | Minimum irreducible water saturation | 0.20 – 0.27 |
| Critical Water Saturation | SWCR | Minimum water saturation for water to flow | 0.22 – 0.35 |
| Maximum Water Saturation | SWU | Maximum possible water saturation | 1.00 (constant) |
| Connate Gas Saturation | SGL | Minimum trapped gas saturation | 0.00 (constant) |
| Critical Gas Saturation | SGCR | Minimum gas saturation for gas to flow | 0.08 – 0.14 |
| Residual Oil to Water | SOWCR | Oil left behind after water displacement | 0.18 – 0.27 |

#### Dynamic Properties (state at initial conditions, t=0)

| Column | Abbrev | Unit | Description | Active Cell Range |
| --- | --- | --- | --- | --- |
| Pressure | PRESSURE | psi | Initial reservoir pressure | 4,247 – 4,892 |
| Water Saturation | SWAT | fraction | Fraction of pore space filled with water | 0.20 – 1.00 |
| Oil Saturation | SOIL | fraction | Fraction of pore space filled with oil | 0.00 – 0.80 |
| Gas Saturation | SGAS | fraction | Fraction of pore space filled with gas | 0.00 (constant — undersaturated reservoir) |
| Solution Gas-Oil Ratio | RS | MSCF/STB | Dissolved gas per unit oil | 0.3633 (constant) |
| Vapor Oil-Gas Ratio | RV | STB/MSCF | Oil dissolved in gas phase | 0.00 (constant — no free gas) |

**Key insight:** This is an **undersaturated oil reservoir** with water drive:
- No free gas (SGAS = 0 everywhere, RS constant)
- Water saturation varies 0.20–1.00 (oil-water system)
- Active water injection (waterflood)
- Gas production comes only from solution gas

---

## 5. Target Variables

### What to predict
For 10 selected forecast cases, predict **cumulative production** at quarterly timesteps:

| Target | Unit | Typical History-End Value |
| --- | --- | --- |
| Gas production cumulative | MSCF | 10.9M – 14.6M |
| Oil production cumulative | STB | 35.2M – 39.5M |
| Water production cumulative | STB | 11.0M – 19.1M |

### Forecast Horizons
- **Case 1**: 10-year forecast (2008 → 2018), 40 quarterly timesteps
- **Cases 2–10**: 20-year forecast (2008 → 2028), 80 quarterly timesteps each

### Important Notes
- Values are **cumulative** (monotonically increasing), not rates
- The forecast starts where history ends (2008-01-01)
- You must select which 10 of your cases to submit as the "best" forecasts
- Quarterly resolution (Jan, Apr, Jul, Oct each year)

---

## 6. Data Quality Notes

### Sentinel Values
- **-999.25** marks inactive/null grid cells across all 29 property columns
- 53% of all grid cells are inactive (47,853 out of 90,365)
- Filter these out before any analysis: `df[df['Porosity (PORO)'] != -999.25]`
- Active cell count is **consistent** across all 100 cases (always 42,512)

### Missing Data
- No NaN values in the grid data — inactive cells use the sentinel instead
- Template deliverable has all-NaN target columns (by design — you fill these)
- Field Production sheet has no missing values

### Date Format Inconsistencies
- `02 Field Production.xlsx` → dates as datetime objects (e.g., `1998-01-01 00:00:00`)
- `09 Template Deliverable.xlsx` → mixed format: some strings (`"04/01/2008"`) and some datetime (`2028-01-01 00:00:00`)
- **Normalize all dates to `datetime` on load**

### Production Time Series Header Structure (02 file)
The 6 production time-series sheets have a **2-row header** that must be handled:
- Row 0: Case identifiers (`Case_1`, `Case_2`, ..., `Case_100`, `RAMP_History`)
- Row 1: Metric labels (e.g., `Oil production cumulative [STB]`)
- Row 2 onward: Actual numeric data
- Column 0 (`X`): Date values

**To parse correctly:** skip the first 2 rows as data, or read with `header=[0,1]` for multi-level columns.

### Column Name Quirks
- Column ` Connateg  Gas Saturation (SGL)` has leading space and double space (typo in original data)
- PERMX = PERMY in Case 1 data (isotropic horizontal permeability): verify if this holds for all cases
- NTG, SWU, SGL, SGAS, RS, RV are constant across all active cells in Case 1 — may be constant across all cases

---

## 7. Suggested ML Approaches

### Why This is Hard
1. **Input dimensionality**: 42,512 active cells × 29 features = ~1.2M features per case, but only 70 training examples
2. **Spatial structure**: The 3D grid has physical adjacency that matters for fluid flow
3. **Temporal extrapolation**: Training data covers 10 years; must predict 10–20 years beyond
4. **Cumulative targets**: Production curves are smooth, monotonically increasing — this is helpful

### Approach 1: Uncertainty Parameters → Production (Direct Surrogate)
The simplest path — ignore the 90K grid cells entirely and map the 4 uncertainty parameters directly to production curves.

**Input:** 4 uncertainty values per case
**Output:** Production time series (cumulative oil, gas, water)
**Methods:** Gaussian Process Regression, Neural Network, LSTM, polynomial regression
**Pros:** Simple, fast, only 4 input features
**Cons:** Ignores spatial heterogeneity in the grid data

### Approach 2: Grid Summary Statistics → Production
Compress each case's 42,512 × 29 grid into summary statistics (mean, std, percentiles of each property), then map to production.

**Input:** ~100-300 summary features per case
**Output:** Production curves
**Methods:** XGBoost, Random Forest, MLP, or LSTM (for time-series output)

### Approach 3: Graph Neural Network (GNN) + LSTM
Treat the 3D grid as a graph (cells = nodes, adjacency = edges based on I/J/K neighbors), encode with GNN, then decode production with LSTM.

**Referenced paper:** Hu Huang et al., "A Deep-Learning-Based GNN-LSTM Model for Reservoir Simulation", SPE 2023

### Approach 4: Autoencoder for Grid Compression
Train an autoencoder to compress the 42K-cell grid into a low-dimensional latent vector, then map latent + uncertainty params to production.

### Approach 5: Dynamic Mode Decomposition (DMD)
A data-driven technique for extracting dynamic modes from time-series data. Treat each case's production curves as a dynamical system.

### Recommended Starting Point
**Approach 1 or 2** for rapid baseline, then **Approach 3 or 4** for competitive accuracy. The 70-case training set is small, so simpler models may outperform complex ones. Feature engineering from the grid data (spatial aggregates, property distributions, connectivity metrics) is likely more impactful than architecture complexity.

---

## 8. Quick Reference

### Key Numbers

| Item | Value |
| --- | --- |
| Total simulation cases | 100 |
| Train / Val / Test split | 70 / 15 / 15 |
| Grid dimensions (I × J × K) | 53 × 55 × 31 |
| Total cells per case | 90,365 |
| Active cells per case | 42,512 (47%) |
| Properties per cell | 29 (6 spatial + 9 static + 8 endpoint + 6 dynamic) |
| Uncertainty parameters | 4 |
| History period | 1998-01-01 → 2008-01-01 (10 years) |
| History resolution | ~11,096 daily + 41 quarterly |
| Forecast period | 2008 → 2018 (Case 1) or 2028 (Cases 2-10) |
| Forecast resolution | Quarterly (40 or 80 timesteps) |
| Target variables | 3 (cumulative oil, gas, water) |
| Output template | 10 cases |

### File Paths
```
/MWAP/01 Introduction.xlsx          → Uncertainty parameters
/MWAP/02 Field Production.xlsx      → Observed + simulated production
/MWAP/03 Train Cases 1.xlsx         → Grid data Cases 1–20
/MWAP/04 Train Cases 2.xlsx         → Grid data Cases 21–40
/MWAP/05 Train Cases 3.xlsx         → Grid data Cases 41–60
/MWAP/06 Train Cases 4.xlsx         → Grid data Cases 61–70
/MWAP/07 Validation Cases.xlsx      → Grid data Cases 71–85
/MWAP/08 Test Cases.xlsx            → Grid data Cases 86–100
/MWAP/09 Template Deliverable.xlsx  → Output template (fill with predictions)
```
