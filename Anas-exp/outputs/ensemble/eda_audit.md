# Comprehensive EDA Audit Report

**Location:** `Anas-exp/outputs/ensemble/eda_audit.md`  
**Generated:** 2026-10-09  
**Auditor:** Antigravity AI Engineering Assistant  
**Task:** Independent Verification of EDA Findings for Leakage-Safe Stacked Ensemble Modeling  

---

## 1. Verified Dataset Characteristics & Dimensions

### 1.1 Independent Sample Size vs Expanded Time-Series
- **Training Cases:** Exactly **70 independent cases** (Cases 1 to 70).
- **Validation Cases:** Exactly **15 independent cases** (Cases 71 to 85).
- **Blind Test Cases:** Exactly **15 independent cases** (Cases 86 to 100).
- **Observed Field History:** Exactly **1 case** (`Case 0`, `RAMP_History`, 1998-01-01 to 2008-01-01).
- **Independent Experimental Unit:** $N = 70$ simulation runs.
- **Methodological Warning:** While `production_timeseries.parquet` contains **6,724,340 rows** and each case has **121 quarterly timestamps** across 30 years, individual quarterly rows are **not independent observations**. They are autocorrelated observations from a single deterministic simulator execution. Cross-validation MUST be strictly partitioned at the **case level** ($N=70$).

### 1.2 Static Input Uncertainty Parameters
Exactly **4 scalar parameters** characterize each simulation run:
1. `Porosity Multiplier`: Mean 1.1345, Range [0.8012, 1.4940]
2. `Permeability Multiplier`: Mean 5.2930, Range [0.5375, 9.5020] (log-uniform, spans 2 orders of magnitude)
3. `Fault Transmissibility`: Mean 0.0993, Range [0.050790, 0.1500] (log-uniform)
4. `Aquifer Pore Volume`: Mean 127.2388, Range [53.9529, 199.8352] (positively skewed)

### 1.3 3D Petrophysical Grid Redundancy Audit
In `02 EDA.ipynb` Cell 13, aggregate properties from the 42,512 reservoir grid cells correlate at:
- Pearson $r(\text{PORV\_total}, \text{Porosity Multiplier}) = \mathbf{1.0000}$
- Pearson $r(\bar{K}_{geom}, \text{Permeability Multiplier}) = \mathbf{1.0000}$
**Verification:** The 3D grid properties are exact affine scalings of a single static base grid. The 3D grid contains zero independent stochastic realizations. Effective input dimensionality is strictly 4.

---

## 2. Independent Numerical Verification of EDA Claims

### 2.1 Parameter-to-Endpoint Spearman Correlations (Training Cases 1-70)
| Reservoir Uncertainty Parameter | Oil Cumulative $\rho$ | Gas Cumulative $\rho$ | Water Cumulative $\rho$ |
| :--- | :---: | :---: | :---: |
| `Porosity Multiplier` | **+0.9931** | **+0.9931** | **-0.9941** |
| `Permeability Multiplier` | **+0.0026** | **+0.0026** | **+0.0926** |
| `Fault Transmissibility` | **+0.1976** | **+0.1976** | **-0.2024** |
| `Aquifer Pore Volume` | **-0.2206** | **-0.2206** | **+0.2906** |

**Verification & Critical Nuance:**
- Porosity Multiplier alone explains $>98\%$ of cumulative production variance.
- Permeability Multiplier has near-zero correlation with cumulative endpoints ($\rho = +0.0026$).
- **CRITICAL AUDIT FINDING:** Weak endpoint correlation does **NOT** mean permeability is uninformative for intermediate trajectory dynamics. In `02 Temporal EDA` Cell 06 (`04_inputs_vs_temporal_parameters.csv`), permeability correlates moderately with intermediate liquid rates ($q_{liquid}$, $\rho = +0.4275$ at 2008), GOR ($\rho = +0.3946$ at 2003), and liquid decline rate ($D_{liquid}$, $\rho = -0.2718$). Models predicting intermediate rate profiles must retain permeability.

### 2.2 SVD / PCA Explained Variance: Raw vs Shape-Normalized Curves
| Metric | Raw Centered Comp 1 | Raw Centered Comp 2 | Shape-Normalized Comp 1 | Shape-Normalized Comp 2 | Shape-Normalized Comp 3 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `oil_cum` | **99.954%** | 0.034% | **99.864%** | 0.072% | 0.037% |
| `gas_cum` | **99.957%** | 0.032% | **99.871%** | 0.068% | 0.036% |
| `water_cum` | **99.910%** | 0.074% | **99.146%** | 0.756% | 0.064% |

**Verification & Interpretation:**
- In raw cumulative curves, Component 1 explains **99.95%** of oil variance.
- When all curves are normalized by their individual final cumulative endpoint (strictly isolating trajectory shape from magnitude), Component 1 still explains **99.86%** of the variance, and Components 1 + 2 explain **99.94%**.
- **Conclusion:** The low-rank structure is **genuine**. It is not merely an artifact of production scale. The underlying physical trajectory shape follows a boundary-dominated decline mode across all cases.

### 2.3 Rate Curve SVD Spectrum
| Metric | Component 1 | Component 2 | Cumulative (1 + 2) |
| :--- | :---: | :---: | :---: |
| `oil_rate` | **98.81%** | 0.73% | **99.54%** |
| `gas_rate` | **98.83%** | 0.73% | **99.57%** |
| `water_rate` | **98.05%** | 1.28% | **99.33%** |

Derived rate curves also exhibit strong low-rank structure ($>98\%$ in Component 1), confirming that complex, multi-state neural networks are prone to severe overfitting on this dataset.

---

## 3. Audit of Temporal Mechanics & Stability Findings

### 3.1 Free-$b$ Arps Collapse vs Strict $b=0$ Exponential
- In `02 Temporal EDA` Cell 07 (`05_arps_b_diagnostic.csv`), 100% of cases at 2003 cutoff fitted $b < 0.01$ (median $b = 1.04 \times 10^{-7}$) with a Jacobian condition number of $6.37 \times 10^{15}$.
- In `03 Timeline Mechanics` Cell 05 (`timeline_matched_ablation_summary.csv`), Free-$b$ Arps completed **$0/12$ cases (100% failure)** on 2003 (3y and 5y) backtests, and only $2/12$ cases on 2004 (3y).
- In contrast, Strict Exponential ($b=0$) completed **$12/12$ cases (100% convergence)** across all origins, achieving $0.0411$ NRMSE on 2005 (3y) vs $0.0763$ for Free-$b$.
- On 2028 extrapolations, Free-$b$ over-predicted cumulative oil by **$+29.50\%$** on average (up to $+32.57\%$).
- **Conclusion:** Free-$b$ Arps is completely disqualified. Strict exponential ($b=0$) is the only viable parametric formulation.

### 3.2 Decoder Architecture Comparison (114,240 Crossed Backtest Rows)
- Across 4 experiments:
  - `Separate` Decoder: Mean development NRMSE = **$0.1525$** (Worst experiment = $0.2935$).
  - `Liquid-WC-volume`: Mean development NRMSE = **$0.2117$** (Worst experiment = $0.4448$).
  - `Liquid-WC-time`: Mean development NRMSE = **$0.2225$** (Worst experiment = $0.4392$).
  - `Constant`: Mean development NRMSE = **$0.2757$** (Worst experiment = $0.4037$).
- **Statistical Significance:** Paired comparisons (`06_paired_decoder_comparison.csv`) show `Separate` beats `Constant` in **$89.8\%$ of cases**, beats `Liquid-WC-time` in **$63.7\%$ of cases**, and beats `Liquid-WC-volume` in **$56.0\%$ of cases**.
- **Conclusion:** Decoupled independent-phase modeling is superior because coupled liquid/water-cut decoders leak early breakthrough noise into oil and gas predictions.

### 3.3 Water-Rate Capping Impact
- In `03 Timeline Mechanics` Cell 08, unconstrained water extrapolations reach up to **$16,000,000$ STB/day** (observed 2008 field rate = $7,926$ STB/d), resulting in cumulative water production of **$110.94\text{M}$ STB**.
- Imposing a $2.0\times$ observed rate cap ($15,853$ STB/day) reduces 2028 cumulative water to **$16.43\text{M}$ STB** (**$85.26\%$ reduction**).
- **Critical Audit Distinction:** The $2\times$ water rate cap only binds on forward 20-year extrapolations (2008–2028); historical water rates prior to 2008 never exceeded the cap. It functions strictly as an extrapolation stabilizer, not a historical curve adjuster.

---

## 4. Modeling Implications for Ensemble Architecture

1. **Strict Case-Level Cross-Validation:** Folds must be grouped by `case_num` across all 70 training cases. No preprocessing, scaling, SVD basis, or meta-learner fitting may see held-out cases.
2. **Feature Set for Static Surrogate (Task A):**
   - Inputs: 4 scalar parameters (`Porosity Multiplier`, `Permeability Multiplier`, `Fault Transmissibility`, `Aquifer Pore Volume`).
   - Discard 3D grid properties as 100% redundant.
3. **Candidate Family Roles:**
   - **Family 1 (Parametric Decline):** Separate exponential ($b=0$) anchored to historical terminal rate and cumulative total.
   - **Family 2 (Low-Rank Functional Regression):** SVD/PCA basis fitted inside training folds, with Ridge regression mapping 4 parameters to component weights.
   - **Family 3 (Gaussian Process Regression):** ARD Matérn GP mapping 4 parameters to curve coefficients with analytical uncertainty estimates.
   - **Family 4 (Regularized Nonlinear Regression):** Polynomial Ridge (degree 2) and compact regularized MLP (4 -> 8 -> 2) to test if non-linear interactions add value.
4. **Physical Constraint Layer:**
   - Universal non-negativity and monotonic cumulative accumulation.
   - Exact historical anchor preservation at cutoff date: $N_p(t) = N_p(t_0) + \Delta N_p(t)$.
   - Water rate ceiling at $2.0\times$ observed rate.
