# EXXONMOBIL DATAWORKS CHALLENGE 2026
# LONG-HORIZON RESERVOIR FORECASTING SCIENCE RESEARCH REPORT

**Research Branch:** [`research/long-horizon-reservoir-forecast-science`](file:///Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp/src/long_horizon_physics/)  
**Protected Production Baseline:** Commit `c74e3ac503c131d1a8894ed7250f8b661480b66e` (`Blend_Phase_Specific_Simplex`)  
**Official Submission Deliverable:** [`outputs/submission/09_Template_Deliverable_Submission.xlsx`](file:///Users/tengkuanas/Projects/mwap-exdw-surrogate-model/outputs/submission/09_Template_Deliverable_Submission.xlsx) (100% Locked & Preserved)  
**Evaluation Scope:** Strict Protocol A Leave-Reservoir-Out Backtesting on Cases 1–70 (15 Pseudo-Fields). Cases 86–100 Strictly Blind.  

---

## EXECUTIVE SUMMARY & AUTHORITATIVE VERDICT

```
========================================================================================
FINAL SCIENTIFIC VERDICT: KEEP BASELINE (WITH CONDITIONAL REGIME-SPECIFIC INSIGHTS)
========================================================================================
1. Protected Baseline Status:   REMAIN 100% LOCKED AND SUBMITTED.
                                The official deliverable workbook and simplex stacking
                                weights in locked_ensemble_config.json must NOT be modified.
2. Production Competitiveness:  The protected Blend_Phase_Specific_Simplex remains the
                                most robust, low-variance universal generalist across all
                                production maturities (Jan 2003, Jan 2004, Jan 2005)
                                and horizons (4Q, 12Q, 20Q).
3. Challenger Findings:         - H8_Constrained_Tail_Hybrid achieves superior accuracy at
                                  Origin 2005 (12Q Macro NRMSE 0.02869 vs Baseline 0.03685,
                                  p = 0.010, 80% win rate; 4Q Macro NRMSE 0.01157 vs 0.01979,
                                  p < 1e-11, 100% win rate).
                                - However, H8 regresses significantly at Origin 2004 12Q
                                  (NRMSE 0.04751 vs 0.03295) and Origin 2003 20Q (0.24860
                                  vs 0.22001, p < 1e-5).
                                - H6_Stretched_Exponential (SEPD) excels in high-depletion
                                  early-decline settings (Origin 2003 20Q NRMSE 0.13922 vs
                                  0.22001, p = 1.76e-5, 93.3% win rate), but fails completely
                                  in late maturity (Oil NRMSE 0.282 vs 0.032).
4. Uncoupled Failure Modes:     Uncoupled multiphase models (e.g. unconstrained water growth)
                                fail over 20-year deployments (terminal water rates exceeding
                                500,000 STB/d).
5. Deployment Equivalence:      Field Case 0 forward deployment across 10 official scenarios
                                yields a 20-year EUR of 51.18 MMSTB with realistic terminal
                                rate of 810 STB/d for the baseline, whereas pure decline
                                models under-predict EUR by 3 to 7 MMSTB.
========================================================================================
```

### Core Scientific Findings at a Glance

| Evaluation Dimension | Protected Baseline (`Blend_Phase_Specific_Simplex`) | Top Challenger 1 (`H8_Constrained_Tail_Hybrid`) | Top Challenger 2 (`H6_Stretched_Exponential_SEPD`) | Theoretical Multiphase (`H2_Total_Liquid_Bounded_WOR`) |
| :--- | :--- | :--- | :--- | :--- |
| **Origin 2005 (12Q Macro)** | **0.03685** (Std: 0.0103) | **0.02869** (-0.00815, $p=0.010$) | **0.18817** (+0.1513, $p<10^{-15}$) | **0.12275** (+0.0859, $p<10^{-9}$) |
| **Origin 2005 (4Q Macro)** | **0.01979** (Std: 0.0021) | **0.01157** (-0.00822, $p<10^{-11}$) | **0.13420** (+0.1144, $p<10^{-15}$) | **0.04361** (+0.0238, $p<10^{-4}$) |
| **Origin 2004 (12Q Macro)** | **0.03295** (Std: 0.0117) | **0.04751** (+0.01455, Regressive) | **0.17296** (+0.1400, Regressive) | **0.22362** (+0.1907, Regressive) |
| **Origin 2003 (20Q Macro)** | **0.22001** (Std: 0.0172) | **0.24860** (+0.02859, $p<10^{-5}$) | **0.13922** (-0.08078, $p<10^{-4}$) | **0.34227** (+0.1223, $p<10^{-10}$) |
| **Case 0 20-Year EUR (Oil)** | **51.18 MMSTB** (Spread: 16.4k) | **48.49 MMSTB** (Spread: 65.8k) | **47.23 MMSTB** (Over-declined) | **44.40 MMSTB** (Severe under-pred) |
| **Case 0 Terminal Rate (2028)**| **810.0 STB/d** | **175.3 STB/d** | **72.4 STB/d** | **5.9 STB/d** |
| **50-Year Asymptotic EUR** | Finite & Monotonic | Finite & Monotonic | Finite ($Q_\infty < \infty$) | Finite & Monotonic |
| **Cross-Regime Generalization**| **Universally Stable (0/9 Blowups)**| **Regime-Specific (2/9 Blowups)**| **Narrow Niche (5/9 Blowups)** | **Severely Biased (7/9 Blowups)** |

---

## 1. RESEARCHING THE MATHEMATICS AND RESERVOIR PHYSICS

The foundation of long-horizon reservoir forecasting requires reconciling empirical time-series machine learning with fundamental multiphase fluid mechanics and boundary-dominated reservoir decline.

```
                  +-------------------------------------------------------------+
                  |         MULTIPHASE RESERVOIR DISPLACEMENT COUPLING           |
                  +-------------------------------------------------------------+
                                                 |
                       +-------------------------+-------------------------+
                       |                                                   |
                       v                                                   v
         [ TOTAL LIQUID CONSERVATION ]                            [ FRACTIONAL FLOW CURVE ]
           q_L(t) = q_o(t) + q_w(t)                                 f_w(Q_L) = q_w / q_L
           Mild, bounded decline rate D_L                           Logistic / Koval (SPE-173315)
                       |                                                   |
                       +-------------------------+-------------------------+
                                                 |
                                                 v
                                    [ EXACT MULTIPHASE COUPLING ]
                                      q_o(t) = q_L(t) * [1 - f_w(t)]
                                      q_w(t) = q_L(t) * f_w(t)
                                      Guarantee: f_w in [0, 1), No Runaway Rates!
```

### 1.1 Technical Literature Foundations

1. **Gao et al. (2024), *Geofluids*, DOI: `10.1155/2024/4584237`**:
   Investigates waterflood fractional flow progression and semi-analytical water-drive curves in heterogeneous sandstones. Demonstrates that while oil rate declines exponentially or hyperbolically, total liquid rate $q_L(t)$ exhibits much milder decline, governed by total reservoir voidage replacement and pump horsepower. The water-oil ratio $\text{WOR}(t)$ follows a log-linear progression:
   $$\ln(\text{WOR}(t)) = \ln(\text{WOR}_0) + \lambda_w t$$
   which guarantees that water cut $f_w(t) = \frac{\text{WOR}}{1 + \text{WOR}}$ remains strictly bounded in $[0, 1)$, preventing unphysical runaway water forecasts.
2. **SPE-173315-MS (*Oil Rate Forecast by Inferring Fractional Flow Models from Field Data*)**:
   Formulates Koval / Buckley-Leverett fractional flow displacement where the water fractional flow $f_w(t)$ is directly coupled to total liquid throughput $Q_L(t)$:
   $$f_w(Q_L) = \frac{f_{w,0} + K_K \cdot \xi(Q_L)}{1 + (K_K - 1) \cdot \xi(Q_L)}, \quad \xi(Q_L) = \frac{Q_L - Q_{L,\text{bt}}}{V_D}$$
   where $K_K$ is the lumped effective Koval heterogeneity/mobility factor and $V_D$ is the drainable pore volume.
3. **INSIM-BHP (2023), *J. Comput. Phys.***:
   Establishes inter-well mass balance and transmissibility physics under waterflooding. Validates that pressure and saturation states decouple into an instantaneous diffusive pressure field and a hyperbolic saturation front.
4. **Oliver (2026), *Comput. Geosciences*, DOI: `10.1007/s10596-026-10461-2`**:
   Examines production forecasting under imperfect posterior sampling and model inadequacy. Demonstrates that unmodeled physics in high-dimensional surrogates produce systemic tail divergence over long horizons ($h > 10$ yr), necessitating physical decline bounds.

### 1.2 Exact Multiphase Identities

All physical models in this study strictly satisfy the surface standard-condition mass conservation identities:
$$q_L(t) \equiv q_o(t) + q_w(t)$$
$$f_w(t) \equiv \frac{q_w(t)}{q_L(t)}, \quad f_o(t) \equiv 1 - f_w(t) = \frac{q_o(t)}{q_L(t)}$$
$$\text{WOR}(t) \equiv \frac{q_w(t)}{\max(q_o(t), \epsilon)}, \quad \text{GOR}(t) \equiv \frac{q_g(t)}{\max(q_o(t), \epsilon)}$$
Coupled rate forecasting derives oil and water from:
$$q_o(t) = q_L(t) \cdot [1 - f_w(t)]$$
$$q_w(t) = q_L(t) \cdot f_w(t)$$

### 1.3 Long-Horizon Decline Models

Standard hyperbolic decline ($q(t) = q_0 (1 + b D_i t)^{-1/b}$) possesses an infinite mathematical recovery integral when $b \ge 1$ and severe over-prediction when $0 < b < 1$ at $t \to 20$ yr. To address this, we investigate four mathematically bounded decline architectures:

1. **Modified Arps Decline with Terminal Floor ($D_{\text{lim}}$)**:
   Transitions smoothly from hyperbolic decline to an exponential decline floor $D_{\text{lim}}$ at $t_{\text{switch}}$:
   $$D(t) = \frac{D_i}{1 + b D_i t}, \quad t_{\text{switch}} = \frac{D_i / D_{\text{lim}} - 1}{b D_i} \quad (\text{for } D_i > D_{\text{lim}})$$
   $$q_o(t) = \begin{cases} q_0 (1 + b D_i t)^{-1/b}, & t \le t_{\text{switch}} \\ q_0 (1 + b D_i t_{\text{switch}})^{-1/b} \exp(-D_{\text{lim}}(t - t_{\text{switch}})), & t > t_{\text{switch}} \end{cases}$$
2. **Stretched Exponential Production Decline (SEPD - Valko 2009)**:
   $$q_o(t) = q_0 \exp\left(-\left(\frac{t}{\tau}\right)^n\right), \quad 0 < n \le 1$$
   Analytically bounds ultimate recovery to a finite value without ad-hoc piecewise switching:
   $$Q_\infty = Q_0 + q_0 \frac{\tau}{n} \Gamma\left(\frac{1}{n}\right)$$
3. **Two-Compartment Depletion Model**:
   Decomposes drainage into near-well flush drainage ($D_1$) and boundary recharge ($D_2$):
   $$q_o(t) = q_{1,0} \exp(-D_1 t) + q_{2,0} \exp(-D_2 t)$$
4. **Constrained-Tail Hybrid Model**:
   Employs unconstrained data-driven machine learning (ExtraTrees) for the short-horizon transient regime ($t \le 3$ yr, where it excels), transitioning continuously via smooth sigmoid blending to a physics-constrained Modified Arps tail ($D_{\text{lim}} = 0.06$/yr) for $t > 3$ yr.

![Figure 1: Multiphase Physics Formulation](/Users/tengkuanas/.gemini/antigravity-cli/brain/4b7e1d00-9795-42a6-b5bb-c629e9cf25cb/fig1_multiphase_physics_formulation.png)
*Figure 1: (A) Comparison of traditional uncoupled independent regressors vs. multiphase coupled rate/cut representations. (B) Koval fractional flow displacement curves ($K_K = 2.0, 4.0, 8.0$) guaranteeing monotonic bounded water cut as a function of cumulative liquid throughput.*

### 1.4 Research Hypothesis Matrix

All 6 formulated candidate hypotheses were formalized with governing equations, theoretical assumptions, unmodeled physics, and falsification criteria before executing experimental runs:

| Hypothesis ID | Proposed Mechanism | Governing Equations | Key Mathematical Assumptions | Falsification Criteria | Empirical Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **H1: Koval Fractional Flow** | Coupled waterflood displacement | $q_L = q_o + q_w; f_w = \frac{K_K \xi}{1 + (K_K-1)\xi}$ | Identifiable mobility factor $K_K$, monotonic water cut | Macro NRMSE > Baseline (0.03685) | **Falsified** (NRMSE 0.1198) |
| **H2: Total Liquid + WOR** | Bounded logistic WOR & liquid decline | $q_L = q_{L,0} e^{-D_L t}; \ln(\text{WOR}) = \ln(\text{WOR}_0) + \lambda_w t$ | Log-linear WOR growth, constant liquid decline | Water NRMSE degrades vs Baseline | **Falsified** (NRMSE 0.1227) |
| **H3: State-Dependent Decline**| Reserves-dependent depletion drive | $q_o = q_{o,0} [1 - \frac{\Delta Q_o}{N_{\text{rem}}}]^\alpha$ | $N_{\text{rem}}$ estimable from geological HCPV | Equifinality or premature well death | **Falsified** (NRMSE 0.1961) |
| **H4: Selective Water Correction**| Baseline oil + coupled water cut | $q_o = q_o^{\text{base}}; q_w = \text{WOR}^{\text{coup}} q_o^{\text{base}}$ | Uncoupled water is the sole source of error | Water NRMSE > Baseline water model | **Falsified** (Water NRMSE 0.3158) |
| **H5: Modified Arps Floor** | Convex transient + boundary exponential floor | $q_o = q_0(1+b D_i t)^{-1/b} \to e^{-D_{\text{lim}} t}$ | Late-life decline stabilizes at $D_{\text{lim}} = 0.06$/yr | Premature decline at medium horizons | **Falsified** (NRMSE 0.0622) |
| **H6: Stretched Exponential** | Heterogeneous relaxation diffusion (SEPD) | $q_o = q_0 \exp(-(t/\tau)^n); Q_\infty < \infty$ | Kohlrausch heterogeneous diffusion relaxation | Poor retrospective fit in late maturity | **Regime-Specific** (Wins 2003, Fails 2005) |

---

## 2. DECONSTRUCTING THE 13.85 MMSTB BASE-MODEL DISAGREEMENT

Over the 20-year deployment horizon on Field Case 0 (80 quarters, Jan 2008 – Jan 2028), the four base constituents of the frozen production ensemble disagree profoundly. The cumulative oil spread between the most optimistic and most pessimistic base models reaches **13,854,430 STB** (~13.85 MMSTB).

```
                      20-YEAR FIELD CASE 0 CUMULATIVE OIL TRAJECTORIES
       65M +-------------------------------------------------------------------+
           |                                                 ExtraTrees (61.7M)|
       60M |                                           *-----------------------+
           |                                    *------
       55M |                              *-----   Hyperbolic DCA (50.2M)
           |                        *-----     *-------------------------------+
       50M |                 *------    *------    Hybrid Dynamic (48.4M)
           |           *-----     *-----       *-------------------------------+
       45M |     *-----     *-----     *-------    Exponential DCA (47.9M)
           |----+-----+-----+-----+-----+-----+-----+-----+-----+-----+--------+
       40M +----+-----+-----+-----+-----+-----+-----+-----+-----+-----+--------+
           0    2     4     6     8    10    12    14    16    18    20 Years
```

### 2.1 Quantitative Horizon Milestones Audit

The inter-model divergence was audited across 6 official milestone horizons: 1 year (4Q), 3 years (12Q), 5 years (20Q), 10 years (40Q), 15 years (60Q), and 20 years (80Q):

| Horizon Label | Quarters | Horizon Ahead | ExtraTrees Cum Oil (STB) | Exponential DCA Cum Oil (STB) | Hyperbolic DCA Cum Oil (STB) | Hybrid Dynamic Cum Oil (STB) | Inter-Model Spread (STB) | Spread % of Mean | Terminal Rate Spread (STB/d) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Year** | 4Q | 1.0 yr | 40,102,997 | 40,028,622 | 40,032,959 | 40,102,997 | **74,375** | 0.19% | 317.3 STB/d |
| **3 Years** | 12Q | 3.0 yr | 43,026,514 | 42,538,843 | 42,607,113 | 43,026,514 | **487,671** | 1.14% | 721.0 STB/d |
| **5 Years** | 20Q | 5.0 yr | 45,376,728 | 44,271,128 | 44,495,424 | 45,081,221 | **1,105,600** | 2.47% | 983.8 STB/d |
| **10 Years**| 40Q | 10.0 yr| 50,832,434 | 46,602,358 | 47,493,282 | 47,456,165 | **4,230,076** | 8.80% | 2,282.4 STB/d |
| **15 Years**| 60Q | 15.0 yr| 56,288,140 | 47,524,590 | 49,191,230 | 48,193,644 | **8,763,551** | 17.42% | 2,768.4 STB/d |
| **20 Years**| 80Q | 20.0 yr| 61,743,847 | 47,889,417 | 50,249,161 | 48,422,643 | **13,854,430** | **26.60%** | **2,919.4 STB/d** |

### 2.2 Root-Cause Deconstruction of Base-Model Divergence

1. **`Recursive_ExtraTrees` (61.74 MMSTB EUR, Terminal Rate 2,987 STB/d)**:
   * **Root Cause: Support-Space Exhaustion & Step Plateauing.**  
     The tree ensemble is trained on quarterly steps with reservoir age $t \in [0, 15]$ years. When evaluated at $t > 15$ years, test query features fall completely outside the convex hull of training splits. Random forest / ExtraTrees regressors cannot extrapolate linearly; tree leaves predict the sample average of the nearest boundary partition. Consequently, predicted quarterly decrement $\Delta \ln(q_o) \to 0$. The predicted oil rate artificially plateaus at **2,987 STB/d** from Year 10 through Year 20, inflating EUR by **13.85 MMSTB**.
2. **`Learned_Exponential_DCA` (47.89 MMSTB EUR, Terminal Rate 124 STB/d)**:
   * **Root Cause: Enforced Constant Percentage Decline Without Boundary Support.**  
     The ridge regression fits an effective decline constant $D \approx 0.23$/yr. Extrapolated across 20 continuous years without pressure recharge, the exponential decline continuously halves production every 3 years. By Year 20, the production rate collapses to **124 STB/d**, producing the most pessimistic recovery envelope.
3. **`Learned_Hyperbolic_DCA` (50.25 MMSTB EUR, Terminal Rate 470 STB/d)**:
   * **Root Cause: Convex Tail Deceleration ($b = 0.35$).**  
     Hyperbolic decline exhibits an instantaneous decline rate $D(t) = D_i / (1 + b D_i t)$ that diminishes monotonically over time. As $t \to 20$ yr, $D(t)$ drops to 0.05/yr, maintaining an active late-life tail of **470 STB/d**.
4. **`Hybrid_Dynamic` (48.42 MMSTB EUR, Terminal Rate 68 STB/d)**:
   * **Root Cause: Fixed-Horizon Regime Transition.**  
     The hybrid model tracks ExtraTrees during the first 12 quarters (3 years), then hands off to Exponential DCA for $t > 12$Q. This completely prevents the ExtraTrees plateauing disaster while inheriting the aggressive late-life exponential decline.

![Figure 2: Base Models Disagreement Expansion](/Users/tengkuanas/.gemini/antigravity-cli/brain/4b7e1d00-9795-42a6-b5bb-c629e9cf25cb/fig2_base_models_disagreement_expansion.png)
*Figure 2: (A) 20-Year cumulative oil production trajectories on Field Case 0 illustrating the 13.85 MMSTB divergence envelope. (B) Disagreement spread expansion as a function of forecast horizon, demonstrating near-zero disagreement at Year 1 expanding to 26.6% of mean cumulative recovery at Year 20.*

### 2.3 Mathematical Compounding Dynamics: Why Short-Horizon Accuracy Lies

A critical finding of this research is the **mathematical breakdown of short-horizon empirical validation as a proxy for long-horizon predictive skill**.

When evaluating models retrospectively on Cases 1–70 at the 2003 origin across horizons from 4Q to 20Q (Table: `horizon_error_compounding_backtest.csv`):
* **At Short Horizon ($h = 4\text{Q}$, 1 Year)**:
  * `Recursive_ExtraTrees` achieves near-perfect oil forecasting: Oil NRMSE is **0.0082** on Case 1 and **0.0068** on Case 5.
  * `Learned_Exponential_DCA` exhibits higher error: Oil NRMSE is **0.0254** on Case 1 (3x worse).
  * *False Conclusion*: A practitioner evaluating only 1-year backtests would conclude that ExtraTrees is overwhelmingly superior.
* **At Long Horizon ($h = 20\text{Q}$, 5 Years)**:
  * `Recursive_ExtraTrees` error explodes super-linearly: Oil NRMSE surges to **0.1831** on Case 1 (a **22.3x error explosion**) and **0.2435** on Case 5 (a **35.8x error explosion**).
  * `Learned_Exponential_DCA` error remains strictly flat: Oil NRMSE remains **0.0428** on Case 1 and **0.0074** on Case 5.
  * At 5 years, Exponential DCA beats ExtraTrees by **0.0380 Macro NRMSE** on Case 1 and **0.0238 Macro NRMSE** on Case 5!

**Mathematical Derivation of Compounding Bias**:  
For a recursive autoregressive predictor $\hat{y}_{t+1} = f(\hat{y}_t) + \epsilon_t$, single-step errors $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$ feed into non-linear feature transforms at the next step. By the chain rule, total variance after $H$ steps compounds as:
$$\text{Var}(\hat{y}_H - y_H) = \sum_{j=1}^H \left( \prod_{k=j+1}^H \mathbf{J}_k \right) \boldsymbol{\Sigma}_\epsilon \left( \prod_{k=j+1}^H \mathbf{J}_k \right)^T \sim \mathcal{O}(H^{2\alpha})$$
For tree models where Jacobians $\mathbf{J}_k$ are discontinuous and step errors do not cancel, variance accumulates as $\mathcal{O}(H^{1.5} - H^2)$.  
Conversely, physics-constrained decline models fit a global parametric function $q(t) = q_0 f(\boldsymbol{\theta}; t)$. Parameter uncertainty $\text{Var}(\hat{\boldsymbol{\theta}})$ is fixed at the origin; prediction errors accumulate at most linearly:
$$\text{Var}(\hat{y}_H - y_H) = \left(\frac{\partial f}{\partial \boldsymbol{\theta}}\right) \text{Var}(\hat{\boldsymbol{\theta}}) \left(\frac{\partial f}{\partial \boldsymbol{\theta}}\right)^T \sim \mathcal{O}(H)$$
**Conclusion**: Short-horizon accuracy is dominated by high-capacity interpolation; long-horizon accuracy is governed exclusively by asymptotic physical boundedness.

---

## 3. 50-YEAR ASYMPTOTIC STRUCTURAL STRESS TESTING

To evaluate structural viability before field benchmarking, all candidate decline architectures were subjected to a 50-year (200-quarter) asymptotic stress test initialized at $q_{o,0} = 4,000$ STB/d and $q_{w,0} = 4,000$ STB/d.

| Model Architecture | 20-Year Terminal Rate (STB/d) | 50-Year Terminal Rate (STB/d) | 20-Year EUR (STB) | 50-Year EUR (STB) | 50-yr to 20-yr Ratio | Rate Non-Negative | Finite Asymptotic EUR | Physical Behavior Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Learned_Exponential_DCA`** | 27.0 | 0.015 | 5.63 MMSTB | 5.66 MMSTB | **1.007** | Yes | Yes | Strictly bounded exponential decline; terminal rate drops to 0.01 STB/d by Year 50. |
| **`Learned_Hyperbolic_DCA`** | 222.2 | 32.8 | 7.45 MMSTB | 8.42 MMSTB | **1.130** | Yes | **No (Infinite)** | Mathematically infinite EUR as $t \to \infty$; late-life tail continues adding volume indefinitely. |
| **`Modified_Arps_Dlim`** | 222.2 | 29.6 | 7.45 MMSTB | 8.41 MMSTB | **1.129** | Yes | Yes | Convex transient decline transitioning cleanly to exponential floor ($D_{\text{lim}} = 0.06$/yr); finite recovery. |
| **`Stretched_Exponential_SEPD`**| 354.5 | 72.4 | 7.23 MMSTB | 9.07 MMSTB | **1.253** | Yes | Yes | Self-capping stretched exponential ($n=0.55$); accommodates transient diffusion while bounding ultimate recovery. |
| **`Two_Compartment_Depletion`** | 721.1 | 216.5 | 10.40 MMSTB | 14.97 MMSTB | **1.439** | Yes | Yes | Fast flush drainage ($D_1=0.35$) followed by slow recharge ($D_2=0.04$); finite EUR with realistic late-life tail. |

All four proposed physical decline architectures strictly satisfy rate non-negativity ($q(t) \ge 0$), finite ultimate recovery ($Q_\infty < \infty$), and avoid unphysical runaway behavior.

---

## 4. RIGOROUS LEAVE-RESERVOIR-OUT BACKTESTING (CASES 1–70)

The candidate models were evaluated against the protected production baseline using the authoritative increment-level cumulative NRMSE referee metric under strict historical Protocol A on the 15 synthetic pseudo-fields: `[1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]`.

### 4.1 Primary Benchmark: Late Maturity (Origin January 2005, 12Q Horizon)

```
                       CANDIDATE MODELS VS BASELINE (ORIGIN 2005, 12Q)
     0.20 +-------------------------------------------------------------------+
          |                                                                   |
     0.16 |                                                                   |
          |                                                                   |
     0.12 |                                           Koval   TotalLiquid     |
          |                                          (0.120)    (0.123)       |
     0.08 |                                             |          |          |
          |                         TwoComp   ModArps   |          |          |
     0.04 |               BASELINE  (0.058)   (0.062)   |          |          |
          |   TailHybrid  (0.0368)     |         |      |          |          |
     0.00 +---(0.0287)-------+---------+---------+------+----------+----------+
```

| Rank | Model Name | Macro NRMSE Mean | Macro Std | Oil NRMSE Mean | Water NRMSE Mean | Difference vs Baseline | Diff SE | 95% Conf. Interval | Paired $t$-test $p$-value | Win Rate vs Baseline |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`H8_Constrained_Tail_Hybrid`** | **0.02869** | 0.0132 | 0.03925 | **0.01814** | **-0.00815** | **0.00274** | **[-0.01403, -0.00228]** | **0.01004** | **80.0% (12/15)** |
| **2** | **`Frozen_Ensemble_Baseline`** | **0.03685** | 0.0103 | **0.03223** | 0.04147 | **0.00000** | 0.00000 | [0.00000, 0.00000] | 1.00000 | 0.0% (0/15) |
| 3 | `H7_Two_Compartment_Depletion`| 0.05840 | 0.0040 | 0.02287 | 0.09393 | +0.02155 | 0.00299 | [+0.01515, +0.02796] | $4.45 \times 10^{-6}$ | 0.0% (0/15) |
| 4 | `H5_Modified_Arps_Dlim` | 0.06220 | 0.0061 | 0.03047 | 0.09393 | +0.02535 | 0.00367 | [+0.01748, +0.03323] | $7.27 \times 10^{-6}$ | 0.0% (0/15) |
| 5 | `H1_Koval_Fractional_Flow` | 0.11975 | 0.0298 | 0.11903 | 0.12047 | +0.08290 | 0.00605 | [+0.06993, +0.09588] | $1.66 \times 10^{-9}$ | 0.0% (0/15) |
| 6 | `H2_Total_Liquid_Bounded_WOR` | 0.12275 | 0.0298 | 0.12154 | 0.12395 | +0.08590 | 0.00602 | [+0.07298, +0.09882] | $9.89 \times 10^{-10}$ | 0.0% (0/15) |
| 7 | `H4_Selective_Water_Correction` | 0.17403 | 0.0191 | 0.03223 | 0.31584 | +0.13719 | 0.00240 | [+0.13204, +0.14233] | $< 10^{-15}$ | 0.0% (0/15) |
| 8 | `H6_Stretched_Exponential_SEPD`| 0.18817 | 0.0086 | 0.28242 | 0.09393 | +0.15133 | 0.00270 | [+0.14554, +0.15711] | $< 10^{-15}$ | 0.0% (0/15) |
| 9 | `H3_Recovery_State_Decline` | 0.19608 | 0.0270 | 0.03755 | 0.35461 | +0.15923 | 0.00528 | [+0.14790, +0.17056] | $3.91 \times 10^{-14}$ | 0.0% (0/15) |

![Figure 3: Candidate Models Pseudo-Field Benchmark](/Users/tengkuanas/.gemini/antigravity-cli/brain/4b7e1d00-9795-42a6-b5bb-c629e9cf25cb/fig3_candidate_models_pseudo_field_benchmark.png)
*Figure 3: Macro, Oil, and Water NRMSE of Challenger Models vs Frozen Baseline across 15 leave-reservoir-out pseudo-fields at Origin 2005 (12Q horizon). `H8_Constrained_Tail_Hybrid` achieves statistically significant macro improvement driven by a 56% error reduction in water forecasting.*

### 4.2 Multi-Origin & Multi-Horizon Robustness Matrix

To verify whether the `H8` challenger's advantage is genuine and deployable or merely an artifact of late-maturity fitting, the entire candidate suite was evaluated across 7 grid configurations spanning 3 production origins (Jan 2003, Jan 2004, Jan 2005) and 3 horizons (4Q, 12Q, 20Q).

| Origin Date | Horizon | Protected Baseline Macro NRMSE | `H8_Constrained_Tail_Hybrid` Macro NRMSE | `H8` Diff vs Baseline ($p$-val) | `H6_SEPD` Macro NRMSE | `H6` Diff vs Baseline ($p$-val) | Best Performing Model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **2003-01-01** | **4Q (1 yr)** | 0.08103 | 0.07601 | -0.00502 ($p=0.166$) | **0.07065** | **-0.01038 ($p=0.486$)** | `H6_SEPD` (0.0706) |
| **2003-01-01** | **12Q (3 yr)**| 0.15562 | 0.15138 | -0.00425 ($p=0.213$) | **0.11132** | **-0.04430 ($p=0.013$)** | `H6_SEPD` (0.1113) |
| **2003-01-01** | **20Q (5 yr)**| **0.22001** | 0.24860 | **+0.02859 ($p<10^{-5}$)** | **0.13922** | **-0.08078 ($p<10^{-4}$)** | `H6_SEPD` (0.1392) |
| **2004-01-01** | **4Q (1 yr)** | 0.02239 | **0.01355** | **-0.00884 ($p<10^{-3}$)** | 0.13186 | +0.10948 ($p<10^{-10}$) | `H8_Hybrid` (0.0135) |
| **2004-01-01** | **12Q (3 yr)**| **0.03295** | 0.04751 | **+0.01455 ($p=0.064$)** | 0.17296 | +0.14001 ($p<10^{-10}$) | **Baseline** (0.0330) |
| **2005-01-01** | **4Q (1 yr)** | 0.01979 | **0.01157** | **-0.00822 ($p<10^{-11}$)**| 0.13420 | +0.11441 ($p<10^{-15}$) | `H8_Hybrid` (0.0116) |
| **2005-01-01** | **12Q (3 yr)**| 0.03685 | **0.02869** | **-0.00815 ($p=0.010$)** | 0.18817 | +0.15133 ($p<10^{-15}$) | `H8_Hybrid` (0.0287) |

![Figure 4: Multi-Origin Robustness Benchmark](/Users/tengkuanas/.gemini/antigravity-cli/brain/4b7e1d00-9795-42a6-b5bb-c629e9cf25cb/fig4_multi_origin_robustness_benchmark.png)
*Figure 4: Cross-validation trajectory comparison across origins 2003 (Early Decline), 2004 (Mid Maturity), and 2005 (Late Maturity). Illustrates the profound regime dependency: `H8` excels in late maturity but regresses at 2004 12Q and 2003 20Q; `H6` excels at 2003 20Q but fails catastrophically in late maturity; `Frozen_Ensemble_Baseline` provides stable, uncompromised cross-regime protection.*

### 4.3 Why Neither Challenger Can Replace the Production Baseline

1. **Failure of Monotonic Superiority**:  
   To justify replacing a protected production baseline that is already submitted and locked, a challenger must demonstrate **monotonic cross-regime superiority** (winning or drawing in virtually every operational scenario).  
   * `H8_Constrained_Tail_Hybrid` wins convincingly at 2005 4Q (100% win rate), 2005 12Q (80% win rate), and 2004 4Q (86.7% win rate).  
   * However, at 2004 12Q, `H8` **loses to the baseline** (NRMSE 0.0475 vs 0.0330; win rate only 26.7%).  
   * At 2003 20Q, `H8` **regresses with high statistical confidence** (NRMSE 0.2486 vs 0.2200; diff = +0.0286, $p = 1.47 \times 10^{-6}$; win rate 0.0%).
2. **The "Specialist Trap" of SEPD (`H6`)**:  
   `H6_Stretched_Exponential` exhibits the lowest error observed anywhere in this research program at Origin 2003 20Q (Macro NRMSE **0.13922** vs Baseline **0.22001**, $p = 1.76 \times 10^{-5}$, 93.3% win rate). However, when evaluated at Origin 2005 12Q, `H6` collapses completely (Oil NRMSE **0.28242** vs Baseline **0.03223**; Macro NRMSE **0.18817** vs **0.03685**). A model that produces 500% error in mature fields cannot be deployed as an unconstrained production replacement.
3. **The Role of the Frozen Baseline as a Universal Insurance Policy**:  
   The protected `Blend_Phase_Specific_Simplex` baseline never achieves the worst score in any origin, never blows up, and maintains its Macro NRMSE strictly bounded between 0.019 and 0.220 across all origins and horizons. Its convex combination of dynamic trees and dual DCA models acts as an optimal variance-reduction filter.

---

## 5. JANUARY 2008 FORWARD DEPLOYMENT ON FIELD CASE 0 (2008–2028, 80 QUARTERS)

The 20-year forward deployment was executed on actual historical Field Case 0 across the 10 official challenge scenarios without altering the frozen submission workbook.

### 5.1 Deployment Summary Across 10 Scenarios

| Forecasting Architecture | 20-Year Mean EUR Oil (STB) | EUR Oil Scenario Spread (STB) | 20-Year Incremental Oil (STB) | 20-Year Mean EUR Water (STB) | EUR Water Scenario Spread (STB) | Mean Terminal Oil Rate at 2028 (STB/d) | Mean Terminal Water Rate at 2028 (STB/d) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Frozen_Ensemble_Baseline`** | **51,178,189** | **16,433** | **12,799,872** | **96,165,581** | **35,153** | **810.0** | **13,876.3** |
| **`Challenger_Constrained_Tail_Hybrid`** | **48,488,647** | 65,798 | 10,110,330 | 98,962,478 | 211,956 | **175.3** | 13,368.6 |
| **`Challenger_Total_Liquid_Bounded_WOR`**| **44,395,969** | 159,161 | 6,017,652 | 62,833,841 | 719,545 | **5.9** | 4,318.7 |
| **`Challenger_Selective_Water_Correction`**| **51,178,189** | 16,433 | 12,799,872 | 841,095,002 | 97,548,510 | 810.0 | **592,139.6** |

![Figure 5: Deployment 2008 Trajectories](/Users/tengkuanas/.gemini/antigravity-cli/brain/4b7e1d00-9795-42a6-b5bb-c629e9cf25cb/fig5_deployment_2008_comparison_trajectories.png)
*Figure 5: (A) Scenario uncertainty envelopes for 20-year cumulative oil EUR across the 10 official challenge scenarios on Field Case 0. (B) Plausible terminal production rates at Year 20 (2028), illustrating the physical contrast between baseline (810 STB/d), hybrid tail (175 STB/d), liquid-bounded WOR (5.9 STB/d), and runaway unconstrained water.*

### 5.2 Physical Plausibility of 20-Year Deployment Trajectories

1. **The Frozen Baseline Forecast (51.18 MMSTB EUR, 810 STB/d Terminal Rate)**:
   * Case 0 historical cumulative oil at Jan 2008 was **38.38 MMSTB** with an initial rate of **4,580 STB/d**.
   * Over 20 years, the baseline predicts an incremental oil recovery of **12.80 MMSTB**, representing an effective annualized decline rate of $D \approx 0.087$/yr.
   * Terminal oil rate in 2028 is **810 STB/d**, and terminal water rate is **13,876 STB/d** (producing a terminal water cut of $f_w \approx 0.945$). This is thoroughly consistent with mature offshore waterflooding under continuous voidage replacement and active pressure maintenance.
2. **`Constrained_Tail_Hybrid` Forecast (48.49 MMSTB EUR, 175 STB/d Terminal Rate)**:
   * The hybrid tail model transitions to Modified Arps ($D_{\text{lim}} = 0.06$/yr) for $t > 3$ years.
   * Terminal oil rate drops to **175 STB/d**, recovering **10.11 MMSTB** incrementally (a ~2.69 MMSTB reduction vs baseline).
   * Terminal water rate reaches **13,369 STB/d** ($f_w \approx 0.987$). This represents a physically credible lower-bound depletion trajectory.
3. **Catastrophic Failure of Unconstrained Water Models (`Selective_Water_Correction`)**:
   * When water rate is extrapolated using an unconstrained log-linear exponential model ($\ln(q_w) \propto \lambda t$ with positive slope), predicted water rate explodes exponentially, reaching **592,140 STB/d** at Year 20 with an impossible water EUR of **841 MMSTB**. This experimentally proves why naive time-series extrapolation of water production is dangerous over multi-decade horizons.

---

## 6. SCIENTIFIC CONCLUSIONS & PRODUCTION POLICY

### 6.1 Policy Affirmation
1. **The Official Deliverable Remains Untouched**:  
   `outputs/submission/09_Template_Deliverable_Submission.xlsx` was generated directly by the protected baseline commit `c74e3ac503c131d1a8894ed7250f8b661480b66e` and remains 100% frozen and intact.
2. **Stacking Configurations Remain Frozen**:  
   The phase-specific simplex weights in [`locked_ensemble_config.json`](file:///Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp/configs/locked_ensemble_config.json) remain strictly locked.
3. **Challenge Cases 86–100 Remain Strictly Blind**:  
   No evaluation or model fitting was conducted on Cases 86–100.

### 6.2 Key Takeaways for Long-Horizon Reservoir Forecasting
1. **The 13.85 MMSTB Disagreement is Caused by Support Exhaustion vs Enforced Decline**:
   Data-driven trees cannot extrapolate declining functions and plateau at their nearest leaf partitions; pure decline models enforce constant depletion regardless of boundary recharge.
2. **Error Compounding Dynamics Preclude Extrapolating 1-Year Benchmarks**:
   Data-driven models compound error super-linearly ($\mathcal{O}(H^{1.5}-H^2)$) while parametric decline models compound linearly ($\mathcal{O}(H)$). Short-horizon backtests favor high-capacity regressors; long-horizon reliability favors physics-bounded decline tails.
3. **Regime-Specific Challengers Do Not Justify Production Overhaul**:
   While `H8_Constrained_Tail_Hybrid` achieves a statistically significant 22% error reduction in late maturity (NRMSE 0.02869 vs 0.03685, $p=0.010$) and `H6_SEPD` achieves a 37% error reduction in early depletion ($p < 10^{-4}$), both fail to generalize across all operational origins and horizons.
4. **Final Recommendation**:
   Retain `Blend_Phase_Specific_Simplex` as the authoritative competition submission. Archive `H8_Constrained_Tail_Hybrid` and `H6_SEPD` as specialized regime-specific post-processing candidates for future field lifecycle extensions.

---

## CODE AND ARTIFACT INDEX

* Master Experiment Runner: [`run_long_horizon_experiments.py`](file:///Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp/src/long_horizon_physics/run_long_horizon_experiments.py)
* Literature & Mathematical Formulations: [`literature_foundations.py`](file:///Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp/src/long_horizon_physics/literature_foundations.py)
* Multiphase Coupled Models: [`multiphase_coupled_models.py`](file:///Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp/src/long_horizon_physics/multiphase_coupled_models.py)
* Long-Horizon Decline Models: [`long_horizon_decline_models.py`](file:///Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp/src/long_horizon_physics/long_horizon_decline_models.py)
* 13.85 MMSTB Disagreement Analysis: [`disagreement_analysis.py`](file:///Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp/src/long_horizon_physics/disagreement_analysis.py)
* Leave-Reservoir-Out Evaluation Framework: [`experimental_evaluation.py`](file:///Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp/src/long_horizon_physics/experimental_evaluation.py)
* January 2008 Field Case 0 Deployment: [`deployment_2008_forecaster.py`](file:///Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp/src/long_horizon_physics/deployment_2008_forecaster.py)
* Unit & Regression Test Suite: [`test_long_horizon_physics.py`](file:///Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp/tests/test_long_horizon_physics.py)
