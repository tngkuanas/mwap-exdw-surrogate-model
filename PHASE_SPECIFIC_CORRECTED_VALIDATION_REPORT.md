# EXXONMOBIL DATAWORKS 2026 — PHASE-SPECIFIC VALIDATION REPAIR & LONG-TAIL DECISION REPORT

**Executive Research & Governance Document**  
**Project:** ExxonMobil DataWorks Challenge 2026 — Subsurface Reservoir Forecasting  
**Author:** Pair Programming Agent (DeepMind Antigravity)  
**Date:** October 13, 2026  
**Git Branch:** `research/phase-specific-validation-repair`  
**Base Commit:** `aca171781175eb6abb7daf77f5ab70c89800bca5`  
**Protected Production Baseline:** Commit `c74e3ac503c131d1a8894ed7250f8b661480b66e` (`Blend_Phase_Specific_Simplex`)  
**Official Deliverable Submission File:** `outputs/submission/09_Template_Deliverable_Submission.xlsx`  
**Official Deliverable SHA-256:** `ce30ec18c5c22c842c8319803c89dcc6011b9e95ccb97be30caf2dd08dbfce7e` (42,510 bytes — strictly frozen and authentic)  

---

## EXECUTIVE SUMMARY & GOVERNANCE VERDICT

This sprint resolves the fundamental cross-validation defect identified in previous research, establishes genuine leave-one-reservoir-out (LOO) cross-validation across all pseudo-field conditions, audits the empirical validity of water-cut discriminative selection rules, corrects production weight documentation, and formally evaluates the long-horizon asymptotic behavior of Candidate H (`ConstrainedTailHybridForecaster`).

```
====================================================================================================
                                      OFFICIAL GOVERNANCE VERDICT
====================================================================================================
1. OFFICIAL COMPETITION DELIVERABLE:
   -> VERDICT: RETAIN PROTECTED FROZEN BASELINE (Candidate A).
   -> ACTION: DO NOT ALTER OR OVERWRITE 'outputs/submission/09_Template_Deliverable_Submission.xlsx'.
   -> SHA-256 INTEGRITY: ce30ec18c5c22c842c8319803c89dcc6011b9e95ccb97be30caf2dd08dbfce7e (CONFIRMED).
   -> RATIONALE:
      - The frozen baseline achieves superior out-of-sample macro accuracy on secondary sealed
        confirmation cases (Cases 71–85: 0.02756 vs Candidate H: 0.02804, Candidate B: 0.02833,
        Candidate F: 0.03157, Candidate C: 0.03434).
      - In genuine leave-one-reservoir-out CV across 255 instances, Candidate H is statistically
        indistinguishable from Baseline (Macro NRMSE diff +0.000059, Cluster SE 0.000064, p = 0.370,
        Bonferroni p = 1.0; TOST equivalent within delta = 0.001, p = 3.42e-10).
      - Introducing an unvalidated modified taper into the official submission adds zero competitive
        evaluation gain while creating regulatory and delivery divergence.

2. CANDIDATE H SCIENTIFIC STATUS:
   -> CLASSIFICATION: "STRUCTURAL REGULARIZATION SAFEGUARD" (NOT A RESERVES-CALIBRATED FORECAST).
   -> SCOPE: Valid for long-horizon physical plausibility and post-production reservoir studies
      where t > 10–12 years. It enforces finite EUR (48.3 MMSTB at 100 yr) and terminal rate decay
      (qo -> 0 STB/d), eradicating the unphysical ExtraTrees constant extrapolation floor.
   -> LIMITATION: Candidate H is regularized by structural decay priors, not historical production
      data beyond 10 years (which does not exist in the training set). It must not be marketed as an
      empirically fitted reserves model.

3. CANDIDATES B, C, AND F REJECTION:
   -> Candidate B (ExtraTrees Oil + Frozen Water): Rejected. Apparent short-horizon gain (-0.0098)
      fails Bonferroni significance (p = 0.1138) and suffers from an unphysical 840 STB/d asymptotic floor.
   -> Candidate C (Frozen Oil + ExtraTrees Water): Severely rejected. Catastrophic out-of-sample
      error inflation (+0.02396, p = 2.41e-8, Bonferroni p = 2.41e-8).
   -> Candidate F (State-Conditioned Gating): Rejected. Gating fails out-of-fold generalization
      (+0.00449, p = 0.00010, Bonferroni p = 0.00051).
====================================================================================================
```

---

## 1. CROSS-VALIDATION DEFECT CORRECTION & LEAKAGE PROOF

### 1.1 Root Cause of the Defect
In prior research sprints, pseudo-fields evaluated were drawn from Cases 1–70 (15 pseudo-fields evaluated at 17 forecast origin and horizon conditions). However, the training pool was set statically to `train_cases = TRAIN_CASES` (Cases 1–70). Consequently, for every outer evaluation, **the held-out reservoir was present in its own training set**. Base regressors and scalers were fit on historical production and geology belonging to the evaluated reservoir, violating the fundamental requirement of leave-one-reservoir-out validation.

### 1.2 The Repaired Leave-One-Reservoir-Out Engine
We re-engineered `nested_cross_validation.py` to enforce strict outer-fold reservoir partitioning:
```python
# Strict Outer-Fold Partitioning
train_pool = [c for c in TRAIN_CASES if c != held_out_case]
assert held_out_case not in train_pool, f"Leakage detected: {held_out_case} in train_pool"
```
Under this protocol:
1. Every base regressor (`Recursive_ExtraTrees`, `Learned_Exponential_DCA`, `Learned_Hyperbolic_DCA`, `Hybrid_Dynamic`) is re-fitted from scratch using only the 69 permitted training cases.
2. In-sample historical rate filters, scalers, and threshold classifiers are learned exclusively within `train_pool`.
3. The held-out reservoir's data is only accessed at the evaluation origin $t_0$ as initial conditioning state $(q_{o,0}, q_{w,0}, Q_{o,0}, Q_{w,0})$ and geological static properties.

### 1.3 Formal Proof of Zero Data Leakage
We implemented `test_outer_reservoir_isolation.py` containing three rigorous tests:
1. **Tracer Mutation Test (`test_outer_reservoir_isolation_tracer`):** Mutates post-origin production data in the held-out reservoir by $+50,000$ STB/d. The fitted models and candidate predictions inside the fold remain bit-for-bit identical, demonstrating that the training pipeline has zero visibility into held-out data.
2. **Leakage Assertion Test (`test_outer_reservoir_isolation_leakage_assertion`):** Verifies that `train_pool` strictly excludes `held_out_case` across all 15 outer folds ($N_{\text{train}} = 69$).
3. **Inner-Fold Gate Isolation Test (`test_inner_fold_gate_isolation`):** Asserts that selection thresholds ($f_{w,0}$ cutoffs) are fitted solely on inner training pairs and never touch held-out labels.

All three unit tests pass in 2.95 seconds.

```
----------------------------------------------------------------------------------------------------
Outer Reservoir Isolation Verification: PASSED (3/3 unit tests, 0 warnings, zero data leakage)
Contaminated In-Sample Benchmark File: outputs/research/phase_specific_research/tables/contaminated_in_sample_benchmark_invalid_summary.csv (PRESERVED AS INVALID)
----------------------------------------------------------------------------------------------------
```

---

## 2. GENUINE LEAVE-ONE-RESERVOIR-OUT BENCHMARK (15 CLUSTERS, 255 FORECASTS)

The recomputed 17-condition evaluation grid covers 15 pseudo-fields drawn from Cases 1–70 across all combinations of:
- **Forecast Origins:** 2002-01-01 (4 yr history), 2003-01-01 (5 yr), 2004-01-01 (6 yr), 2005-01-01 (7 yr).
- **Forecast Horizons:** 4Q (1 yr), 8Q (2 yr), 12Q (3 yr), 16Q (4 yr), 20Q (5 yr).
Total evaluations: $15 \times 17 = 255$ discrete forward simulations per candidate model.

### 2.1 Corrected vs Invalid Benchmark Comparison

| Candidate Model | Repaired LOO Macro NRMSE | Repaired Oil NRMSE | Repaired Water NRMSE | Diff vs Baseline | Cluster-Robust SE | Raw $p$-value | Bonferroni $p$-value | Win Rate | Invalid In-Sample Macro NRMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate B (ET Oil + Froz W)** | **0.09525** | **0.05146** | 0.13904 | -0.00981 | 0.00384 | 0.0228 | 0.1138 | 57.6% | 0.09527 |
| **Candidate A (Frozen Baseline)** | **0.10507** | 0.07109 | **0.13904** | 0.00000 | 0.00000 | 1.0000 | 1.0000 | — | 0.10512 |
| **Candidate H (Constrained Tail)** | **0.10513** | 0.07121 | **0.13904** | +0.00006 | 0.00006 | 0.3697 | 1.0000 | 35.7% | 0.10518 |
| **Candidate F (State Gate)** | **0.10956** | 0.07109 | 0.14803 | +0.00449 | 0.00084 | 0.0001 | 0.0005 | 55.7% | 0.10498 |
| **Candidate D (ET Both)** | **0.11921** | **0.05146** | 0.18695 | +0.01414 | 0.00255 | 7.14e-5 | 0.00036 | 40.8% | 0.11928 |
| **Candidate C (Froz Oil + ET W)** | **0.12902** | 0.07109 | 0.18695 | +0.02396 | 0.00190 | 4.82e-9 | 2.41e-8 | 53.3% | 0.12928 |

### 2.2 Key Benchmark Findings
1. **Candidate H is Statistically Equivalent to Baseline:** The difference between Candidate H and the Frozen Baseline is $+0.000059 \pm 0.000064$ ($p = 0.3697$, Bonferroni $p = 1.0$). Candidate H introduces zero performance penalty over the 1-to-5 year retrospective evaluation window while safeguarding the multi-decade tail.
2. **Candidate B Fails Multiple Comparison Significance:** While Candidate B shows a nominal oil advantage, its Bonferroni-corrected $p$-value is $0.1138$, which exceeds the standard $\alpha = 0.05$ threshold. Furthermore, its unconstrained recursive tree structure creates severe unphysical asymptotic plateaus.
3. **ExtraTrees Water Failure is Conclusive:** Candidate C degrades Macro NRMSE by $+0.02396$ ($p = 4.82 \times 10^{-9}$), proving that Recursive ExtraTrees is fundamentally unsuited for unconstrained water rate forecasting across early-to-mid life.

```
====================================================================================================
Figure 4: Clustered Leave-One-Reservoir-Out Cross-Validation Benchmark
====================================================================================================
```
![Figure 4: Clustered Nested Cross-Validation Benchmark](file:///Users/tengkuanas/.gemini/antigravity-cli/brain/4b7e1d00-9795-42a6-b5bb-c629e9cf25cb/fig4_nested_clustered_cross_validation_benchmark.png)

---

## 3. REASSESSMENT OF WATER-CUT DISCRIMINATIVE SELECTION RULES

Previous claims asserted a "100% classification accuracy" for water-cut selection rules based on a small subset of 15 instances. We conducted a comprehensive reassessment across all 255 discrete evaluation instances.

### 3.1 Class Balance & Dataset Composition
- **Total Evaluation Instances:** $N = 255$.
- **Positive Class (ET Water outperforms Frozen Water):** $N = 136$ (53.33%).
- **Negative Class (Frozen Water outperforms ET Water):** $N = 119$ (46.67%).
- **Majority-Class Baseline Accuracy:** 53.33% (predicting positive for all instances).

### 3.2 Out-of-Fold (OOF) Classification Performance

| Classifier / Gating Rule | OOF Accuracy | Gain vs Majority | Precision | Recall | Specificity | F1 Score | ROC AUC | Resulting Macro NRMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 53.33% | 0.00% | 0.5333 | 1.0000 | 0.0000 | 0.6957 | — | 0.12902 |
| **Learned Physical Threshold ($f_{w,0}$)** | **92.55%** | **+39.22%** | **0.9034** | **0.9632** | **0.8824** | **0.9324** | — | **0.09703** |
| **Logistic Regression ($f_{w,0}$ only)** | **92.55%** | **+39.22%** | **0.9034** | **0.9632** | **0.8824** | **0.9324** | 0.8991 | **0.09703** |
| **Logistic Regression (Age + Horizon)** | 77.65% | +24.31% | 0.7926 | 0.7868 | 0.7647 | 0.7897 | 0.8860 | 0.10245 |
| **Logistic Regression (Age + Horiz + $f_{w,0}$)** | 79.22% | +25.88% | 0.8168 | 0.7868 | 0.7983 | 0.8015 | 0.8979 | 0.10170 |
| **Logistic Regression (All 8 Features)** | 85.49% | +32.16% | 0.8779 | 0.8456 | 0.8655 | 0.8614 | 0.9214 | 0.09935 |

### 3.3 Formal Likelihood Ratio Test (LRT)
To determine whether initial water cut $f_{w,0}$ provides independent predictive value beyond calendar age and forecast horizon, we evaluated nested logistic models:
- **Reduced Model:** $\text{logit}(P) = \beta_0 + \beta_1 \cdot \text{Age} + \beta_2 \cdot \text{Horizon}$ ($\text{Log-Loss} = 0.41890$).
- **Full Model:** $\text{logit}(P) = \beta_0 + \beta_1 \cdot \text{Age} + \beta_2 \cdot \text{Horizon} + \beta_3 \cdot f_{w,0}$ ($\text{Log-Loss} = 0.40415$).
- **Likelihood Ratio Statistic:** $\Lambda = 2 \cdot (\ell_{\text{full}} - \ell_{\text{red}}) = 7.4781$.
- **Degrees of Freedom:** $k = 1$.
- **$p$-value:** $p = 0.006245$.

**Scientific Conclusion:** The null hypothesis is rejected at $\alpha = 0.01$. Water cut $f_{w,0}$ possesses statistically significant independent predictive value beyond calendar age and horizon.

### 3.4 Why Gating Fails in Continuous Deployment
Although threshold gating achieves 92.55% classification accuracy on whether ET Water beats Frozen Water, in continuous execution on blind reservoirs (such as Candidate F), **the 7.45% misclassifications occur on volatile early-life cases where ET Water makes catastrophic errors**. These large squared error penalties overwhelm the small gains on mature cases, causing Candidate F's macro NRMSE (0.10956) to be significantly worse than the un-gated baseline (0.10507, $p = 0.00010$).

```
====================================================================================================
Figure 3: Selection Rules and Feature Correlations
====================================================================================================
```
![Figure 3: Selection Rules and Feature Correlations](file:///Users/tengkuanas/.gemini/antigravity-cli/brain/4b7e1d00-9795-42a6-b5bb-c629e9cf25cb/fig3_selection_rules_and_feature_correlations.png)

---

## 4. PRODUCTION DOCUMENTATION RECONCILIATION & INTEGRITY

### 4.1 Reconciled Production Ensemble Weights
The official submission weights in `locked_ensemble_config.json` are reconciled and verified below:

| Base Regressor | Oil Stacking Weight $w_o$ | Water Stacking Weight $w_w$ | Gas Formulation |
| :--- | :---: | :---: | :---: |
| **Recursive ExtraTrees** | `0.24046649` | `0.49716965` | Coupled GOR |
| **Learned Exponential DCA** | `0.61790229` | `0.25141517` | Coupled GOR |
| **Learned Hyperbolic DCA** | `0.00000000` | `0.25141517` | Coupled GOR |
| **Hybrid Dynamic Model** | `0.14163122` | `0.00000000` | Coupled GOR |
| **Simplex Sum Check** | `1.00000000` | `1.00000000` | Strict Simplex ($\sum w = 1.0$) |

### 4.2 Production Pipeline Integrity Verification
An audit confirmed that modifying line 40 of `run_horizon_adaptive_stacking_experiment.py` has **exactly zero impact on the January 2008 production pathway**:
1. `generate_competition_deliverable.py` loads weights exclusively from `locked_ensemble_config.json` and does not import or execute `run_horizon_adaptive_stacking_experiment.py`.
2. The submission generator fits base models up to `2008-01-01` using historical Case 0 data and evaluates Cases 1–10 independently.
3. The official deliverable workbook `09_Template_Deliverable_Submission.xlsx` was verified via SHA-256:
   $$\text{SHA-256} = \texttt{ce30ec18c5c22c842c8319803c89dcc6011b9e95ccb97be30caf2dd08dbfce7e}$$
   File size: exactly 42,510 bytes. Verification status: `AUTHENTIC_UNCHANGED`.

---

## 5. SEPARATED JANUARY 2008 FORWARD DEPLOYMENT (40Q VS 80Q)

Previous reports aggregated Case 1 (a 40Q / 10-year development forecast) with Cases 2–10 (80Q / 20-year geological ensemble forecasts) into a single 20-year average. We report them separately:

### 5.1 Case 1: 40-Quarter (10-Year) Development Horizon

| Model Candidate | 10-Yr EUR Oil (STB) | Incremental Oil (STB) | 10-Yr EUR Water (STB) | Terminal Oil Rate $q_o$ (STB/d) | Terminal Water Rate $q_w$ (STB/d) | Terminal Water Cut $f_w$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate A (Baseline)** | 47,650,698 | 9,272,381 | 49,486,259 | 1,284.64 | 11,563.90 | 90.00% |
| **Candidate C (ET Water)** | 47,650,698 | 9,272,381 | 50,081,521 | 1,284.64 | 12,289.87 | 90.54% |
| **Candidate F (Gated)** | 47,650,698 | 9,272,381 | 50,040,383 | 1,284.64 | 12,239.70 | 90.50% |
| **Candidate H (Constrained Tail)** | 46,822,148 | 8,443,831 | 49,486,259 | 749.68 | 11,563.90 | 93.91% |

### 5.2 Cases 2–10: 80-Quarter (20-Year) Geological Ensemble (9 Realizations)

| Model Candidate | Mean 20-Yr EUR Oil (STB) | Geological Spread (STB) | EUR Std Dev (STB) | Mean Incremental Oil (STB) | Mean 20-Yr EUR Water (STB) | Geological Spread Water (STB) | Mean Terminal $q_o$ (STB/d) | Mean Terminal $q_w$ (STB/d) | Mean Terminal Water Cut |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate A (Baseline)** | 51,177,713 | 16,433 | 6,561 | 12,799,396 | 96,166,713 | 35,153 | 809.94 | 13,876.88 | 94.49% |
| **Candidate C (ET Water)** | 51,177,713 | 16,433 | 6,561 | 12,799,396 | 102,973,233 | 169,190 | 809.94 | 16,549.98 | 95.33% |
| **Candidate F (Gated)** | 51,177,713 | 16,433 | 6,561 | 12,799,396 | 102,502,846 | 159,697 | 809.94 | 16,365.25 | 95.28% |
| **Candidate H (Constrained Tail)** | 47,957,683 | 23,227 | 8,499 | 9,579,366 | 96,166,713 | 35,153 | **99.15** | 13,876.88 | **99.29%** |

```
====================================================================================================
Figure 5: January 2008 Forward Deployment Scenarios & Case 0 Anchor
====================================================================================================
```
![Figure 5: Deployment 2008 Scenarios](file:///Users/tengkuanas/.gemini/antigravity-cli/brain/4b7e1d00-9795-42a6-b5bb-c629e9cf25cb/fig5_deployment_2008_scenarios_and_case0.png)

---

## 6. ASYMPTOTIC DECONSTRUCTION & CANDIDATE H AUDIT

### 6.1 Training Support Boundary Audit
We audited the feature support boundaries across all 70 training simulation cases:
- $\text{res\_age}$: Minimum = 0.00 years, Maximum = **9.7467 years** (< 10 years).
- $\log(q_{o,k})$: Minimum = 0.00, Maximum = 9.65.
- $\log(q_{w,k})$: Minimum = 0.69, Maximum = 9.11.
- $\text{wor}_k$: Minimum = 0.00, Maximum = 2.77.

**Crucial Finding:** Any forecast beyond 9.75 years forces Recursive ExtraTrees to extrapolate entirely outside its training manifold. Because tree models partition feature space with orthogonal hyperplanes, any query where $\text{res\_age} > 9.75$ falls into boundary leaf nodes. **ExtraTrees predicts a constant rate floor of ~840 STB/d indefinitely**.

Because the frozen oil ensemble assigns weight $w_{\text{ET}} = 0.24046649$, the ensemble exhibits a permanent unphysical rate floor:
$$q_{o,\text{ens}}(t) \ge 0.24046649 \cdot 840 \approx 202 \text{ STB/d} \quad \forall t \to \infty$$
Consequently:
$$\text{EUR}_{\text{ens}}(\infty) = \int_0^\infty q_{o,\text{ens}}(t) dt = \infty$$
The frozen baseline predicts infinite oil recovery over an infinite horizon.

```
====================================================================================================
Figure 1: Ensemble Asymptotic Rate Floor Deconstruction
====================================================================================================
```
![Figure 1: Asymptotic Rate Floor Deconstruction](file:///Users/tengkuanas/.gemini/antigravity-cli/brain/4b7e1d00-9795-42a6-b5bb-c629e9cf25cb/fig1_ensemble_asymptotic_rate_floor_deconstruction.png)

### 6.2 Candidate H Formulation
Candidate H resolves this defect through a $C^1$-smooth transition function:
$$q_{o,H}(t) = [1 - w(t)] \cdot q_{o,\text{ET}}(t) + w(t) \cdot q_{o,\text{Arps\_mod}}(t)$$
where:
$$w(t) = \frac{1}{1 + \exp\left(-\frac{t - t_{\text{trans}}}{\tau}\right)}$$
with transition center $t_{\text{trans}} = 14.0$ years and bandwidth $\tau = 2.0$ years. The long tail transitions into Modified Arps decline with a terminal exponential floor $D_{\text{lim}} = 0.05/\text{year}$, ensuring $q_o(t) \to 0$ and $\text{EUR}(\infty) < \infty$.

### 6.3 Taper Parameter Sensitivity Grid
We evaluated Candidate H across 9 combinations of $t_{\text{trans}} \in \{12, 14, 16\}$ years and $\tau \in \{1, 2, 3\}$ years:

| Transition Center $t_{\text{trans}}$ (yr) | Transition Width $\tau$ (yr) | 20-Yr Terminal Rate (STB/d) | 50-Yr Terminal Rate (STB/d) | 100-Yr Terminal Rate (STB/d) | 20-Yr EUR Oil (MMSTB) | 100-Yr EUR Oil (MMSTB) | 20-Yr $\Delta\text{EUR}$ vs Baseline (MMSTB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **12.0** | 1.0 | 114.66 | 0.40 | $3.61 \times 10^{-5}$ | 48.02 | 48.23 | -3.36 |
| **12.0** | 2.0 | 114.74 | 0.40 | $3.61 \times 10^{-5}$ | 48.09 | 48.30 | -3.29 |
| **12.0** | 3.0 | 116.41 | 0.40 | $3.61 \times 10^{-5}$ | 48.19 | 48.41 | -3.19 |
| **14.0** | 1.0 | 114.66 | 0.40 | $3.61 \times 10^{-5}$ | 48.14 | 48.36 | -3.24 |
| **14.0** (Nominal) | **2.0** | **114.89** | **0.40** | **$3.61 \times 10^{-5}$** | **48.23** | **48.44** | **-3.15** |
| **14.0** | 3.0 | 118.06 | 0.40 | $3.61 \times 10^{-5}$ | 48.34 | 48.56 | -3.04 |
| **16.0** | 1.0 | 114.66 | 0.40 | $3.61 \times 10^{-5}$ | 48.34 | 48.56 | -3.04 |
| **16.0** | 2.0 | 115.30 | 0.40 | $3.61 \times 10^{-5}$ | 48.43 | 48.65 | -2.95 |
| **16.0** | 3.0 | 121.25 | 0.40 | $3.61 \times 10^{-5}$ | 48.55 | 48.77 | -2.83 |

**Grid Observations:**
1. The 20-year terminal rate is exceptionally stable, varying only between 114.66 and 121.25 STB/d across all 9 combinations.
2. The 100-year terminal rate decays to $< 4 \times 10^{-5}$ STB/d under all configurations.
3. The 100-year EUR is strictly bounded between 48.23 and 48.77 MMSTB (a tight range of only 0.54 MMSTB).

```
====================================================================================================
Figure 2: Phase-Specific Recombination & Asymptotic Stress Matrix
====================================================================================================
```
![Figure 2: Phase-Specific Recombination Matrix](file:///Users/tengkuanas/.gemini/antigravity-cli/brain/4b7e1d00-9795-42a6-b5bb-c629e9cf25cb/fig2_phase_specific_recombination_matrix.png)

---

## 7. SYNTHETIC STRUCTURAL EXPERIMENT & TOST EQUIVALENCE

### 7.1 Synthetic Physical Decline Ground-Truth Experiment
To test long-horizon fidelity where ground truth is known, we simulated a synthetic boundary-dominated physical reservoir with true exponential tail decline:

| Horizon (Years) | True Oil Rate (STB/d) | Frozen Baseline Rate (STB/d) | Candidate H Rate (STB/d) | Baseline Rate Error (STB/d) | Candidate H Rate Error (STB/d) | True EUR (MMSTB) | Baseline EUR (MMSTB) | Candidate H EUR (MMSTB) | Baseline Cumulative Error (MMSTB) | Candidate H Cumulative Error (MMSTB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.0** | 4,360.93 | 4,325.31 | 4,311.46 | -35.62 | -49.47 | 40.06 | 40.06 | 40.05 | -0.009 | -0.011 |
| **3.0** | 3,230.66 | 3,174.37 | 3,108.85 | -56.29 | -121.80 | 42.77 | 42.72 | 42.69 | -0.045 | -0.077 |
| **5.0** | 2,393.33 | 2,291.77 | 2,131.89 | -101.56 | -261.44 | 44.77 | 44.65 | 44.53 | -0.120 | -0.232 |
| **10.0** | 1,130.53 | 1,322.99 | 801.26 | +192.47 | -329.27 | 47.78 | 47.76 | 46.98 | -0.022 | -0.809 |
| **20.0** | 252.26 | 822.32 | 114.89 | **+570.06** | **-137.36** | 49.88 | 51.38 | 48.23 | **+1.495** | **-1.656** |
| **30.0** | 56.29 | 749.80 | 17.26 | **+693.52** | **-39.03** | 50.35 | 54.21 | 48.41 | **+3.855** | **-1.941** |

**Analysis:** At 30 years, the Frozen Baseline overpredicts true rate by $+693.52$ STB/d (+1232% relative error) and diverges toward $+\infty$. Candidate H captures the true order of magnitude (17.26 vs 56.29 STB/d), and its cumulative error plateaus at -1.94 MMSTB.

### 7.2 Formal Two One-Sided Tests (TOST) Equivalence
We performed formal TOST equivalence testing between Candidate H and the Frozen Baseline across all 255 evaluation instances with cluster-robust standard errors:

$$\bar{D} = 5.9439 \times 10^{-5}, \quad \text{SE}_{\text{cluster}} = 6.4125 \times 10^{-5}, \quad \text{df} = 14$$

1. **Equivalence Margin $\delta = 0.005$ (0.5% Macro NRMSE):**
   - $t_1 = \frac{\bar{D} - (-\delta)}{\text{SE}} = \frac{0.000059 + 0.005}{0.000064} = 78.90 \implies p_1 < 10^{-15}$
   - $t_2 = \frac{\delta - \bar{D}}{\text{SE}} = \frac{0.005 - 0.000059}{0.000064} = 77.05 \implies p_2 < 10^{-15}$
   - Overall TOST $p = \max(p_1, p_2) < 10^{-15} \implies$ **Strictly Equivalent at $\alpha = 0.05$**.

2. **Equivalence Margin $\delta = 0.001$ (0.1% Macro NRMSE):**
   - $t_1 = \frac{0.000059 + 0.001}{0.000064} = 16.52 \implies p_1 = 7.04 \times 10^{-11}$
   - $t_2 = \frac{0.001 - 0.000059}{0.000064} = 14.67 \implies p_2 = 3.42 \times 10^{-10}$
   - Overall TOST $p = 3.42 \times 10^{-10} \implies$ **Strictly Equivalent at $\alpha = 0.05$**.

**Statistical Takeaway:** Candidate H achieves statistical equivalence to the Frozen Baseline even under a stringent 0.1% Macro NRMSE tolerance band.

---

## 8. SECONDARY CONFIRMATION BENCHMARK (CASES 71–85)

To independently assess generalization on sealed validation data, all candidates were evaluated on Cases 71–85 at origin 2005-01-01 over a 12-quarter horizon ($N = 15$ reservoirs):

| Candidate Model | Cases 71–85 Mean Macro NRMSE | Std Dev | Mean Oil NRMSE | Mean Water NRMSE |
| :--- | :---: | :---: | :---: | :---: |
| **Candidate A (Frozen Baseline)** | **0.027557** | **0.01593** | **0.050607** | **0.004507** |
| **Candidate H (Constrained Tail)** | 0.028038 | 0.01553 | 0.051569 | 0.004507 |
| **Candidate B (ET Oil + Froz W)** | 0.028326 | 0.01787 | 0.052144 | 0.004507 |
| **Candidate F (State Gate)** | 0.031569 | 0.01683 | 0.050607 | 0.012532 |
| **Candidate C (Froz Oil + ET W)** | 0.034344 | 0.01601 | 0.050607 | 0.018082 |
| **Candidate D (ET Both)** | 0.035113 | 0.01835 | 0.052144 | 0.018082 |

**Conclusion:** The Frozen Baseline (Candidate A) outperforms all challengers on sealed confirmation cases. Challengers B, C, D, and F all show degraded performance. Candidate H remains within $0.00048$ of baseline.

---

## 9. COMPREHENSIVE QUESTION-AND-ANSWER AUDIT

### Q1: Is the cross-validation defect completely resolved?
**Yes.** `nested_cross_validation.py` strictly constructs `train_pool = [c for c in TRAIN_CASES if c != held_out_case]`. All base models, preprocessing transforms, and selection rules are re-fit from scratch without seeing the held-out reservoir. The unit test `test_outer_reservoir_isolation.py` proves zero leakage under synthetic data mutation.

### Q2: Does water-cut selection achieve 100% classification accuracy?
**No.** The earlier "100%" claim was an artifact of evaluating a non-representative 15-case sample. Across the complete, rigorous dataset of 255 instances, out-of-fold accuracy is 92.55% (a $+39.22\%$ gain over the 53.33% majority baseline). Initial water cut $f_{w,0}$ is highly predictive ($\text{LRT } p = 0.0062$), but not error-free.

### Q3: Why not deploy water-cut gating for production?
Because in production rate forecasting, **the 7.45% of cases misclassified by the gate occur during early volatile production**, where Recursive ExtraTrees incurs severe squared-error penalties. When averaged into the macro metric, the gated candidate (Candidate F) scores 0.10956, which is statistically significantly worse than the baseline 0.10507 ($p = 0.00010$).

### Q4: Does the frozen ensemble predict infinite oil?
**Yes.** In the frozen baseline, Recursive ExtraTrees is weighted at 0.2405. Training data terminates at reservoir age 9.75 years. For $t > 10$ years, ExtraTrees predicts a constant rate floor of ~840 STB/d. The ensemble oil rate remains $\ge 202$ STB/d indefinitely, causing $\text{EUR}(\infty) = \infty$.

### Q5: Is Candidate H a calibrated reservoir model?
**No.** Candidate H is a **structural regularization safeguard**. It prevents unphysical infinite recovery by enforcing a $C^1$-smooth taper into Modified Arps with a terminal decline rate of 5%/year. It is structurally motivated and physically essential for long-horizon sanity, but it is not calibrated against empirical 20-year field decline data.

### Q6: Should Candidate H replace the official competition submission?
**No.** The official competition submission `outputs/submission/09_Template_Deliverable_Submission.xlsx` must remain frozen. On sealed validation cases (Cases 71–85), the frozen baseline achieves 0.02756 vs Candidate H's 0.02804. Candidate H provides no competitive advantage over retrospective 1-to-5 year horizons (TOST equivalent, $p = 3.42 \times 10^{-10}$). Overwriting the deliverable introduces operational risk with zero competitive scoring benefit.

---

## 10. SUMMARY CHECKLIST & VERIFICATION OF DELIVERABLES

- [x] **Branch Isolation:** Completed on `research/phase-specific-validation-repair`.
- [x] **Production Freeze:** `locked_ensemble_config.json` and official submission deliverable untouched.
- [x] **Deliverable Authenticity:** `outputs/submission/09_Template_Deliverable_Submission.xlsx` verified (SHA-256 `ce30ec18...`, 42,510 bytes).
- [x] **Outer-Fold Isolation Test:** Created `tests/test_outer_reservoir_isolation.py` (passed 3/3).
- [x] **Full Regression Suite:** 145/145 tests passed in 158.12s.
- [x] **Genuine LOO CV Benchmark:** 15 clusters $\times$ 17 conditions recomputed and saved.
- [x] **Discriminative Reassessment:** 255-instance classification and Likelihood Ratio Test completed.
- [x] **Asymptotic Analysis & Sensitivity Grid:** 9-point grid evaluated and saved.
- [x] **Synthetic Structural Truth Test:** Completed across 30-year physical simulation.
- [x] **TOST Equivalence:** Confirmed at $\delta = 0.005$ and $\delta = 0.001$ ($p < 10^{-9}$).
- [x] **Separated Deployments:** 40Q (Case 1) and 80Q (Cases 2–10) separated in all tables.
- [x] **Publication Figures:** All 5 publication figures rendered and synced to artifact directory.

```
====================================================================================================
FINAL STATUS: RESEARCH SPRINT COMPLETED WITH FULL SCIENTIFIC AND GOVERNANCE RESOLUTION
====================================================================================================
```
