# ExxonMobil DataWorks Challenge 2026 — Methodology Presentation Outline

**Suggested Presentation Slides & Experimental Evidence Mapping**

---

### Slide 1: Executive Overview & Study Objectives
- **Title:** Physics-Coupled Horizon-Adaptive Stacked Surrogate for 20-Year Reservoir Forecasting
- **Core Challenge:** Accurately history-matching 10 years of observed reservoir dynamics (1998–2008) and reliably extrapolating multi-phase production over 10- and 20-year horizons (to 2028).
- **Major Breakthrough:** Transition from static SVD-GP curve fitting (which suffered from historical overfitting and synthetic polynomial extrapolation) to a true dynamic forward forecasting ensemble combining Recursive ML, Learned Decline Curve Analysis (DCA), and exact physical solution-GOR coupling.
- **Headline Result:** 73.7% macro NRMSE reduction vs standalone hyperbolic DCA; verified 0.04924 validation NRMSE and 0.04960 nested OOF NRMSE; 100% win rate across holdout validation cases ($p = 0.0001$).

---

### Slide 2: Background & Forensic Discovery of Baseline Defects
- **The Pitfalls of Unconstrained SVD Extrapolation:**
  - Audit demonstrated that earlier 80-quarter GP forecasts were purely mathematical polynomial extensions decoupled from dynamic reservoir states.
  - SVD basis fitted on short historical windows cannot extrapolate forward dynamics without strong physical constraints.
- **Hyperbolic DCA Tail Runaway:**
  - Standard Arps hyperbolic decline with $b > 0$ generates fat, unphysical late-time production tails when extrapolated to 20 years.
  - Case 71 and other development cases revealed that unconstrained DCA led to unbounded EUR accumulation.
- **Cross-Phase Decoupling:**
  - Independent fluid modeling violated the invariant physical solution-GOR ratio ($R_s = 0.3633$ MSCF/STB), resulting in distorted gas predictions.

---

### Slide 3: Model Architecture — `Blend_Phase_Specific_Simplex`
- **Dynamic Multi-Model Diversity:**
  1. **Recursive ExtraTrees:** High-accuracy short-horizon non-linear state transitions conditioned on dynamic lag features ($q_o, q_w, f_w, t_{\text{lead}}$) and static geology.
  2. **Learned Exponential DCA:** Bounds late-time production with constant fractional decline $D$, preventing late-time runaway.
  3. **Learned Hyperbolic DCA:** Captures moderate transient boundary-dominated decline in water production.
  4. **Hybrid Dynamic:** Seamlessly bridges early-time machine-learned rates into physical boundary decline.
- **Phase-Specific Simplex Stacking Weights (Frozen):**
  - **Oil Phase:**
    - Recursive ExtraTrees: $24.05\%$
    - Learned Exponential DCA: $61.79\%$ (primary physical anchor)
    - Learned Hyperbolic DCA: $0.00\%$ (eliminated to prevent oil tail divergence)
    - Hybrid Dynamic: $14.16\%$
  - **Water Phase:**
    - Recursive ExtraTrees: $49.72\%$
    - Learned Exponential DCA: $25.14\%$
    - Learned Hyperbolic DCA: $25.14\%$
    - Hybrid Dynamic: $0.00\%$
  - **Gas Phase:**
    - Coupled via exact physical solution GOR: $Q_g(t) = Q_{g,0} + 0.3633 \times [Q_o(t) - Q_{o,0}]$ ($<10^{-6}$ residual).

---

### Slide 4: Validation Methodology & Rigorous Cross-Validation
- **Nested Outer/Inner Cross-Validation:**
  - 5 Outer Folds $\times$ 4 Inner Folds across Cases 1–70 with independent random seeds (Seed 42 & Seed 123).
  - Strictly prevents target and meta-learner leakage.
- **Statistical Significance Verification:**
  - 10,000-resample paired bootstrap testing on holdout validation cohort (Cases 71–85).
  - 95% Confidence Interval for macro error difference: $[-0.1431, -0.1290]$, confirming unambiguous superiority ($p = 0.0001$).
  - 15 out of 15 case-level wins against standalone hyperbolic DCA.

---

### Slide 5: Physical Consistency & Quality Assurance
- **Strict Monotonicity & Non-Negativity:**
  - Production rates $q(t) \ge 0$ guaranteed at every quarter.
  - Cumulative totals $Q(t)$ strictly non-decreasing across all phases.
- **Historical Anchor Preservation:**
  - Trajectories anchored at 2008-01-01 historical cumulative base; zero unphysical first-quarter jumps.
- **Rate-to-Cumulative Reconciliation:**
  - Numerical differentiation verifies that $\max | (Q_k - Q_{k-1})/\Delta t_k - q_k | < 10^{-6}$.
- **Late-Horizon Tail Stabilization:**
  - $61.8\%$ Exponential DCA oil weighting enforces conservative recovery limits in the 10-to-20-year tail.

---

### Slide 6: Competition Deliverable Mapping & Uncertainty Scenarios
- **Probabilistic Forecast Scenarios (`09 Template Deliverable.xlsx`):**
  - **Case 1:** P10 (Optimistic Scenario, 90th percentile EUR) — exactly 40 quarters (10-year horizon through 2018-01-01).
  - **Case 2:** P50 (Base Case Scenario, median EUR) — exactly 80 quarters (20-year horizon through 2028-01-01).
  - **Case 3:** P90 (Pessimistic Scenario, 10th percentile EUR) — exactly 80 quarters (20-year horizon through 2028-01-01).
  - **Cases 4–10:** 7 representative posterior realization samples spanning the reservoir parameter uncertainty space (P20 to P85) — exactly 80 quarters each.
- **End-to-End Verification:**
  - 10/10 sheets populated, 0 NaNs, 0 negative increments, 100% compliance with official Excel specifications.

---

### Slide 7: Business Impact & Strategic Next Steps
- **Production Forecasting Confidence:**
  - Robust uncertainty bounds enable informed capital allocation and field development planning under reservoir uncertainty.
- **Computational Efficiency:**
  - Inference completes in under 1 second, enabling interactive real-time decision support in Power BI.
- **Extensibility:**
  - Architecture ready for integration with automated well control and water shutoff optimization.
