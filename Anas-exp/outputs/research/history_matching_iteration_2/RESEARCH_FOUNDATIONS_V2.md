# Research Iteration 2: Scientific Foundations and Methodological Mapping

**Project:** ExxonMobil DataWorks Challenge 2026  
**Workspace:** `/Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp`  
**Branch:** `research/history-matching-calibration-v2`  
**Baseline Commit:** `5c614674b573ff9d525f8e14b76ebc0825caf9ce`  

---

## 1. Methodological Mapping: Literature to Experiments

This table establishes the formal mapping from published peer-reviewed methodology to the specific experiments, mathematical formulations, and validation gates implemented in Research Iteration 2.

| Paper / Source | Key Theoretical Insights & Theorems | Required Mathematical Assumptions | Precise Experiment in Iteration 2 | Section / Module |
|---|---|---|---|---|
| **Kennedy & O'Hagan (2001)**<br/>*Bayesian Calibration of Computer Models* | Explicit decomposition of uncertainty: observation error $\boldsymbol{\epsilon}_{\text{obs}}$, simulator discrepancy $\boldsymbol{\delta}_{\text{disc}}$, and emulator epistemic uncertainty $\boldsymbol{\delta}_{\text{emu}}(\boldsymbol{\theta})$. Calibration likelihood requires joint covariance $\boldsymbol{\Sigma}(\boldsymbol{\theta}) = \boldsymbol{\Sigma}_{\text{obs}} + \boldsymbol{\Sigma}_{\text{emu}}(\boldsymbol{\theta})$ and parameter-dependent log-determinant $\log \det \boldsymbol{\Sigma}(\boldsymbol{\theta})$ to penalize artificially inflated variance (Occam's razor). | Additive Gaussian error processes; differentiable emulator covariance; stationary observation noise. | **Experiment L0 vs L1 vs L2 vs L3**: Rigorous Cholesky-based multivariate Gaussian likelihood with propagated SVD-GP epistemic variance and exact log-det normalization. | Section 2<br/>[`history_matching_likelihood_v2.py`](file:///Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp/src/history_matching_v2/history_matching_likelihood_v2.py) |
| **Vehtari et al. (2021)**<br/>*Rank-Normalization, Folding, and Localization: An Improved R-hat* | Traditional $\hat{R}$ fails to detect variance differences and heavy-tailed non-convergence. Rank-normalized split-$\hat{R}$ transforms samples to normal quantiles across split chains. Folded split-$\hat{R}$ transforms $|x - \text{median}|$ to evaluate tail scale. Bulk-ESS and Tail-ESS quantify precision for mean vs extreme quantiles ($P_{10}, P_{90}$). | Independent Markov chains; finite variance of rank-transformed variables. | **Rigorous MCMC Diagnostic Suite**: 4 chains $\times$ 4,000 draws. Compute rank-normalized split-$\hat{R}$, folded $\hat{R}$, bulk-ESS, tail-ESS, and quantile Monte Carlo standard errors (MCSE). | Section 5<br/>[`bayesian_mcmc_v2.py`](file:///Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp/src/history_matching_v2/bayesian_mcmc_v2.py) |
| **Talts et al. (2018/2020)**<br/>*Validating Bayesian Inference via Simulation-Based Calibration* | If inference is statistically sound, posterior draws conditioned on synthetic data generated from prior draws must satisfy uniformity: rank statistic $r = \sum_{l=1}^L \mathbb{I}(\theta^{(l)} < \theta^*) \sim \text{Uniform}(0, L)$. Pseudo-field recovery validates credible interval empirical coverage against true parameters. | Model generative consistency; prior samples cover physically realizable reservoir states. | **SBC & Pseudo-Field Recovery Tests**: Empirical coverage evaluation of 50%, 80%, and 90% credible intervals across held-out simulator cases (Cases 71–85) and synthetic prior realizations. | Section 5<br/>[`simulation_based_calibration.py`](file:///Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp/src/history_matching_v2/simulation_based_calibration.py) |
| **Sacks et al. (1989)**<br/>*Design & Analysis of Computer Experiments (DACE)* | Deterministic simulator outputs have no random replication error; surrogate residuals represent deterministic approximation error. GP validation requires independent cross-validation, rate-vs-cumulative consistency, and checking parameter space interpolation support (convex hull / distance metrics). | Smooth continuous response surface; stationary covariance kernel (Matérn 5/2). | **Surrogate Revalidation**: Time-aligned rate differentiation, 5-fold CV on Cases 1–70, held-out validation on Cases 71–85, empirical variance coverage, and parameter convex hull check. | Section 3<br/>[`emulator_revalidation_v2.py`](file:///Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp/src/history_matching_v2/emulator_revalidation_v2.py) |
| **Evensen (2021)**<br/>*Formulation of Correlated Observation Errors in Data Assimilation* | Cumulative production and rate measurements are strongly auto-correlated. Neglecting temporal correlation ($\rho > 0$) causes variance collapse by treating dependent quarters as independent evidence. AR(1) precision matrix is tridiagonal, filtering smooth offsets and penalizing curvature mismatches. | Stationary temporal autocorrelation parameter $\rho \in [0.5, 0.95]$. | **Observation Covariance Sensitivity**: Sweep $\rho \in [0.50, 0.80, 0.95]$ and relative noise $\sigma_{\text{obs}} \in [0.02, 0.05, 0.10]$ to quantify impact on MAP and posterior dispersion. | Section 2 & 4<br/>[`history_matching_likelihood_v2.py`](file:///Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp/src/history_matching_v2/history_matching_likelihood_v2.py) |
| **Mahjour et al. (2020)**<br/>*Proxy-Assisted History Matching & Clustering* | History-matched scenarios should not be selected using ad-hoc static rate proxies. Representative models must reflect long-horizon forecast spread and capture multi-modal reservoir states without sacrificing match quality. | Forecast model is stationary across the forecast origin; joint parameter covariance is preserved. | **Forecast-Consistent Case Selection**: Evaluate full 80-quarter predictions using the frozen `Blend_Phase_Specific_Simplex` stack across thinned posterior samples. Rank actual 20-yr EUR. | Section 6<br/>[`scenario_selection_v2.py`](file:///Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp/src/history_matching_v2/scenario_selection_v2.py) |

---

## 2. Mathematical Formalization of Likelihood Models

### 2.1 The Observation Vector and Time Alignment
Let $\mathbf{t} = [t_0, t_1, \dots, t_{N-1}]$ represent the 41 quarterly timestamps from `1998-01-01` to `2008-01-01`.
The actual duration of each quarter $i \in \{0, \dots, N-2\}$ is:
$$\Delta t_i = t_{i+1} - t_i \quad \text{days} \quad (\Delta t_i \in \{90, 91, 92\})$$

In `field_history.parquet`:
- `ramp_oil_cum_stb` ($Q_{o, i}$) and `ramp_water_cum_stb` ($Q_{w, i}$) record cumulative production up to timestamp $t_i$.
- `ramp_oil_rate_stbd` ($q_{o, i}$) and `ramp_water_rate_stbd` ($q_{w, i}$) record the **forward production rate** for the quarter starting at $t_i$ and ending at $t_{i+1}$:
$$q_{o, i} = \frac{Q_{o, i+1} - Q_{o, i}}{\Delta t_i}$$
for $i = 0, \dots, N-2$ (40 quarters).
Timestamp $t_{40}$ (`2008-01-01`) records the instantaneous/anchor rate for the quarter beginning at the forecast origin ($q_{o, \text{term}} = 5,066.68$ STB/d).

### 2.2 Likelihood Formulations

#### Model L0: Iteration 1 Heuristic Objective (Reference)
$$\mathcal{M}_{\text{L0}}(\boldsymbol{\theta}) = \frac{1}{2} \left[ \frac{1}{N} \mathbf{r}_{qo}^T \boldsymbol{\Sigma}_{\text{AR1}, qo}^{-1} \mathbf{r}_{qo} + \frac{1}{N} \mathbf{r}_{qw}^T \boldsymbol{\Sigma}_{\text{AR1}, qw}^{-1} \mathbf{r}_{qw} \right] + 0.20 \times e_{\text{term}}^2 + \text{emu\_penalty}$$
*Defects:*
1. Lacks normalizing constant and log-determinant.
2. Inadvertent backward vs forward rate misalignment caused 99.5% of misfit to originate from a single index (Step 3).

#### Model L1: Proper Gaussian Likelihood on Forward Rates with AR(1) Covariance
For phase $p \in \{\text{oil}, \text{water}\}$, the forward rate residual vector is:
$$\mathbf{r}_p(\boldsymbol{\theta}) = \mathbf{q}_{\text{pred}, p}(\boldsymbol{\theta}) - \mathbf{q}_{\text{obs}, p} \in \mathbb{R}^{40}$$
Observation covariance:
$$\boldsymbol{\Sigma}_{\text{obs}, p} = \sigma_{\text{obs}, p}^2 \mathbf{R}_{\rho} \in \mathbb{R}^{40 \times 40}$$
where $[\mathbf{R}_{\rho}]_{ij} = \rho^{|i - j|}$ and $\sigma_{\text{obs}, p} = c_{\text{noise}} \cdot \text{RMS}(\mathbf{q}_{\text{obs}, p})$.
Cholesky factor: $\mathbf{L}_p \mathbf{L}_p^T = \boldsymbol{\Sigma}_{\text{obs}, p}$.
Solve $\mathbf{L}_p \mathbf{z}_p = \mathbf{r}_p(\boldsymbol{\theta})$.
$$\log p_{L1}(\mathbf{y} \mid \boldsymbol{\theta}) = -\frac{1}{2} \sum_{p \in \{o, w\}} \left[ \mathbf{z}_p^T \mathbf{z}_p + 2 \sum_{i=1}^{40} \log L_{p, ii} + 40 \log(2\pi) \right]$$

#### Model L2: Proper Gaussian Likelihood with Propagated SVD-GP Epistemic Uncertainty
Under Kennedy & O'Hagan (2001), the emulator adds epistemic covariance $\boldsymbol{\Sigma}_{\text{emu}, p}(\boldsymbol{\theta})$.
For SVD-GP with components $\mathbf{V}_{K, p} \in \mathbb{R}^{41 \times K}$ and latent GP score standard deviations $\mathbf{s}_{k, p}(\boldsymbol{\theta})$:
$$\boldsymbol{\Sigma}_{\text{emu}, \text{cum}, p}(\boldsymbol{\theta}) = \mathbf{V}_{K, p} \operatorname{diag}(\mathbf{s}_{1..K, p}^2(\boldsymbol{\theta})) \mathbf{V}_{K, p}^T$$
Transforming cumulative covariance to forward rate covariance via differencing operator $\mathbf{D} \in \mathbb{R}^{40 \times 41}$:
$$\mathbf{D}_{i, i} = -\frac{1}{\Delta t_i}, \quad \mathbf{D}_{i, i+1} = \frac{1}{\Delta t_i}$$
$$\boldsymbol{\Sigma}_{\text{emu}, \text{rate}, p}(\boldsymbol{\theta}) = \mathbf{D} \boldsymbol{\Sigma}_{\text{emu}, \text{cum}, p}(\boldsymbol{\theta}) \mathbf{D}^T$$
Total parameter-dependent covariance:
$$\boldsymbol{\Sigma}_p(\boldsymbol{\theta}) = \boldsymbol{\Sigma}_{\text{obs}, p} + \boldsymbol{\Sigma}_{\text{emu}, \text{rate}, p}(\boldsymbol{\theta})$$
Cholesky factor: $\mathbf{L}_p(\boldsymbol{\theta}) \mathbf{L}_p(\boldsymbol{\theta})^T = \boldsymbol{\Sigma}_p(\boldsymbol{\theta})$.
$$\log p_{L2}(\mathbf{y} \mid \boldsymbol{\theta}) = -\frac{1}{2} \sum_{p \in \{o, w\}} \left[ \mathbf{r}_p^T \boldsymbol{\Sigma}_p(\boldsymbol{\theta})^{-1} \mathbf{r}_p + \log \det \boldsymbol{\Sigma}_p(\boldsymbol{\theta}) + 40 \log(2\pi) \right]$$

#### Model L3: Cumulative Increments + Terminal Regularization
Evaluates quarterly incremental volumes $\Delta Q_{p, i} = Q_{p, i+1} - Q_{p, i}$ directly, coupled with terminal rate anchoring:
$$\log p_{L3}(\mathbf{y} \mid \boldsymbol{\theta}) = \log p_{L2}(\Delta \mathbf{Q} \mid \boldsymbol{\theta}) - \frac{1}{2 \sigma_{\text{term}}^2} (q_{\text{pred}, \text{term}}(\boldsymbol{\theta}) - q_{\text{obs}, \text{term}})^2$$
without double-counting cumulative totals.
