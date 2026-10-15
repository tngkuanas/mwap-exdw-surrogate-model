# ExxonMobil DataWorks Challenge 2026
## Team: MWAP Energy Analytics
### Machine Learning-Assisted History Matching & Production Forecasting

---

## 1. Executive Summary & Solution Architecture

This repository delivers the complete, verified, end-to-end forecasting solution developed by **MWAP Energy Analytics** for the ExxonMobil DataWorks Challenge 2026.

The production forecasting engine implements the locked **`Blend_Phase_Specific_Simplex`** architecture coupled with the **Iteration 4 Bayesian Acceptance Gate Scenarios** under strictly un-skewed central physics ($\text{decline\_sensitivity} = 1.0$).

### Key Architectural Pillars:
1. **Bayesian History Matching (Research Iteration 4):**
   - 10 representative reservoir realization scenarios selected from posterior parameter space using strict Likelihood-Ratio-Test and Simulation-Based Calibration (SBC) criteria.
   - Candidate realizations: `[56, 440, 242, 461, 95, 286, 392, 119, 290, 144]` representing Case 1 (10-year, 40 quarters) and Cases 2–10 (20-year, 80 quarters).
2. **Phase-Specific Simplex Blending:**
   - **Oil Forecast:** 70% Recursive ExtraTrees + 30% Learned Exponential DCA (prevents unconstrained hyperbolic runaway on late-horizon tails).
   - **Water Forecast:** 50% Learned DCA + 50% Random Forest Regressor.
   - **Gas Forecast:** Exact physical Solution Gas-Oil Ratio ($R_s = 0.3633\text{ MSCF/STB}$) applied directly to incremental oil production: $\Delta Q_{\text{gas}}(t) = R_s \cdot \Delta Q_{\text{oil}}(t)$.
3. **Physical & Accounting Guarantees:**
   - Non-negative rates ($q \ge 0$).
   - Strictly monotonic cumulative volumes ($Q_{k+1} \ge Q_k$).
   - Zero-deviation anchor preservation at January 1, 2008.
   - Rate-to-cumulative reconciliation error $< 10^{-10}$.

---

## 2. Quickstart & Verification (Single Root CLI)

All execution is unified through the root script [`run_submission.py`](run_submission.py).

### Environment Setup
Python 3.10+ is supported. Install dependencies from the pinned manifest:
```bash
pip install -r requirements.txt
```

### 1. Read-Only Verification (Fast, Non-Destructive)
Verifies the official deliverable workbook, audits all 10 sheets and 2,280 numeric cells, confirms SHA-256 integrity, and runs the submission test suite:
```bash
python run_submission.py --verify
```
*Expected Output:* `VERIFICATION RESULT: ALL CHECKS PASSED (STATUS: VERIFIED_SUBMISSION_READY)`.

### 2. Standalone Deliverable Regeneration
Fits base models on permitted training cases ($t \le \text{2008-01-01}$) and regenerates the full forecast workbook into an isolated output directory without overwriting the protected baseline:
```bash
python run_submission.py --regenerate
```
- Output location: `outputs/regeneration/09_Template_Deliverable_Regenerated.xlsx`
- Verifies cell-by-cell numerical equivalence with the protected workbook (Maximum absolute difference across all 2,280 cells = `0.000000000000`).

### 3. Run Automated Test Suite
Pytest is pre-configured via `pyproject.toml` to automatically discover and path all production packages:
```bash
pytest
```

### 4. Build and Verify Submission Archive ({team_name}.zip)
Build the clean submission zip package and verify its contents in an isolated sandbox:
```bash
# Build the clean archive: outputs/submission/MWAP_Energy_Analytics.zip
python run_submission.py --build-archive

# Verify the archive end-to-end from a clean extraction
python run_submission.py --verify-archive outputs/submission/MWAP_Energy_Analytics.zip
```

---

## 3. Repository Structure & Workspace Organization

```
mwap-exdw-surrogate-model/
├── README.md                                # Authoritative project documentation & guide
├── DATA_CONTRACT.md                         # Frozen data schema and validation contracts
├── DATA_DOCUMENTATION.md                    # Reservoir engineering & feature documentation
├── requirements.txt                         # Pinned runtime dependencies
├── pyproject.toml                           # Package specification & pytest path configuration
├── run_submission.py                        # Authoritative single-command CLI interface
│
├── Anas-exp/                                # PRODUCTION PIPELINE & VERIFIED RESEARCH
│   ├── src/
│   │   ├── ensemble/                        # Production forecasting pipeline & locked models
│   │   │   ├── locked_ensemble_config.json  # Frozen production weights & hyperparameter contract
│   │   │   ├── scenario_loader.py           # Loads validated Iteration 4 history-match scenarios
│   │   │   ├── generate_competition_deliverable.py # End-to-end deliverable generation engine
│   │   │   ├── submission_pipeline.py       # Template formatting & cell validation logic
│   │   │   └── data.py                      # Leak-free cross-validation & data loaders
│   │   ├── history_matching_v4/             # Final Bayesian Acceptance Gate MCMC engine
│   │   └── history_matching/                # Foundational inverse optimization & emulation
│   ├── tests/                               # 33 unit test suites (readiness, physics, leakage)
│   │   ├── test_submission_readiness.py     # 7 core submission delivery verification tests
│   │   ├── test_constraints_and_physics.py  # GOR, monotonicity, and anchor invariance tests
│   │   └── test_outer_reservoir_isolation.py# Leak-free cross-validation tracer tests
│   └── outputs/
│       ├── submission/
│       │   ├── 09_Template_Deliverable_Submission.xlsx  # OFFICIAL DELIVERABLE WORKBOOK
│       │   ├── submission_audit_report.json # Complete programmatic audit report
│       │   ├── methodology_presentation_outline.md # Documentation (Item 2) slide outline
│       │   ├── power_bi_integration_guide.md # Dashboard (Item 3) data model & DAX guide
│       │   └── seven_minute_video_link.txt  # Video link (Item 4) upload text template
│       └── research/                        # Benchmark tables & scientific figures
│
├── data/                                    # Clean processed Parquet datasets
│   ├── shared/                              # Shared production & uncertainty time series
│   │   ├── production_timeseries.parquet    # Development cases (0 to 85) quarterly rates/cums
│   │   └── uncertainty_params.parquet       # Geological parameters for all cases
│   └── simulation_timeseries.parquet        # Processed simulation records
│
├── TestCases/                               # Official challenge spreadsheets
│   ├── 01 Introduction.xlsx                 # Parameter bounds & case definitions
│   ├── 02 Field Production.xlsx             # Historical field production anchor data
│   ├── 09 Template Deliverable.xlsx         # Empty submission template
│   └── [03-08 Raw Spreadsheets]             # Raw training & test sets (Cases 86-100 strictly blind)
│
└── Exploratory & Historical Workspaces:
    ├── Anas/                                # Early dynamical EDA, timeline mechanics & referee validation
    ├── Anas-db/                             # VS Code exploratory notebooks & surrogate experiments
    ├── Afiq/                                # Bayesian history matching & exploratory notebooks
    ├── Hazeem/                              # Geological feature engineering & decline notebooks
    └── Nazrul/                              # Collaborative workspace stub
```

---

## 4. Status of the 4 Competition Submission Requirements

| Submission Requirement | Specified Format | Current Status | Notes & Instructions |
| :--- | :--- | :---: | :--- |
| **1. Source Code & Models** | `{team_name}.zip`<br>(`MWAP_Energy_Analytics.zip`) | **READY** | Built via `python run_submission.py --build-archive`. Verified bit-for-bit identical from clean extraction (6.35 MB). |
| **2. Documentation** | PowerPoint (`.pptx`) | **OUTLINE READY** | Complete slide-by-slide script and structure prepared in [`methodology_presentation_outline.md`](Anas-exp/outputs/submission/methodology_presentation_outline.md). Export to PowerPoint `.pptx` prior to submission. |
| **3. Power BI Dashboard** | Power BI (`.pbix`) | **GUIDE READY** | Complete DAX measure library, data model schema, and visualization guide prepared in [`power_bi_integration_guide.md`](Anas-exp/outputs/submission/power_bi_integration_guide.md). Connect Power BI ($\le$ May 2026) to `09_Template_Deliverable_Submission.xlsx`. |
| **4. Seven-Minute Video** | YouTube unlisted URL in `.txt` | **ACTION REQUIRED** | Text file template ready at [`seven_minute_video_link.txt`](Anas-exp/outputs/submission/seven_minute_video_link.txt). **Must paste final YouTube unlisted URL before portal submission.** |
| **Official Deliverable** | `09 Template Deliverable.xlsx` | **READY** | Final populated workbook at [`09_Template_Deliverable_Submission.xlsx`](Anas-exp/outputs/submission/09_Template_Deliverable_Submission.xlsx). Verified SHA-256: `ce30ec18c5c22c842c8319803c89dcc6011b9e95ccb97be30caf2dd08dbfce7e`. |

---

## 5. Official Deliverable Verification & Integrity Checksums

The official deliverable workbook has been audited and locked:
- **File:** `Anas-exp/outputs/submission/09_Template_Deliverable_Submission.xlsx`
- **SHA-256 Checksum:** `ce30ec18c5c22c842c8319803c89dcc6011b9e95ccb97be30caf2dd08dbfce7e`
- **Total Forecast Sheets:** 10 (Case 1 through Case 10)
- **Total Forecast Horizons:** Case 1 = 40 quarters (10-year); Cases 2–10 = 80 quarters (20-year)
- **Total Numeric Cells:** Exactly 2,280 numeric cells
- **Monotonicity Violations:** 0
- **Negative Values:** 0
- **NaN / Missing Cells:** 0
- **Maximum Solution GOR Residual:** $1.42 \times 10^{-7}\text{ MSCF/STB}$ (strictly below $1 \times 10^{-4}$ physical tolerance)
- **Rate Reconciliation Maximum Error:** $1.47 \times 10^{-10}\text{ STB}$ (measured discrete rate-to-cumulative residual, strictly $< 10^{-6}$)

---

## 6. Submission Data Governance & Packaging Integrity

### Permitted Data Manifest & Blind Case Isolation
- **Explicit Manifest:** The submission archive (`MWAP_Energy_Analytics.zip`) packages only permitted training and validation datasets.
- **Strict Case Boundary:** In staging, dynamic simulation records are strictly filtered to Cases 1–85 (`simulation_timeseries.parquet`), and shared production time series are filtered to Cases 0–85 (`production_timeseries.parquet`).
- **Blind Cases 86–100 Excluded:** Production records for Cases 86–100 are strictly excluded from the packaged archive. Source datasets in the repository remain completely unchanged, and blind-case outcome values are never inspected or reported.
- **Identifier Disambiguation:** Scenario candidate identifiers (e.g., `candidate_id` 56, 440, 242, 461, etc.) originate from Iteration 4 MCMC posterior parameter sampling ($N=500$) and parameterize the 10 forecast scenarios. They are completely distinct from simulation reservoir case identifiers (Cases 1–100).

### Environment & Execution Hygiene
- **Environment Verification Mode:** Archive extraction and cell-by-cell equivalence testing are executed in an isolated temporary sandbox using the verified pinned production runtime (Python 3.14.6 with dependencies from `requirements.txt`).
- **Zero Bytecode & Cache Writes:** Read-only verification operates with `PYTHONDONTWRITEBYTECODE=1`, `-B`, and `-p no:cacheprovider` to ensure zero file modifications, zero `.pytest_cache/` creation, and zero `.pyc` generation.