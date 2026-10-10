# ExxonMobil DataWorks Challenge 2026 — Power BI Integration Guide

**Dashboard Data Source & Visualization Architecture**

---

### 1. Data Connection Architecture
- **Source File:** `outputs/submission/09_Template_Deliverable_Submission.xlsx`
- **Power BI Compatibility:** Version 2.154.1260.0 (64-bit, May 2026) or earlier.
- **Data Load Step:**
  - Load all 10 sheets (`Case 1` through `Case 10`).
  - In Power Query, unpivot or append the tables with a added column `Scenario_ID`:
    - `Case 1`: "P10 (Optimistic)"
    - `Case 2`: "P50 (Base Case)"
    - `Case 3`: "P90 (Pessimistic)"
    - `Case 4`–`Case 10`: "Posterior Sample 1" through "Posterior Sample 7"

---

### 2. Recommended Visualizations
1. **Multi-Phase Production Fan Chart (P10 / P50 / P90 Area/Line Chart):**
   - **X-Axis:** `Date` (Quarterly intervals 2008–2028).
   - **Y-Axis:** Cumulative Production (`Oil [STB]`, `Gas [MSCF]`, `Water [STB]`).
   - **Visual Elements:**
     - Central line: `Case 2` (P50 Base Case).
     - Upper boundary: `Case 1` (P10 Optimistic, through 2018).
     - Lower boundary: `Case 3` (P90 Pessimistic).
     - Shaded uncertainty ribbon: P10 to P90 envelope.

2. **Quarterly Incremental Rate Dynamics:**
   - DAX Measure:
     ```dax
     Quarterly_Oil_Rate_STBD = 
     VAR CurrentDate = SELECTEDVALUE('Production'[Date])
     VAR PrevDate = CALCULATE(MAX('Production'[Date]), FILTER(ALL('Production'), 'Production'[Date] < CurrentDate && 'Production'[Scenario_ID] = SELECTEDVALUE('Production'[Scenario_ID])))
     VAR CurrentCum = SELECTEDVALUE('Production'[Oil production cumulative [STB]])
     VAR PrevCum = CALCULATE(MAX('Production'[Oil production cumulative [STB]]), FILTER(ALL('Production'), 'Production'[Date] = PrevDate && 'Production'[Scenario_ID] = SELECTEDVALUE('Production'[Scenario_ID])))
     RETURN (CurrentCum - PrevCum) / 91.3125
     ```

3. **Water-Oil Ratio (WOR) & Water Cut Evolution:**
   - Line chart showing water cut $f_w(t) = q_w / (q_o + q_w)$ across scenarios over time.
   - Highlights late-time water breakthrough and water management constraints.

4. **Gas-to-Oil Ratio Physical Verification Card:**
   - Measure: `Cumulative_Rs = DIVIDE([Cum_Gas_MSCF], [Cum_Oil_STB])`
   - Verification KPI Card displaying constant $0.3633$ MSCF/STB ratio on incremental production.
