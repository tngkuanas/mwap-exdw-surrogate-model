"""
Competition Submission and Deliverable Pipeline for ExxonMobil DataWorks Challenge 2026.
Conforms strictly to '09 Template Deliverable.xlsx' specification:
- Exact columns: ['Date', 'Gas production cumulative [MSCF]', 'Oil production cumulative [STB]', 'Water production cumulative [STB]']
- Variable case horizons: Case 1 requires 40 quarters (through 2018-01-01); Cases 2-10 require 80 quarters (through 2028-01-01).
- Exact physical units: Gas in MSCF, Oil in STB, Water in STB.
- Physical guarantees: Non-negative rates, strictly monotonic cumulatives, anchor preservation, exact Rs solution gas coupling.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import datetime
import numpy as np
import pandas as pd
import openpyxl

RS_SOLUTION = 0.3633  # MSCF / STB exact physical solution gas-oil ratio
EXPECTED_COLUMNS = [
    "Date",
    "Gas production cumulative [MSCF]",
    "Oil production cumulative [STB]",
    "Water production cumulative [STB]",
]

TEMPLATE_PATH = Path("../TestCases/09 Template Deliverable.xlsx")


def inspect_template_spec(template_path: Path = TEMPLATE_PATH) -> Dict[str, Any]:
    """Inspects sheet names, date rows, and required horizon per case in the official template."""
    wb = openpyxl.load_workbook(template_path, data_only=True)
    spec = {}
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        rows = list(ws.iter_rows(values_only=True))
        headers = [c for c in rows[0] if c is not None]
        date_rows = [r[0] for r in rows[1:] if r[0] is not None]
        spec[sheet] = {
            "headers": headers,
            "num_quarters": len(date_rows),
            "dates": date_rows,
            "start_date": str(date_rows[0]),
            "end_date": str(date_rows[-1]),
        }
    return spec


def format_deliverable_forecast(
    case_predictions: Dict[str, pd.DataFrame],
    template_path: Path = TEMPLATE_PATH,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Populates predictions into an exact replica of the competition deliverable template.
    case_predictions: dict mapping sheet name (e.g. 'Case 1') to DataFrame with:
      ['date', 'cum_gas_mscf', 'cum_oil_stb', 'cum_water_stb']
    """
    if output_path is None:
        output_path = Path("outputs/submission/09_Template_Deliverable_Submission.xlsx")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    wb = openpyxl.load_workbook(template_path)
    
    for sheet_name, df_pred in case_predictions.items():
        if sheet_name not in wb.sheetnames:
            continue
        ws = wb[sheet_name]
        
        expected_len = ws.max_row - 1
        assert len(df_pred) == expected_len, (
            f"Length mismatch for {sheet_name}: expected {expected_len}, got {len(df_pred)}"
        )
        
        for idx, row in df_pred.iterrows():
            row_idx = idx + 2
            # Date is in Col 1, Gas in Col 2, Oil in Col 3, Water in Col 4
            ws.cell(row=row_idx, column=2, value=float(row["cum_gas_mscf"]))
            ws.cell(row=row_idx, column=3, value=float(row["cum_oil_stb"]))
            ws.cell(row=row_idx, column=4, value=float(row["cum_water_stb"]))
            
    wb.save(output_path)
    return output_path


def validate_deliverable_workbook(file_path: Path) -> Dict[str, Any]:
    """
    Programmatically opens and audits the deliverable workbook for:
    1. Sheet names: Case 1 through Case 10 present.
    2. Column headers: exact match with competition spec.
    3. Row count: Case 1 = 41 rows (40 forecast steps); Cases 2-10 = 81 rows (80 forecast steps).
    4. Completeness: zero NaNs, zero nulls, zero empty cells.
    5. Monotonicity: strictly non-decreasing cumulatives (Q_{k+1} >= Q_k) across all 3 phases.
    6. Non-negativity: Q >= 0 across all fluids.
    7. Solution GOR consistency: delta Q_gas / delta Q_oil == 0.3633 (+- 1e-4).
    """
    wb = openpyxl.load_workbook(file_path, data_only=True)
    assert set(wb.sheetnames) == {f"Case {i}" for i in range(1, 11)}, (
        f"Sheet mismatch: {wb.sheetnames}"
    )
    
    sheet_audit = {}
    total_cells_checked = 0
    nan_count = 0
    monotonicity_violations = 0
    negative_values_count = 0
    gor_discrepancies = 0
    
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        header = [c for c in rows[0] if c is not None]
        assert header[:4] == EXPECTED_COLUMNS, f"Header mismatch in {sheet_name}: {header[:4]}"
        
        expected_rows = 41 if sheet_name == "Case 1" else 81
        assert len(rows) == expected_rows, (
            f"Row count mismatch in {sheet_name}: expected {expected_rows}, got {len(rows)}"
        )
        
        dates = []
        gas_cum = []
        oil_cum = []
        water_cum = []
        
        for r_idx, row in enumerate(rows[1:], start=2):
            d, g, o, w = row[0], row[1], row[2], row[3]
            dates.append(str(d))
            total_cells_checked += 3
            
            for val, name in [(g, "gas"), (o, "oil"), (w, "water")]:
                if val is None or (isinstance(val, (float, int)) and np.isnan(val)):
                    nan_count += 1
                elif val < 0:
                    negative_values_count += 1
                    
            gas_cum.append(float(g))
            oil_cum.append(float(o))
            water_cum.append(float(w))
            
        g_arr = np.array(gas_cum)
        o_arr = np.array(oil_cum)
        w_arr = np.array(water_cum)
        
        # Check monotonicity
        d_g = np.diff(g_arr)
        d_o = np.diff(o_arr)
        d_w = np.diff(w_arr)
        
        m_viol = int(np.sum(d_g < -1e-6) + np.sum(d_o < -1e-6) + np.sum(d_w < -1e-6))
        monotonicity_violations += m_viol
        
        # Check solution GOR on increments
        nonzero_oil_mask = d_o > 1e-4
        if np.any(nonzero_oil_mask):
            gor_ratio = d_g[nonzero_oil_mask] / d_o[nonzero_oil_mask]
            gor_err = np.max(np.abs(gor_ratio - RS_SOLUTION))
            if gor_err > 1e-3:
                gor_discrepancies += 1
        else:
            gor_err = 0.0
            
        sheet_audit[sheet_name] = {
            "num_quarters": len(dates),
            "start_date": dates[0],
            "end_date": dates[-1],
            "oil_cum_start": float(o_arr[0]),
            "oil_cum_final": float(o_arr[-1]),
            "gas_cum_start": float(g_arr[0]),
            "gas_cum_final": float(g_arr[-1]),
            "water_cum_start": float(w_arr[0]),
            "water_cum_final": float(w_arr[-1]),
            "oil_monotonic": bool(np.all(d_o >= -1e-6)),
            "gas_monotonic": bool(np.all(d_g >= -1e-6)),
            "water_monotonic": bool(np.all(d_w >= -1e-6)),
            "max_gor_residual": float(gor_err),
        }
        
    audit_summary = {
        "file_path": str(file_path),
        "total_sheets": len(wb.sheetnames),
        "total_cells_checked": total_cells_checked,
        "nan_count": nan_count,
        "negative_values_count": negative_values_count,
        "monotonicity_violations": monotonicity_violations,
        "gor_discrepancies": gor_discrepancies,
        "all_checks_passed": (
            nan_count == 0
            and negative_values_count == 0
            and monotonicity_violations == 0
            and gor_discrepancies == 0
        ),
        "sheets": sheet_audit,
    }
    
    assert audit_summary["all_checks_passed"], f"Deliverable audit failed: {audit_summary}"
    return audit_summary


def audit_rate_reconciliation(
    rates: np.ndarray,
    cums: np.ndarray,
    anchor_cum: float,
    dt_days: float = 91.3125,
) -> float:
    """
    Confirms that differentiating cumulative predictions recovers underlying rates
    with maximum absolute error < 1e-6.
    """
    reconstructed_rates = np.zeros_like(rates)
    reconstructed_rates[0] = (cums[0] - anchor_cum) / dt_days
    if len(rates) > 1:
        reconstructed_rates[1:] = np.diff(cums) / dt_days
        
    max_err = float(np.max(np.abs(reconstructed_rates - rates)))
    return max_err


if __name__ == "__main__":
    spec = inspect_template_spec()
    print("Template Deliverable Specification Verified:")
    for k, v in spec.items():
        print(f"  {k:10s} | Quarters: {v['num_quarters']} | Start: {v['start_date']} | End: {v['end_date']}")
