"""
00_prepare_data.py — rebuild the team's 3 shared parquet files locally.

Replicates notebooks/01 Data Preparation.ipynb (Anas) so the outputs have the
same schema as the team's Databricks files:
    uncertainty_params.parquet     100 rows
    production_timeseries.parquet  ~6.7M rows (long format)
    grid_properties.parquet        ~4.25M rows (42,512 active cells x 100 cases)

Usage:
    python 00_prepare_data.py --raw "/path/to/excel/folder" --out "/path/to/data"

Needs: pandas, pyarrow, and python-calamine (fast) or openpyxl (slow fallback).
Grid files are processed one sheet at a time and saved per file, so a crash can
resume from where it stopped.
"""
import argparse
import gc
import glob
import os
import re
import time

import numpy as np
import pandas as pd

SENTINEL = -999.25

CONSTANT_COLS = [
    "Net-to-Gross Ratio (NTG)",
    "Maximum Water Saturation (SWU)",
    " Connateg  Gas Saturation (SGL)",
    "Gas Saturation (SGAS)",
    "Solution Gas-Oil Ratio (RS)",
    "Vapor Oil-Gas Ratio (RV)",
]
COL_RENAME = {" Connateg  Gas Saturation (SGL)": "Connate Gas Saturation (SGL)"}

GRID_FILES = [
    ("03", list(range(1, 21))),
    ("04", list(range(21, 41))),
    ("05", list(range(41, 61))),
    ("06", list(range(61, 71))),
    ("07", list(range(71, 86))),
    ("08", list(range(86, 101))),
]

PROD_SHEETS = [
    ("Oil production cumulative", "oil_cum", "STB"),
    ("Oil production rate", "oil_rate", "STB/d"),
    ("Gas production cumulative", "gas_cum", "MSCF"),
    ("Gas production rate", "gas_rate", "MSCF/d"),
    ("Water production cumulative", "water_cum", "STB"),
    ("Water production rate", "water_rate", "STB/d"),
]


def get_split(case_num):
    if case_num == 0:
        return "history"
    if case_num <= 70:
        return "train"
    if case_num <= 85:
        return "validation"
    return "test"


def pick_engine():
    try:
        import python_calamine  # noqa: F401
        return "calamine"
    except ImportError:
        return "openpyxl"


def find_file(raw, prefix):
    """Match files by their 2-digit prefix (names have stray spaces, e.g. '07  Validation')."""
    hits = sorted(
        p for p in glob.glob(os.path.join(raw, f"{prefix}*.xls*"))
        if not os.path.basename(p).startswith("~$")
    )
    if not hits:
        raise FileNotFoundError(f"No Excel file starting with '{prefix}' in {raw}")
    return hits[0]


def build_uncertainty(raw, out, engine):
    df = pd.read_excel(find_file(raw, "01"), sheet_name="Uncertainty Values", engine=engine)
    df = df.rename(columns={"Item": "case_id"})
    df["case_num"] = df["case_id"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    df["split"] = df["case_num"].map(get_split)
    df = df[["case_num", "case_id", "split", "Fault Transmissibility",
             "Porosity Multiplier", "Permeability Multiplier", "Aquifer Pore Volume"]]
    df = df.sort_values("case_num").reset_index(drop=True)
    assert len(df) == 100, len(df)
    df.to_parquet(os.path.join(out, "uncertainty_params.parquet"), index=False)
    print(f"uncertainty_params: {df.shape}")


def build_production(raw, out, engine):
    path = find_file(raw, "02")
    parts = []
    for sheet, key, unit in PROD_SHEETS:
        t0 = time.time()
        rawdf = pd.read_excel(path, sheet_name=sheet, engine=engine)
        names = rawdf.iloc[0, 1:].values
        data = rawdf.iloc[2:].copy()
        data.columns = ["date"] + list(names)
        data["date"] = pd.to_datetime(data["date"])
        m = data.melt(id_vars="date", var_name="case_raw", value_name="value")
        m["value"] = pd.to_numeric(m["value"], errors="coerce")
        m["metric"], m["unit"] = key, unit
        num = m["case_raw"].astype(str).str.extract(r"(\d+)")[0].astype(float)
        num[m["case_raw"].astype(str) == "RAMP_History"] = 0
        m["case_num"] = num.astype(int)
        m["split"] = m["case_num"].map(get_split)
        parts.append(m.drop(columns="case_raw"))
        print(f"  {sheet}: {len(m):,} rows ({time.time() - t0:.0f}s)")

    fp = pd.read_excel(path, sheet_name="Field Production", engine=engine)
    fp["Date"] = pd.to_datetime(fp["Date"])
    rows = []
    for _, r in fp.iterrows():
        for col, key, unit in [
            ("Oil production rate (STB/d)", "oil_rate_quarterly", "STB/d"),
            ("Gas production rate (MSCF/d)", "gas_rate_quarterly", "MSCF/d"),
            ("Water production rate (STB/d)", "water_rate_quarterly", "STB/d"),
            ("Water injection rate (STB/d)", "water_inj_quarterly", "STB/d"),
        ]:
            rows.append({"date": r["Date"], "value": r[col], "metric": key,
                         "unit": unit, "case_num": 0, "split": "history"})
    parts.append(pd.DataFrame(rows))

    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["case_num", "metric", "date"]).reset_index(drop=True)
    df.to_parquet(os.path.join(out, "production_timeseries.parquet"), index=False)
    print(f"production_timeseries: {df.shape}")


def build_grid(raw, out, engine):
    part_files = []
    for i, (prefix, cases) in enumerate(GRID_FILES, 1):
        part = os.path.join(out, f"grid_part{i}.parquet")
        part_files.append(part)
        if os.path.exists(part):
            print(f"  [{i}/6] {prefix} already done, skipping")
            continue
        path = find_file(raw, prefix)
        print(f"  [{i}/6] {os.path.basename(path)}")
        frames = []
        for c in cases:
            t0 = time.time()
            df = pd.read_excel(path, sheet_name=f"Case {c}", engine=engine)
            df = df.rename(columns=COL_RENAME)
            df = df[df["Porosity (PORO)"] != SENTINEL].copy()
            df = df.drop(columns=[x for x in CONSTANT_COLS if x in df.columns])
            df["case_num"], df["split"] = c, get_split(c)
            frames.append(df)
            print(f"    Case {c:3d}: {len(df):,} active cells ({time.time() - t0:.0f}s)")
        pd.concat(frames, ignore_index=True).to_parquet(part, index=False)
        del frames
        gc.collect()

    df = pd.concat([pd.read_parquet(p) for p in part_files], ignore_index=True)
    if np.allclose(df["Permeability I (PERMX)"], df["Permeability J (PERMY)"]):
        df = df.drop(columns=["Permeability J (PERMY)"])
        print("  PERMX == PERMY everywhere -> dropped PERMY")
    df.to_parquet(os.path.join(out, "grid_properties.parquet"), index=False)
    for p in part_files:
        os.remove(p)
    counts = df.groupby("case_num").size().unique()
    print(f"grid_properties: {df.shape}, active cells per case: {counts}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder with the 01-08 Excel files")
    ap.add_argument("--out", required=True, help="output folder for parquet files")
    ap.add_argument("--only", choices=["unc", "prod", "grid"], help="build one file only")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    eng = pick_engine()
    print("Excel engine:", eng)
    t = time.time()
    if a.only in (None, "unc"):
        build_uncertainty(a.raw, a.out, eng)
    if a.only in (None, "prod"):
        build_production(a.raw, a.out, eng)
    if a.only in (None, "grid"):
        build_grid(a.raw, a.out, eng)
    print(f"Done in {(time.time() - t) / 60:.1f} min -> {a.out}")
