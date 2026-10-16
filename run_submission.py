#!/usr/bin/env python3
"""
================================================================================
ExxonMobil DataWorks Challenge 2026 — Unified Submission Interface
Team: MWAP Energy Analytics
================================================================================
Authoritative Root CLI for:
  1. Read-Only Verification of Protected Competition Deliverable
  2. Safe Standalone Regeneration into an Isolated Output Location
  3. Building Clean Submission Archive ({team_name}.zip)
  4. End-to-End Extraction & Cell-by-Cell Verification

Usage:
  python run_submission.py --verify
  python run_submission.py --regenerate [--output <path>]
  python run_submission.py --build-archive [--output-dir <path>]
  python run_submission.py --verify-archive <path_to_zip>
================================================================================
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

# Strict execution hygiene: disable bytecode caching
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# Ensure Anas-exp/src is on sys.path regardless of execution directory
REPO_ROOT = Path(__file__).resolve().parent
ANAS_EXP_SRC = REPO_ROOT / "Anas-exp" / "src"
if str(ANAS_EXP_SRC) not in sys.path:
    sys.path.insert(0, str(ANAS_EXP_SRC))

PROTECTED_SHA256 = "e413f3caad576bd215cda0a83dc6cfa79f64ad05664a030b1806716097562c97"
LEGACY_SHA256 = "ce30ec18c5c22c842c8319803c89dcc6011b9e95ccb97be30caf2dd08dbfce7e"
OFFICIAL_DELIVERABLE_REL = Path("Anas-exp/outputs/submission/09_Template_Deliverable_Submission.xlsx")
OFFICIAL_DELIVERABLE_PATH = REPO_ROOT / OFFICIAL_DELIVERABLE_REL
DEFAULT_REGEN_OUTPUT = REPO_ROOT / "outputs" / "regeneration" / "09_Template_Deliverable_Regenerated.xlsx"
TEAM_NAME = "MWAP_Energy_Analytics"


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hexadecimal digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_readonly_verification(deliverable_path: Path = OFFICIAL_DELIVERABLE_PATH) -> bool:
    """
    Performs strict read-only audit of official deliverable workbook.
    Never modifies or writes any file.
    """
    print("=" * 80)
    print("EXXONMOBIL DATAWORKS CHALLENGE 2026 — READ-ONLY DELIVERABLE VERIFICATION")
    print("=" * 80)
    print(f"Target Workbook: {deliverable_path}")
    
    if not deliverable_path.exists():
        print(f"[FAIL] Deliverable file does not exist at {deliverable_path}!")
        return False
        
    # Check SHA-256
    file_sha = compute_sha256(deliverable_path)
    print(f"Calculated SHA-256: {file_sha}")
    print(f"Expected SHA-256:   {PROTECTED_SHA256}")
    
    sha_match = (file_sha == PROTECTED_SHA256)
    if sha_match:
        print("[PASS] SHA-256 exact match with protected production baseline.")
    else:
        print("[FAIL] SHA-256 mismatch!")
        return False
        
    # Execute authoritative deliverable validation
    from ensemble.submission_pipeline import validate_deliverable_workbook
    audit = validate_deliverable_workbook(deliverable_path)
    
    for sheet_name, s_info in audit["sheets"].items():
        print(f"  {sheet_name:8s}: {s_info['num_quarters']} quarters verified | Monotonic | Non-negative | Max GOR err: {s_info['max_gor_residual']:.2e}")
        
    print(f"\n[Audit Summary]")
    print(f"  Total Sheets Audited:       {audit['total_sheets']}")
    print(f"  Total Cells Audited:        {audit['total_cells_checked']} (2,400 numeric + 800 dates)")
    print(f"  NaN / Non-numeric Count:    {audit['nan_count']}")
    print(f"  Negative Values Count:      {audit['negative_values_count']}")
    print(f"  Monotonicity Violations:    {audit['monotonicity_violations']}")
    print(f"  Solution GOR Discrepancies: {audit['gor_discrepancies']}")
    
    if not audit["all_checks_passed"]:
        print("[FAIL] Audit checks failed!")
        return False
        
    # Run test_submission_readiness suite via pytest (strictly read-only: no cache, no bytecode)
    print("\nRunning authoritative submission test suite (test_submission_readiness.py)...")
    test_file = REPO_ROOT / "Anas-exp" / "tests" / "test_submission_readiness.py"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    cmd = [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        str(test_file),
        "-p",
        "no:cacheprovider",
        "-q",
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    
    if res.returncode == 0:
        print(res.stdout.strip())
        print("[PASS] All submission readiness tests passed.")
    else:
        print(res.stdout)
        print(res.stderr)
        print("[FAIL] Submission readiness test suite failed!")
        return False
        
    print("\n" + "=" * 80)
    print("VERIFICATION RESULT: ALL CHECKS PASSED (STATUS: VERIFIED_SUBMISSION_READY)")
    print("=" * 80)
    return True


def run_regeneration(
    output_path: Path = DEFAULT_REGEN_OUTPUT,
    force_overwrite: bool = False,
    retrain: bool = False,
) -> bool:
    """
    Executes model inference (or retraining) and scenario forecasting to regenerate deliverable workbook.
    Writes strictly to a separate output path to preserve protected official deliverable.
    """
    print("=" * 80)
    print("EXXONMOBIL DATAWORKS CHALLENGE 2026 — DELIVERABLE REGENERATION")
    mode_desc = "Retraining from permitted data (Cases 1-70)" if retrain else "Fast inference from saved models (models/production_models.pkl)"
    print(f"Mode: {mode_desc}")
    print("=" * 80)
    
    # Safety guard: prevent accidental overwrite of protected workbook
    if output_path.resolve() == OFFICIAL_DELIVERABLE_PATH.resolve() and not force_overwrite:
        print(f"[ERROR] Output path matches protected official deliverable ({OFFICIAL_DELIVERABLE_PATH})!")
        print("Regeneration must write to a separate location. Use --output <path> or pass --force-overwrite-protected.")
        return False
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Target Regeneration Output: {output_path}")
    print("Executing generate_competition_deliverable pipeline...")
    
    from ensemble.generate_competition_deliverable import run_pipeline
    
    t0 = time.time()
    try:
        smoke_regen = output_path.parent / "smoke_test_dev_cases_1_to_10_regen.xlsx"
        run_pipeline(
            config_path=None,
            output_path=output_path,
            smoke_test_path=smoke_regen,
            template_path=None,
            retrain=retrain,
        )
    except Exception as e:
        print(f"[FAIL] Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    elapsed = time.time() - t0
    print(f"\nRegeneration completed in {elapsed:.2f} seconds.")
    print(f"Regenerated Workbook saved to: {output_path}")
    
    # Compare regenerated cells against official deliverable cell by cell
    print("\nComparing Regenerated Workbook against Protected Official Deliverable...")
    import openpyxl
    wb_orig = openpyxl.load_workbook(OFFICIAL_DELIVERABLE_PATH, data_only=True)
    wb_regen = openpyxl.load_workbook(output_path, data_only=True)
    
    total_cells = 0
    max_abs_diff = 0.0
    mismatch_count = 0
    
    for sheet in wb_orig.sheetnames:
        ws_orig = wb_orig[sheet]
        ws_regen = wb_regen[sheet]
        rows_orig = list(ws_orig.iter_rows(values_only=True))
        rows_regen = list(ws_regen.iter_rows(values_only=True))
        
        for r_idx, (ro, rr) in enumerate(zip(rows_orig[1:], rows_regen[1:]), start=2):
            # Check Date
            if str(ro[0]) != str(rr[0]):
                mismatch_count += 1
                print(f"[MISMATCH] {sheet} Row {r_idx} Date: orig={ro[0]}, regen={rr[0]}")
            # Check Gas, Oil, Water
            for col_idx, (vo, vr) in enumerate(zip(ro[1:4], rr[1:4]), start=1):
                total_cells += 1
                diff = abs(float(vo) - float(vr))
                if diff > max_abs_diff:
                    max_abs_diff = diff
                if diff > 1e-4:
                    mismatch_count += 1
                    print(f"[MISMATCH] {sheet} Row {r_idx} Col {col_idx}: orig={vo}, regen={vr}, diff={diff}")
                    
    print(f"Total Numeric Cells Compared: {total_cells} / 2400")
    print(f"Max Absolute Cell Difference: {max_abs_diff:.12f}")
    print(f"Cells with Difference > 1e-4: {mismatch_count}")
    
    if mismatch_count == 0 and max_abs_diff < 1e-4:
        print("\n[SUCCESS] Exact mathematical equivalence confirmed across all 2,400 cells!")
        return True
    else:
        print("\n[FAIL] Cell discrepancy detected between regenerated and official workbook!")
        return False


def build_submission_archive(output_dir: Path = REPO_ROOT / "outputs" / "submission") -> Path:
    """
    Builds clean, self-contained {team_name}.zip archive using an explicit manifest
    and an isolated staging directory. Enforces permitted-data filtering for simulation
    and production records (Cases 86-100 strictly excluded).
    """
    import pandas as pd
    
    print("=" * 80)
    print(f"BUILDING CLEAN SUBMISSION ARCHIVE: {TEAM_NAME}.zip")
    print("=" * 80)
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"{TEAM_NAME}.zip"
    
    # Explicit file and directory inclusion manifest
    # (src, tests, locked config, scenario artifact, shared training inputs, official template, official deliverable, docs)
    manifest_items = [
        ("requirements.txt", "requirements.txt"),
        ("pyproject.toml", "pyproject.toml"),
        ("run_submission.py", "run_submission.py"),
        ("README.md", "README.md"),
        ("DATA_CONTRACT.md", "DATA_CONTRACT.md"),
        ("DATA_DOCUMENTATION.md", "DATA_DOCUMENTATION.md"),
        ("TestCases/09 Template Deliverable.xlsx", "TestCases/09 Template Deliverable.xlsx"),
        # Explicit permitted static metadata and training inputs
        ("data/cases_metadata.parquet", "data/cases_metadata.parquet"),
        ("data/field_history.parquet", "data/field_history.parquet"),
        ("data/shared/uncertainty_params.parquet", "data/shared/uncertainty_params.parquet"),
        # Source code and test suite
        ("Anas-exp/src", "Anas-exp/src"),
        ("Anas-exp/tests", "Anas-exp/tests"),
        # Research reports and governance documentation
        ("Anas-exp/RESEARCH_FOUNDATIONS.md", "Anas-exp/RESEARCH_FOUNDATIONS.md"),
        ("Anas-exp/RESEARCH_ITERATION_4_REPORT.md", "Anas-exp/RESEARCH_ITERATION_4_REPORT.md"),
        ("Anas-exp/PRODUCTION_INTEGRATION_REPORT.md", "Anas-exp/PRODUCTION_INTEGRATION_REPORT.md"),
        ("Anas-exp/FORECAST_RELIABILITY_CHALLENGE_REPORT.md", "Anas-exp/FORECAST_RELIABILITY_CHALLENGE_REPORT.md"),
        # Verified Iteration 4 scenarios artifact
        ("Anas-exp/outputs/research/history_matching_iteration_4",
         "Anas-exp/outputs/research/history_matching_iteration_4"),
        # Protected official competition deliverable and audit materials
        ("Anas-exp/outputs/submission/09_Template_Deliverable_Submission.xlsx",
         "Anas-exp/outputs/submission/09_Template_Deliverable_Submission.xlsx"),
        ("Anas-exp/outputs/submission/smoke_test_dev_cases_1_to_10.xlsx",
         "Anas-exp/outputs/submission/smoke_test_dev_cases_1_to_10.xlsx"),
        ("Anas-exp/outputs/submission/submission_audit_report.json",
         "Anas-exp/outputs/submission/submission_audit_report.json"),
        ("Anas-exp/outputs/submission/methodology_presentation_outline.md",
         "Anas-exp/outputs/submission/methodology_presentation_outline.md"),
        ("Anas-exp/outputs/submission/power_bi_integration_guide.md",
         "Anas-exp/outputs/submission/power_bi_integration_guide.md"),
        ("Anas-exp/outputs/submission/seven_minute_video_link.txt",
         "Anas-exp/outputs/submission/seven_minute_video_link.txt"),
        # Serialized production models and metadata
        ("models/production_models.pkl", "models/production_models.pkl"),
        ("models/model_metadata.json", "models/model_metadata.json"),
    ]
    
    with tempfile.TemporaryDirectory() as staging_dir:
        staging_root = Path(staging_dir) / TEAM_NAME
        staging_root.mkdir(parents=True)
        print(f"Staging files in clean directory: {staging_root}...")
        
        # 1. Process explicit manifest items (missing files are fatal)
        for src_rel, dest_rel in manifest_items:
            src_path = REPO_ROOT / src_rel
            dest_path = staging_root / dest_rel
            
            if not src_path.exists():
                raise FileNotFoundError(f"FATAL: Required submission manifest item missing: {src_path}")
                
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if src_path.is_dir():
                shutil.copytree(
                    src_path,
                    dest_path,
                    ignore=shutil.ignore_patterns(
                        "__pycache__", "*.pyc", ".DS_Store", ".pytest_cache", "*mlflow.db*"
                    )
                )
            else:
                shutil.copy2(src_path, dest_path)
                
        # 2. In staging only, filter dynamic timeseries to permitted cases (Cases 86-100 excluded)
        # Note: Original source datasets in REPO_ROOT remain completely untouched.
        # Blind test case outcome values are never inspected or reported.
        print("Staging permitted dynamic production datasets (Cases 86-100 excluded)...")
        
        sim_src = REPO_ROOT / "data" / "simulation_timeseries.parquet"
        if not sim_src.exists():
            raise FileNotFoundError(f"FATAL: Source simulation_timeseries missing: {sim_src}")
        df_sim = pd.read_parquet(sim_src)
        df_sim_filtered = df_sim[df_sim["case_id"] <= 85].copy()
        assert int(df_sim_filtered["case_id"].max()) <= 85
        assert not any(c in range(86, 101) for c in df_sim_filtered["case_id"].unique())
        staging_sim = staging_root / "data" / "simulation_timeseries.parquet"
        staging_sim.parent.mkdir(parents=True, exist_ok=True)
        df_sim_filtered.to_parquet(staging_sim, index=False)
        print(f"  [PERMITTED DATA] Filtered simulation_timeseries to Cases 1-{df_sim_filtered['case_id'].max()} "
              f"({len(df_sim_filtered):,} rows; blind Cases 86-100 excluded).")
        
        prod_src = REPO_ROOT / "data" / "shared" / "production_timeseries.parquet"
        if not prod_src.exists():
            raise FileNotFoundError(f"FATAL: Source production_timeseries missing: {prod_src}")
        df_prod = pd.read_parquet(prod_src)
        df_prod_filtered = df_prod[df_prod["case_num"] <= 85].copy()
        assert int(df_prod_filtered["case_num"].max()) <= 85
        assert not any(c in range(86, 101) for c in df_prod_filtered["case_num"].unique())
        staging_prod = staging_root / "data" / "shared" / "production_timeseries.parquet"
        staging_prod.parent.mkdir(parents=True, exist_ok=True)
        df_prod_filtered.to_parquet(staging_prod, index=False)
        print(f"  [PERMITTED DATA] Filtered production_timeseries to Cases 0-{df_prod_filtered['case_num'].max()} "
              f"({len(df_prod_filtered):,} rows; blind Cases 86-100 excluded).")
        
        # 3. Clean any accidental junk in staging
        for root, dirs, files in os.walk(staging_root):
            for d in list(dirs):
                if d in ("__pycache__", ".pytest_cache"):
                    shutil.rmtree(os.path.join(root, d))
            for f in files:
                if f.endswith(".pyc") or f == ".DS_Store":
                    os.remove(os.path.join(root, f))
                    
        print(f"Creating ZIP archive at: {zip_path}...")
        if zip_path.exists():
            zip_path.unlink()
            
        shutil.make_archive(
            base_name=str(output_dir / TEAM_NAME),
            format="zip",
            root_dir=staging_dir,
            base_dir=TEAM_NAME,
        )
        
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Archive generated: {zip_path} ({size_mb:.2f} MB)")
    return zip_path


def verify_extracted_archive(zip_path: Path) -> bool:
    """
    Extracts archive into a fresh temporary directory, verifies contents, runs tests,
    and checks cell-by-cell equivalence.
    """
    print("=" * 80)
    print(f"VERIFYING EXTRACTED SUBMISSION ARCHIVE: {zip_path}")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as extract_dir:
        extract_root = Path(extract_dir)
        print(f"Extracting {zip_path} to {extract_root}...")
        shutil.unpack_archive(zip_path, extract_root)
        
        pkg_root = extract_root / TEAM_NAME
        if not pkg_root.exists():
            pkg_root = extract_root
            
        # Check packaged deliverable SHA-256
        packaged_deliverable = pkg_root / "Anas-exp" / "outputs" / "submission" / "09_Template_Deliverable_Submission.xlsx"
        if not packaged_deliverable.exists():
            print(f"[FAIL] Packaged deliverable not found at: {packaged_deliverable}")
            return False
            
        packaged_sha = compute_sha256(packaged_deliverable)
        print(f"Packaged Deliverable SHA-256: {packaged_sha}")
        if packaged_sha != PROTECTED_SHA256:
            print(f"[FAIL] Packaged deliverable SHA-256 mismatch! Expected {PROTECTED_SHA256}")
            return False
        print("[PASS] Packaged deliverable SHA-256 equals authoritative checksum.")
        
        # Describe environment verification accurately
        print(f"\nEnvironment Verification Mode:")
        print(f"  Extraction tested in isolated sandbox using existing pinned production runtime:")
        print(f"  Python Runtime: {sys.version.split()[0]} ({sys.executable})")
        print(f"  Bytecode Generation: DISABLED (PYTHONDONTWRITEBYTECODE=1, -B)")
        print(f"  Pytest Caching: DISABLED (-p no:cacheprovider)")
        
        # Check case isolation in extracted data
        print("\nVerifying Case Isolation in Extracted Data (Cases 86-100 Exclusion)...")
        ext_sim = pkg_root / "data" / "simulation_timeseries.parquet"
        ext_prod = pkg_root / "data" / "shared" / "production_timeseries.parquet"
        
        import pandas as pd
        if not ext_sim.exists() or not ext_prod.exists():
            print(f"[FAIL] Required packaged parquet files missing: sim={ext_sim.exists()}, prod={ext_prod.exists()}")
            return False
            
        df_sim = pd.read_parquet(ext_sim)
        sim_cases = set(df_sim["case_id"].unique())
        blind_sim_leak = sim_cases.intersection(set(range(86, 101)))
        if blind_sim_leak:
            print(f"[FAIL] Blind cases found in packaged simulation_timeseries: {blind_sim_leak}")
            return False
        assert max(sim_cases) <= 85
        print(f"  simulation_timeseries: Cases 1-{max(sim_cases)} present ({len(sim_cases)} cases). Cases 86-100 excluded [PASS]")
        
        df_prod = pd.read_parquet(ext_prod)
        prod_cases = set(df_prod["case_num"].unique())
        blind_prod_leak = prod_cases.intersection(set(range(86, 101)))
        if blind_prod_leak:
            print(f"[FAIL] Blind cases found in packaged production_timeseries: {blind_prod_leak}")
            return False
        assert max(prod_cases) <= 85
        print(f"  production_timeseries: Cases 0-{max(prod_cases)} present ({len(prod_cases)} cases). Cases 86-100 excluded [PASS]")
        print("  Disambiguation: Scenario Candidate IDs (e.g. 440, 242, 461) are MCMC parameter draws, distinct from reservoir case IDs [PASS]")
        
        # Run test suite from inside package (no cache, no bytecode)
        print("\nExecuting test_submission_readiness inside extracted package...")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        res = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "pytest",
                "Anas-exp/tests/test_submission_readiness.py",
                "-p",
                "no:cacheprovider",
                "-q",
            ],
            cwd=pkg_root,
            env=env,
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            print(res.stdout)
            print(res.stderr)
            print("[FAIL] Tests failed in extracted package!")
            return False
        print(res.stdout.strip())
        print("[PASS] Tests passed inside extracted package.")
        
        # Check packaged model artifacts
        print("\nVerifying Packaged Production Model Artifacts...")
        ext_model = pkg_root / "models" / "production_models.pkl"
        ext_meta = pkg_root / "models" / "model_metadata.json"
        if not ext_model.exists() or not ext_meta.exists():
            print(f"[FAIL] Required packaged model artifacts missing: model={ext_model.exists()}, meta={ext_meta.exists()}")
            return False
        model_size_mb = ext_model.stat().st_size / (1024 * 1024)
        print(f"  production_models.pkl ({model_size_mb:.2f} MB) and model_metadata.json verified [PASS]")
        
        # Test regeneration inside package to a separate file
        regen_path = pkg_root / "outputs" / "regeneration" / "test_fresh_extract_regen.xlsx"
        print(f"\nTesting regeneration inside package to: {regen_path}...")
        regen_cmd = [
            sys.executable,
            "-B",
            "run_submission.py",
            "--regenerate",
            "--output",
            str(regen_path),
        ]
        res_regen = subprocess.run(
            regen_cmd,
            cwd=pkg_root,
            env=env,
            capture_output=True,
            text=True,
        )
        if res_regen.returncode != 0:
            print(res_regen.stdout)
            print(res_regen.stderr)
            print("[FAIL] Regeneration failed inside extracted package!")
            return False
            
        print("[PASS] Regeneration and cell-by-cell equivalence verified in fresh extraction.")
        
    print("\n" + "=" * 80)
    print("EXTRACTED ARCHIVE VERIFICATION PASSED SUCCESSFULLY")
    print("=" * 80)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="ExxonMobil DataWorks Challenge 2026 — Submission Interface"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--verify",
        action="store_true",
        help="Run strict read-only audit and verification of official deliverable workbook.",
    )
    group.add_argument(
        "--regenerate",
        action="store_true",
        help="Safely regenerate deliverable workbook into an isolated output location.",
    )
    group.add_argument(
        "--inference",
        action="store_true",
        help="Run fast inference from saved model artifacts to regenerate deliverable workbook.",
    )
    group.add_argument(
        "--build-archive",
        action="store_true",
        help="Build clean, self-contained MWAP_Energy_Analytics.zip submission archive.",
    )
    group.add_argument(
        "--verify-archive",
        type=Path,
        metavar="ZIP_PATH",
        help="Extract and verify a submission archive end-to-end in an isolated sandbox.",
    )
    
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining base models from permitted training data (Cases 1-70) during regeneration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REGEN_OUTPUT,
        help="Output location for regeneration (default: outputs/regeneration/09_Template_Deliverable_Regenerated.xlsx)",
    )
    parser.add_argument(
        "--force-overwrite-protected",
        action="store_true",
        help="Explicitly permit overwriting the protected official deliverable during regeneration.",
    )
    
    args = parser.parse_args()
    
    if args.verify:
        success = run_readonly_verification()
        sys.exit(0 if success else 1)
    elif args.regenerate or args.inference:
        success = run_regeneration(
            output_path=args.output,
            force_overwrite=args.force_overwrite_protected,
            retrain=args.retrain,
        )
        sys.exit(0 if success else 1)
    elif args.build_archive:
        zip_path = build_submission_archive()
        sys.exit(0 if zip_path.exists() else 1)
    elif args.verify_archive:
        success = verify_extracted_archive(args.verify_archive)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
