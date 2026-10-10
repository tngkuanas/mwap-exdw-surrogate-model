#!/usr/bin/env bash
# Reproducibility script for the ExxonMobil DataWorks Challenge 2026 Ensemble & Adversarial Audit
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/Anas-exp/src:${PYTHONPATH:-}"

echo "1. Running unit test suite..."
"$REPO_ROOT/.venv/bin/python" -m pytest "$REPO_ROOT/Anas-exp/tests" -q

echo "2. Running Adversarial Audit & Paired Bootstrap Tests..."
"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/Anas-exp/src/ensemble/adversarial_audit.py"

echo "3. Running GP Sensitivity and Target Representation Studies..."
"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/Anas-exp/src/ensemble/run_gp_and_target_studies.py"

echo "✅ All ensemble pipeline stages and adversarial audits rerun successfully."
