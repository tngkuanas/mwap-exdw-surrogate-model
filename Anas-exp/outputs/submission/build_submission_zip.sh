#!/usr/bin/env bash
# ==============================================================================
# Build Submission Code Archive: {team_name}.zip
# ExxonMobil DataWorks Challenge 2026
# Conforms to official README.md: "Zip into one folder named {team_name}.zip"
# ==============================================================================
set -euo pipefail

TEAM_NAME="MWAP_Energy_Analytics"
ZIP_NAME="${TEAM_NAME}.zip"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/Anas-exp/outputs/submission"

echo "Building clean submission archive: ${OUTPUT_DIR}/${ZIP_NAME}..."

cd "${PROJECT_ROOT}"

# Exclude virtual environments, git objects, large raw caches, and bytecode
zip -r "${OUTPUT_DIR}/${ZIP_NAME}" \
    Anas-exp/src \
    Anas-exp/tests \
    Anas-exp/outputs/submission \
    Anas-exp/outputs/stacked_forecasting \
    data/shared \
    DATA_DOCUMENTATION.md \
    DATA_CONTRACT.md \
    README.md \
    -x "*.venv*" \
    -x "*__pycache__*" \
    -x "*.pytest_cache*" \
    -x "*.git*" \
    -x "*.DS_Store*" \
    -x "*mlflow.db*" \
    -x "*03 Train Cases 1.xlsx*" \
    -x "*04 Train Cases 2.xlsx*" \
    -x "*05 Train Cases 3.xlsx*" \
    -x "*06 Train Cases 4.xlsx*" \
    -x "*07  Validation Cases.xlsx*" \
    -x "*08 Test Cases.xlsx*"

echo "Archive successfully generated at: ${OUTPUT_DIR}/${ZIP_NAME}"
ls -lh "${OUTPUT_DIR}/${ZIP_NAME}"
