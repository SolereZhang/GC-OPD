#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT_DIR}/outputs/reference_analysis}
BUNDLE_DIR=${OUTPUT_ROOT}/bundle
ANALYSIS_DIR=${OUTPUT_ROOT}/analysis

mkdir -p "${OUTPUT_ROOT}"

PYTHONPATH="${ROOT_DIR}/verl" python "${ROOT_DIR}/scripts/make_example_bundle.py" \
    --output "${BUNDLE_DIR}"
PYTHONPATH="${ROOT_DIR}/verl" python \
    "${ROOT_DIR}/verl/examples/gc_opd/analyze_gc_opd_frozen_replay.py" \
    --bundle "${BUNDLE_DIR}" \
    --output-dir "${ANALYSIS_DIR}" \
    --bootstrap-reps 50 \
    --seed 7

echo "Reference analysis written to ${ANALYSIS_DIR}"
