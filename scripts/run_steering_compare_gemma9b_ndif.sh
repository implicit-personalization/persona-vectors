#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-/Users/hengxuli/Repos/synth-persona/.env}"
MODEL="${MODEL:-google/gemma-2-9b-it}"
PERSONAS="${PERSONAS:-30}"
QUESTIONS="${QUESTIONS:-56}"
LAYER="${LAYER:-20}"
QA_TYPE="${QA_TYPE:-implicit}"
ALPHA_SCALE="${ALPHA_SCALE:-1.0}"
CONFIGS="${CONFIGS:-}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/artifacts/experiments/steering_compare/${RUN_ID}__gemma2-9b-it__p${PERSONAS}__q${QUESTIONS}}"

mkdir -p "${OUT_ROOT}"

run_cfg() {
  echo
  echo "============================================================"
  echo "[run] cache-first steering compare"
  echo "============================================================"
  local -a extra_args=()
  if [[ -n "${CONFIGS}" ]]; then
    extra_args+=(--configs "${CONFIGS}")
  fi
  uv run --env-file "${ENV_FILE}" python experiments/04_steering_compare.py \
    --model "${MODEL}" \
    --personas "${PERSONAS}" \
    --questions-per-persona "${QUESTIONS}" \
    --qa-type "${QA_TYPE}" \
    --alpha-scale "${ALPHA_SCALE}" \
    "${extra_args[@]}" \
    --remote \
    --out-dir "${OUT_ROOT}"
}

cd "${ROOT}"

run_cfg

echo
echo "[done] steering compare complete"
echo "[done] outputs -> ${OUT_ROOT}"
