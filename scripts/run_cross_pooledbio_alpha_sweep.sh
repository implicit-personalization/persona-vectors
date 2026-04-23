#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/hengxuli/Repos/implicit-personalization/persona-vectors"
ENV_FILE="/Users/hengxuli/Repos/synth-persona/.env"
MODEL="google/gemma-2-9b-it"
PERSONAS=3
QUESTIONS=20
QA_TYPE="implicit"
OUT_BASE="artifacts/experiments/cross_persona_steering/alpha_sweep_pooledbio"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)__gemma2-9b-it__implicit__p${PERSONAS}__q${QUESTIONS}"
RUN_ROOT="${OUT_BASE}/${RUN_ID}"

mkdir -p "${ROOT}/${RUN_ROOT}"

for ALPHA in 0.5 1.0 2.0 3.0; do
  SAFE_ALPHA="${ALPHA//./p}"
  echo "=== pooled_biography alpha=${ALPHA} ==="
  cd "${ROOT}"
  uv run --env-file "${ENV_FILE}" python experiments/03_cross_persona_steering.py \
    --model "${MODEL}" \
    --personas "${PERSONAS}" \
    --questions-per-persona "${QUESTIONS}" \
    --qa-type "${QA_TYPE}" \
    --remote \
    --all-layers \
    --negative-variant pooled_biography \
    --method mean \
    --center \
    --alpha-override "${ALPHA}" \
    --out-dir "${RUN_ROOT}/alpha_${SAFE_ALPHA}"
done

echo "Saved sweep outputs to ${RUN_ROOT}"
