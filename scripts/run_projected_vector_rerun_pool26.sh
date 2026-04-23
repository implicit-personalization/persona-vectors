#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-artifacts/experiments/cross_persona_contract_rerun/20260423T061811Z__google__gemma-2-9b-it__implicit__evalp3__vecp30__q20}"
VECTOR_TRANSFORM="${VECTOR_TRANSFORM:-project_mean}"
LAYERS="${LAYERS:-all}"
ALPHAS="${ALPHAS:-1.0,2.0}"
QUESTIONS_PER_PERSONA="${QUESTIONS_PER_PERSONA:-20}"
LAYERS_LABEL="$(echo "$LAYERS" | tr ',' '_' | tr '-' 'to')"
OUT_ROOT="${OUT_ROOT:-artifacts/experiments/projected_vector_rerun/$(date -u +%Y%m%dT%H%M%SZ)__google__gemma-2-9b-it__${VECTOR_TRANSFORM}__layers_${LAYERS_LABEL}__q${QUESTIONS_PER_PERSONA}}"

mkdir -p "$OUT_ROOT"

uv run python experiments/07_projected_vector_rerun.py \
  --source-run-root "$SOURCE_RUN_ROOT" \
  --vector-transform "$VECTOR_TRANSFORM" \
  --layers "$LAYERS" \
  --questions-per-persona "$QUESTIONS_PER_PERSONA" \
  --question-batch-size 5 \
  --alphas "$ALPHAS" \
  --remote \
  --out-dir "$OUT_ROOT" \
  2>&1 | tee -a "$OUT_ROOT/launch.log"
