#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-artifacts/experiments/cross_persona_contract_rerun/20260423T061811Z__google__gemma-2-9b-it__implicit__evalp3__vecp30__q20}"
CONSTRUCTION="${CONSTRUCTION:-unit_mean}"
LAYERS="${LAYERS:-37-41}"
ALPHAS="${ALPHAS:-8.0}"
QUESTIONS_PER_PERSONA="${QUESTIONS_PER_PERSONA:-20}"
QUESTION_BATCH_SIZE="${QUESTION_BATCH_SIZE:-5}"
LAYERS_LABEL="$(echo "$LAYERS" | tr ',' '_' | tr '-' 'to')"
OUT_ROOT="${OUT_ROOT:-artifacts/experiments/static_vector_construction_rerun/$(date -u +%Y%m%dT%H%M%SZ)__google__gemma-2-9b-it__${CONSTRUCTION}__layers_${LAYERS_LABEL}__q${QUESTIONS_PER_PERSONA}}"

mkdir -p "$OUT_ROOT"

uv run python experiments/09_static_construction_rerun.py \
  --source-run-root "$SOURCE_RUN_ROOT" \
  --construction "$CONSTRUCTION" \
  --layers "$LAYERS" \
  --questions-per-persona "$QUESTIONS_PER_PERSONA" \
  --question-batch-size "$QUESTION_BATCH_SIZE" \
  --alphas "$ALPHAS" \
  --remote \
  --out-dir "$OUT_ROOT" \
  2>&1 | tee -a "$OUT_ROOT/launch.log"
