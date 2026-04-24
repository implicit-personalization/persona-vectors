#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/Users/hengxuli/Repos/implicit-personalization/persona-vectors}"
ENV_FILE="${ENV_FILE:-/Users/hengxuli/Repos/synth-persona/.env}"
MODEL="${MODEL:-google/gemma-2-9b-it}"
LAYER="${LAYER:-41}"
TRAIN_PER_CLASS="${TRAIN_PER_CLASS:-4}"
ALPHAS="${ALPHAS:-0.25,0.5,1.0}"
EXTRACTION_BATCH_SIZE="${EXTRACTION_BATCH_SIZE:-1}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/artifacts/experiments/attribute_direction_smoke/${RUN_ID}__gemma2-9b-it__political_views__layer_${LAYER}}"

cd "$REPO_ROOT"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

uv run python experiments/13_attribute_direction_smoke.py \
  --model "$MODEL" \
  --attribute political_views \
  --layer "$LAYER" \
  --train-per-class "$TRAIN_PER_CLASS" \
  --alphas "$ALPHAS" \
  --extraction-batch-size "$EXTRACTION_BATCH_SIZE" \
  --env-file "$ENV_FILE" \
  --remote \
  --out-dir "$OUT_DIR"
