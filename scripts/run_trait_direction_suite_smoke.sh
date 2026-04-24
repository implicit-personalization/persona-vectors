#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/Users/hengxuli/Repos/implicit-personalization/persona-vectors}"
ENV_FILE="${ENV_FILE:-/Users/hengxuli/Repos/synth-persona/.env}"
MODEL="${MODEL:-google/gemma-2-9b-it}"
LAYER="${LAYER:-41}"
TRAITS="${TRAITS:-political_views,total_wealth}"
EVAL_TRAITS="${EVAL_TRAITS:-same}"
ACTIVATION_SOURCE="${ACTIVATION_SOURCE:-prompt_last}"
TRAIN_PER_CLASS="${TRAIN_PER_CLASS:-4}"
SEEDS="${SEEDS:-1337}"
ALPHAS="${ALPHAS:-0.25,0.5,1.0}"
EVAL_OPTION_ROTATIONS="${EVAL_OPTION_ROTATIONS:-0}"
EXTRACTION_BATCH_SIZE="${EXTRACTION_BATCH_SIZE:-1}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-6}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/artifacts/experiments/trait_direction_suite/${RUN_ID}__gemma2-9b-it__layer_${LAYER}__${ACTIVATION_SOURCE}}"

cd "$REPO_ROOT"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

uv run python experiments/14_trait_direction_suite.py \
  --model "$MODEL" \
  --traits "$TRAITS" \
  --eval-traits "$EVAL_TRAITS" \
  --activation-source "$ACTIVATION_SOURCE" \
  --layer "$LAYER" \
  --train-per-class "$TRAIN_PER_CLASS" \
  --seeds "$SEEDS" \
  --alphas "$ALPHAS" \
  --eval-option-rotations "$EVAL_OPTION_ROTATIONS" \
  --extraction-batch-size "$EXTRACTION_BATCH_SIZE" \
  --score-batch-size "$SCORE_BATCH_SIZE" \
  --env-file "$ENV_FILE" \
  --remote \
  --out-dir "$OUT_DIR"
