#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/Users/hengxuli/Repos/implicit-personalization/persona-vectors}"
ENV_FILE="${ENV_FILE:-/Users/hengxuli/Repos/synth-persona/.env}"
MODEL="${MODEL:-google/gemma-2-9b-it}"
LAYER="${LAYER:-41}"
MODE="${MODE:-persona}"
ATTRIBUTE="${ATTRIBUTE:-political_views}"
QA_FILTER="${QA_FILTER:-all}"
CONTEXT_MODE="${CONTEXT_MODE:-none}"
MAX_CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-0}"
QA_PER_PERSONA="${QA_PER_PERSONA:-4}"
PERSONA_LIMIT="${PERSONA_LIMIT:-6}"
MC_ITEMS_PER_PERSONA="${MC_ITEMS_PER_PERSONA:-8}"
TRAIN_PER_CLASS="${TRAIN_PER_CLASS:-2}"
SEEDS="${SEEDS:-1337}"
ALPHAS="${ALPHAS:-0.5}"
ATTRIBUTE_MC_EVAL="${ATTRIBUTE_MC_EVAL:-0}"
ATTRIBUTE_MC_ROTATIONS="${ATTRIBUTE_MC_ROTATIONS:-0,1,2,3}"
EXTRACTION_BATCH_SIZE="${EXTRACTION_BATCH_SIZE:-2}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-4}"
STEERING_POSITIONS="${STEERING_POSITIONS:-last}"
DRY_RUN="${DRY_RUN:-0}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/artifacts/experiments/response_mean_direction_suite/${RUN_ID}__gemma2-9b-it__layer_${LAYER}__${MODE}__${QA_FILTER}_smoke}"

cd "$REPO_ROOT"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

cmd=(
  uv run python experiments/15_response_mean_direction_suite.py
  --model "$MODEL" \
  --layer "$LAYER" \
  --mode "$MODE" \
  --attribute "$ATTRIBUTE" \
  --qa-filter "$QA_FILTER" \
  --context-mode "$CONTEXT_MODE" \
  --max-context-chars "$MAX_CONTEXT_CHARS" \
  --qa-per-persona "$QA_PER_PERSONA" \
  --persona-limit "$PERSONA_LIMIT" \
  --mc-items-per-persona "$MC_ITEMS_PER_PERSONA" \
  --train-per-class "$TRAIN_PER_CLASS" \
  --seeds "$SEEDS" \
  --alphas "$ALPHAS" \
  --extraction-batch-size "$EXTRACTION_BATCH_SIZE" \
  --score-batch-size "$SCORE_BATCH_SIZE" \
  --steering-positions "$STEERING_POSITIONS" \
  --env-file "$ENV_FILE" \
  --remote \
  --out-dir "$OUT_DIR"
)

if [[ "$ATTRIBUTE_MC_EVAL" == "1" || "$ATTRIBUTE_MC_EVAL" == "true" ]]; then
  cmd+=(--attribute-mc-eval --attribute-mc-rotations "$ATTRIBUTE_MC_ROTATIONS")
fi

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  cmd+=(--dry-run)
fi

"${cmd[@]}"
