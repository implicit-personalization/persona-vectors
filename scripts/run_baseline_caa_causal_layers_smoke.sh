#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/Users/hengxuli/Repos/implicit-personalization/persona-vectors}"
ENV_FILE="${ENV_FILE:-/Users/hengxuli/Repos/synth-persona/.env}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-$REPO_ROOT/artifacts/experiments/baseline_caa_smoke/20260423T215240__gemma9b__prompt_last__baseline__all_layers__p3q5}"
LAYERS="${LAYERS:-26-41}"
ALPHAS="${ALPHAS:-0.5,1.0,2.0,4.0}"
QUESTIONS_PER_PERSONA="${QUESTIONS_PER_PERSONA:-5}"
QUESTION_BATCH_SIZE="${QUESTION_BATCH_SIZE:-5}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/artifacts/experiments/baseline_caa_causal_layers_smoke/${RUN_ID}__gemma2-9b-it__layers_${LAYERS//-/_to_}__q${QUESTIONS_PER_PERSONA}}"

cd "$REPO_ROOT"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

uv run python experiments/07_projected_vector_rerun.py \
  --source-run-root "$SOURCE_RUN_ROOT" \
  --layers "$LAYERS" \
  --vector-transform none \
  --alphas "$ALPHAS" \
  --questions-per-persona "$QUESTIONS_PER_PERSONA" \
  --question-batch-size "$QUESTION_BATCH_SIZE" \
  --remote \
  --out-dir "$OUT_DIR"
