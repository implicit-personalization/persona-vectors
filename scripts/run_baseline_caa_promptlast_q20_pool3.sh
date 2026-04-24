#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/Users/hengxuli/Repos/implicit-personalization/persona-vectors}"
ENV_FILE="${ENV_FILE:-/Users/hengxuli/Repos/synth-persona/.env}"
MODEL="${MODEL:-google/gemma-2-9b-it}"
PERSONAS="${PERSONAS:-3}"
VECTOR_PERSONAS="${VECTOR_PERSONAS:-3}"
QUESTIONS_PER_PERSONA="${QUESTIONS_PER_PERSONA:-20}"
ALPHAS="${ALPHAS:-0.25,0.5,1.0}"
QUESTION_BATCH_SIZE="${QUESTION_BATCH_SIZE:-5}"
EXTRACTION_BATCH_SIZE="${EXTRACTION_BATCH_SIZE:-5}"
CENTER="${CENTER:-false}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/artifacts/experiments/baseline_caa_promptlast_q20/${RUN_ID}__gemma2-9b-it__p${PERSONAS}__q${QUESTIONS_PER_PERSONA}__center_${CENTER}}"

cd "$REPO_ROOT"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

CENTER_ARGS=()
if [[ "$CENTER" == "true" ]]; then
  CENTER_ARGS=(--center)
elif [[ "$CENTER" == "false" ]]; then
  CENTER_ARGS=(--no-center)
else
  echo "CENTER must be one of: true, false" >&2
  exit 2
fi

uv run python experiments/06_cross_persona_contract_rerun.py \
  --model "$MODEL" \
  --personas "$PERSONAS" \
  --vector-personas "$VECTOR_PERSONAS" \
  --questions-per-persona "$QUESTIONS_PER_PERSONA" \
  --qa-type implicit \
  --remote \
  --negative-variant baseline \
  --method mean \
  "${CENTER_ARGS[@]}" \
  --mask-strategy prompt_last \
  --all-layers \
  --alphas "$ALPHAS" \
  --question-batch-size "$QUESTION_BATCH_SIZE" \
  --extraction-batch-size "$EXTRACTION_BATCH_SIZE" \
  --out-dir "$OUT_DIR"
