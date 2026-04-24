#!/usr/bin/env bash
set -euo pipefail

SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-artifacts/experiments/cross_persona_contract_rerun/20260423T202734Z__google__gemma-2-9b-it__implicit__evalp3__vecp30__q20__prompt_mean}"
QUESTIONS_PER_PERSONA="${QUESTIONS_PER_PERSONA:-5}"
ALPHAS="${ALPHAS:-2.0}"
LAYERS="${LAYERS:-37-41}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4}"
QUESTION_BATCH_SIZE="${QUESTION_BATCH_SIZE:-5}"
ENV_FILE="${ENV_FILE:-/Users/hengxuli/Repos/synth-persona/.env}"

uv run python experiments/10_generation_actadd_smoke.py \
  --source-run-root "$SOURCE_RUN_ROOT" \
  --questions-per-persona "$QUESTIONS_PER_PERSONA" \
  --alphas "$ALPHAS" \
  --layers "$LAYERS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --question-batch-size "$QUESTION_BATCH_SIZE" \
  --env-file "$ENV_FILE" \
  --remote
