#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/Users/hengxuli/Repos/implicit-personalization/persona-vectors}"
ENV_FILE="${ENV_FILE:-/Users/hengxuli/Repos/synth-persona/.env}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-$REPO_ROOT/artifacts/experiments/baseline_caa_promptlast_q20/20260424T052630Z__gemma2-9b-it__p3__q20__center_false}"
LAYER="${LAYER:-41}"
ALPHAS="${ALPHAS:-1.0}"
QUESTIONS_PER_PERSONA="${QUESTIONS_PER_PERSONA:-20}"
CENTER="${CENTER:-false}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/artifacts/experiments/leave_one_out_delta_steering/${RUN_ID}__gemma2-9b-it__layer_${LAYER}__q${QUESTIONS_PER_PERSONA}__center_${CENTER}}"

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

uv run python experiments/12_leave_one_out_delta_steering.py \
  --source-run-root "$SOURCE_RUN_ROOT" \
  --layer "$LAYER" \
  --questions-per-persona "$QUESTIONS_PER_PERSONA" \
  --alphas "$ALPHAS" \
  "${CENTER_ARGS[@]}" \
  --remote \
  --out-dir "$OUT_DIR"
