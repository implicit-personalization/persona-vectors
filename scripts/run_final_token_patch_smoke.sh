#!/usr/bin/env bash
set -euo pipefail

REFERENCE_JSONL="${REFERENCE_JSONL:-artifacts/experiments/steering_compare/20260422T192442Z__gemma2-9b-it__p3__q20/shared_reference/per_example.jsonl}"
LAYERS="${LAYERS:-30-41}"
LIMIT="${LIMIT:-3}"
PATCH_MODE="${PATCH_MODE:-replace_clean}"
ENV_FILE="${ENV_FILE:-/Users/hengxuli/Repos/synth-persona/.env}"

uv run python experiments/11_final_token_patch_smoke.py \
  --reference-jsonl "$REFERENCE_JSONL" \
  --layers "$LAYERS" \
  --limit "$LIMIT" \
  --patch-mode "$PATCH_MODE" \
  --env-file "$ENV_FILE" \
  --remote
