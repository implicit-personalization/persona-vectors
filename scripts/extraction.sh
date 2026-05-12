#!/usr/bin/env bash
# Extract baseline_assistant + first N personas, then push to HF Hub.

set -euo pipefail

MODEL="${MODEL:-google/gemma-2-9b-it}"
REPO="${REPO:-implicit-personalization/synth-persona-vectors}"
N="${N:-100}"
BACKEND="${BACKEND:-remote}"
# NOTE: Currently working only with the templateed to simplify things and avoid OOM
VARIANT="${VARIANT:-templated}"
SKIP_FAILED="${SKIP_FAILED:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR%/*}"

EXTRA_ARGS=()
if [[ "$SKIP_FAILED" == "1" ]]; then
    EXTRA_ARGS+=(--skip-failed)
fi

run_extract() {
    local persona_id="$1"

    echo "=== Extracting: ${persona_id} ==="

    uv run python main.py extract \
        --model "$MODEL" \
        --persona-id "$persona_id" \
        --backend "$BACKEND" \
        --variants "$VARIANT" \
        "${EXTRA_ARGS[@]}"
}

echo "Model=$MODEL Repo=$REPO N=$N Backend=$BACKEND Variant=$VARIANT SkipFailed=$SKIP_FAILED"

run_extract baseline_assistant

uv run python main.py extract \
    --model "$MODEL" \
    --backend "$BACKEND" \
    --variants "$VARIANT" \
    --sample-size "$N" \
    "${EXTRA_ARGS[@]}"

echo "=== Pushing to Hugging Face Hub: $REPO ==="
uv run python main.py push --model "$MODEL" --repo "$REPO"
