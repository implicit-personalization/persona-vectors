#!/usr/bin/env bash
# Extract baseline_assistant + first N personas, then push to HF Hub.

set -euo pipefail

MODEL="${MODEL:-google/gemma-2-9b-it}"
REPO="${REPO:-implicit-personalization/synth-persona-vectors}"
N="${N:-100}"
BACKEND="${BACKEND:-remote}"
N_TRAIN="${N_TRAIN:-50}"
QA_TYPE="${QA_TYPE:-all}"
# NOTE: Currently working only with the templateed to simplify things and avoid OOM
VARIANT="${VARIANT:-templated}"
SKIP_FAILED="${SKIP_FAILED:-0}"
BASE_ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-artifacts/activations}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR%/*}"

EXTRA_ARGS=()
if [[ "$SKIP_FAILED" == "1" ]]; then
    EXTRA_ARGS+=(--skip-failed)
fi
if [[ "$N_TRAIN" != "all" ]]; then
    EXTRA_ARGS+=(--n-train "$N_TRAIN")
fi
if [[ "$QA_TYPE" != "all" ]]; then
    EXTRA_ARGS+=(--qa-type "$QA_TYPE")
fi

run_extract() {
    local persona_id="$1"

    echo "=== Extracting: ${persona_id} ==="

    uv run python main.py extract \
        --model "$MODEL" \
        --activations-dir "$BASE_ACTIVATIONS_DIR" \
        --persona-id "$persona_id" \
        --backend "$BACKEND" \
        --variants "$VARIANT" \
        "${EXTRA_ARGS[@]}"
}

echo "Model=$MODEL Repo=$REPO N=$N N_TRAIN=$N_TRAIN QA_TYPE=$QA_TYPE Backend=$BACKEND Variant=$VARIANT SkipFailed=$SKIP_FAILED BaseDir=$BASE_ACTIVATIONS_DIR"

run_extract baseline_assistant

uv run python main.py extract \
    --model "$MODEL" \
    --activations-dir "$BASE_ACTIVATIONS_DIR" \
    --backend "$BACKEND" \
    --variants "$VARIANT" \
    --sample-size "$N" \
    "${EXTRA_ARGS[@]}"

echo "=== Pushing to Hugging Face Hub: $REPO ==="
uv run python main.py push --model "$MODEL" --repo "$REPO" --activations-dir "$BASE_ACTIVATIONS_DIR"
