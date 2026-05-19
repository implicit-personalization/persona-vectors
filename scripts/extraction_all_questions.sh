#!/usr/bin/env bash
# Extract baseline_assistant + first N personas across all questions, push to the
# HF Hub, and refresh the dataset card. Writes under artifacts/persona-vectors/.

set -euo pipefail

MODEL="${MODEL:-google/gemma-2-9b-it}"
N="${N:-100}"
BACKEND="${BACKEND:-remote}"
QA_TYPE="${QA_TYPE:-explicit}"
VARIANT="${VARIANT:-templated}"
SKIP_FAILED="${SKIP_FAILED:-0}"

REPO="${REPO:-implicit-personalization/synth-persona-vectors}"
# NOTE: Keep this run separate from the shared activations tree.
BASE_ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-artifacts/persona-vectors}"
CONFIG="${MODEL//\//__}__answer_mean"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR%/*}"

EXTRA_ARGS=()
if [[ "$SKIP_FAILED" == "1" ]]; then
    EXTRA_ARGS+=(--skip-failed)
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

echo "Model=$MODEL N=$N QA_TYPE=$QA_TYPE Backend=$BACKEND Variant=$VARIANT SkipFailed=$SKIP_FAILED BaseDir=$BASE_ACTIVATIONS_DIR"

run_extract baseline_assistant

uv run python main.py extract \
    --model "$MODEL" \
    --activations-dir "$BASE_ACTIVATIONS_DIR" \
    --backend "$BACKEND" \
    --variants "$VARIANT" \
    --sample-size "$N" \
    "${EXTRA_ARGS[@]}"

echo "=== Pushing to Hugging Face Hub: $REPO ==="
uv run python main.py push --model "$MODEL" --repo "$REPO" --activations-dir "$BASE_ACTIVATIONS_DIR" --variants "$VARIANT"

# Refresh the Hub dataset card from the uploaded parquet files. Counts come
# from Hugging Face, not local artifacts.
echo "=== Updating Hugging Face README: $REPO ==="
uv run python scripts/upload_hf_readme.py \
    --repo "$REPO" \
    --current-config "$CONFIG" \
    --question-set "all explicit questions" \
    --qa-filter "$QA_TYPE"
