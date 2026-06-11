#!/usr/bin/env bash
# Extract persona_mean vectors (one forward pass over the persona text itself;
# the masked mean covers only the system-prompt tokens, so the result is
# independent of the QA pair) for all personas, both prompt variants.
# Feeds notebooks/unsupervised/datamap_persona_mean.py. Resumable: personas
# already extracted for this selection are skipped on re-run.

set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
BACKEND="${BACKEND:-remote}"
VARIANTS="${VARIANTS:-templated biography}"
# NOTE: the shared activations tree, not artifacts/persona-vectors - these
# are text-reading embeddings, not the answer_mean persona vectors pushed to
# the Hub by extraction_all_questions.sh.
ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-artifacts/activations}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR%/*}"

echo "Model=$MODEL Backend=$BACKEND Variants=$VARIANTS Dir=$ACTIVATIONS_DIR"

# shellcheck disable=SC2086
uv run python main.py extract \
    --model "$MODEL" \
    --mask-strategy persona_mean \
    --backend "$BACKEND" \
    --variants $VARIANTS \
    --activations-dir "$ACTIVATIONS_DIR" \
    --skip-failed
