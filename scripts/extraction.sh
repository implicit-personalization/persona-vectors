#!/usr/bin/env bash
# Extract baseline_assistant + first N personas, then push to HF Hub.

set -euo pipefail

MODEL="${MODEL:-google/gemma-2-9b-it}"
REPO="${REPO:-implicit-personalization/synth-persona-vectors}"
N="${N:-100}"
BACKEND="${BACKEND:-remote}"
# NOTE: Currently working only with the templateed to simplify things and avoid OOM
VARIANT="${VARIANT:-templated}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR%/*}"

run_extract() {
    local persona_id="$1"

    echo "=== Extracting: ${persona_id} ==="

    uv run python main.py extract \
        --model "$MODEL" \
        --persona-id "$persona_id" \
        --backend "$BACKEND" \
        --variants "$VARIANT"
}

echo "Model=$MODEL Repo=$REPO N=$N Backend=$BACKEND Variant=$VARIANT"

run_extract baseline_assistant

mapfile -t PERSONA_IDS < <(uv run python - "$N" <<'EOF'
import sys
from persona_data.synth_persona import SynthPersonaDataset

n = int(sys.argv[1])
for persona in list(SynthPersonaDataset())[:n]:
    print(persona.id)
EOF
)

for persona_id in "${PERSONA_IDS[@]}"; do
    run_extract "$persona_id"
done

echo "=== Pushing to Hugging Face Hub: $REPO ==="
uv run python scripts/push_to_hf.py --model "$MODEL" --repo "$REPO"
