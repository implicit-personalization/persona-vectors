#!/usr/bin/env bash
# Extract activations for baseline_assistant + first 100 personas, then push to HF.
set -euo pipefail

MODEL="${MODEL:-google/gemma-2-9b-it}"
REPO="${REPO:-implicit-personalization/synth-persona-vectors}"
N="${N:-100}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=== Extracting baseline_assistant ==="
uv run python main.py extract --model "$MODEL" --persona-id baseline_assistant --backend remote

echo "=== Extracting first $N personas ==="
mapfile -t PERSONA_IDS < <(uv run python - "$N" <<'EOF'
import sys
from persona_data.synth_persona import SynthPersonaDataset
ds = SynthPersonaDataset()
for p in list(ds)[:int(sys.argv[1])]:
    print(p.id)
EOF
)

for pid in "${PERSONA_IDS[@]}"; do
    echo "--- $pid ---"
    uv run python main.py extract --model "$MODEL" --persona-id "$pid" --backend remote
done

echo "=== Pushing to HuggingFace Hub: $REPO ==="
uv run python scripts/push_to_hf.py --model "$MODEL" --repo "$REPO"

echo "Done."
