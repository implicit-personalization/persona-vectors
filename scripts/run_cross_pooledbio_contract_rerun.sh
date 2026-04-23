#!/usr/bin/env bash
set -euo pipefail

cd /Users/hengxuli/Repos/implicit-personalization/persona-vectors

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="artifacts/experiments/cross_persona_contract_rerun/${RUN_ID}__google__gemma-2-9b-it__implicit__p3__q20"
mkdir -p "${OUT_ROOT}"
echo "out_dir=${OUT_ROOT}"

uv run --env-file /Users/hengxuli/Repos/synth-persona/.env \
  python experiments/06_cross_persona_contract_rerun.py \
  --model google/gemma-2-9b-it \
  --personas 3 \
  --questions-per-persona 20 \
  --qa-type implicit \
  --remote \
  --all-layers \
  --negative-variant pooled_biography \
  --method mean \
  --center \
  --alphas 1.0,2.0 \
  --question-batch-size 5 \
  --out-dir "${OUT_ROOT}" | tee "${OUT_ROOT}/launch.log"
