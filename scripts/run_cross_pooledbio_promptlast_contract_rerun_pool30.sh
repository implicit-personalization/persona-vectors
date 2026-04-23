#!/usr/bin/env bash
set -euo pipefail

cd /Users/hengxuli/Repos/implicit-personalization/persona-vectors

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
MASK_STRATEGY="${MASK_STRATEGY:-prompt_last}"
QUESTIONS_PER_PERSONA="${QUESTIONS_PER_PERSONA:-20}"
ALPHAS="${ALPHAS:-1.0,2.0}"
OUT_ROOT="${OUT_ROOT:-artifacts/experiments/cross_persona_contract_rerun/${RUN_ID}__google__gemma-2-9b-it__implicit__evalp3__vecp30__q${QUESTIONS_PER_PERSONA}__${MASK_STRATEGY}}"
mkdir -p "${OUT_ROOT}"
echo "out_dir=${OUT_ROOT}"

uv run --env-file /Users/hengxuli/Repos/synth-persona/.env \
  python experiments/06_cross_persona_contract_rerun.py \
  --model google/gemma-2-9b-it \
  --personas 3 \
  --vector-personas 30 \
  --questions-per-persona "${QUESTIONS_PER_PERSONA}" \
  --qa-type implicit \
  --remote \
  --all-layers \
  --negative-variant pooled_biography \
  --method mean \
  --center \
  --mask-strategy "${MASK_STRATEGY}" \
  --alphas "${ALPHAS}" \
  --skip-extraction-failures \
  --extraction-batch-size 2 \
  --question-batch-size 5 \
  --out-dir "${OUT_ROOT}" | tee -a "${OUT_ROOT}/launch.log"
