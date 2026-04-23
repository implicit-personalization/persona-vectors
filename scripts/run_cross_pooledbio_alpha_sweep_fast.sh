#!/usr/bin/env bash
set -euo pipefail

cd /Users/hengxuli/Repos/implicit-personalization/persona-vectors

uv run --env-file /Users/hengxuli/Repos/synth-persona/.env \
  python experiments/05_cross_persona_alpha_sweep_fast.py \
  --model google/gemma-2-9b-it \
  --personas 3 \
  --questions-per-persona 20 \
  --qa-type implicit \
  --remote \
  --all-layers \
  --negative-variant pooled_biography \
  --method mean \
  --center \
  --alphas 0.5,1.0,2.0,3.0
