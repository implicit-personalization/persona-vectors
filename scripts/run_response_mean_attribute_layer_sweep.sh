#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/Users/hengxuli/Repos/implicit-personalization/persona-vectors}"
ENV_FILE="${ENV_FILE:-/Users/hengxuli/Repos/synth-persona/.env}"
MODEL="${MODEL:-google/gemma-2-9b-it}"
LAYERS="${LAYERS:-32 35 38}"
MODE="${MODE:-attribute}"
ATTRIBUTE="${ATTRIBUTE:-political_views}"
QA_FILTER="${QA_FILTER:-attribute_keywords}"
CONTEXT_MODE="${CONTEXT_MODE:-none}"
MAX_CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-0}"
QA_PER_PERSONA="${QA_PER_PERSONA:-4}"
PERSONA_LIMIT="${PERSONA_LIMIT:-0}"
MC_ITEMS_PER_PERSONA="${MC_ITEMS_PER_PERSONA:-0}"
TRAIN_PER_CLASS="${TRAIN_PER_CLASS:-4}"
SEEDS="${SEEDS:-1337}"
ALPHAS="${ALPHAS:-1.0,2.0,5.0}"
ATTRIBUTE_MC_EVAL="${ATTRIBUTE_MC_EVAL:-1}"
ATTRIBUTE_MC_ROTATIONS="${ATTRIBUTE_MC_ROTATIONS:-0,1,2,3}"
EXTRACTION_BATCH_SIZE="${EXTRACTION_BATCH_SIZE:-2}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-8}"
STEERING_POSITIONS="${STEERING_POSITIONS:-all_prompt}"
DRY_RUN="${DRY_RUN:-0}"
LAYER_TIMEOUT_SECONDS="${LAYER_TIMEOUT_SECONDS:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/artifacts/experiments/response_mean_direction_suite}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/response_mean_direction_suite}"

cd "$REPO_ROOT"
mkdir -p "$LOG_DIR"

NORMALIZED_LAYERS="${LAYERS//,/ }"
LAYER_SLUG="${NORMALIZED_LAYERS// /-}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${RUN_ID}__attribute_${ATTRIBUTE}__layers_${LAYER_SLUG}__${STEERING_POSITIONS}.log}"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "run_id=$RUN_ID"
echo "repo_root=$REPO_ROOT"
echo "log_file=$LOG_FILE"
echo "model=$MODEL"
echo "layers=$NORMALIZED_LAYERS"
echo "attribute=$ATTRIBUTE"
echo "qa_filter=$QA_FILTER"
echo "steering_positions=$STEERING_POSITIONS"
echo "alphas=$ALPHAS"
echo "attribute_mc_rotations=$ATTRIBUTE_MC_ROTATIONS"
echo "dry_run=$DRY_RUN"
echo "layer_timeout_seconds=$LAYER_TIMEOUT_SECONDS"
echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ "$MODE" != "attribute" ]]; then
  echo "This runner is only for attribute sweeps; got MODE=$MODE" >&2
  exit 2
fi

for LAYER in $NORMALIZED_LAYERS; do
  OUT_DIR="$OUT_ROOT/${RUN_ID}__gemma2-9b-it__layer_${LAYER}__attribute_${ATTRIBUTE}__${QA_FILTER}__${STEERING_POSITIONS}"
  if [[ -e "$OUT_DIR" ]]; then
    echo "Refusing to reuse existing OUT_DIR=$OUT_DIR" >&2
    exit 3
  fi

  echo
  echo "===== layer=$LAYER started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) ====="

  layer_cmd=(
    env
    REPO_ROOT="$REPO_ROOT" \
    ENV_FILE="$ENV_FILE" \
    MODEL="$MODEL" \
    LAYER="$LAYER" \
    MODE="$MODE" \
    ATTRIBUTE="$ATTRIBUTE" \
    QA_FILTER="$QA_FILTER" \
    CONTEXT_MODE="$CONTEXT_MODE" \
    MAX_CONTEXT_CHARS="$MAX_CONTEXT_CHARS" \
    QA_PER_PERSONA="$QA_PER_PERSONA" \
    PERSONA_LIMIT="$PERSONA_LIMIT" \
    MC_ITEMS_PER_PERSONA="$MC_ITEMS_PER_PERSONA" \
    TRAIN_PER_CLASS="$TRAIN_PER_CLASS" \
    SEEDS="$SEEDS" \
    ALPHAS="$ALPHAS" \
    ATTRIBUTE_MC_EVAL="$ATTRIBUTE_MC_EVAL" \
    ATTRIBUTE_MC_ROTATIONS="$ATTRIBUTE_MC_ROTATIONS" \
    EXTRACTION_BATCH_SIZE="$EXTRACTION_BATCH_SIZE" \
    SCORE_BATCH_SIZE="$SCORE_BATCH_SIZE" \
    STEERING_POSITIONS="$STEERING_POSITIONS" \
    DRY_RUN="$DRY_RUN" \
    OUT_DIR="$OUT_DIR" \
    bash scripts/run_response_mean_direction_suite_smoke.sh
  )

  if [[ "$LAYER_TIMEOUT_SECONDS" != "0" ]]; then
    "${layer_cmd[@]}" &
    layer_pid=$!
    (
      sleep "$LAYER_TIMEOUT_SECONDS"
      if kill -0 "$layer_pid" 2>/dev/null; then
        echo "Layer $LAYER timed out after ${LAYER_TIMEOUT_SECONDS}s; sending SIGINT"
        kill -INT "$layer_pid" 2>/dev/null || true
        sleep 15
        if kill -0 "$layer_pid" 2>/dev/null; then
          echo "Layer $LAYER still running after SIGINT; sending SIGTERM"
          kill -TERM "$layer_pid" 2>/dev/null || true
        fi
      fi
    ) &
    watchdog_pid=$!
    set +e
    wait "$layer_pid"
    layer_status=$?
    set -e
    kill "$watchdog_pid" 2>/dev/null || true
    wait "$watchdog_pid" 2>/dev/null || true
    if [[ "$layer_status" != "0" ]]; then
      exit "$layer_status"
    fi
  else
    "${layer_cmd[@]}"
  fi

  echo "===== layer=$LAYER finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  if [[ -f "$OUT_DIR/summary.json" ]]; then
    echo "summary=$OUT_DIR/summary.json"
  else
    echo "Missing summary for layer=$LAYER at $OUT_DIR/summary.json" >&2
    exit 4
  fi
done

echo
echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
