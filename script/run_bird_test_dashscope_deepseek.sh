#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONUTF8=1
export CONFIG_PATH="${CONFIG_PATH:-config/local/config-bird-test.toml}"
PYTHON="${PYTHON:-python}"
RUN_ROOT="${RUN_ROOT:-workspace/runs/optsql-dashscope-deepseek-bird-test}"
LOG_DIR="$RUN_ROOT/logs"
MARKER_DIR="$RUN_ROOT/completed_stages"
BASE_SNAPSHOT="$RUN_ROOT/sql_selection.snapshot"
REPAIR_CHECKPOINT="$RUN_ROOT/controller/repair_checkpoint.jsonl"
CONTRACT_CHECKPOINT="$RUN_ROOT/controller/contract_checkpoint.jsonl"
FINAL_SNAPSHOT="$RUN_ROOT/controller/final.snapshot"
COLUMN_MEANING_PATH="${COLUMN_MEANING_PATH:-data/bird/test/column_meaning.json}"
CONTROLLER_WORKERS="${CONTROLLER_WORKERS:-8}"

mkdir -p "$LOG_DIR" "$MARKER_DIR" "$RUN_ROOT/controller"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "$name is required" >&2
    exit 2
  fi
}

require_env DASHSCOPE_API_KEY
require_env DEEPSEEK_API_KEY

for required in \
  "$CONFIG_PATH" \
  "$COLUMN_MEANING_PATH" \
  "data/bird/test/test.json" \
  "data/bird/test/test_tables.json" \
  "data/bird/test/test_databases" \
  "data/bird/train/train.json" \
  "models/Qwen3-Embedding-0.6B"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 2
  fi
done

run_stage() {
  local name="$1"
  shift
  local marker="$MARKER_DIR/$name.done"
  if [[ -f "$marker" ]]; then
    echo "[$(date -Iseconds)] reuse completed stage: $name"
    return 0
  fi
  echo "[$(date -Iseconds)] starting $name" | tee -a "$LOG_DIR/run.log"
  "$@" 2>&1 | tee -a "$LOG_DIR/$name.log"
  touch "$marker"
  echo "[$(date -Iseconds)] completed $name" | tee -a "$LOG_DIR/run.log"
}

run_stage offline_preflight "$PYTHON" scripts/validate_bird_submission.py \
  --config "$CONFIG_PATH"
run_stage api_preflight "$PYTHON" scripts/check_submission_apis.py
run_stage prepare_input "$PYTHON" scripts/prepare_bird_input.py \
  --bird-root data/bird \
  --split test \
  --column-meaning "$COLUMN_MEANING_PATH"
run_stage preprocess "$PYTHON" runner/preprocess_dataset.py
run_stage vector_index "$PYTHON" runner/create_vector_db_parallel.py
run_stage few_shot_index "$PYTHON" runner/build_few_shot_index.py \
  --config "$CONFIG_PATH"
run_stage value_retrieval "$PYTHON" runner/run_value_retrieval.py
run_stage few_shot_preparation "$PYTHON" runner/run_few_shot_preparation.py \
  --config "$CONFIG_PATH"
run_stage schema_linking "$PYTHON" runner/run_schema_linking.py
run_stage sql_generation "$PYTHON" runner/run_sql_generation.py
run_stage sql_revision "$PYTHON" runner/run_sql_revision.py
run_stage sql_selection "$PYTHON" runner/run_sql_selection.py
run_stage export_base "$PYTHON" runner/convert_snapshot_to_sql.py \
  --snapshot_path "$BASE_SNAPSHOT" \
  --output "$RUN_ROOT/base_predictions.json"
run_stage repair_candidates "$PYTHON" integration/run_consensus_repair.py \
  --snapshot "$BASE_SNAPSHOT" \
  --config "$CONFIG_PATH" \
  --llm-profile deepseek_controller \
  --checkpoint "$REPAIR_CHECKPOINT" \
  --output "$RUN_ROOT/controller/repair.snapshot" \
  --model deepseek-v4-pro \
  --workers "$CONTROLLER_WORKERS" \
  --sql-timeout-seconds 5
run_stage contract_controller "$PYTHON" integration/run_rag_contract_controller.py \
  --snapshot "$BASE_SNAPSHOT" \
  --source-checkpoint "$REPAIR_CHECKPOINT" \
  --config "$CONFIG_PATH" \
  --llm-profile deepseek_controller \
  --checkpoint "$CONTRACT_CHECKPOINT" \
  --output "$FINAL_SNAPSHOT" \
  --model deepseek-v4-pro \
  --workers "$CONTROLLER_WORKERS" \
  --sql-timeout-seconds 5
run_stage export_final "$PYTHON" runner/convert_snapshot_to_sql.py \
  --snapshot_path "$FINAL_SNAPSHOT" \
  --output "$RUN_ROOT/final_predictions.json"
run_stage validate_base "$PYTHON" scripts/validate_predictions.py \
  --test-json data/bird/test/test.json \
  --predictions "$RUN_ROOT/base_predictions.json"
run_stage validate_final "$PYTHON" scripts/validate_predictions.py \
  --test-json data/bird/test/test.json \
  --predictions "$RUN_ROOT/final_predictions.json"

echo "Base checkpoint:  $RUN_ROOT/base_predictions.json"
echo "Final checkpoint: $RUN_ROOT/final_predictions.json"
