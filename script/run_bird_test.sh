#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONUTF8=1
export CONFIG_PATH="${CONFIG_PATH:-config/local/config-bird-test.toml}"
export DS_BASE_URL="${DS_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export OPTSQL_ROOT="${OPTSQL_ROOT:-$PROJECT_ROOT/vendor/optsql_controller}"

COLUMN_MEANING_PATH="${COLUMN_MEANING_PATH:-data/bird/test/column_meaning.json}"
RUN_ROOT="${RUN_ROOT:-workspace/runs/qwen3-coder-plus-bird-test}"
SNAPSHOT="$RUN_ROOT/sql_selection.snapshot"
CONTROLLER_OUTPUT="$RUN_ROOT/optsql_batch"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"

if [[ -z "${DS_API_KEY:-}" ]]; then
  echo "DS_API_KEY is required" >&2
  exit 2
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config: $CONFIG_PATH" >&2
  exit 2
fi
if [[ ! -f "$COLUMN_MEANING_PATH" ]]; then
  echo "Missing column meanings: $COLUMN_MEANING_PATH" >&2
  exit 2
fi
if [[ ! -d "$OPTSQL_ROOT" ]]; then
  echo "Missing controller runtime: $OPTSQL_ROOT" >&2
  exit 2
fi

run_stage() {
  local name="$1"
  shift
  echo "[$(date -Iseconds)] starting $name" | tee -a "$LOG_DIR/run.log"
  "$@" 2>&1 | tee -a "$LOG_DIR/${name}.log"
  echo "[$(date -Iseconds)] completed $name" | tee -a "$LOG_DIR/run.log"
}

run_stage prepare_input python scripts/prepare_bird_input.py \
  --bird-root data/bird --split test --column-meaning "$COLUMN_MEANING_PATH"
run_stage preprocess python runner/preprocess_dataset.py
run_stage vector_index python runner/create_vector_db_parallel.py
run_stage few_shot_index python runner/build_few_shot_index.py --config "$CONFIG_PATH"
run_stage value_retrieval python runner/run_value_retrieval.py
run_stage few_shot_preparation python runner/run_few_shot_preparation.py --config "$CONFIG_PATH"
run_stage schema_linking python runner/run_schema_linking.py
run_stage sql_generation python runner/run_sql_generation.py
run_stage sql_revision python runner/run_sql_revision.py
run_stage sql_selection python runner/run_sql_selection.py
run_stage export_base python runner/convert_snapshot_to_sql.py \
  --snapshot_path "$SNAPSHOT" --output "$RUN_ROOT/base_predictions.json"
run_stage controller python integration/run_optsql_bridge_batch.py \
  --snapshot "$SNAPSHOT" --optsql-root "$OPTSQL_ROOT" --output-dir "$CONTROLLER_OUTPUT"

cp "$CONTROLLER_OUTPUT/predictions.json" "$RUN_ROOT/predictions.json"
python scripts/validate_predictions.py \
  --test-json data/bird/test/test.json \
  --predictions "$RUN_ROOT/predictions.json"
echo "Final predictions: $RUN_ROOT/predictions.json"
