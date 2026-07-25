# OptSQL BIRD Test Submission

This package contains the BIRD test entry point for OptSQL.
It produces two checkpoints from one run:

1. `base_predictions.json`: Qwen3-Coder-Plus base generation and SQL selection.
2. `final_predictions.json`: the base checkpoint after conservative OptSQL
   repair, answer-contract judging, semantic guarding, and fallback.

## Team

- Method: OptSQL
- Team: OptSQL-Repro
- Lead contributor: Lin Li, Zhejiang University
- Primary contact: Xuyijie, Zhejiang University
- Contact: xu.yijie@qq.com

## Declared Models

| Role | Provider | Model |
| --- | --- | --- |
| Base Text-to-SQL pipeline | Alibaba Cloud Model Studio / DashScope | `qwen3-coder-plus` |
| Repair and Controller | DeepSeek official API | `deepseek-v4-pro` |
| Embedding and retrieval | Local | `Qwen/Qwen3-Embedding-0.6B` |

API credentials are read only from environment variables. SQLite databases
are opened locally and are never uploaded. Model prompts contain the question,
evidence, selected schema, retrieved value snippets, and retrieved training
examples required for Text-to-SQL inference. Qwen requests are sent only to
the declared DashScope workspace endpoint, and Controller requests are sent
only to the official DeepSeek endpoint.

## Development Result

The development experiment used all 1,534 BIRD 2023 dev examples:

| Checkpoint | Correct | EX |
| --- | ---: | ---: |
| Qwen3-Coder-Plus base | 1,109 | 72.2947% |
| OptSQL final | 1,121 | 73.0769% |

The Controller fixed 12 examples and broke 0 on the development set. These are
development results; official test performance must be determined by BIRD.
The evaluation used the original 1,534-example BIRD 2023 dev split, not the
2025 `bird-sql-dev-1106` refresh.

Development API usage:

| Stage | Prompt tokens | Completion tokens | Total tokens |
| --- | ---: | ---: | ---: |
| Qwen base pipeline | 140,554,036 | 31,105,581 | 171,659,617 |
| DeepSeek repair and Controller | 15,401,177 | 6,592,048 | 21,993,225 |

Gold-free development prediction files are included under `results/bird-dev/`.

## Requirements

- Ubuntu 22.04 or a compatible Linux distribution
- Python 3.12
- One CUDA-capable GPU with at least 8 GB VRAM
- At least 32 GB system RAM
- Alibaba Cloud Model Studio credit for `qwen3-coder-plus`
- DeepSeek official API credit for `deepseek-v4-pro`
- BIRD train and hidden test files supplied by the evaluator

Verified development hardware was one RTX 4060 Laptop GPU with 8 GB VRAM and
32 GB RAM. Expected test runtime is approximately 48-72 hours with base
parallelism 2 and Controller parallelism 8, excluding provider queue time.

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p config/local models
cp config/template/dashscope-deepseek/config-bird-test.toml \
  config/local/config-bird-test.toml

python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-0.6B', local_dir='models/Qwen3-Embedding-0.6B')"
```

Set the dedicated evaluation credentials in the launching shell:

```bash
export DASHSCOPE_API_KEY='temporary-dashscope-evaluation-key'
export DEEPSEEK_API_KEY='temporary-deepseek-evaluation-key'
```

The keys are not stored in configuration files, checkpoints, logs, or the
submission archive.

## BIRD Input Layout

```text
data/bird/
├── train/
│   ├── train.json
│   └── train_databases/
└── test/
    ├── test.json
    ├── test_tables.json
    ├── column_meaning.json
    └── test_databases/
```

`column_meaning.json` is required and used to create the schema descriptions
consumed by the pipeline. The hidden test records may omit `question_id`,
`difficulty`, and gold `SQL`. The test entry point never reads gold SQL.

## Preflight

Offline configuration and secret scan:

```bash
python scripts/validate_bird_submission.py \
  --config config/local/config-bird-test.toml
```

Minimal provider health checks:

```bash
python scripts/check_submission_apis.py
```

The checks hide all credential values.

## Run

```bash
CONFIG_PATH=config/local/config-bird-test.toml \
PYTHON=.venv/bin/python \
bash script/run_bird_test_dashscope_deepseek.sh
```

The entry point runs:

1. BIRD input and `column_meaning.json` preparation.
2. Local value and few-shot indexes.
3. Qwen schema linking, candidate generation, revision, and base selection.
4. Three independent DeepSeek repair proposals per example.
5. Execution-result consensus.
6. Candidate-blind answer-contract planning and position-swapped judging.
7. AST semantic guard; every rejected or failed change falls back to base SQL.
8. Gold-free completeness validation for both output checkpoints.

Final files:

```text
workspace/runs/optsql-dashscope-deepseek-bird-test/base_predictions.json
workspace/runs/optsql-dashscope-deepseek-bird-test/final_predictions.json
```

## Resume and Failure Handling

Every expensive pipeline stage stores snapshots or item-level JSONL
checkpoints. A stage marker is written only after successful completion.
Re-running the same command reuses completed stages and retries failed
Controller items. If a Controller request fails, that item remains on the base
SQL until it is retried successfully.

To intentionally rerun one completed stage, delete only its marker under:

```text
workspace/runs/optsql-dashscope-deepseek-bird-test/completed_stages/
```

Do not delete snapshots or JSONL checkpoints when resuming.
