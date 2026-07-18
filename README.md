# OptSQL

OptSQL is a reproducible Text-to-SQL pipeline for BIRD. This repository contains
the Qwen3-Coder-Plus generation workflow, resumable stage snapshots, the
meta-controller and optimization adapter, and development-set predictions.

**Affiliations:** Zhejiang University and Alibaba Cloud  
**Team:** OptSQL-Repro  
**Contact:** Xuyijie (徐一杰), xu.yijie@qq.com

## Development Result

| Dataset | Model | Examples | EX |
| --- | --- | ---: | ---: |
| BIRD dev | Qwen3-Coder-Plus | 1,534 | 72.19% |

The generation and controller prediction files are available under
`results/bird-dev/`. Detailed local measurements are documented in
[`RESULTS_zh.md`](RESULTS_zh.md).

## Repository Layout

| Path | Purpose |
| --- | --- |
| `app/` | Schema linking, retrieval, SQL generation, revision, and selection |
| `runner/` | Stage entry points and snapshot export |
| `integration/` | OptSQL controller adapter and metric evaluation |
| `config/template/qwen3-coder-plus/` | Reproducible BIRD dev/test configurations |
| `scripts/` | Resumable PowerShell orchestration |
| `script/run_bird_test.sh` | Linux BIRD-test entry point |
| `vendor/optsql_controller/` | Controller runtime used by the final stage |
| `results/bird-dev/` | Public development predictions |

## Requirements

- Python 3.12
- `uv`
- BIRD dev data
- DashScope-compatible `qwen3-coder-plus` API access
- `Qwen/Qwen3-Embedding-0.6B`
- Linux evaluation: CUDA 12.2/12.3 compatible environment

Large datasets, model weights, vector indexes, snapshots, and credentials are
intentionally excluded from this repository.

## Setup

```powershell
uv sync
New-Item -ItemType Directory -Force config/local | Out-Null
Copy-Item config/template/qwen3-coder-plus/config-bird-dev.toml `
  config/local/config-bird-dev.toml

$env:DS_API_KEY = "your-temporary-key"
$env:DS_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:OPTSQL_ROOT = "$PWD\vendor\optsql_controller"
```

Place the BIRD data under `data/bird` and the embedding model under
`models/Qwen3-Embedding-0.6B`, or edit the copied local configuration.

## BIRD Test Submission

This is a **Combined Models** submission. SQL generation uses the declared
DashScope API, while embedding/retrieval and SQLite execution run locally. The
only model-service network endpoint used by the BIRD pipeline is
`DS_BASE_URL`. Database files are opened read-only and are never uploaded; API
prompts contain the question, evidence, selected schema, and retrieved value
snippets needed for Text-to-SQL inference.

Expected official input layout:

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

`column_meaning.json` is **required and used**. The submission entry point
converts it into per-table schema descriptions before preprocessing. Test
records may omit `question_id`, `difficulty`, and gold `SQL`; deterministic
defaults are assigned and no evaluation stage reads gold SQL.

Linux setup and execution:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p config/local models
cp config/template/qwen3-coder-plus/config-bird-test.toml \
  config/local/config-bird-test.toml
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-0.6B', local_dir='models/Qwen3-Embedding-0.6B')"

export DS_API_KEY='temporary-evaluation-key'
export DS_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
bash script/run_bird_test.sh
```

The final SQL file is written to:

```text
workspace/runs/qwen3-coder-plus-bird-test/predictions.json
```

Every expensive stage stores item-level artifacts and logs under the same run
directory. After an API, SSH, or machine interruption, run the same command
again; completed items are reused. The final validator fails if any prediction
is missing, empty, or has an unexpected ID.

### Resources and Runtime

- Local model: `Qwen/Qwen3-Embedding-0.6B`
- API model: `qwen3-coder-plus`
- Verified dev hardware: one RTX 4060 Laptop GPU (8 GB VRAM), 32 GB RAM
- CUDA required by the provided config: one GPU with at least 8 GB VRAM
- BIRD dev active runtime: approximately 32 hours with `parallelism = 2`
- Estimated BIRD test runtime: approximately 36-48 hours, excluding queue time
- BIRD dev prompt tokens: 140,554,036
- BIRD dev completion tokens: 31,105,581
- BIRD dev total tokens: 171,659,617

The API key must be supplied by the submitter as an environment variable and
can be revoked immediately after the evaluation.

## Run

Each stage writes resumable snapshots. A stopped run can continue from its last
completed item.

```powershell
./scripts/run_full_stage.ps1 -Stage FewShotIndex
./scripts/run_full_stage.ps1 -Stage ValueRetrieval
./scripts/run_full_stage.ps1 -Stage FewShotPreparation
./scripts/run_full_stage.ps1 -Stage SchemaLinking
./scripts/run_full_stage.ps1 -Stage SqlGeneration
./scripts/run_full_stage.ps1 -Stage SqlRevision
./scripts/run_full_stage.ps1 -Stage SqlSelection
./scripts/run_full_stage.ps1 -Stage Export
./scripts/run_full_stage.ps1 -Stage Controller
```

Run every stage in order with:

```powershell
./scripts/run_full_sequence.ps1
```

## Credentials

API credentials are read only from environment variables. Never commit `.env`,
local TOML files, logs, snapshots, or API keys.
