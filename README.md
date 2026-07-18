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
| `config/template/qwen3-coder-plus/` | Reproducible BIRD configuration |
| `scripts/` | Resumable PowerShell orchestration |
| `results/bird-dev/` | Public development predictions |

## Requirements

- Python 3.12
- `uv`
- BIRD dev data
- DashScope-compatible `qwen3-coder-plus` API access
- `Qwen/Qwen3-Embedding-0.6B`
- An OptSQL controller checkout from <https://github.com/OptSQL/OptSQL>

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
$env:OPTSQL_ROOT = "D:\path\to\OptSQL"
```

Place the BIRD data under `data/bird` and the embedding model under
`models/Qwen3-Embedding-0.6B`, or edit the copied local configuration.

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
