import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)

# Project-wide base directory for BIRD benchmark data. Override this path if
# your data lives outside the repository default location.
PROJECT_ROOT = Path(__file__).resolve().parent
BIRD_BASE = PROJECT_ROOT / "data"
EESQLBENCH_DB_ROOT = Path(
    os.getenv("EESQLBENCH_DB_ROOT", "~/Desktop/data/dev_databases")
).expanduser()
EESQLBENCH_TABLE_ROW_COUNT_CACHE = Path(
    os.getenv(
        "EESQLBENCH_TABLE_ROW_COUNT_CACHE",
        str(PROJECT_ROOT / "data" / "eesqlbench_table_row_counts.json"),
    )
).expanduser()

# OpenAI-compatible SDK configuration for EESQLBench runs.
DS_API_KEY = os.getenv("DS_API_KEY", "")
DS_BASE_URL = os.getenv("DS_BASE_URL", "")
DS_MODEL = os.getenv("DS_MODEL", "deepseek-v4-pro")
GPT_API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("OEPENAI_API_KEY", ""))
GPT_BASE_URL = os.getenv("OPENAI_GPT_BASE_URL", "")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-5.4")
DS_MAX_TOKENS = _env_int("DS_MAX_TOKENS", 3200)
GPT5_MAX_COMPLETION_TOKENS = _env_int(
    "GPT5_MAX_COMPLETION_TOKENS",
    DS_MAX_TOKENS,
)


def resolve_model_endpoint(model: str | None) -> tuple[str, str]:
    """Return the API key and base URL for the requested model."""
    normalized = (model or "").strip().lower()
    if normalized.startswith("gpt-5"):
        return GPT_API_KEY, GPT_BASE_URL
    return DS_API_KEY, DS_BASE_URL
