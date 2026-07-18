import sqlite3
from pathlib import Path

from config import BIRD_BASE
from config import EESQLBENCH_DB_ROOT
from config import EESQLBENCH_TABLE_ROW_COUNT_CACHE

_CUSTOM_DATABASE_PATHS: dict[str, Path] = {}
_BENCHMARK_PROFILES: dict[str, dict[str, Path]] = {
    "bird": {
        "db_root": Path(BIRD_BASE) / "dev_20240627" / "dev_databases",
        "table_row_count_cache": Path(BIRD_BASE) / "dev_20240627" / "bird_table_row_counts.json",
    },
    "eesqlbench": {
        "db_root": Path(EESQLBENCH_DB_ROOT),
        "table_row_count_cache": Path(EESQLBENCH_TABLE_ROW_COUNT_CACHE),
    },
}
_ACTIVE_BENCHMARK = "bird"


def list_supported_benchmarks() -> list[str]:
    return sorted(_BENCHMARK_PROFILES)


def set_active_benchmark(benchmark_name: str) -> None:
    normalized_name = str(benchmark_name).strip().lower()
    if normalized_name not in _BENCHMARK_PROFILES:
        supported = ", ".join(list_supported_benchmarks())
        raise ValueError(
            f"Unsupported benchmark '{benchmark_name}'. Supported benchmarks: {supported}"
        )
    global _ACTIVE_BENCHMARK
    _ACTIVE_BENCHMARK = normalized_name
    try:
        from utils.bird_table_stats_cache import clear_bird_table_row_count_cache

        clear_bird_table_row_count_cache()
    except Exception:
        pass


def get_active_benchmark() -> str:
    return _ACTIVE_BENCHMARK


def get_active_db_root() -> Path:
    return _BENCHMARK_PROFILES[_ACTIVE_BENCHMARK]["db_root"]


def get_active_table_row_count_cache_path() -> Path:
    return _BENCHMARK_PROFILES[_ACTIVE_BENCHMARK]["table_row_count_cache"]


def register_database_path(db_name: str, db_path: str | Path) -> None:
    """Register a SQLite database path for a non-BIRD task."""
    if not isinstance(db_name, str) or not db_name.strip():
        raise ValueError("db_name must be a non-empty string.")

    resolved_path = Path(db_path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"SQLite database does not exist: {resolved_path}")

    _CUSTOM_DATABASE_PATHS[db_name.strip()] = resolved_path


def get_bird_db_path(db_name: str) -> Path:
    """Return the SQLite file path for the active benchmark database."""
    if not isinstance(db_name, str) or not db_name.strip():
        raise ValueError("db_name must be a non-empty string.")

    normalized_name = db_name.strip()
    custom_path = _CUSTOM_DATABASE_PATHS.get(normalized_name)
    if custom_path is not None:
        return custom_path

    db_path = (
        get_active_db_root()
        / normalized_name
        / f"{normalized_name}.sqlite"
    )

    if not db_path.is_file():
        raise FileNotFoundError(
            f"SQLite database '{normalized_name}' does not exist for benchmark "
            f"'{get_active_benchmark()}': {db_path}"
        )

    return db_path


def connect_bird_database(db_name: str) -> sqlite3.Connection:
    """Create and return a SQLite connection for the active benchmark database."""
    db_path = get_bird_db_path(db_name)
    return sqlite3.connect(db_path)


def get_bird_cursor(db_name: str) -> sqlite3.Cursor:
    """Create and return a cursor for the given BIRD database."""
    conn = connect_bird_database(db_name)
    return conn.cursor()
