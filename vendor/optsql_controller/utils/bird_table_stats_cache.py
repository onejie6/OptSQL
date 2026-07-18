"""Helpers for reading precomputed BIRD table row-count cache."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from utils.db import get_active_table_row_count_cache_path


DEFAULT_BIRD_TABLE_ROW_COUNT_CACHE: Path | None = None


@lru_cache(maxsize=1)
def load_bird_table_row_count_cache(
    cache_path: str | Path | None = DEFAULT_BIRD_TABLE_ROW_COUNT_CACHE,
) -> dict[str, dict[str, int]]:
    path = Path(cache_path) if cache_path is not None else get_active_table_row_count_cache_path()
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    dbs = payload.get("dbs") if isinstance(payload, dict) else None
    if not isinstance(dbs, dict):
        return {}
    result: dict[str, dict[str, int]] = {}
    for db_id, table_map in dbs.items():
        if not isinstance(db_id, str) or not isinstance(table_map, dict):
            continue
        clean_table_map: dict[str, int] = {}
        for table, row_count in table_map.items():
            if isinstance(table, str) and isinstance(row_count, int):
                clean_table_map[table] = row_count
        result[db_id] = clean_table_map
    return result


def get_cached_bird_table_row_count(
    db_id: str,
    table: str,
    *,
    cache_path: str | Path | None = DEFAULT_BIRD_TABLE_ROW_COUNT_CACHE,
) -> int | None:
    cache = load_bird_table_row_count_cache(cache_path)
    return cache.get(str(db_id), {}).get(str(table))


def get_cached_bird_db_table_row_counts(
    db_id: str,
    *,
    cache_path: str | Path | None = DEFAULT_BIRD_TABLE_ROW_COUNT_CACHE,
) -> dict[str, int]:
    cache = load_bird_table_row_count_cache(cache_path)
    row_counts = cache.get(str(db_id), {})
    return dict(row_counts) if isinstance(row_counts, dict) else {}


def clear_bird_table_row_count_cache() -> None:
    load_bird_table_row_count_cache.cache_clear()
