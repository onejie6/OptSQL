"""Common helpers for Explain Analyser tools.

This module keeps connection handling, identifier quoting, SQL safety checks,
and DB-API row conversion out of individual tools so each tool can stay focused
on its own contract.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator

from utils.db import connect_bird_database
from utils.sql_safety import ensure_select_sql


SQLITE_PROGRESS_INTERVAL = 1000


def dialect_for_dbms(dbms: str) -> str:
    if dbms == "sqlite":
        return "sqlite"
    if dbms == "mysql":
        return "mysql"
    raise ValueError(f"Unsupported dbms: {dbms}")


def ensure_readonly_select(sql: str, dbms: str) -> None:
    """Reject multi-statement and non-read-only SQL before explain/stat calls."""
    ensure_select_sql(sql, dialect=dialect_for_dbms(dbms))


def quote_identifier(identifier: str, dbms: str) -> str:
    escaped = identifier.replace('"', '""').replace("`", "``")
    if dbms == "mysql":
        return f"`{escaped}`"
    return f'"{escaped}"'


@contextmanager
def sqlite_connection(db_id: str, timeout_ms: int | None = None) -> Iterator[sqlite3.Connection]:
    conn = connect_bird_database(db_id)
    if timeout_ms is not None:
        import time

        deadline = time.monotonic() + timeout_ms / 1000.0

        def progress_handler() -> int:
            if time.monotonic() > deadline:
                return 1
            return 0

        conn.set_progress_handler(progress_handler, SQLITE_PROGRESS_INTERVAL)
    try:
        yield conn
    finally:
        conn.set_progress_handler(None, 0)
        conn.close()


def require_mysql_connection(connection: Any | None) -> Any:
    if connection is None:
        raise ValueError(
            "A MySQL DB-API connection must be passed via the 'connection' argument."
        )
    return connection


def rows_to_dicts(cursor: Any, rows: list[Any]) -> list[dict[str, Any]]:
    columns = [desc[0] for desc in (cursor.description or [])]
    result: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            result.append(row)
        else:
            result.append({columns[idx]: value for idx, value in enumerate(row)})
    return result


def fetch_all_dicts(cursor: Any) -> list[dict[str, Any]]:
    rows = cursor.fetchall()
    return rows_to_dicts(cursor, list(rows))


def mysql_placeholders(values: list[str]) -> str:
    if not values:
        return "NULL"
    return ", ".join(["%s"] * len(values))


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
