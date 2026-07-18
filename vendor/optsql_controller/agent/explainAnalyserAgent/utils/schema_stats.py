"""Tool: collect_schema_stats.

Collects table, column, index, and foreign-key metadata needed to interpret
PlanIR risk tags. SQLite is implemented against BIRD databases; MySQL is
implemented against information_schema through a caller-provided DB-API
connection.
"""

from __future__ import annotations

from typing import Any

from agent.explainAnalyserAgent.utils.common import fetch_all_dicts
from agent.explainAnalyserAgent.utils.common import mysql_placeholders
from agent.explainAnalyserAgent.utils.common import quote_identifier
from agent.explainAnalyserAgent.utils.common import require_mysql_connection
from agent.explainAnalyserAgent.utils.common import sqlite_connection
from agent.explainAnalyserAgent.utils.common import unique_preserve_order
from agent.explainAnalyserAgent.utils.models import CollectSchemaStatsInput
from agent.explainAnalyserAgent.utils.models import CollectSchemaStatsOutput
from agent.explainAnalyserAgent.utils.models import ColumnStats
from agent.explainAnalyserAgent.utils.models import ForeignKeyStats
from agent.explainAnalyserAgent.utils.models import IndexStats
from agent.explainAnalyserAgent.utils.models import TableStats
from utils.bird_table_stats_cache import get_cached_bird_table_row_count


def collect_schema_stats(
    input_data: CollectSchemaStatsInput,
    *,
    connection: Any | None = None,
) -> CollectSchemaStatsOutput:
    """Collect schema statistics for tables and columns referenced by a SQL query."""
    if input_data.dbms == "sqlite":
        return _collect_sqlite_schema_stats(input_data)
    if input_data.dbms == "mysql":
        return _collect_mysql_schema_stats(input_data, connection=connection)
    raise ValueError(f"Unsupported dbms: {input_data.dbms}")


def _collect_sqlite_schema_stats(
    input_data: CollectSchemaStatsInput,
) -> CollectSchemaStatsOutput:
    warnings: list[str] = []
    table_stats: dict[str, TableStats] = {}
    column_stats: dict[str, ColumnStats] = {}
    indexes: dict[str, list[IndexStats]] = {}
    foreign_keys: list[ForeignKeyStats] = []
    requested_columns = _columns_by_table(input_data.columns)

    try:
        with sqlite_connection(input_data.db_id, timeout_ms=input_data.timeout_ms) as conn:
            for table in input_data.tables:
                quoted_table = quote_identifier(table, "sqlite")
                try:
                    table_info = conn.execute(f"PRAGMA table_info({quoted_table})").fetchall()
                except Exception as exc:
                    warnings.append(f"Failed to read SQLite table_info for {table}: {exc}")
                    continue

                primary_key = [row[1] for row in table_info if row[5]]
                row_count, row_count_kind = _sqlite_count_rows(
                    conn,
                    input_data.db_id,
                    table,
                    warnings,
                )
                table_stats[table] = TableStats(
                    table=table,
                    row_count=row_count,
                    row_count_kind=row_count_kind,
                    primary_key=primary_key,
                )

                allowed_columns = requested_columns.get(table)
                for row in table_info:
                    column = row[1]
                    if allowed_columns is not None and column not in allowed_columns:
                        continue
                    sample_values = (
                        _sqlite_sample_values(conn, table, column, warnings)
                        if input_data.include_samples
                        else None
                    )
                    column_stats[f"{table}.{column}"] = ColumnStats(
                        table=table,
                        column=column,
                        data_type=row[2],
                        nullable=not bool(row[3]),
                        approx_distinct=None,
                        sample_values=sample_values,
                    )

                indexes[table] = _sqlite_indexes(conn, table, warnings)
                foreign_keys.extend(_sqlite_foreign_keys(conn, table, warnings))
    except Exception as exc:
        warnings.append(f"Failed to collect SQLite schema stats: {exc}")

    return CollectSchemaStatsOutput(
        tables=table_stats,
        columns=column_stats,
        indexes=indexes,
        foreign_keys=foreign_keys,
        warnings=warnings,
    )


def _sqlite_count_rows(
    conn: Any,
    db_id: str,
    table: str,
    warnings: list[str],
) -> tuple[int | None, str]:
    cached = get_cached_bird_table_row_count(db_id, table)
    if cached is not None:
        return cached, "exact"
    warnings.append(
        f"Missing precomputed row-count cache for SQLite table {db_id}.{table}; "
        "row_count left unknown."
    )
    return None, "unknown"


def _sqlite_sample_values(
    conn: Any,
    table: str,
    column: str,
    warnings: list[str],
    limit: int = 5,
) -> list[Any] | None:
    try:
        quoted_table = quote_identifier(table, "sqlite")
        quoted_column = quote_identifier(column, "sqlite")
        rows = conn.execute(
            f"SELECT DISTINCT {quoted_column} FROM {quoted_table} "
            f"WHERE {quoted_column} IS NOT NULL LIMIT {int(limit)}"
        ).fetchall()
        return [row[0] for row in rows]
    except Exception as exc:
        warnings.append(f"Failed to sample SQLite column {table}.{column}: {exc}")
        return None


def _sqlite_indexes(conn: Any, table: str, warnings: list[str]) -> list[IndexStats]:
    result: list[IndexStats] = []
    try:
        quoted_table = quote_identifier(table, "sqlite")
        index_rows = conn.execute(f"PRAGMA index_list({quoted_table})").fetchall()
    except Exception as exc:
        warnings.append(f"Failed to read SQLite indexes for {table}: {exc}")
        return result

    for row in index_rows:
        index_name = row[1]
        unique = bool(row[2])
        origin = row[3] if len(row) > 3 else None
        try:
            quoted_index = quote_identifier(index_name, "sqlite")
            column_rows = conn.execute(f"PRAGMA index_info({quoted_index})").fetchall()
            columns = [col_row[2] for col_row in column_rows]
        except Exception as exc:
            warnings.append(f"Failed to read SQLite index_info for {index_name}: {exc}")
            columns = []
        result.append(
            IndexStats(
                table=table,
                index_name=index_name,
                columns=columns,
                unique=unique,
                origin=origin,
            )
        )
    return result


def _sqlite_foreign_keys(conn: Any, table: str, warnings: list[str]) -> list[ForeignKeyStats]:
    result: list[ForeignKeyStats] = []
    try:
        quoted_table = quote_identifier(table, "sqlite")
        rows = conn.execute(f"PRAGMA foreign_key_list({quoted_table})").fetchall()
    except Exception as exc:
        warnings.append(f"Failed to read SQLite foreign keys for {table}: {exc}")
        return result

    grouped: dict[tuple[int, str], dict[str, list[str] | str]] = {}
    for row in rows:
        fk_id = row[0]
        target_table = row[2]
        grouped.setdefault(
            (fk_id, target_table),
            {
                "source_columns": [],
                "target_columns": [],
                "target_table": target_table,
            },
        )
        grouped[(fk_id, target_table)]["source_columns"].append(row[3])
        grouped[(fk_id, target_table)]["target_columns"].append(row[4])

    for value in grouped.values():
        result.append(
            ForeignKeyStats(
                source_table=table,
                source_columns=list(value["source_columns"]),
                target_table=str(value["target_table"]),
                target_columns=list(value["target_columns"]),
            )
        )
    return result


def _collect_mysql_schema_stats(
    input_data: CollectSchemaStatsInput,
    *,
    connection: Any | None,
) -> CollectSchemaStatsOutput:
    warnings: list[str] = []
    try:
        conn = require_mysql_connection(connection)
    except Exception as exc:
        return CollectSchemaStatsOutput(warnings=[str(exc)])

    tables = unique_preserve_order(input_data.tables)
    if not tables:
        return CollectSchemaStatsOutput()

    table_stats = _mysql_table_stats(conn, tables, warnings)
    column_stats = _mysql_column_stats(
        conn,
        tables,
        input_data.columns,
        input_data.include_samples,
        warnings,
    )
    indexes = _mysql_indexes(conn, tables, warnings)
    foreign_keys = _mysql_foreign_keys(conn, tables, warnings)
    return CollectSchemaStatsOutput(
        tables=table_stats,
        columns=column_stats,
        indexes=indexes,
        foreign_keys=foreign_keys,
        warnings=warnings,
    )


def _mysql_table_stats(
    conn: Any,
    tables: list[str],
    warnings: list[str],
) -> dict[str, TableStats]:
    result: dict[str, TableStats] = {}
    placeholders = mysql_placeholders(tables)
    sql = (
        "SELECT table_name, table_rows "
        "FROM information_schema.tables "
        "WHERE table_schema = DATABASE() AND table_name IN (" + placeholders + ")"
    )
    try:
        cursor = conn.cursor()
        try:
            cursor.execute(sql, tuple(tables))
            rows = fetch_all_dicts(cursor)
        finally:
            _close_cursor(cursor)
        for row in rows:
            table = row.get("TABLE_NAME") or row.get("table_name")
            table_rows = row.get("TABLE_ROWS") or row.get("table_rows")
            if table:
                result[str(table)] = TableStats(
                    table=str(table),
                    row_count=_safe_int(table_rows),
                    row_count_kind="estimated",
                    primary_key=[],
                )
    except Exception as exc:
        warnings.append(f"Failed to read MySQL table stats: {exc}")
    return result


def _mysql_column_stats(
    conn: Any,
    tables: list[str],
    requested_columns: list[str],
    include_samples: bool,
    warnings: list[str],
) -> dict[str, ColumnStats]:
    result: dict[str, ColumnStats] = {}
    requested_by_table = _columns_by_table(requested_columns)
    placeholders = mysql_placeholders(tables)
    sql = (
        "SELECT table_name, column_name, data_type, is_nullable, column_key "
        "FROM information_schema.columns "
        "WHERE table_schema = DATABASE() AND table_name IN (" + placeholders + ")"
    )
    try:
        cursor = conn.cursor()
        try:
            cursor.execute(sql, tuple(tables))
            rows = fetch_all_dicts(cursor)
        finally:
            _close_cursor(cursor)
        for row in rows:
            table = str(row.get("TABLE_NAME") or row.get("table_name"))
            column = str(row.get("COLUMN_NAME") or row.get("column_name"))
            allowed_columns = requested_by_table.get(table)
            if allowed_columns is not None and column not in allowed_columns:
                continue
            samples = (
                _mysql_sample_values(conn, table, column, warnings)
                if include_samples
                else None
            )
            result[f"{table}.{column}"] = ColumnStats(
                table=table,
                column=column,
                data_type=row.get("DATA_TYPE") or row.get("data_type"),
                nullable=(
                    str(row.get("IS_NULLABLE") or row.get("is_nullable")).upper()
                    == "YES"
                ),
                approx_distinct=None,
                sample_values=samples,
            )
    except Exception as exc:
        warnings.append(f"Failed to read MySQL column stats: {exc}")
    return result


def _mysql_indexes(
    conn: Any,
    tables: list[str],
    warnings: list[str],
) -> dict[str, list[IndexStats]]:
    result: dict[str, list[IndexStats]] = {table: [] for table in tables}
    placeholders = mysql_placeholders(tables)
    sql = (
        "SELECT table_name, index_name, column_name, seq_in_index, non_unique "
        "FROM information_schema.statistics "
        "WHERE table_schema = DATABASE() AND table_name IN (" + placeholders + ") "
        "ORDER BY table_name, index_name, seq_in_index"
    )
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        cursor = conn.cursor()
        try:
            cursor.execute(sql, tuple(tables))
            rows = fetch_all_dicts(cursor)
        finally:
            _close_cursor(cursor)
        for row in rows:
            table = str(row.get("TABLE_NAME") or row.get("table_name"))
            index_name = str(row.get("INDEX_NAME") or row.get("index_name"))
            column = str(row.get("COLUMN_NAME") or row.get("column_name"))
            non_unique = (
                row.get("NON_UNIQUE") if "NON_UNIQUE" in row else row.get("non_unique")
            )
            grouped.setdefault(
                (table, index_name),
                {"columns": [], "unique": _safe_int(non_unique) == 0},
            )
            grouped[(table, index_name)]["columns"].append(column)
        for (table, index_name), payload in grouped.items():
            result.setdefault(table, []).append(
                IndexStats(
                    table=table,
                    index_name=index_name,
                    columns=payload["columns"],
                    unique=bool(payload["unique"]),
                    origin=None,
                )
            )
    except Exception as exc:
        warnings.append(f"Failed to read MySQL indexes: {exc}")
    return result


def _mysql_foreign_keys(
    conn: Any,
    tables: list[str],
    warnings: list[str],
) -> list[ForeignKeyStats]:
    result: list[ForeignKeyStats] = []
    placeholders = mysql_placeholders(tables)
    sql = (
        "SELECT table_name, column_name, referenced_table_name, referenced_column_name, "
        "constraint_name, ordinal_position "
        "FROM information_schema.key_column_usage "
        "WHERE table_schema = DATABASE() "
        "AND referenced_table_name IS NOT NULL "
        "AND table_name IN (" + placeholders + ") "
        "ORDER BY table_name, constraint_name, ordinal_position"
    )
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    try:
        cursor = conn.cursor()
        try:
            cursor.execute(sql, tuple(tables))
            rows = fetch_all_dicts(cursor)
        finally:
            _close_cursor(cursor)
        for row in rows:
            source_table = str(row.get("TABLE_NAME") or row.get("table_name"))
            target_table = str(
                row.get("REFERENCED_TABLE_NAME") or row.get("referenced_table_name")
            )
            constraint_name = str(row.get("CONSTRAINT_NAME") or row.get("constraint_name"))
            key = (source_table, target_table, constraint_name)
            grouped.setdefault(
                key,
                {
                    "source_columns": [],
                    "target_columns": [],
                    "source_table": source_table,
                    "target_table": target_table,
                },
            )
            grouped[key]["source_columns"].append(
                str(row.get("COLUMN_NAME") or row.get("column_name"))
            )
            grouped[key]["target_columns"].append(
                str(row.get("REFERENCED_COLUMN_NAME") or row.get("referenced_column_name"))
            )
        for payload in grouped.values():
            result.append(
                ForeignKeyStats(
                    source_table=payload["source_table"],
                    source_columns=payload["source_columns"],
                    target_table=payload["target_table"],
                    target_columns=payload["target_columns"],
                )
            )
    except Exception as exc:
        warnings.append(f"Failed to read MySQL foreign keys: {exc}")
    return result


def _mysql_sample_values(
    conn: Any,
    table: str,
    column: str,
    warnings: list[str],
    limit: int = 5,
) -> list[Any] | None:
    try:
        sql = (
            f"SELECT DISTINCT {quote_identifier(column, 'mysql')} "
            f"FROM {quote_identifier(table, 'mysql')} "
            f"WHERE {quote_identifier(column, 'mysql')} IS NOT NULL "
            f"LIMIT {int(limit)}"
        )
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
        finally:
            _close_cursor(cursor)
        return [
            row[0] if not isinstance(row, dict) else next(iter(row.values()))
            for row in rows
        ]
    except Exception as exc:
        warnings.append(f"Failed to sample MySQL column {table}.{column}: {exc}")
        return None


def _columns_by_table(columns: list[str]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for column_ref in columns:
        if "." not in column_ref:
            continue
        table, column = column_ref.split(".", 1)
        result.setdefault(table, set()).add(column)
    return result


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _close_cursor(cursor: Any) -> None:
    try:
        cursor.close()
    except Exception:
        pass
