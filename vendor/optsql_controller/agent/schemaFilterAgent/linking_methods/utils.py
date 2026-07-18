"""Utilities for embedded schema linking contracts."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import sqlglot
import sqlglot.expressions as exp


def merge_schema_linking_results(
    results: list[dict[str, list[str]] | None],
) -> dict[str, list[str]]:
    """Merge multiple table -> columns selections, skipping failed linkers."""
    merged: dict[str, set[str]] = {}
    for result in results:
        if not result:
            continue
        for table_name, columns in result.items():
            merged.setdefault(table_name, set()).update(columns)
    return {
        table_name: sorted(columns)
        for table_name, columns in sorted(merged.items())
    }


def build_schema_profile(schema_columns: list[dict[str, Any]]) -> str:
    """Render a compact schema profile for LLM prompts."""
    tables: dict[str, list[dict[str, Any]]] = defaultdict(list)
    table_comments: dict[str, str] = {}
    for column in schema_columns:
        table_name = str(column["table_name"])
        tables[table_name].append(column)
        table_comments.setdefault(table_name, str(column.get("table_comment") or table_name))

    lines: list[str] = []
    for table_name in sorted(tables):
        table_comment = table_comments.get(table_name) or table_name
        lines.append(f'Table "{table_name}" ({table_comment})')
        for column in sorted(tables[table_name], key=lambda item: item["column_name"]):
            flags = []
            if column.get("is_primary_key"):
                flags.append("primary key")
            if column.get("is_foreign_key"):
                flags.append("foreign key")
            flag_text = f" [{' ; '.join(flags)}]" if flags else ""
            description = (
                column.get("column_comment")
                or column.get("semantic_column_name")
                or column.get("column_description")
                or ""
            )
            data_type = column.get("data_type") or "unknown"
            lines.append(
                f'  - "{column["column_name"]}" ({data_type}){flag_text}: {description}'
            )
            value_description = column.get("value_description")
            if value_description:
                lines.append(f"    Value notes: {value_description}")
        lines.append("")
    return "\n".join(lines).strip()


def parse_direct_linking_response(
    response: str,
    schema_columns: list[dict[str, Any]],
) -> dict[str, list[str]] | None:
    """Parse XML table/column selection output."""
    answer_match = re.search(r"<result>(.*?)</result>", response or "", re.DOTALL)
    content = answer_match.group(1).strip() if answer_match else response or ""

    known_tables = _known_tables(schema_columns)
    known_columns = _known_columns(schema_columns)
    result: dict[str, list[str]] = {}
    table_matches = re.findall(
        r'<table\s+table_name\s*=\s*["\']([^"\']+)["\'][^>]*>(.*?)</table>',
        content,
        re.DOTALL,
    )
    for raw_table_name, table_content in table_matches:
        table_name = _resolve_table(raw_table_name, known_tables)
        if table_name is None:
            continue
        columns: list[str] = []
        column_matches = re.findall(
            r'<column\s+column_name\s*=\s*["\']([^"\']+)["\']\s*/?>',
            table_content,
        )
        for raw_column_name in column_matches:
            column_name = _resolve_column(table_name, raw_column_name, known_columns)
            if column_name is not None and column_name not in columns:
                columns.append(column_name)
        result.setdefault(table_name, [])
        result[table_name].extend(column for column in columns if column not in result[table_name])

    return result or None


def parse_sql_linking_response(
    response: str,
    schema_columns: list[dict[str, Any]],
) -> dict[str, list[str]] | None:
    """Parse a generated SQL candidate and extract referenced schema elements."""
    sql = _extract_result_payload(response)
    if not sql:
        return None

    known_tables = _known_tables(schema_columns)
    known_columns = _known_columns(schema_columns)
    result: dict[str, list[str]] = {}
    alias_map: dict[str, str] = {}
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        ast = None

    if ast is not None:
        for table in ast.find_all(exp.Table):
            table_name = _resolve_table(table.name, known_tables)
            if table_name is None:
                continue
            result.setdefault(table_name, [])
            alias_map[table.name] = table_name
            if table.alias:
                alias_map[table.alias] = table_name

        for column in ast.find_all(exp.Column):
            raw_column = column.name
            raw_table = column.table
            table_candidates: list[str]
            if raw_table:
                resolved_table = alias_map.get(raw_table) or _resolve_table(raw_table, known_tables)
                table_candidates = [resolved_table] if resolved_table else []
            else:
                table_candidates = list(result) or sorted(known_tables.values())
            for table_name in table_candidates:
                column_name = _resolve_column(table_name, raw_column, known_columns)
                if column_name is None:
                    continue
                result.setdefault(table_name, [])
                if column_name not in result[table_name]:
                    result[table_name].append(column_name)
                break

    if result:
        return result

    # Fallback to broad containment when SQL parsing fails.
    lowered_sql = sql.lower()
    for table_key, table_name in known_tables.items():
        if table_key not in lowered_sql:
            continue
        matched_columns = [
            column_name
            for column_key, column_name in known_columns.get(table_name, {}).items()
            if column_key in lowered_sql
        ]
        result[table_name] = matched_columns
    return result or None


def _extract_result_payload(response: str) -> str:
    answer_match = re.search(r"<result>(.*?)</result>", response or "", re.DOTALL)
    text = answer_match.group(1).strip() if answer_match else (response or "").strip()
    if text.startswith("```sql") and text.endswith("```"):
        text = text[len("```sql") : -len("```")].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[len("```") : -len("```")].strip()
    return text


def _known_tables(schema_columns: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(column["table_name"]).lower(): str(column["table_name"])
        for column in schema_columns
    }


def _known_columns(schema_columns: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    columns: dict[str, dict[str, str]] = defaultdict(dict)
    for column in schema_columns:
        table_name = str(column["table_name"])
        column_name = str(column["column_name"])
        columns[table_name][column_name.lower()] = column_name
    return columns


def _resolve_table(raw_table_name: str, known_tables: dict[str, str]) -> str | None:
    normalized = str(raw_table_name or "").strip().strip('"`[]').lower()
    if normalized in known_tables:
        return known_tables[normalized]
    if "." in normalized:
        tail = normalized.split(".")[-1]
        return known_tables.get(tail)
    return None


def _resolve_column(
    table_name: str,
    raw_column_name: str,
    known_columns: dict[str, dict[str, str]],
) -> str | None:
    normalized = str(raw_column_name or "").strip().strip('"`[]').lower()
    return known_columns.get(table_name, {}).get(normalized)
