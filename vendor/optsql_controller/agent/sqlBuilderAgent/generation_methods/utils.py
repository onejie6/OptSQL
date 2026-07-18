"""Shared helpers for SQL generation methods."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from myTypes import VerifiedContextBlueprint


def parse_xml_result(response: str) -> str | None:
    """Extract SQL from an XML `<result>` block."""
    answer_match = re.search(r"<result>(.*?)</result>", response or "", re.DOTALL)
    content = answer_match.group(1).strip() if answer_match else (response or "").strip()
    if content.startswith("```sql") and content.endswith("```"):
        content = content[len("```sql") : -len("```")].strip()
    elif content.startswith("```") and content.endswith("```"):
        content = content[len("```") : -len("```")].strip()
    return content or None


def normalize_sql(sql: str) -> str:
    return " ".join((sql or "").split()).strip().lower()


def hash_rows(rows: list[Any]) -> str:
    normalized_rows = [tuple(row) if not isinstance(row, tuple) else row for row in rows]
    payload = repr(sorted(normalized_rows, key=repr)).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


def rows_to_table_string(rows: list[Any], max_rows: int = 20) -> str:
    if not rows:
        return "<empty result>"
    return "\n".join(repr(tuple(row)) for row in rows[:max_rows])


def build_blueprint_schema_profile(blueprint: VerifiedContextBlueprint) -> str:
    tables: dict[str, list[str]] = {table: [] for table in blueprint.selected_tables}
    comments: dict[tuple[str, str], str | None] = {}
    types: dict[tuple[str, str], str | None] = {}
    for column in blueprint.selected_columns:
        tables.setdefault(column.table_name, []).append(column.column_name)
        comments[(column.table_name, column.column_name)] = column.comment
        types[(column.table_name, column.column_name)] = column.data_type

    lines: list[str] = []
    for table_name in sorted(tables):
        lines.append(f'Table "{table_name}"')
        for column_name in sorted(tables[table_name]):
            data_type = types.get((table_name, column_name)) or "unknown"
            comment = comments.get((table_name, column_name)) or ""
            lines.append(f'  - "{column_name}" ({data_type}): {comment}')
        lines.append("")

    if blueprint.value_mappings:
        lines.append("Verified value mappings:")
        for mapping in blueprint.value_mappings:
            lines.append(
                f'  - {mapping.keyword}: "{mapping.table_name}"."{mapping.column_name}" = {mapping.value!r}'
            )
        lines.append("")

    if blueprint.join_topology.edges:
        lines.append("Allowed joins:")
        for edge in blueprint.join_topology.edges:
            lines.append(
                f'  - {edge.join_type}: "{edge.source_table}"."{edge.source_column}" '
                f'= "{edge.target_table}"."{edge.target_column}"'
            )
        lines.append("")

    if blueprint.predicate_hints:
        lines.append("Predicate hints:")
        for hint in blueprint.predicate_hints:
            lines.append(f"  - {hint.predicate_type}: {hint.expression} ({hint.source_text})")

    return "\n".join(lines).strip()
