"""Schema Filter evaluation metrics.

The metrics in this module evaluate retrieved schema columns against the
ground-truth SQL attached to a BIRD task.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from myTypes import BirdTask
from myTypes import ColumnRef
from myTypes import VerifiedContextBlueprint
from utils.schema_grounding import list_schema_columns


SchemaColumn = tuple[str, str]

_SQL_STRING_RE = re.compile(r"'(?:''|[^'])*'")
_TABLE_REF_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+"
    r"(?P<table>`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\s+(?:AS\s+)?(?P<alias>`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*))?",
    re.IGNORECASE,
)
_QUALIFIED_COLUMN_RE = re.compile(
    r"(?P<owner>`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)"
    r"\s*\.\s*"
    r"(?P<column>`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)"
)
_QUOTED_IDENTIFIER_RE = re.compile(r"(?<!\.)`([^`]+)`|(?<!\.)\"([^\"]+)\"|(?<!\.)\[([^\]]+)\]")
_BARE_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_SQL_KEYWORDS = {
    "and",
    "as",
    "asc",
    "avg",
    "between",
    "by",
    "case",
    "cast",
    "count",
    "desc",
    "distinct",
    "else",
    "end",
    "from",
    "group",
    "having",
    "in",
    "inner",
    "is",
    "join",
    "left",
    "like",
    "limit",
    "max",
    "min",
    "not",
    "null",
    "on",
    "or",
    "order",
    "outer",
    "real",
    "right",
    "select",
    "sum",
    "then",
    "where",
    "when",
}


def extract_ground_truth_schema_columns(
    task: BirdTask,
    split: str = "dev",
) -> set[SchemaColumn]:
    """Extract the ground-truth schema column set from a BIRD task SQL string.

    The returned set uses original SQL identifiers as `(table_name, column_name)`.
    It includes columns used in SELECT, WHERE, ORDER BY, GROUP BY, HAVING, and
    JOIN predicates when those columns can be resolved from metadata.
    """
    schema_columns = list_schema_columns(task.db_id, split=split)
    columns_by_table = _columns_by_table(schema_columns)
    table_lookup = {table.lower(): table for table in columns_by_table}
    sql = _strip_sql_strings(task.sql)
    alias_to_table = _extract_aliases(sql, table_lookup)
    active_tables = list(dict.fromkeys(alias_to_table.values()))
    resolved: set[SchemaColumn] = set()

    for match in _QUALIFIED_COLUMN_RE.finditer(sql):
        owner = _unquote_identifier(match.group("owner"))
        column = _unquote_identifier(match.group("column"))
        table_name = alias_to_table.get(owner.lower()) or table_lookup.get(owner.lower())
        if table_name:
            resolved_column = _resolve_column_name(columns_by_table, table_name, column)
            if resolved_column:
                resolved.add((table_name, resolved_column))

    for identifier in _extract_unqualified_identifiers(sql):
        resolved_column = _resolve_unqualified_column(columns_by_table, active_tables, identifier)
        if resolved_column:
            resolved.add(resolved_column)

    return resolved


def normalize_schema_columns(columns: Any) -> set[SchemaColumn]:
    """Normalize SchemaFilter outputs to a set of `(table_name, column_name)`."""
    if isinstance(columns, VerifiedContextBlueprint):
        return normalize_schema_columns(columns.selected_columns)

    normalized: set[SchemaColumn] = set()
    for column in columns or []:
        normalized_column = _normalize_schema_column(column)
        if normalized_column:
            normalized.add(normalized_column)
    return normalized


def calculate_fpr(retrieved_columns: Any, ground_truth_columns: set[SchemaColumn]) -> float:
    """Calculate false positive rate for one task.

    FPR = retrieved irrelevant columns / total retrieved columns.
    """
    retrieved = normalize_schema_columns(retrieved_columns)
    if not retrieved:
        return 0.0

    false_positives = retrieved - ground_truth_columns
    return len(false_positives) / len(retrieved)


def average_fpr(task_results: Iterable[tuple[Any, set[SchemaColumn]]]) -> float:
    """Calculate the mean FPR across task results."""
    fprs = [
        calculate_fpr(retrieved_columns, ground_truth_columns)
        for retrieved_columns, ground_truth_columns in task_results
    ]
    if not fprs:
        return 0.0
    return sum(fprs) / len(fprs)


def calculate_slr(task_results: Iterable[tuple[Any, set[SchemaColumn]]]) -> float:
    """Calculate SchemaLinkingRecall across tasks.

    SLR is the proportion of tasks whose retrieved schema columns fully cover
    all ground-truth schema columns required by the task SQL.
    """
    total = 0
    covered = 0
    for retrieved_columns, ground_truth_columns in task_results:
        total += 1
        retrieved = normalize_schema_columns(retrieved_columns)
        if ground_truth_columns <= retrieved:
            covered += 1

    if total == 0:
        return 0.0
    return covered / total


def _strip_sql_strings(sql: str) -> str:
    return _SQL_STRING_RE.sub("''", sql)


def _extract_aliases(sql: str, table_lookup: dict[str, str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for match in _TABLE_REF_RE.finditer(sql):
        table_name = _unquote_identifier(match.group("table"))
        resolved_table = table_lookup.get(table_name.lower())
        if not resolved_table:
            continue

        aliases[resolved_table.lower()] = resolved_table
        alias = match.group("alias")
        if alias:
            normalized_alias = _unquote_identifier(alias)
            if normalized_alias.lower() not in _SQL_KEYWORDS:
                aliases[normalized_alias.lower()] = resolved_table

    return aliases


def _extract_unqualified_identifiers(sql: str) -> list[str]:
    identifiers = []
    for match in _QUOTED_IDENTIFIER_RE.finditer(sql):
        identifiers.append(next(group for group in match.groups() if group is not None))

    for match in _BARE_IDENTIFIER_RE.finditer(_QUALIFIED_COLUMN_RE.sub(" ", sql)):
        identifier = match.group(0)
        if identifier.lower() not in _SQL_KEYWORDS:
            identifiers.append(identifier)

    return identifiers


def _columns_by_table(schema_columns: list[dict]) -> dict[str, dict[str, str]]:
    grouped: dict[str, dict[str, str]] = {}
    for column in schema_columns:
        table_name = column["table_name"]
        column_name = column["column_name"]
        grouped.setdefault(table_name, {})[column_name.lower()] = column_name
    return grouped


def _resolve_column_name(
    columns_by_table: dict[str, dict[str, str]],
    table_name: str,
    column_name: str,
) -> str | None:
    return columns_by_table.get(table_name, {}).get(column_name.lower())


def _resolve_unqualified_column(
    columns_by_table: dict[str, dict[str, str]],
    active_tables: list[str],
    column_name: str,
) -> SchemaColumn | None:
    matches = []
    for table_name in active_tables:
        resolved_column = _resolve_column_name(columns_by_table, table_name, column_name)
        if resolved_column:
            matches.append((table_name, resolved_column))

    if len(matches) == 1:
        return matches[0]
    return None


def _normalize_schema_column(column: Any) -> SchemaColumn | None:
    if isinstance(column, ColumnRef):
        return column.table_name, column.column_name

    if isinstance(column, dict):
        table_name = column.get("table_name")
        column_name = column.get("column_name")
        if table_name and column_name:
            return str(table_name), str(column_name)
        return None

    if isinstance(column, tuple) and len(column) == 2:
        return str(column[0]), str(column[1])

    if isinstance(column, str) and "." in column:
        table_name, column_name = column.split(".", 1)
        return table_name, column_name

    return None


def _unquote_identifier(identifier: str) -> str:
    identifier = identifier.strip()
    if len(identifier) >= 2 and (
        (identifier[0] == "`" and identifier[-1] == "`")
        or (identifier[0] == '"' and identifier[-1] == '"')
        or (identifier[0] == "[" and identifier[-1] == "]")
    ):
        return identifier[1:-1]
    return identifier
