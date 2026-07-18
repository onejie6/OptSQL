"""Schema and value grounding utilities for BIRD SQLite databases."""

import csv
import io
import json
import re
import sqlite3
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path

from config import BIRD_BASE
from utils.db import connect_bird_database


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_SPLIT_DIR_MAP = {
    "dev": "dev_20240627",
    "train": "train_20240627",
}
_CUSTOM_DATABASE_METADATA: dict[str, dict] = {}
_CUSTOM_DESCRIPTION_METADATA: dict[str, dict[str, dict[str, dict]]] = {}
_SQL_COMMENT_RE = re.compile(r"/\*(.*?)\*/|--([^\n\r]*)", re.DOTALL)
_TABLE_CONSTRAINT_PREFIXES = {
    "constraint",
    "primary",
    "foreign",
    "unique",
    "check",
    "exclude",
}


def normalize_text(value: object) -> str:
    """Normalize text for lightweight schema and value matching."""
    return " ".join(_TOKEN_RE.findall(str(value).lower()))


def tokenize(value: object) -> set[str]:
    """Return normalized alphanumeric tokens."""
    return set(normalize_text(value).split())


def text_similarity(left: object, right: object) -> float:
    """Score lexical similarity with token overlap and sequence matching."""
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0

    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    overlap = 0.0
    if left_tokens and right_tokens:
        overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    sequence = SequenceMatcher(None, left_norm, right_norm).ratio()
    return max(overlap, sequence)


def quote_identifier(identifier: str) -> str:
    """Quote a SQLite identifier."""
    return '"' + identifier.replace('"', '""') + '"'


def _sqlite_type_to_bird_type(raw_type: object) -> str:
    normalized_type = str(raw_type or "").strip().lower()
    if "int" in normalized_type:
        return "integer"
    if any(token in normalized_type for token in ("real", "floa", "doub", "dec", "num")):
        return "real"
    if "bool" in normalized_type:
        return "boolean"
    if "date" in normalized_type or "time" in normalized_type:
        return "date"
    return "text"


def _semantic_label(name: str) -> str:
    label = re.sub(r"[_\-]+", " ", str(name)).strip()
    label = re.sub(r"\s+", " ", label)
    return label or str(name)


def _table_comment_from_ddl(ddl: str) -> str | None:
    open_paren_index = ddl.find("(")
    if open_paren_index < 0:
        return None
    comments = _extract_sql_comments(ddl[:open_paren_index])
    return comments[-1] if comments else None


def _column_comments_from_ddl(ddl: str) -> dict[str, str]:
    body = _ddl_parenthesized_body(ddl)
    if not body:
        return {}

    comments: dict[str, str] = {}
    for item in _split_sql_list(body):
        item_comments = _extract_sql_comments(item)
        if not item_comments:
            continue
        uncommented = _strip_sql_comments(item).strip()
        if not uncommented:
            continue
        column_name = _first_sql_identifier(uncommented)
        if column_name is None or column_name.lower() in _TABLE_CONSTRAINT_PREFIXES:
            continue
        comment = " ".join(item_comments).strip()
        if comment:
            comments[column_name] = comment
            comments[column_name.lower()] = comment
    return comments


def _ddl_parenthesized_body(ddl: str) -> str:
    start = ddl.find("(")
    end = ddl.rfind(")")
    if start < 0 or end <= start:
        return ""
    return ddl[start + 1 : end]


def _split_sql_list(value: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    index = 0
    while index < len(value):
        char = value[index]
        if quote:
            current.append(char)
            if char == quote:
                if index + 1 < len(value) and value[index + 1] == quote:
                    current.append(value[index + 1])
                    index += 1
                else:
                    quote = None
            index += 1
            continue
        if char in {"'", '"', "`"}:
            quote = char
            current.append(char)
        elif char == "[":
            quote = "]"
            current.append(char)
        elif char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth = max(0, depth - 1)
            current.append(char)
        elif char == "," and depth == 0:
            items.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        index += 1
    tail = "".join(current).strip()
    if tail:
        items.append(tail)
    return items


def _extract_sql_comments(sql: str) -> list[str]:
    comments: list[str] = []
    for match in _SQL_COMMENT_RE.finditer(sql):
        comment = match.group(1) if match.group(1) is not None else match.group(2)
        cleaned = " ".join(str(comment or "").strip().split())
        if cleaned:
            comments.append(cleaned)
    return comments


def _strip_sql_comments(sql: str) -> str:
    return _SQL_COMMENT_RE.sub(" ", sql)


def _first_sql_identifier(sql: str) -> str | None:
    stripped = sql.strip()
    if not stripped:
        return None
    if stripped[0] in {'"', "`", "["}:
        close_char = "]" if stripped[0] == "[" else stripped[0]
        escaped = stripped[0] != "["
        chars: list[str] = []
        index = 1
        while index < len(stripped):
            char = stripped[index]
            if char == close_char:
                if escaped and index + 1 < len(stripped) and stripped[index + 1] == close_char:
                    chars.append(close_char)
                    index += 2
                    continue
                return "".join(chars)
            chars.append(char)
            index += 1
        return None
    return stripped.split(maxsplit=1)[0]


def _first_primary_key_index(
    primary_keys: list[int | list[int]],
    target_table: str,
    column_names_original: list[list[int | str]],
    table_names_original: list[str],
) -> int | None:
    try:
        target_table_index = table_names_original.index(target_table)
    except ValueError:
        return None
    flattened_primary_keys = [
        column_index
        for primary_key in primary_keys
        for column_index in (primary_key if isinstance(primary_key, list) else [primary_key])
    ]
    for column_index in flattened_primary_keys:
        table_index, _ = column_names_original[column_index]
        if table_index == target_table_index:
            return column_index
    return None


def normalize_split(split: str) -> str:
    """Validate and normalize a BIRD split name."""
    normalized_split = split.strip().lower()
    if normalized_split not in _SPLIT_DIR_MAP:
        raise ValueError("split must be one of: dev, train.")
    return normalized_split


def register_database_metadata(
    db_id: str,
    metadata: dict,
    description_metadata: dict[str, dict[str, dict]] | None = None,
) -> None:
    """Register schema metadata for a non-BIRD database."""
    if not db_id or not db_id.strip():
        raise ValueError("db_id must be a non-empty string.")
    normalized_db_id = db_id.strip()
    _CUSTOM_DATABASE_METADATA[normalized_db_id] = metadata
    _CUSTOM_DESCRIPTION_METADATA[normalized_db_id] = description_metadata or {}


def build_sqlite_metadata_from_ddl(
    db_id: str,
    db_path: str | Path,
) -> tuple[dict, dict[str, dict[str, dict]]]:
    """Build BIRD-style metadata by inspecting SQLite DDL.

    SQLite does not have a native COMMENT clause, but comments embedded in the
    CREATE TABLE DDL are preserved in sqlite_master.sql. When present, these
    comments are used as table/column descriptions.
    """
    if not db_id or not db_id.strip():
        raise ValueError("db_id must be a non-empty string.")

    db_path = Path(db_path).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"SQLite database does not exist: {db_path}")

    table_names_original: list[str] = []
    table_names: list[str] = []
    column_names_original: list[list[int | str]] = [[-1, "*"]]
    column_names: list[list[int | str]] = [[-1, "*"]]
    column_types: list[str] = ["text"]
    primary_keys: list[int | list[int]] = []
    foreign_key_refs: list[tuple[str, str, str, str | None]] = []
    description_metadata: dict[str, dict[str, dict]] = {}

    with sqlite3.connect(db_path) as conn:
        table_rows = conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()

        column_index_by_table_column: dict[tuple[str, str], int] = {}
        for table_index, (table_name, ddl) in enumerate(table_rows):
            ddl = ddl or ""
            table_names_original.append(table_name)
            table_comment = _table_comment_from_ddl(ddl) or _semantic_label(table_name)
            table_names.append(table_comment)

            column_comments = _column_comments_from_ddl(ddl)
            table_pk_indexes: list[tuple[int, int]] = []
            table_descriptions: dict[str, dict] = {}
            pragma_rows = conn.execute(
                f"PRAGMA table_info({quote_identifier(table_name)})"
            ).fetchall()

            for pragma_row in pragma_rows:
                _, column_name, raw_type, _, _, pk_position = pragma_row
                column_index = len(column_names_original)
                column_index_by_table_column[(table_name, column_name)] = column_index
                column_names_original.append([table_index, column_name])

                comment = (
                    column_comments.get(column_name)
                    or column_comments.get(column_name.lower())
                    or _semantic_label(column_name)
                )
                column_names.append([table_index, comment])
                column_types.append(_sqlite_type_to_bird_type(raw_type))
                if pk_position:
                    table_pk_indexes.append((int(pk_position), column_index))
                table_descriptions[column_name] = {
                    "original_column_name": column_name,
                    "semantic_column_name": comment,
                    "column_description": (
                        comment if comment != _semantic_label(column_name) else None
                    ),
                    "data_format": None,
                    "value_description": None,
                }

            if table_pk_indexes:
                pk_indexes = [
                    column_index
                    for _, column_index in sorted(table_pk_indexes, key=lambda item: item[0])
                ]
                primary_keys.append(pk_indexes[0] if len(pk_indexes) == 1 else pk_indexes)
            description_metadata[table_name] = table_descriptions

            fk_rows = conn.execute(
                f"PRAGMA foreign_key_list({quote_identifier(table_name)})"
            ).fetchall()
            for fk_row in fk_rows:
                # id, seq, table, from, to, on_update, on_delete, match
                foreign_key_refs.append((table_name, fk_row[3], fk_row[2], fk_row[4]))

    foreign_keys: list[list[int]] = []
    for source_table, source_column, target_table, target_column in foreign_key_refs:
        source_index = column_index_by_table_column.get((source_table, source_column))
        if source_index is None:
            continue
        if target_column is None:
            target_index = _first_primary_key_index(
                primary_keys,
                target_table,
                column_names_original,
                table_names_original,
            )
        else:
            target_index = column_index_by_table_column.get((target_table, target_column))
        if target_index is None:
            continue
        foreign_keys.append([source_index, target_index])

    metadata = {
        "db_id": db_id.strip(),
        "table_names_original": table_names_original,
        "table_names": table_names,
        "column_names_original": column_names_original,
        "column_names": column_names,
        "column_types": column_types,
        "primary_keys": primary_keys,
        "foreign_keys": foreign_keys,
    }
    return metadata, description_metadata


def get_bird_tables_path(split: str = "dev") -> Path:
    """Return the BIRD tables metadata path for a split."""
    normalized_split = normalize_split(split)

    tables_path = Path(BIRD_BASE) / _SPLIT_DIR_MAP[normalized_split] / f"{normalized_split}_tables.json"
    if not tables_path.is_file():
        raise FileNotFoundError(f"BIRD tables file does not exist: {tables_path}")

    return tables_path


def get_bird_database_description_dir(db_id: str, split: str = "dev") -> Path:
    """Return the BIRD per-table database description directory."""
    normalized_split = normalize_split(split)
    return (
        Path(BIRD_BASE)
        / _SPLIT_DIR_MAP[normalized_split]
        / f"{normalized_split}_databases"
        / db_id
        / "database_description"
    )


def load_table_description_metadata(
    db_id: str,
    table_name: str,
    split: str = "dev",
) -> dict[str, dict]:
    """Load BIRD column descriptions for one table keyed by original column name."""
    description_path = get_bird_database_description_dir(db_id, split=split) / f"{table_name}.csv"
    if not description_path.is_file():
        return {}

    descriptions: dict[str, dict] = {}
    try:
        text = description_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = description_path.read_text(encoding="cp1252")

    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        original_column_name = (row.get("original_column_name") or "").strip()
        if not original_column_name:
            continue

        semantic_column_name = (row.get("column_name") or "").strip() or original_column_name
        descriptions[original_column_name] = {
            "original_column_name": original_column_name,
            "semantic_column_name": semantic_column_name,
            "column_description": (row.get("column_description") or "").strip() or None,
            "data_format": (row.get("data_format") or "").strip() or None,
            "value_description": (row.get("value_description") or "").strip() or None,
        }

    return descriptions


def load_database_description_metadata(db_id: str, split: str = "dev") -> dict[str, dict[str, dict]]:
    """Load all BIRD table description CSV files for one database."""
    if db_id in _CUSTOM_DESCRIPTION_METADATA:
        return _CUSTOM_DESCRIPTION_METADATA[db_id]

    description_dir = get_bird_database_description_dir(db_id, split=split)
    if not description_dir.is_dir():
        return {}

    metadata: dict[str, dict[str, dict]] = {}
    for description_path in sorted(description_dir.glob("*.csv")):
        table_name = description_path.stem
        metadata[table_name] = load_table_description_metadata(
            db_id,
            table_name,
            split=split,
        )

    return metadata


def load_bird_tables_metadata(split: str = "dev") -> list[dict]:
    """Load BIRD table metadata."""
    tables_path = get_bird_tables_path(split)
    with tables_path.open("r", encoding="utf-8") as file:
        raw_metadata = json.load(file)

    if not isinstance(raw_metadata, list):
        raise ValueError(f"BIRD tables file must contain a list: {tables_path}")

    return raw_metadata


def get_bird_database_metadata(db_id: str, split: str = "dev") -> dict:
    """Return BIRD table metadata for one database."""
    if db_id in _CUSTOM_DATABASE_METADATA:
        return _CUSTOM_DATABASE_METADATA[db_id]

    for metadata in load_bird_tables_metadata(split=split):
        if metadata.get("db_id") == db_id:
            return metadata

    raise LookupError(f"BIRD database metadata does not exist for db_id: {db_id}")


def list_schema_columns(db_id: str, split: str = "dev") -> list[dict]:
    """List schema columns with original SQL names and semantic labels."""
    metadata = get_bird_database_metadata(db_id, split=split)
    description_metadata = load_database_description_metadata(db_id, split=split)
    table_names_original = metadata["table_names_original"]
    table_names = metadata.get("table_names", table_names_original)
    column_names_original = metadata["column_names_original"]
    column_names = metadata.get("column_names", column_names_original)
    column_types = metadata.get("column_types", [])
    pk_raw = metadata.get("primary_keys", [])
    primary_keys = {
        column_index
        for pk in pk_raw
        for column_index in (pk if isinstance(pk, list) else [pk])
    }
    foreign_key_indexes = {
        column_index
        for pair in metadata.get("foreign_keys", [])
        for column_index in pair
    }

    columns: list[dict] = []
    for column_index, (table_index, column_name) in enumerate(column_names_original):
        if table_index == -1:
            continue

        semantic_column = column_names[column_index][1]
        table_name = table_names_original[table_index]
        description = description_metadata.get(table_name, {}).get(column_name, {})
        column_description = description.get("column_description")
        semantic_column_name = description.get("semantic_column_name") or semantic_column
        columns.append(
            {
                "db_id": db_id,
                "table_index": table_index,
                "column_index": column_index,
                "table_name": table_name,
                "table_comment": table_names[table_index],
                "column_name": column_name,
                "column_comment": column_description or semantic_column_name,
                "semantic_column_name": semantic_column_name,
                "column_description": column_description,
                "data_format": description.get("data_format"),
                "value_description": description.get("value_description"),
                "data_type": column_types[column_index] if column_index < len(column_types) else None,
                "is_primary_key": column_index in primary_keys,
                "is_foreign_key": column_index in foreign_key_indexes,
            }
        )

    return columns


def list_schema_tables(db_id: str, split: str = "dev") -> list[dict]:
    """List schema tables with original SQL names and semantic labels."""
    metadata = get_bird_database_metadata(db_id, split=split)
    table_names_original = metadata["table_names_original"]
    table_names = metadata.get("table_names", table_names_original)
    return [
        {
            "db_id": db_id,
            "table_index": index,
            "table_name": table_name,
            "table_comment": table_names[index],
        }
        for index, table_name in enumerate(table_names_original)
    ]


def list_foreign_key_edges(db_id: str, split: str = "dev") -> list[dict]:
    """Return foreign key edges using original table and column names."""
    metadata = get_bird_database_metadata(db_id, split=split)
    columns_by_index = {
        column["column_index"]: column
        for column in list_schema_columns(db_id, split=split)
    }
    edges: list[dict] = []

    for source_index, target_index in metadata.get("foreign_keys", []):
        source = columns_by_index[source_index]
        target = columns_by_index[target_index]
        edges.append(
            {
                "source_table": source["table_name"],
                "source_column": source["column_name"],
                "target_table": target["table_name"],
                "target_column": target["column_name"],
                "join_type": "inner",
            }
        )

    return edges


def inspect_column_info(
    db_id: str,
    table_name: str,
    column_name: str,
    split: str = "dev",
) -> dict:
    """Return DDL and metadata for a schema column."""
    matching_columns = [
        column
        for column in list_schema_columns(db_id, split=split)
        if column["table_name"] == table_name and column["column_name"] == column_name
    ]
    if not matching_columns:
        raise LookupError(f"Column does not exist in metadata: {table_name}.{column_name}")

    with connect_bird_database(db_id) as conn:
        ddl_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()

    column_info = dict(matching_columns[0])
    column_info["ddl"] = ddl_row[0] if ddl_row else None
    return column_info


def verify_exact_value(
    db_id: str,
    table_name: str,
    column_name: str,
    value: object,
    limit: int = 5,
) -> dict:
    """Verify whether a value exists exactly in a SQLite column."""
    sql = (
        f"SELECT {quote_identifier(column_name)} "
        f"FROM {quote_identifier(table_name)} "
        f"WHERE lower(CAST({quote_identifier(column_name)} AS TEXT)) = lower(?) "
        f"LIMIT ?"
    )
    with connect_bird_database(db_id) as conn:
        rows = conn.execute(sql, (str(value), limit)).fetchall()

    return {
        "exists": bool(rows),
        "value": value,
        "examples": [row[0] for row in rows],
    }


def probe_similar_values(
    db_id: str,
    table_name: str,
    column_name: str,
    value: object,
    top_k: int = 5,
    scan_limit: int = 2000,
) -> list[dict]:
    """Return the top-k distinct column values most similar to a fuzzy value."""
    sql = (
        f"SELECT DISTINCT {quote_identifier(column_name)} "
        f"FROM {quote_identifier(table_name)} "
        f"WHERE {quote_identifier(column_name)} IS NOT NULL "
        f"LIMIT ?"
    )
    with connect_bird_database(db_id) as conn:
        rows = conn.execute(sql, (scan_limit,)).fetchall()

    ranked = []
    for row in rows:
        candidate_value = row[0]
        score = text_similarity(value, candidate_value)
        if score > 0:
            ranked.append({"value": candidate_value, "score": score})

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


def get_column_enums(
    db_id: str,
    table_name: str,
    column_name: str,
    limit: int = 50,
) -> list[dict]:
    """Return frequent distinct values for a column."""
    sql = (
        f"SELECT {quote_identifier(column_name)}, COUNT(*) AS frequency "
        f"FROM {quote_identifier(table_name)} "
        f"WHERE {quote_identifier(column_name)} IS NOT NULL "
        f"GROUP BY {quote_identifier(column_name)} "
        f"ORDER BY frequency DESC "
        f"LIMIT ?"
    )
    with connect_bird_database(db_id) as conn:
        rows = conn.execute(sql, (limit,)).fetchall()

    return [{"value": row[0], "frequency": row[1]} for row in rows]


def sample_column_format(
    db_id: str,
    table_name: str,
    column_name: str,
    sample_size: int = 10,
) -> list[object]:
    """Sample non-null values from a column."""
    sql = (
        f"SELECT {quote_identifier(column_name)} "
        f"FROM {quote_identifier(table_name)} "
        f"WHERE {quote_identifier(column_name)} IS NOT NULL "
        f"LIMIT ?"
    )
    with connect_bird_database(db_id) as conn:
        rows = conn.execute(sql, (sample_size,)).fetchall()

    return [row[0] for row in rows]


def find_similar_columns(
    db_id: str,
    value_or_description: str,
    top_k: int = 10,
    split: str = "dev",
) -> list[dict]:
    """Find columns whose names or semantic labels match a description."""
    ranked = []
    for column in list_schema_columns(db_id, split=split):
        column_text = " ".join(
            [
                column["table_name"],
                column["table_comment"],
                column["column_name"],
                column["column_comment"],
                column.get("semantic_column_name") or "",
                column.get("column_description") or "",
                column.get("data_format") or "",
                column.get("value_description") or "",
                column["data_type"] or "",
            ]
        )
        score = text_similarity(value_or_description, column_text)
        if score > 0:
            ranked.append({**column, "score": score})

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


def route_topology(
    db_id: str,
    source_table: str,
    target_table: str,
    split: str = "dev",
) -> list[dict]:
    """Find the shortest foreign-key path between two tables."""
    if source_table == target_table:
        return []

    adjacency: dict[str, list[tuple[str, dict]]] = {}
    for edge in list_foreign_key_edges(db_id, split=split):
        adjacency.setdefault(edge["source_table"], []).append((edge["target_table"], edge))
        reverse_edge = {
            "source_table": edge["target_table"],
            "source_column": edge["target_column"],
            "target_table": edge["source_table"],
            "target_column": edge["source_column"],
            "join_type": edge["join_type"],
        }
        adjacency.setdefault(edge["target_table"], []).append((edge["source_table"], reverse_edge))

    queue = deque([(source_table, [])])
    visited = {source_table}

    while queue:
        current_table, path = queue.popleft()
        for next_table, edge in adjacency.get(current_table, []):
            if next_table in visited:
                continue

            next_path = path + [edge]
            if next_table == target_table:
                return next_path

            visited.add(next_table)
            queue.append((next_table, next_path))

    return []


def build_topology_closure(
    db_id: str,
    selected_tables: list[str],
    split: str = "dev",
) -> dict:
    """Connect selected tables with shortest foreign-key paths when possible."""
    unique_tables = list(dict.fromkeys(selected_tables))
    if len(unique_tables) <= 1:
        return {"tables": unique_tables, "edges": []}

    connected_tables = {unique_tables[0]}
    topology_edges: list[dict] = []

    for table_name in unique_tables[1:]:
        best_path = None
        for connected_table in connected_tables:
            path = route_topology(db_id, connected_table, table_name, split=split)
            if path and (best_path is None or len(path) < len(best_path)):
                best_path = path

        if not best_path:
            connected_tables.add(table_name)
            continue

        for edge in best_path:
            topology_edges.append(edge)
            connected_tables.add(edge["source_table"])
            connected_tables.add(edge["target_table"])

    all_tables = list(dict.fromkeys(
        unique_tables
        + [
            table
            for edge in topology_edges
            for table in (edge["source_table"], edge["target_table"])
        ]
    ))
    unique_edges = list({
        (
            edge["source_table"],
            edge["source_column"],
            edge["target_table"],
            edge["target_column"],
        ): edge
        for edge in topology_edges
    }.values())
    return {"tables": all_tables, "edges": unique_edges}
