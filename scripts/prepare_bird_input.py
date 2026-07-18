"""Convert BIRD column meanings into per-database schema descriptions."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any


CSV_FIELDS = (
    "original_column_name",
    "column_name",
    "column_description",
    "data_format",
    "value_description",
)


def _flatten_meanings(payload: Any) -> dict[tuple[str, str, str], str]:
    flattened: dict[tuple[str, str, str], str] = {}
    if not isinstance(payload, dict):
        raise ValueError("column_meaning.json must contain a JSON object")

    for key, value in payload.items():
        if isinstance(key, str) and key.count("|") >= 2:
            db_id, table_name, column_name = key.split("|", 2)
            flattened[(db_id.casefold(), table_name.casefold(), column_name.casefold())] = str(value or "")
            continue
        if not isinstance(value, dict):
            continue
        for table_name, columns in value.items():
            if not isinstance(columns, dict):
                continue
            for column_name, description in columns.items():
                flattened[(str(key).casefold(), str(table_name).casefold(), str(column_name).casefold())] = str(
                    description or ""
                )
    return flattened


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _database_file(database_dir: Path, db_id: str) -> Path:
    expected = database_dir / f"{db_id}.sqlite"
    if expected.is_file():
        return expected
    candidates = sorted(database_dir.glob("*.sqlite"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Could not identify one SQLite file in {database_dir}")
    return candidates[0]


def write_database_descriptions(
    database_root: Path,
    meanings: dict[tuple[str, str, str], str],
) -> tuple[int, int]:
    database_count = 0
    described_columns = 0
    for database_dir in sorted(path for path in database_root.iterdir() if path.is_dir()):
        db_id = database_dir.name
        db_path = _database_file(database_dir, db_id)
        description_dir = database_dir / "database_description"
        description_dir.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
        try:
            table_names = [
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            for table_name in table_names:
                columns = connection.execute(
                    f"PRAGMA table_info({_quote_identifier(table_name)})"
                ).fetchall()
                with (description_dir / f"{table_name}.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                    writer.writeheader()
                    for column in columns:
                        column_name = str(column[1])
                        description = meanings.get(
                            (db_id.casefold(), table_name.casefold(), column_name.casefold()),
                            "",
                        )
                        described_columns += int(bool(description))
                        writer.writerow(
                            {
                                "original_column_name": column_name,
                                "column_name": column_name,
                                "column_description": description,
                                "data_format": str(column[2] or ""),
                                "value_description": "",
                            }
                        )
        finally:
            connection.close()
        database_count += 1
    return database_count, described_columns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bird-root", type=Path, default=Path("data/bird"))
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--column-meaning", type=Path, required=True)
    args = parser.parse_args()

    database_root = args.bird_root / args.split / f"{args.split}_databases"
    if not database_root.is_dir():
        raise FileNotFoundError(f"Missing database directory: {database_root}")
    payload = json.loads(args.column_meaning.read_text(encoding="utf-8"))
    database_count, described_columns = write_database_descriptions(
        database_root, _flatten_meanings(payload)
    )
    print(
        json.dumps(
            {"databases": database_count, "described_columns": described_columns},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
