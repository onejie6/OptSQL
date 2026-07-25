"""Gold-free snapshot iteration and read-only SQLite execution helpers."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable


def iter_snapshot_items(snapshot_path: Path) -> Iterable[dict[str, Any]]:
    manifest = json.loads(snapshot_path.read_text(encoding="utf-8"))
    items_path = snapshot_path.with_name(f"{snapshot_path.name}.data") / manifest.get(
        "items_file", "items.jsonl"
    )
    with items_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            item = dict(record.get("input") or {})
            for artifact in (record.get("pipeline_artifacts") or {}).values():
                if isinstance(artifact, dict):
                    item.update(artifact)
            yield item


def execute_rows(
    db_path: str,
    sql: str,
    timeout_seconds: float,
) -> tuple[set[tuple[Any, ...]] | None, str | None]:
    started = time.monotonic()
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only = ON")
    connection.set_progress_handler(
        lambda: 1 if time.monotonic() - started >= timeout_seconds else 0,
        10_000,
    )
    try:
        return set(connection.execute(sql).fetchall()), None
    except Exception as exc:
        message = str(exc)
        kind = "timeout" if "interrupted" in message.lower() else "execution_error"
        return None, f"{kind}: {type(exc).__name__}: {message}"
    finally:
        connection.close()
