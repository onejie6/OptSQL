"""Tool: detect_db_capabilities.

Detects the target DBMS version and the explain-plan features available to the
Explain Analyser. SQLite is supported through the project's BIRD connection
helper; MySQL is supported through an optional DB-API connection supplied by the
caller.
"""

from __future__ import annotations

from typing import Any

from agent.explainAnalyserAgent.utils.common import require_mysql_connection
from agent.explainAnalyserAgent.utils.common import sqlite_connection
from agent.explainAnalyserAgent.utils.models import DetectDbCapabilitiesInput
from agent.explainAnalyserAgent.utils.models import DetectDbCapabilitiesOutput


def detect_db_capabilities(
    input_data: DetectDbCapabilitiesInput,
    *,
    connection: Any | None = None,
) -> DetectDbCapabilitiesOutput:
    """Return explain/profiling capability flags for SQLite or MySQL."""
    if input_data.dbms == "sqlite":
        return _detect_sqlite_capabilities(input_data)
    if input_data.dbms == "mysql":
        return _detect_mysql_capabilities(input_data, connection=connection)
    raise ValueError(f"Unsupported dbms: {input_data.dbms}")


def _detect_sqlite_capabilities(
    input_data: DetectDbCapabilitiesInput,
) -> DetectDbCapabilitiesOutput:
    notes: list[str] = []
    version: str | None = None
    try:
        with sqlite_connection(input_data.db_id) as conn:
            row = conn.execute("SELECT sqlite_version()").fetchone()
            version = row[0] if row else None
    except Exception as exc:
        notes.append(f"Failed to read SQLite version: {exc}")

    return DetectDbCapabilitiesOutput(
        dbms="sqlite",
        version=version,
        supports_explain_query_plan=True,
        supports_explain_json=False,
        supports_explain_analyze=False,
        supports_runtime_timeout=True,
        supports_optimizer_trace=False,
        default_explain_mode="estimated",
        notes=notes,
    )


def _detect_mysql_capabilities(
    input_data: DetectDbCapabilitiesInput,
    *,
    connection: Any | None,
) -> DetectDbCapabilitiesOutput:
    notes: list[str] = []
    version: str | None = None
    supports_explain_analyze = False
    try:
        conn = require_mysql_connection(connection)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT VERSION()")
            row = cursor.fetchone()
            version = row[0] if row else None
        finally:
            try:
                cursor.close()
            except Exception:
                pass
    except Exception as exc:
        notes.append(f"Failed to read MySQL version: {exc}")

    if version:
        supports_explain_analyze = _mysql_version_supports_explain_analyze(version)
    else:
        notes.append("MySQL version is unknown; EXPLAIN ANALYZE support is disabled.")

    return DetectDbCapabilitiesOutput(
        dbms="mysql",
        version=version,
        supports_explain_query_plan=False,
        supports_explain_json=True,
        supports_explain_analyze=supports_explain_analyze,
        supports_runtime_timeout=True,
        supports_optimizer_trace=True,
        default_explain_mode="estimated",
        notes=notes,
    )


def _mysql_version_supports_explain_analyze(version: str) -> bool:
    """Best-effort MySQL version check without binding to a specific driver."""
    numeric = version.split("-")[0]
    parts: list[int] = []
    for chunk in numeric.split(".")[:3]:
        try:
            parts.append(int(chunk))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    major, minor, patch = parts
    return (major, minor, patch) >= (8, 0, 18)
