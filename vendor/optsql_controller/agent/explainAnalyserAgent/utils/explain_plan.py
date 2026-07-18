"""Tool: get_explain_plan.

Fetches raw DBMS execution plans. SQLite uses EXPLAIN QUERY PLAN. MySQL uses
EXPLAIN FORMAT=JSON by default and falls back to tabular EXPLAIN if JSON
explain fails. The tool keeps raw evidence and does not normalize plan details.
"""

from __future__ import annotations

import json
from typing import Any

from agent.explainAnalyserAgent.utils.common import ensure_readonly_select
from agent.explainAnalyserAgent.utils.common import fetch_all_dicts
from agent.explainAnalyserAgent.utils.common import require_mysql_connection
from agent.explainAnalyserAgent.utils.common import sqlite_connection
from agent.explainAnalyserAgent.utils.models import GetExplainPlanInput
from agent.explainAnalyserAgent.utils.models import GetExplainPlanOutput


def get_explain_plan(
    input_data: GetExplainPlanInput,
    *,
    connection: Any | None = None,
) -> GetExplainPlanOutput:
    """Return the raw SQLite/MySQL explain plan for a read-only query."""
    try:
        ensure_readonly_select(input_data.sql, input_data.dbms)
    except Exception as exc:
        return GetExplainPlanOutput(
            dbms=input_data.dbms,
            mode=input_data.mode,
            explain_sql="",
            raw_plan=None,
            error=str(exc),
        )

    if input_data.dbms == "sqlite":
        return _get_sqlite_explain_plan(input_data)
    if input_data.dbms == "mysql":
        return _get_mysql_explain_plan(input_data, connection=connection)
    raise ValueError(f"Unsupported dbms: {input_data.dbms}")


def _get_sqlite_explain_plan(input_data: GetExplainPlanInput) -> GetExplainPlanOutput:
    explain_sql = f"EXPLAIN QUERY PLAN {input_data.sql}"
    warnings: list[str] = []
    if input_data.mode == "analyze":
        warnings.append("SQLite does not support EXPLAIN ANALYZE; using estimated mode.")
    try:
        with sqlite_connection(input_data.db_id, timeout_ms=input_data.timeout_ms) as conn:
            cursor = conn.execute(explain_sql)
            rows = cursor.fetchall()
            raw_plan = [
                {
                    "id": row[0],
                    "parent": row[1],
                    "notused": row[2],
                    "detail": row[3],
                }
                for row in rows
            ]
        return GetExplainPlanOutput(
            dbms="sqlite",
            mode="estimated",
            explain_sql=explain_sql,
            raw_plan=raw_plan,
            warnings=warnings,
        )
    except Exception as exc:
        return GetExplainPlanOutput(
            dbms="sqlite",
            mode="estimated",
            explain_sql=explain_sql,
            raw_plan=None,
            warnings=warnings,
            error=str(exc),
        )


def _get_mysql_explain_plan(
    input_data: GetExplainPlanInput,
    *,
    connection: Any | None,
) -> GetExplainPlanOutput:
    warnings: list[str] = []
    try:
        conn = require_mysql_connection(connection)
    except Exception as exc:
        return GetExplainPlanOutput(
            dbms="mysql",
            mode=input_data.mode,
            explain_sql="",
            raw_plan=None,
            error=str(exc),
        )

    if input_data.mode == "analyze":
        analyze_sql = f"EXPLAIN ANALYZE {input_data.sql}"
        try:
            cursor = conn.cursor()
            try:
                cursor.execute(analyze_sql)
                rows = fetch_all_dicts(cursor)
            finally:
                try:
                    cursor.close()
                except Exception:
                    pass
            return GetExplainPlanOutput(
                dbms="mysql",
                mode="analyze",
                explain_sql=analyze_sql,
                raw_plan=rows,
                warnings=["EXPLAIN ANALYZE executes the query."],
            )
        except Exception as exc:
            warnings.append(f"EXPLAIN ANALYZE failed; falling back to JSON explain: {exc}")

    explain_sql = f"EXPLAIN FORMAT=JSON {input_data.sql}"
    try:
        cursor = conn.cursor()
        try:
            cursor.execute(explain_sql)
            row = cursor.fetchone()
        finally:
            try:
                cursor.close()
            except Exception:
                pass
        raw_value = row[0] if row else "{}"
        raw_plan = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
        return GetExplainPlanOutput(
            dbms="mysql",
            mode="estimated",
            explain_sql=explain_sql,
            raw_plan=raw_plan,
            warnings=warnings,
        )
    except Exception as json_exc:
        warnings.append(f"EXPLAIN FORMAT=JSON failed; falling back to tabular EXPLAIN: {json_exc}")

    fallback_sql = f"EXPLAIN {input_data.sql}"
    try:
        cursor = conn.cursor()
        try:
            cursor.execute(fallback_sql)
            rows = fetch_all_dicts(cursor)
        finally:
            try:
                cursor.close()
            except Exception:
                pass
        return GetExplainPlanOutput(
            dbms="mysql",
            mode="estimated",
            explain_sql=fallback_sql,
            raw_plan=rows,
            warnings=warnings,
        )
    except Exception as exc:
        return GetExplainPlanOutput(
            dbms="mysql",
            mode="estimated",
            explain_sql=fallback_sql,
            raw_plan=None,
            warnings=warnings,
            error=str(exc),
        )
