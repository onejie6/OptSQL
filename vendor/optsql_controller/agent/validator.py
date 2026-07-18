"""Validator Agent implementation.

Validator is the optimization loop safety gate: it checks candidate SQL
execution, result equivalence, performance delta, and Blueprint guardrails.
It does not rewrite SQL and does not expand schema context.
"""

from __future__ import annotations

import json
import subprocess
import time
import re
from pathlib import Path
from typing import Any

import sqlglot
import sqlglot.expressions as exp

from agent.base import BaseAgent
from myTypes import (
    AgentRequest,
    AgentResponse,
    ColumnRef,
    ExecutionMetrics,
    JoinEdge,
    JoinGraph,
    ResultComparison,
    SQLVersion,
    ValidationReport,
    VerifiedContextBlueprint,
)
from utils.db import connect_bird_database
from utils.db import get_bird_db_path
from utils.sql_safety import ensure_select_sql
from utils.sql_comparison import compare_sql_results


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQLITE_SCANSTATUS_SOURCE = PROJECT_ROOT / "tools" / "sqlite_scanstatus" / "scanstatus_probe.c"
SQLITE_SCANSTATUS_BINARY = PROJECT_ROOT / "tools" / "sqlite_scanstatus" / "scanstatus_probe"


class ValidatorAgent(BaseAgent):
    """Validate syntax, equivalence, performance, and semantic guardrails."""

    name = "validator"

    def __init__(
        self,
        *,
        min_improvement_ratio: float | None = None,
        min_scan_row_improvement_ratio: float | None = None,
        min_latency_improvement_percent: float = 10.0,
        min_latency_improvement_ms: float = 5.0,
        min_scan_row_improvement_percent: float = 3.0,
        latency_regression_tolerance: float = 0.05,
        timeout_ms: int = 5000,
    ) -> None:
        self.min_improvement_ratio = (
            float(min_improvement_ratio)
            if min_improvement_ratio is not None
            else _percent_to_ratio(min_latency_improvement_percent)
        )
        self.min_scan_row_improvement_ratio = (
            float(min_scan_row_improvement_ratio)
            if min_scan_row_improvement_ratio is not None
            else _percent_to_ratio(min_scan_row_improvement_percent)
        )
        self.min_latency_improvement_ms = float(min_latency_improvement_ms)
        self.latency_regression_tolerance = latency_regression_tolerance
        self.timeout_ms = timeout_ms
        self._last_equivalence: ResultComparison | None = None
        self._last_performance_delta: dict | None = None
        self._last_guardrail_errors: list[str] = []

    # ------------------------------------------------------------------
    # Input selection
    # ------------------------------------------------------------------

    def select_source_sql(self, request: AgentRequest) -> SQLVersion:
        for container in (request.input_artifacts, request.runtime_state):
            for key in ("source_sql_version", "source_sql", "current_sql_version", "sql_version"):
                sql_version = _coerce_sql_version(container.get(key))
                if sql_version is not None:
                    return sql_version
        raise ValueError("ValidatorAgent requires a source SQLVersion.")

    def select_candidate_sql(self, request: AgentRequest) -> SQLVersion:
        for container in (request.input_artifacts, request.runtime_state):
            for key in ("candidate_sql_version", "candidate_sql"):
                sql_version = _coerce_sql_version(container.get(key))
                if sql_version is not None:
                    return sql_version
        raise ValueError("ValidatorAgent requires a candidate SQLVersion.")

    def select_blueprint(self, request: AgentRequest) -> VerifiedContextBlueprint:
        for container in (request.input_artifacts, request.runtime_state):
            blueprint = _coerce_blueprint(container.get("blueprint"))
            if blueprint is not None:
                return blueprint
        raise ValueError("ValidatorAgent requires input_artifacts['blueprint'].")

    def select_db_context(self, request: AgentRequest) -> tuple[str, str]:
        db_id = str(
            request.input_artifacts.get("db_id")
            or request.constraints.get("db_id")
            or request.task.db_id
        )
        dbms = str(
            request.input_artifacts.get("dbms")
            or request.constraints.get("dbms")
            or request.task.dbms
        ).lower()
        if dbms != "sqlite":
            raise ValueError(f"ValidatorAgent currently supports sqlite only, got {dbms}.")
        return db_id, dbms

    def apply_threshold_constraints(self, request: AgentRequest) -> None:
        """Allow per-request percentage thresholds to override defaults."""
        constraints = request.constraints or {}
        if "min_latency_improvement_percent" in constraints:
            self.min_improvement_ratio = _percent_to_ratio(
                constraints["min_latency_improvement_percent"]
            )
        elif "min_improvement_ratio" in constraints:
            self.min_improvement_ratio = float(constraints["min_improvement_ratio"])
        if "min_latency_improvement_ms" in constraints:
            self.min_latency_improvement_ms = float(
                constraints["min_latency_improvement_ms"]
            )

        if "min_scan_row_improvement_percent" in constraints:
            self.min_scan_row_improvement_ratio = _percent_to_ratio(
                constraints["min_scan_row_improvement_percent"]
            )
        elif "min_scan_row_improvement_ratio" in constraints:
            self.min_scan_row_improvement_ratio = float(
                constraints["min_scan_row_improvement_ratio"]
            )

    # ------------------------------------------------------------------
    # Validation tools
    # ------------------------------------------------------------------

    def validate_syntax(self, sql_version: SQLVersion, db_id: str, dbms: str) -> ExecutionMetrics:
        """Execute SQL to prove it parses/runs on the target DBMS."""
        if dbms.lower() != "sqlite":
            return ExecutionMetrics(
                executable=False,
                latency_ms=None,
                row_count=None,
                explain_plan=None,
                plan_features={},
                error_message=f"Unsupported dbms: {dbms}",
            )
        start = time.perf_counter()
        conn = None
        try:
            ensure_select_sql(sql_version.sql)
            conn = connect_bird_database(db_id)
            cursor = conn.execute(sql_version.sql)
            rows = cursor.fetchall()
            plan_snapshot = _collect_sqlite_plan_snapshot(conn, sql_version, db_id=db_id)
            return ExecutionMetrics(
                executable=True,
                latency_ms=round((time.perf_counter() - start) * 1000.0, 3),
                row_count=len(rows),
                explain_plan=plan_snapshot["explain_plan"],
                plan_features={
                    "sql_version_id": sql_version.version_id,
                    "dbms": dbms,
                    "sql": sql_version.sql,
                    "raw_explain_plan": plan_snapshot["raw_explain_plan"],
                    "explain_sql": plan_snapshot["explain_sql"],
                    "total_scanned_rows": plan_snapshot["total_scanned_rows"],
                    "scan_details": plan_snapshot["scan_details"],
                    "scan_row_source": plan_snapshot["scan_row_source"],
                    "scan_row_kind": plan_snapshot.get("scan_row_kind"),
                    "scanstatus_warnings": plan_snapshot.get("warnings", []),
                    "explain_plan_cacheable": True,
                },
                error_message=None,
            )
        except Exception as exc:
            return ExecutionMetrics(
                executable=False,
                latency_ms=round((time.perf_counter() - start) * 1000.0, 3),
                row_count=None,
                explain_plan=None,
                plan_features={
                    "sql_version_id": sql_version.version_id,
                    "dbms": dbms,
                    "sql": sql_version.sql,
                },
                error_message=str(exc),
            )
        finally:
            if conn is not None:
                conn.close()

    def check_equivalence(
        self,
        source_sql: SQLVersion,
        candidate_sql: SQLVersion,
        db_id: str,
    ) -> bool:
        """Execution-based result equivalence using source SQL order semantics."""
        comparison = compare_sql_results(
            gold_sql=source_sql.sql,
            generated_sql=candidate_sql.sql,
            db_id=db_id,
        )
        self._last_equivalence = comparison
        return comparison.equivalent

    def measure_performance_delta(
        self,
        source_metrics: ExecutionMetrics,
        candidate_metrics: ExecutionMetrics,
    ) -> dict:
        """Return actual latency and scanned-row comparison."""
        old_latency = source_metrics.latency_ms
        new_latency = candidate_metrics.latency_ms
        latency_improvement_ratio = 0.0
        latency_improvement_ms = 0.0
        latency_better = False
        latency_regressed = False
        if old_latency and new_latency and old_latency > 0:
            latency_improvement_ms = old_latency - new_latency
            latency_improvement_ratio = (old_latency - new_latency) / old_latency
            latency_better = (
                latency_improvement_ratio >= self.min_improvement_ratio
                and latency_improvement_ms >= self.min_latency_improvement_ms
            )
            latency_regressed = new_latency > old_latency * (1 + self.latency_regression_tolerance)

        old_scan_rows = _numeric_feature(source_metrics.plan_features or {}, "total_scanned_rows")
        new_scan_rows = _numeric_feature(candidate_metrics.plan_features or {}, "total_scanned_rows")
        plan_regression_reasons = _plan_regression_reasons(
            source_metrics.plan_features or {},
            candidate_metrics.plan_features or {},
        )
        scan_rows_improvement_ratio = 0.0
        scan_rows_better = False
        if old_scan_rows and new_scan_rows is not None and old_scan_rows > 0:
            scan_rows_improvement_ratio = (old_scan_rows - new_scan_rows) / old_scan_rows
            scan_rows_better = scan_rows_improvement_ratio >= self.min_scan_row_improvement_ratio

        scan_rows_better_without_latency_regression = scan_rows_better and not latency_regressed
        latency_better_without_plan_regression = latency_better and not plan_regression_reasons
        performance_better = (
            latency_better_without_plan_regression
            or scan_rows_better_without_latency_regression
        )
        delta = {
            "old_latency_ms": old_latency,
            "new_latency_ms": new_latency,
            "latency_improvement_ms": round(latency_improvement_ms, 6),
            "min_latency_improvement_ms": self.min_latency_improvement_ms,
            "latency_improvement_ratio": round(latency_improvement_ratio, 6),
            "min_latency_improvement_ratio": self.min_improvement_ratio,
            "min_latency_improvement_percent": round(self.min_improvement_ratio * 100, 6),
            "latency_better": latency_better,
            "latency_better_without_plan_regression": latency_better_without_plan_regression,
            "latency_regressed": latency_regressed,
            "old_total_scanned_rows": old_scan_rows,
            "new_total_scanned_rows": new_scan_rows,
            "plan_regression_reasons": plan_regression_reasons,
            "scan_rows_improvement_ratio": round(scan_rows_improvement_ratio, 6),
            "min_scan_row_improvement_ratio": self.min_scan_row_improvement_ratio,
            "min_scan_row_improvement_percent": round(
                self.min_scan_row_improvement_ratio * 100,
                6,
            ),
            "scan_rows_better": scan_rows_better,
            "scan_rows_better_without_latency_regression": scan_rows_better_without_latency_regression,
            "performance_better": performance_better,
        }
        self._last_performance_delta = delta
        return delta

    def enforce_semantic_guardrails(
        self,
        candidate_sql: SQLVersion,
        blueprint: VerifiedContextBlueprint,
    ) -> bool:
        """Schema reference validation is intentionally disabled."""
        del candidate_sql, blueprint
        self._last_guardrail_errors = []
        return not self._last_guardrail_errors

    def validate(
        self,
        source_sql: SQLVersion,
        candidate_sql: SQLVersion,
        blueprint: VerifiedContextBlueprint,
        db_id: str,
        dbms: str,
    ) -> ValidationReport:
        self._last_equivalence = None
        self._last_performance_delta = None
        self._last_guardrail_errors = []

        guardrails_ok = self.enforce_semantic_guardrails(candidate_sql, blueprint)
        old_metrics = self.validate_syntax(source_sql, db_id, dbms)
        if guardrails_ok:
            new_metrics = self.validate_syntax(candidate_sql, db_id, dbms)
        else:
            new_metrics = ExecutionMetrics(
                executable=False,
                latency_ms=None,
                row_count=None,
                explain_plan=None,
                plan_features={
                    "sql_version_id": candidate_sql.version_id,
                    "dbms": dbms,
                    "sql": candidate_sql.sql,
                },
                error_message="Candidate rejected by semantic guardrails before execution.",
            )

        equivalent = False
        performance_better = False
        failure_reason = None
        if not guardrails_ok:
            failure_reason = "; ".join(self._last_guardrail_errors)
        elif not old_metrics.executable:
            failure_reason = f"Source SQL is not executable: {old_metrics.error_message}"
        elif not new_metrics.executable:
            failure_reason = f"Candidate SQL is not executable: {new_metrics.error_message}"
        else:
            equivalent = self.check_equivalence(source_sql, candidate_sql, db_id)
            if not equivalent:
                diff = self._last_equivalence.diff_summary if self._last_equivalence else None
                failure_reason = f"Candidate SQL is not equivalent: {diff or 'result mismatch'}"
            else:
                delta = self.measure_performance_delta(old_metrics, new_metrics)
                performance_better = bool(delta["performance_better"])
                if not performance_better:
                    failure_reason = "Candidate is equivalent but has no measured performance improvement."

        accepted = bool(new_metrics.executable and equivalent and performance_better and guardrails_ok)
        if accepted:
            failure_reason = None
        return ValidationReport(
            executable=new_metrics.executable,
            equivalent=equivalent,
            performance_better=performance_better,
            old_metrics=old_metrics,
            new_metrics=new_metrics,
            failure_reason=failure_reason,
            accepted=accepted,
        )

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run_validation(self, request: AgentRequest) -> dict:
        source_sql = self.select_source_sql(request)
        candidate_sql = self.select_candidate_sql(request)
        blueprint = self.select_blueprint(request)
        db_id, dbms = self.select_db_context(request)
        self.apply_threshold_constraints(request)
        report = self.validate(source_sql, candidate_sql, blueprint, db_id, dbms)
        explain_plan_cache = (
            _build_explain_plan_cache(candidate_sql, report.new_metrics, db_id, dbms)
            if report.accepted
            else None
        )
        return {
            "source_sql_version": source_sql,
            "candidate_sql_version": candidate_sql,
            "blueprint": blueprint,
            "db_id": db_id,
            "dbms": dbms,
            "validation_report": report,
            "equivalence_comparison": self._last_equivalence,
            "performance_delta": self._last_performance_delta,
            "guardrail_errors": list(self._last_guardrail_errors),
            "explain_plan_cache": explain_plan_cache,
            "tool_calls": [
                {"tool_name": "enforce_semantic_guardrails", "summary": "schema reference check disabled"},
                {"tool_name": "execute_sql", "summary": "source and candidate actual execution"},
                {"tool_name": "explain_query_plan", "summary": "source and candidate scan-row snapshot"},
                {"tool_name": "check_equivalence", "summary": "execution-based result comparison"},
                {"tool_name": "measure_performance_delta", "summary": "actual latency and scanned-row delta"},
            ],
        }

    def run(self, request: AgentRequest) -> AgentResponse:
        try:
            artifacts = self.run_validation(request)
        except Exception as exc:
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status="error",
                output_artifacts={},
                reasoning_summary="Validation failed before producing a ValidationReport.",
                tool_calls=[],
                errors=[str(exc)],
            )
        report = artifacts["validation_report"]
        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            status="success",
            output_artifacts=artifacts,
            reasoning_summary=(
                f"Validation accepted={report.accepted}; "
                f"executable={report.executable}; equivalent={report.equivalent}; "
                f"performance_better={report.performance_better}."
            ),
            tool_calls=artifacts["tool_calls"],
            errors=[] if report.accepted else ([report.failure_reason] if report.failure_reason else []),
        )


def _collect_sqlite_plan_snapshot(conn: Any, sql_version: SQLVersion, db_id: str | None = None) -> dict:
    try:
        ensure_select_sql(sql_version.sql)
    except Exception as exc:
        return {
            "explain_sql": "",
            "explain_plan": None,
            "raw_explain_plan": None,
            "total_scanned_rows": None,
            "scan_details": [],
            "scan_row_source": f"explain_blocked: {exc}",
            "scan_row_kind": "unknown",
            "warnings": [],
        }
    explain_sql = f"EXPLAIN QUERY PLAN {sql_version.sql}"
    try:
        rows = conn.execute(explain_sql).fetchall()
    except Exception as exc:
        return {
            "explain_sql": explain_sql,
            "explain_plan": None,
            "raw_explain_plan": None,
            "total_scanned_rows": None,
            "scan_details": [],
            "scan_row_source": f"explain_failed: {exc}",
            "scan_row_kind": "unknown",
            "warnings": [],
        }

    raw_plan = [
        {"id": row[0], "parent": row[1], "notused": row[2], "detail": row[3]}
        for row in rows
    ]
    actual_scan_snapshot = _collect_sqlite_scanstatus_snapshot(db_id, sql_version)
    if actual_scan_snapshot is not None:
        return {
            "explain_sql": explain_sql,
            "explain_plan": str(raw_plan),
            "raw_explain_plan": raw_plan,
            "total_scanned_rows": actual_scan_snapshot["total_scanned_rows"],
            "scan_details": actual_scan_snapshot["scan_details"],
            "scan_row_source": actual_scan_snapshot["scan_row_source"],
            "scan_row_kind": actual_scan_snapshot["scan_row_kind"],
            "warnings": actual_scan_snapshot.get("warnings", []),
        }

    alias_to_table = _sql_alias_to_table(sql_version.sql)
    table_row_counts: dict[str, int] = {}
    scan_details: list[dict] = []
    total_scanned_rows = 0
    for row in raw_plan:
        detail = str(row.get("detail") or "")
        table = _table_from_sqlite_plan_detail(detail)
        if not table:
            continue
        real_table = alias_to_table.get(table, table)
        if real_table not in table_row_counts:
            table_row_counts[real_table] = _table_row_count(conn, real_table)
        row_count = table_row_counts[real_table]
        total_scanned_rows += row_count
        scan_details.append(
            {
                "plan_node_id": row["id"],
                "detail": detail,
                "table": real_table,
                "estimated_scanned_rows": row_count,
            }
        )

    return {
        "explain_sql": explain_sql,
        "explain_plan": str(raw_plan),
        "raw_explain_plan": raw_plan,
        "total_scanned_rows": total_scanned_rows,
        "scan_details": scan_details,
        "scan_row_source": "sqlite_explain_query_plan_table_row_counts",
        "scan_row_kind": "estimated",
        "warnings": [],
    }


def _collect_sqlite_scanstatus_snapshot(db_id: str | None, sql_version: SQLVersion) -> dict | None:
    if not db_id:
        return None
    probe = _ensure_sqlite_scanstatus_probe()
    if probe is None:
        return None
    try:
        db_path = get_bird_db_path(db_id)
    except Exception:
        return None
    try:
        result = subprocess.run(
            [str(probe), str(db_path), sql_version.sql],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception:
        return None
    if not result.stdout.strip():
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if not payload.get("available"):
        return None
    details = payload.get("scan_details")
    total_scanned_rows = payload.get("total_scanned_rows")
    if not isinstance(details, list) or not isinstance(total_scanned_rows, int):
        return None
    return {
        "total_scanned_rows": total_scanned_rows,
        "scan_details": details,
        "scan_row_source": str(payload.get("scan_row_source") or "sqlite_stmt_scanstatus_v2_nvisit"),
        "scan_row_kind": "actual",
        "warnings": [],
    }


def _ensure_sqlite_scanstatus_probe() -> Path | None:
    if SQLITE_SCANSTATUS_BINARY.is_file():
        return SQLITE_SCANSTATUS_BINARY
    if not SQLITE_SCANSTATUS_SOURCE.is_file():
        return None
    try:
        result = subprocess.run(
            [
                "cc",
                str(SQLITE_SCANSTATUS_SOURCE),
                "-lsqlite3",
                "-o",
                str(SQLITE_SCANSTATUS_BINARY),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0 or not SQLITE_SCANSTATUS_BINARY.is_file():
        return None
    return SQLITE_SCANSTATUS_BINARY


def _plan_regression_reasons(source_features: dict, candidate_features: dict) -> list[str]:
    source_details = _sqlite_plan_details(source_features)
    candidate_details = _sqlite_plan_details(candidate_features)
    reasons: list[str] = []
    if _uses_temp_sort(candidate_details) and not _uses_temp_sort(source_details):
        reasons.append("candidate introduces a temporary sort")
    if _uses_full_scan(candidate_details) and not _uses_full_scan(source_details):
        reasons.append("candidate changes indexed/search access into a full scan")
    return reasons


def _sqlite_plan_details(plan_features: dict) -> list[str]:
    raw_plan = plan_features.get("raw_explain_plan")
    if not isinstance(raw_plan, list):
        return []
    details: list[str] = []
    for row in raw_plan:
        if isinstance(row, dict):
            details.append(str(row.get("detail") or "").upper())
    return details


def _uses_temp_sort(details: list[str]) -> bool:
    return any("USE TEMP B-TREE" in detail for detail in details)


def _uses_full_scan(details: list[str]) -> bool:
    return any(detail.startswith("SCAN ") for detail in details)


def _table_from_sqlite_plan_detail(detail: str) -> str | None:
    match = re.search(r"\b(?:SCAN|SEARCH)\s+([`\"\[]?[\w\s().-]+[`\"\]]?)", detail)
    if not match:
        return None
    name = match.group(1).strip().strip("`\"[]")
    upper_name = name.upper()
    if upper_name in {"CONSTANT ROW", "SUBQUERY"} or upper_name.startswith("SUBQUERY"):
        return None
    return name.split()[0]


def _sql_alias_to_table(sql: str) -> dict[str, str]:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return {}
    result: dict[str, str] = {}
    for table in ast.find_all(exp.Table):
        if not table.name:
            continue
        result[table.name] = table.name
        if table.alias:
            result[table.alias] = table.name
    return result


def _table_row_count(conn: Any, table_name: str) -> int:
    escaped = table_name.replace('"', '""')
    try:
        row = conn.execute(f'SELECT COUNT(*) FROM "{escaped}"').fetchone()
    except Exception:
        return 0
    return int(row[0]) if row and row[0] is not None else 0


def _build_explain_plan_cache(
    sql_version: SQLVersion,
    metrics: ExecutionMetrics,
    db_id: str,
    dbms: str,
) -> dict | None:
    features = metrics.plan_features or {}
    raw_plan = features.get("raw_explain_plan")
    if raw_plan is None:
        return None
    return {
        "sql_version_id": sql_version.version_id,
        "sql": sql_version.sql,
        "db_id": db_id,
        "dbms": dbms,
        "mode": "estimated",
        "explain_sql": features.get("explain_sql"),
        "raw_plan": raw_plan,
        "total_scanned_rows": features.get("total_scanned_rows"),
        "scan_details": features.get("scan_details", []),
        "source": "validator",
    }


def _numeric_feature(features: dict, key: str) -> float | None:
    value = features.get(key)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _percent_to_ratio(value: Any) -> float:
    return float(value) / 100.0


def _blueprint_violations(sql: str, blueprint: VerifiedContextBlueprint) -> list[str]:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception as exc:
        return [f"Candidate SQL failed to parse: {exc}"]
    if isinstance(ast, (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop)):
        return ["Candidate SQL must be a read-only SELECT query."]
    if ast.find(exp.Select) is None:
        return ["Candidate SQL must contain a SELECT query."]

    allowed_tables = set(blueprint.selected_tables)
    allowed_columns = {(c.table_name, c.column_name) for c in blueprint.selected_columns}
    allowed_column_names = {c.column_name for c in blueprint.selected_columns}
    tables, alias_to_table = _extract_table_refs(ast)
    violations: list[str] = []
    for table in sorted(tables):
        if table not in allowed_tables:
            violations.append(f"Table '{table}' is outside the Blueprint.")
    for column in ast.find_all(exp.Column):
        if isinstance(column.this, exp.Star):
            violations.append("Candidate SQL contains SELECT *.")
            continue
        name = column.name
        qualifier = column.table
        if not name:
            continue
        if qualifier:
            resolved_table = alias_to_table.get(qualifier, qualifier)
            if (resolved_table, name) not in allowed_columns:
                violations.append(f"Column '{qualifier}.{name}' is outside the Blueprint.")
        elif name not in allowed_column_names:
            violations.append(f"Column '{name}' is outside the Blueprint.")
    return _unique(violations)


def _extract_table_refs(ast: exp.Expression) -> tuple[set[str], dict[str, str]]:
    tables: set[str] = set()
    alias_to_table: dict[str, str] = {}
    for table in ast.find_all(exp.Table):
        name = table.name
        if not name:
            continue
        tables.add(name)
        alias_to_table[name] = name
        if table.alias:
            alias_to_table[table.alias] = name
    return tables, alias_to_table


def _coerce_sql_version(value: Any) -> SQLVersion | None:
    if isinstance(value, SQLVersion):
        return value
    if isinstance(value, str):
        return SQLVersion(
            version_id="inline",
            parent_id=None,
            sql=value,
            source_agent="inline",
            rewrite_rule_ids=[],
            explanation="Inline SQL from request artifact.",
            created_at="",
        )
    if not isinstance(value, dict):
        return None
    required = {"version_id", "parent_id", "sql", "source_agent", "rewrite_rule_ids", "explanation", "created_at"}
    if not required.issubset(value):
        return None
    return SQLVersion(
        version_id=str(value["version_id"]),
        parent_id=value["parent_id"],
        sql=str(value["sql"]),
        source_agent=str(value["source_agent"]),
        rewrite_rule_ids=list(value["rewrite_rule_ids"]),
        explanation=str(value["explanation"]),
        created_at=str(value["created_at"]),
    )


def _coerce_blueprint(value: Any) -> VerifiedContextBlueprint | None:
    if isinstance(value, VerifiedContextBlueprint):
        return value
    if not isinstance(value, dict):
        return None
    columns = [
        col if isinstance(col, ColumnRef) else ColumnRef(
            table_name=str(col["table_name"]),
            column_name=str(col["column_name"]),
            data_type=col.get("data_type"),
            comment=col.get("comment"),
        )
        for col in value.get("selected_columns", [])
    ]
    topology = value.get("join_topology") or {}
    if isinstance(topology, JoinGraph):
        join_topology = topology
    else:
        edges = [
            edge if isinstance(edge, JoinEdge) else JoinEdge(
                source_table=str(edge["source_table"]),
                source_column=str(edge["source_column"]),
                target_table=str(edge["target_table"]),
                target_column=str(edge["target_column"]),
                join_type=str(edge.get("join_type", "inner")),
            )
            for edge in topology.get("edges", [])
        ]
        join_topology = JoinGraph(
            tables=list(topology.get("tables", value.get("selected_tables", []))),
            edges=edges,
        )
    return VerifiedContextBlueprint(
        db_id=str(value["db_id"]),
        selected_tables=list(value.get("selected_tables", [])),
        selected_columns=columns,
        value_mappings=list(value.get("value_mappings", [])),
        join_topology=join_topology,
        predicate_hints=list(value.get("predicate_hints", [])),
        evidence_trace=list(value.get("evidence_trace", [])),
        confidence=float(value.get("confidence", 0.0)),
    )


def _unique(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
