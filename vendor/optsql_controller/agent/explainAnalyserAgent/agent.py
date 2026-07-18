"""Explain Analyzer Agent implementation.

The Explain Analyzer sits in the Optimization Loop between SQL generation and
SQL rewriting. It collects execution-plan evidence, normalizes DBMS-specific
details, builds a compact BottleneckReport, and returns a deterministic loop
decision for the Controller.

This agent does not generate rewritten SQL and does not accept or reject
candidate SQL. Those responsibilities belong to SQLRewriterAgent,
ValidatorAgent, and MetaCognitiveController.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import sqlglot
import sqlglot.expressions as exp

from agent.base import BaseAgent
from agent.explainAnalyserAgent.utils import collect_schema_stats
from agent.explainAnalyserAgent.utils import decide_optimization_action
from agent.explainAnalyserAgent.utils import detect_db_capabilities
from agent.explainAnalyserAgent.utils import get_explain_plan
from agent.explainAnalyserAgent.utils import normalize_plan
from agent.explainAnalyserAgent.utils import parse_sql_structure
from agent.explainAnalyserAgent.utils.common import unique_preserve_order
from agent.explainAnalyserAgent.utils.decision import _is_index_bound_full_scan
from agent.explainAnalyserAgent.utils.models import CollectSchemaStatsInput
from agent.explainAnalyserAgent.utils.models import CollectSchemaStatsOutput
from agent.explainAnalyserAgent.utils.models import DetectDbCapabilitiesInput
from agent.explainAnalyserAgent.utils.models import DetectDbCapabilitiesOutput
from agent.explainAnalyserAgent.utils.models import GetExplainPlanInput
from agent.explainAnalyserAgent.utils.models import GetExplainPlanOutput
from agent.explainAnalyserAgent.utils.models import NormalizePlanInput
from agent.explainAnalyserAgent.utils.models import OptimizationDecision
from agent.explainAnalyserAgent.utils.models import ParseSqlStructureInput
from agent.explainAnalyserAgent.utils.models import ParseSqlStructureOutput
from agent.explainAnalyserAgent.utils.models import PlanIR
from myTypes import AgentRequest
from myTypes import AgentResponse
from myTypes import BottleneckReport
from myTypes import ExecutionMetrics
from myTypes import RewriteHint
from myTypes import SQLVersion


SUPPORTED_DBMS = {"sqlite", "mysql"}
EXPLAIN_MODES = {"estimated", "analyze"}


class ExplainAnalyzerAgent(BaseAgent):
    """Analyze execution plans and convert them into bottleneck reports."""

    name = "explain_analyzer"

    def __init__(
        self,
        *,
        default_timeout_ms: int = 5000,
        include_schema_samples: bool = False,
    ) -> None:
        self.default_timeout_ms = default_timeout_ms
        self.include_schema_samples = include_schema_samples

    # ------------------------------------------------------------------
    # Stage 1: request and input normalization
    # ------------------------------------------------------------------

    def select_sql_version(self, request: AgentRequest) -> SQLVersion:
        """Return the SQLVersion that should be analyzed."""
        for container in (request.input_artifacts, request.runtime_state):
            for key in ("sql_version", "current_sql_version"):
                candidate = container.get(key)
                sql_version = _coerce_sql_version(candidate)
                if sql_version is not None:
                    return sql_version
        raise ValueError(
            "ExplainAnalyzerAgent requires input_artifacts['sql_version'] "
            "or runtime_state['current_sql_version']."
        )

    def select_db_context(self, request: AgentRequest) -> tuple[str, str, Any | None]:
        """Return ``(db_id, dbms, connection)`` for tool calls."""
        artifacts = request.input_artifacts
        constraints = request.constraints
        db_id = str(artifacts.get("db_id") or constraints.get("db_id") or request.task.db_id)
        dbms = _validate_dbms(
            str(artifacts.get("dbms") or constraints.get("dbms") or request.task.dbms)
        )
        connection = artifacts.get("connection") or artifacts.get("mysql_connection")
        if dbms == "mysql" and connection is None:
            raise ValueError(
                "MySQL explain analysis requires input_artifacts['connection'] "
                "or input_artifacts['mysql_connection']."
            )
        return db_id, dbms, connection

    def select_previous_risk_tags(self, request: AgentRequest) -> list[str]:
        """Return previous optimization risk tags, if available."""
        for container in (request.input_artifacts, request.runtime_state):
            explicit_tags = container.get("previous_risk_tags")
            if isinstance(explicit_tags, list):
                return unique_preserve_order([str(tag) for tag in explicit_tags])

            previous_report = (
                container.get("previous_bottleneck_report")
                or container.get("bottleneck_report")
            )
            tags = _risk_tags_from_report(previous_report)
            if tags:
                return tags
        return []

    # ------------------------------------------------------------------
    # Stage 2: evidence collection tools
    # ------------------------------------------------------------------

    def detect_capabilities(
        self,
        *,
        db_id: str,
        dbms: str,
        connection: Any | None = None,
    ) -> DetectDbCapabilitiesOutput:
        """Call ``detect_db_capabilities`` and return DBMS explain support."""
        dbms = _validate_dbms(dbms)
        return detect_db_capabilities(
            DetectDbCapabilitiesInput(db_id=db_id, dbms=dbms),
            connection=connection,
        )

    def parse_structure(self, *, sql: str, dbms: str) -> ParseSqlStructureOutput:
        """Call ``parse_sql_structure`` and return static SQL structure."""
        dbms = _validate_dbms(dbms)
        return parse_sql_structure(ParseSqlStructureInput(sql=sql, dbms=dbms))

    def explain(
        self,
        sql_version: SQLVersion,
        db_id: str,
        dbms: str,
        *,
        mode: str | None = None,
        timeout_ms: int | None = None,
        connection: Any | None = None,
    ) -> ExecutionMetrics:
        """Collect raw explain evidence for the SQL version."""
        dbms = _validate_dbms(dbms)
        timeout_ms = timeout_ms or self.default_timeout_ms
        capabilities = self.detect_capabilities(
            db_id=db_id,
            dbms=dbms,
            connection=connection,
        )
        explain_mode = _select_explain_mode(mode, capabilities)
        started_at = time.perf_counter()
        raw_explain = self.get_raw_plan(
            sql=sql_version.sql,
            db_id=db_id,
            dbms=dbms,
            mode=explain_mode,
            timeout_ms=timeout_ms,
            connection=connection,
        )
        latency_ms = (time.perf_counter() - started_at) * 1000
        return ExecutionMetrics(
            executable=raw_explain.error is None,
            latency_ms=latency_ms,
            row_count=None,
            explain_plan=str(raw_explain.raw_plan),
            plan_features={
                "dbms": dbms,
                "sql": sql_version.sql,
                "sql_version_id": sql_version.version_id,
                "capabilities": capabilities,
                "raw_explain": raw_explain,
                "explain_mode": raw_explain.mode,
                "warnings": list(raw_explain.warnings),
            },
            error_message=raw_explain.error,
        )

    def get_raw_plan(
        self,
        *,
        sql: str,
        db_id: str,
        dbms: str,
        mode: str,
        timeout_ms: int,
        connection: Any | None = None,
    ) -> GetExplainPlanOutput:
        """Call ``get_explain_plan`` and return DBMS-specific raw plan data."""
        dbms = _validate_dbms(dbms)
        mode = _validate_explain_mode(mode)
        return get_explain_plan(
            GetExplainPlanInput(
                sql=sql,
                db_id=db_id,
                dbms=dbms,
                mode=mode,
                timeout_ms=timeout_ms,
            ),
            connection=connection,
        )

    def normalize_plan(
        self,
        *,
        dbms: str,
        raw_plan: dict | list | str | None,
        sql_structure: ParseSqlStructureOutput,
    ) -> PlanIR:
        """Call ``normalize_plan`` and return DBMS-neutral PlanIR."""
        dbms = _validate_dbms(dbms)
        return normalize_plan(
            NormalizePlanInput(
                dbms=dbms,
                raw_plan=raw_plan,
                sql_structure=sql_structure,
            )
        ).plan_ir

    def collect_schema_stats(
        self,
        *,
        db_id: str,
        dbms: str,
        sql_structure: ParseSqlStructureOutput,
        timeout_ms: int | None = None,
        connection: Any | None = None,
    ) -> CollectSchemaStatsOutput:
        """Call ``collect_schema_stats`` for SQL-referenced tables/columns."""
        dbms = _validate_dbms(dbms)
        return collect_schema_stats(
            CollectSchemaStatsInput(
                db_id=db_id,
                dbms=dbms,
                tables=sql_structure.tables,
                columns=sql_structure.columns,
                include_samples=self.include_schema_samples,
                timeout_ms=timeout_ms or self.default_timeout_ms,
            ),
            connection=connection,
        )

    # ------------------------------------------------------------------
    # Stage 3: deterministic analysis and report assembly
    # ------------------------------------------------------------------

    def extract_plan_features(self, metrics: ExecutionMetrics, dbms: str) -> dict:
        """Extract a stable feature dictionary from collected metrics."""
        features = dict(metrics.plan_features or {})
        features["dbms"] = _validate_dbms(str(features.get("dbms") or dbms))
        features["latency_ms"] = metrics.latency_ms
        features["row_count"] = metrics.row_count
        features["executable"] = metrics.executable
        features["error_message"] = metrics.error_message
        return features

    def classify_risk_tags(self, plan_features: dict) -> list[str]:
        """Return canonical risk tags from PlanIR and static SQL patterns."""
        tags: list[str] = []
        plan_ir = plan_features.get("normalized_plan")
        if isinstance(plan_ir, PlanIR):
            tags.extend(plan_ir.global_flags)
            for node in plan_ir.nodes:
                tags.extend(node.flags)
                if node.operation == "table_scan":
                    tags.append("full_table_scan")
            if _row_estimate_skew_ratio(plan_ir) is not None:
                tags.append("row_estimate_skew")

        sql_structure = plan_features.get("sql_structure")
        if isinstance(sql_structure, ParseSqlStructureOutput):
            tags.extend(
                _sql_structure_risk_tags(
                    sql_structure,
                    sql=str(plan_features.get("sql") or ""),
                    dbms=str(plan_features.get("dbms") or ""),
                    schema_stats=plan_features.get("schema_stats"),
                )
            )

        decision = plan_features.get("optimization_decision")
        if isinstance(decision, OptimizationDecision):
            tags.extend(decision.risk_tags)

        return unique_preserve_order([tag for tag in tags if tag])

    def detect_bottlenecks(self, plan_features: dict) -> list[str]:
        """Return human-readable bottleneck summaries for the report."""
        tags = set(plan_features.get("risk_tags") or self.classify_risk_tags(plan_features))
        plan_ir = plan_features.get("normalized_plan")
        sql_structure = plan_features.get("sql_structure")
        schema_stats = plan_features.get("schema_stats")
        bottlenecks: list[str] = []

        if isinstance(plan_ir, PlanIR):
            full_scan_tables = _full_scan_tables(plan_ir)
            if full_scan_tables:
                bottlenecks.append(
                    "Full table scan detected on "
                    + _format_tables_with_counts(full_scan_tables, schema_stats)
                    + "."
                )
            if {"temp_sort", "filesort"} & tags:
                bottlenecks.append("Plan uses a temporary sort/filesort operator.")
            if "temp_group_by" in tags:
                bottlenecks.append("Plan uses temporary storage for GROUP BY.")
            if "temp_table" in tags:
                bottlenecks.append("MySQL plan uses a temporary table.")
            if "temp_distinct" in tags:
                bottlenecks.append("Plan uses temporary storage for DISTINCT.")
            if "row_estimate_skew" in tags:
                bottlenecks.append(
                    "Estimated rows differ significantly from actual rows; optimizer statistics may be stale or predicates may be poorly modeled."
                )
            if "correlated_subquery" in tags:
                bottlenecks.append("Plan or SQL structure contains a correlated subquery.")
            if "materialized_subquery" in tags:
                bottlenecks.append("Plan materializes a subquery or derived table.")

        if isinstance(sql_structure, ParseSqlStructureOutput):
            if "select_star" in tags:
                bottlenecks.append("SQL selects all columns, which may widen scanned rows.")
            if "function_on_column" in tags:
                bottlenecks.append("Predicate applies a function to a column, which can block index use.")
            if "leading_wildcard_like" in tags:
                bottlenecks.append("LIKE predicate starts with a wildcard and is unlikely to use a normal index.")
            if "or_predicate" in tags:
                bottlenecks.append("OR predicate may prevent selective index access.")
            if "join_without_condition" in tags:
                bottlenecks.append("JOIN without an ON condition may create a cartesian join.")
            if "scalar_maxmin_subquery" in tags:
                bottlenecks.append(
                    "SQL uses a scalar MAX/MIN subquery that may be replaceable with ORDER BY ... LIMIT."
                )
            if "nullable_sort_key" in tags:
                bottlenecks.append("SQL sorts on a nullable key without an explicit NULL guard.")

        if not bottlenecks:
            bottlenecks.append("No deterministic bottleneck detected from the available evidence.")
        return unique_preserve_order(bottlenecks)

    def build_cost_snapshot(self, plan_features: dict) -> dict:
        """Return a compact cost/shape snapshot for Controller decisions."""
        plan_ir = plan_features.get("normalized_plan")
        tags = set(plan_features.get("risk_tags") or self.classify_risk_tags(plan_features))
        estimated_costs: list[float] = []
        estimated_rows: list[float] = []
        actual_plan_rows: list[float] = []
        actual_times: list[float] = []
        loops: list[int] = []
        full_scan_tables: list[str] = []
        table_node_count = 0

        if isinstance(plan_ir, PlanIR):
            for node in plan_ir.nodes:
                if node.table:
                    table_node_count += 1
                if node.estimated_cost is not None:
                    estimated_costs.append(float(node.estimated_cost))
                if node.estimated_rows is not None:
                    estimated_rows.append(float(node.estimated_rows))
                if node.actual_rows is not None:
                    actual_plan_rows.append(float(node.actual_rows))
                if node.actual_time_ms is not None:
                    actual_times.append(float(node.actual_time_ms))
                if node.loops is not None:
                    loops.append(int(node.loops))
            full_scan_tables = _full_scan_tables(plan_ir)
        estimated_vs_actual_ratio = _estimate_actual_ratio(
            sum(estimated_rows) if estimated_rows else None,
            sum(actual_plan_rows) if actual_plan_rows else None,
        )

        return {
            "estimated_cost": sum(estimated_costs) if estimated_costs else None,
            "estimated_rows": sum(estimated_rows) if estimated_rows else None,
            "actual_latency_ms": plan_features.get("latency_ms"),
            "actual_rows": plan_features.get("row_count"),
            "actual_plan_rows": sum(actual_plan_rows) if actual_plan_rows else None,
            "actual_time_ms": sum(actual_times) if actual_times else None,
            "loops": sum(loops) if loops else None,
            "full_scan_tables": full_scan_tables,
            "uses_temp_sort": bool({"temp_sort", "filesort"} & tags),
            "uses_temp_group": "temp_group_by" in tags,
            "uses_covering_index": "covering_index" in tags,
            "uses_filesort": "filesort" in tags,
            "uses_temporary": bool({"temp_table", "temp_group_by"} & tags),
            "uses_index_condition": "index_condition_pushdown" in tags,
            "uses_post_filter": "post_filter" in tags,
            "estimated_vs_actual_row_ratio": estimated_vs_actual_ratio,
            "row_estimate_skew": estimated_vs_actual_ratio is not None and estimated_vs_actual_ratio >= 10,
            "join_strategy": "nested_loop_join" if table_node_count > 1 else None,
            "plan_confidence": plan_ir.confidence if isinstance(plan_ir, PlanIR) else None,
        }

    def build_rewrite_hints(self, plan_features: dict) -> list[RewriteHint]:
        """Return rewrite direction hints without producing rewritten SQL."""
        tags = set(plan_features.get("risk_tags") or self.classify_risk_tags(plan_features))
        dbms = str(plan_features.get("dbms") or "")
        full_scan_tables = ", ".join(plan_features.get("cost_snapshot", {}).get("full_scan_tables", []))
        sql = str(plan_features.get("sql") or "")
        sql_structure = plan_features.get("sql_structure")
        schema_stats = plan_features.get("schema_stats")
        plan_ir = plan_features.get("normalized_plan")
        hints: list[RewriteHint] = []
        if (
            isinstance(sql_structure, ParseSqlStructureOutput)
            and isinstance(schema_stats, CollectSchemaStatsOutput)
            and isinstance(plan_ir, PlanIR)
            and _is_index_bound_full_scan(plan_ir, schema_stats, sql_structure)
        ):
            return _unique_rewrite_hints(
                [
                    RewriteHint(
                        strategy="add_index_candidate",
                        target_fragment=full_scan_tables or None,
                        expected_effect="Improve access paths for filtered columns on the scan-driving table.",
                        risk="low",
                        requires_validation=False,
                        dbms_notes=_dbms_note(dbms),
                    ),
                    RewriteHint(
                        strategy="no_rewrite",
                        target_fragment=full_scan_tables or None,
                        expected_effect="SQL rewrite space is exhausted; scanned rows are unlikely to drop without a new index.",
                        risk="low",
                        requires_validation=False,
                        dbms_notes=_dbms_note(dbms),
                    ),
                ]
            )
        if {"large_table_scan", "full_table_scan", "missing_index", "join_without_index"} & tags:
            hints.append(
                RewriteHint(
                    strategy="add_index_candidate",
                    target_fragment=full_scan_tables or None,
                    expected_effect="Improve access paths for filtered or joined columns.",
                    risk="low",
                    requires_validation=True,
                    dbms_notes=_dbms_note(dbms),
                )
            )
            hints.append(
                RewriteHint(
                    strategy="push_down_filter",
                    target_fragment=(
                        f"Selective predicates on scan-driving tables ({full_scan_tables})"
                        if full_scan_tables
                        else "Selective WHERE/JOIN predicates on the scan-driving side"
                    ),
                    expected_effect=(
                        "Reduce Scan Rows by filtering before wide joins, DISTINCT, GROUP BY, or ORDER BY."
                    ),
                    risk="medium",
                    requires_validation=True,
                    dbms_notes=_dbms_note(dbms),
                )
            )
        if _has_redundant_same_table_lookup_join(sql):
            hints.append(
                RewriteHint(
                    strategy="eliminate_redundant_self_join",
                    target_fragment="Same-table alias join / lookup dimension boundary",
                    expected_effect=(
                        "Remove a same-table lookup join by replacing the filtered alias with a direct key lookup on the retained relation."
                    ),
                    risk="medium",
                    requires_validation=True,
                    dbms_notes=_dbms_note(dbms),
                )
            )
        if {"temp_sort", "filesort"} & tags:
            hints.append(
                RewriteHint(
                    strategy="align_order_by_with_index",
                    target_fragment="ORDER BY / LIMIT / top-k fragment",
                    expected_effect=(
                        "Reduce rows entering sort and prefer top-k evaluation on the smallest equivalent input."
                    ),
                    risk="medium",
                    requires_validation=True,
                    dbms_notes=_dbms_note(dbms),
                )
            )
        if "temp_distinct" in tags:
            hints.append(
                RewriteHint(
                    strategy="simplify_join_graph",
                    target_fragment="DISTINCT / join fanout boundary",
                    expected_effect=(
                        "Avoid join amplification so DISTINCT is unnecessary or runs after fewer rows."
                    ),
                    risk="medium",
                    requires_validation=True,
                    dbms_notes=_dbms_note(dbms),
                )
            )
        if "temp_group_by" in tags:
            hints.append(
                RewriteHint(
                    strategy="pre_aggregate_before_join",
                    target_fragment="GROUP BY / pre-join aggregation boundary",
                    expected_effect=(
                        "Aggregate on the narrowest equivalent input so fewer rows flow into joins and grouping."
                    ),
                    risk="high",
                    requires_validation=True,
                    dbms_notes=_dbms_note(dbms),
                )
            )
        if "correlated_subquery" in tags:
            hints.append(
                RewriteHint(
                    strategy="replace_correlated_subquery",
                    target_fragment="correlated subquery",
                    expected_effect="Avoid repeated subquery execution by using a JOIN or CTE.",
                    risk="high",
                    requires_validation=True,
                    dbms_notes=_dbms_note(dbms),
                )
            )
        if "select_star" in tags:
            hints.append(
                RewriteHint(
                    strategy="reduce_select_columns",
                    target_fragment="SELECT *",
                    expected_effect="Remove unnecessary projected columns and reduce row width.",
                    risk="low",
                    requires_validation=True,
                    dbms_notes=_dbms_note(dbms),
                )
            )
        if "scalar_maxmin_subquery" in tags:
            hints.append(
                RewriteHint(
                    strategy="rewrite_scalar_maxmin_subquery",
                    target_fragment="Scalar MAX/MIN / extreme-value fragment",
                    expected_effect=(
                        "Explore top-k rewrite that reduces rows before extreme-value evaluation when tie semantics stay equivalent."
                    ),
                    risk="high",
                    requires_validation=True,
                    dbms_notes=_dbms_note(dbms),
                )
            )
        if "nullable_sort_key" in tags:
            hints.append(
                RewriteHint(
                    strategy="add_null_guard_for_sort_key",
                    target_fragment="ORDER BY nullable column",
                    expected_effect="Avoid NULL values changing top-k or min/max ordering semantics.",
                    risk="medium",
                    requires_validation=True,
                    dbms_notes=_dbms_note(dbms),
                )
            )
        if {"function_on_column", "leading_wildcard_like", "non_sargable_predicate"} & tags:
            hints.append(
                RewriteHint(
                    strategy="avoid_function_on_column",
                    target_fragment="Column-side function / cast / parsing predicate",
                    expected_effect=(
                        "Rewrite predicates into sargable ranges, constant-side transforms, or narrower prefiltered relations."
                    ),
                    risk="medium",
                    requires_validation=True,
                    dbms_notes=_dbms_note(dbms),
                )
            )
        if "or_predicate" in tags:
            hints.append(
                RewriteHint(
                    strategy="rewrite_or_to_union",
                    target_fragment="OR predicate / disjunctive filter",
                    expected_effect=(
                        "Split disjunctive predicates into separate selective access paths when duplicate semantics remain equivalent."
                    ),
                    risk="high",
                    requires_validation=True,
                    dbms_notes=_dbms_note(dbms),
                )
            )
        if {"join_without_condition", "cartesian_join"} & tags:
            hints.append(
                RewriteHint(
                    strategy="simplify_join_graph",
                    target_fragment="JOIN graph",
                    expected_effect="Remove accidental cartesian joins or add missing join predicates.",
                    risk="high",
                    requires_validation=True,
                    dbms_notes=_dbms_note(dbms),
                )
            )
        if not hints:
            hints.append(
                RewriteHint(
                    strategy="no_rewrite",
                    target_fragment=None,
                    expected_effect="No deterministic rewrite direction found.",
                    risk="low",
                    requires_validation=False,
                    dbms_notes=_dbms_note(dbms),
                )
            )
        return _unique_rewrite_hints(hints)

    def decide_optimization_action(
        self,
        *,
        sql_structure: ParseSqlStructureOutput,
        plan_ir: PlanIR,
        schema_stats: CollectSchemaStatsOutput,
        previous_risk_tags: list[str] | None = None,
    ) -> OptimizationDecision:
        """Call the deterministic decision layer for loop control."""
        return decide_optimization_action(
            sql_structure=sql_structure,
            plan_ir=plan_ir,
            schema_stats=schema_stats,
            previous_risk_tags=previous_risk_tags,
        )

    def build_report(
        self,
        sql_version: SQLVersion,
        metrics: ExecutionMetrics,
        plan_features: dict,
    ) -> BottleneckReport:
        """Build the public BottleneckReport artifact."""
        features = dict(plan_features)
        risk_tags = self.classify_risk_tags(features)
        features["risk_tags"] = risk_tags
        bottlenecks = self.detect_bottlenecks(features)
        cost_snapshot = self.build_cost_snapshot(features)
        features["cost_snapshot"] = cost_snapshot
        rewrite_hints = self.build_rewrite_hints(features)
        decision = features.get("optimization_decision")
        explanation_parts = [
            f"Analyzed SQL version {sql_version.version_id}.",
            f"Executable explain: {metrics.executable}.",
            f"Risk tags: {risk_tags or 'none'}.",
        ]
        if isinstance(decision, OptimizationDecision):
            explanation_parts.append(
                f"Decision: {decision.decision} ({decision.next_action}) because {decision.reason}"
            )
        if metrics.error_message:
            explanation_parts.append(f"Explain error: {metrics.error_message}")
        return BottleneckReport(
            sql_version_id=sql_version.version_id,
            bottlenecks=bottlenecks,
            cost_snapshot=cost_snapshot,
            risk_tags=risk_tags,
            rewrite_hints=rewrite_hints,
            explanation=" ".join(explanation_parts),
        )

    # ------------------------------------------------------------------
    # Stage 4: orchestration
    # ------------------------------------------------------------------

    def run_analysis(self, request: AgentRequest) -> dict:
        """Run the full evidence-to-report pipeline and return raw artifacts."""
        tool_calls: list[dict] = []
        warnings: list[str] = []
        sql_version = self.select_sql_version(request)
        db_id, dbms, connection = self.select_db_context(request)
        previous_risk_tags = self.select_previous_risk_tags(request)
        timeout_ms = int(request.constraints.get("timeout_ms", self.default_timeout_ms))

        capabilities = _record_tool_call(
            tool_calls,
            "detect_db_capabilities",
            {"db_id": db_id, "dbms": dbms},
            lambda: self.detect_capabilities(
                db_id=db_id,
                dbms=dbms,
                connection=connection,
            ),
        )
        warnings.extend(capabilities.notes)

        sql_structure = _record_tool_call(
            tool_calls,
            "parse_sql_structure",
            {"dbms": dbms, "sql_version_id": sql_version.version_id},
            lambda: self.parse_structure(sql=sql_version.sql, dbms=dbms),
        )
        warnings.extend(sql_structure.parse_warnings)

        requested_mode = request.constraints.get("explain_mode") or request.input_artifacts.get(
            "explain_mode"
        )
        explain_mode = _select_explain_mode(requested_mode, capabilities)
        started_at = time.perf_counter()
        cached_raw_explain = _cached_explain_plan_from_request(
            request=request,
            sql_version=sql_version,
            db_id=db_id,
            dbms=dbms,
            mode=explain_mode,
        )
        if cached_raw_explain is not None:
            raw_explain = _record_tool_call(
                tool_calls,
                "get_explain_plan_cache",
                {
                    "db_id": db_id,
                    "dbms": dbms,
                    "mode": explain_mode,
                    "sql_version_id": sql_version.version_id,
                },
                lambda: cached_raw_explain,
            )
        else:
            raw_explain = _record_tool_call(
                tool_calls,
                "get_explain_plan",
                {
                    "db_id": db_id,
                    "dbms": dbms,
                    "mode": explain_mode,
                    "timeout_ms": timeout_ms,
                    "sql_version_id": sql_version.version_id,
                },
                lambda: self.get_raw_plan(
                    sql=sql_version.sql,
                    db_id=db_id,
                    dbms=dbms,
                    mode=explain_mode,
                    timeout_ms=timeout_ms,
                    connection=connection,
                ),
            )
        explain_latency_ms = (time.perf_counter() - started_at) * 1000
        warnings.extend(raw_explain.warnings)

        execution_metrics = ExecutionMetrics(
            executable=raw_explain.error is None,
            latency_ms=explain_latency_ms,
            row_count=None,
            explain_plan=str(raw_explain.raw_plan),
            plan_features={
                "dbms": dbms,
                "sql": sql_version.sql,
                "sql_version_id": sql_version.version_id,
                "capabilities": capabilities,
                "sql_structure": sql_structure,
                "raw_explain": raw_explain,
                "explain_mode": raw_explain.mode,
                "warnings": list(warnings),
            },
            error_message=raw_explain.error,
        )
        if raw_explain.error is not None:
            raise RuntimeError(f"Explain plan collection failed: {raw_explain.error}")

        plan_ir = _record_tool_call(
            tool_calls,
            "normalize_plan",
            {"dbms": dbms, "sql_version_id": sql_version.version_id},
            lambda: self.normalize_plan(
                dbms=dbms,
                raw_plan=raw_explain.raw_plan,
                sql_structure=sql_structure,
            ),
        )
        warnings.extend(plan_ir.warnings)

        schema_stats = _record_tool_call(
            tool_calls,
            "collect_schema_stats",
            {
                "db_id": db_id,
                "dbms": dbms,
                "tables": sql_structure.tables,
                "columns": sql_structure.columns,
                "timeout_ms": timeout_ms,
            },
            lambda: self.collect_schema_stats(
                db_id=db_id,
                dbms=dbms,
                sql_structure=sql_structure,
                timeout_ms=timeout_ms,
                connection=connection,
            ),
        )
        warnings.extend(schema_stats.warnings)
        sql_structure = replace(
            sql_structure,
            risky_patterns=_sql_structure_risk_tags(
                sql_structure,
                sql=sql_version.sql,
                dbms=dbms,
                schema_stats=schema_stats,
            ),
        )

        optimization_decision = _record_tool_call(
            tool_calls,
            "decide_optimization_action",
            {
                "sql_version_id": sql_version.version_id,
                "previous_risk_tags": previous_risk_tags,
            },
            lambda: self.decide_optimization_action(
                sql_structure=sql_structure,
                plan_ir=plan_ir,
                schema_stats=schema_stats,
                previous_risk_tags=previous_risk_tags,
            ),
        )

        plan_features = self.extract_plan_features(execution_metrics, dbms)
        plan_features.update(
            {
                "capabilities": capabilities,
                "sql_structure": sql_structure,
                "raw_explain": raw_explain,
                "normalized_plan": plan_ir,
                "schema_stats": schema_stats,
                "optimization_decision": optimization_decision,
                "previous_risk_tags": previous_risk_tags,
                "warnings": list(warnings),
            }
        )
        risk_tags = self.classify_risk_tags(plan_features)
        plan_features["risk_tags"] = risk_tags
        execution_metrics = ExecutionMetrics(
            executable=True,
            latency_ms=execution_metrics.latency_ms,
            row_count=execution_metrics.row_count,
            explain_plan=execution_metrics.explain_plan,
            plan_features=plan_features,
            error_message=None,
        )
        bottleneck_report = self.build_report(sql_version, execution_metrics, plan_features)

        return {
            "sql_version": sql_version,
            "db_id": db_id,
            "dbms": dbms,
            "capabilities": capabilities,
            "sql_structure": sql_structure,
            "raw_explain": raw_explain,
            "normalized_plan": plan_ir,
            "schema_stats": schema_stats,
            "execution_metrics": execution_metrics,
            "plan_features": plan_features,
            "bottleneck_report": bottleneck_report,
            "optimization_decision": optimization_decision,
            "tool_calls": tool_calls,
            "warnings": unique_preserve_order(warnings),
        }

    def run(self, request: AgentRequest) -> AgentResponse:
        """Analyze the current SQL version and return an AgentResponse."""
        try:
            artifacts = self.run_analysis(request)
        except Exception as exc:
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status="error",
                output_artifacts={},
                reasoning_summary="Explain analysis failed before producing a bottleneck report.",
                tool_calls=[],
                errors=[str(exc)],
            )

        decision = artifacts["optimization_decision"]
        report = artifacts["bottleneck_report"]
        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            status="success",
            output_artifacts=artifacts,
            reasoning_summary=(
                f"Analyzed SQL version {report.sql_version_id}. "
                f"Decision: {decision.decision}; next_action: {decision.next_action}. "
                f"Risk tags: {report.risk_tags or 'none'}."
            ),
            tool_calls=artifacts["tool_calls"],
            errors=[],
        )


def _validate_dbms(dbms: str) -> str:
    normalized = str(dbms).strip().lower()
    if normalized not in SUPPORTED_DBMS:
        raise ValueError(f"Unsupported dbms: {dbms}")
    return normalized


def _validate_explain_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in EXPLAIN_MODES:
        raise ValueError(f"Unsupported explain mode: {mode}")
    return normalized


def _select_explain_mode(
    requested_mode: str | None,
    capabilities: DetectDbCapabilitiesOutput,
) -> str:
    mode = (
        _validate_explain_mode(requested_mode)
        if requested_mode is not None
        else capabilities.default_explain_mode
    )
    if mode == "analyze" and not capabilities.supports_explain_analyze:
        return "estimated"
    return mode


def _coerce_sql_version(value: Any) -> SQLVersion | None:
    if isinstance(value, SQLVersion):
        return value
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


def _risk_tags_from_report(report: Any) -> list[str]:
    if isinstance(report, BottleneckReport):
        return unique_preserve_order([str(tag) for tag in report.risk_tags])
    if isinstance(report, dict) and isinstance(report.get("risk_tags"), list):
        return unique_preserve_order([str(tag) for tag in report["risk_tags"]])
    return []


def _cached_explain_plan_from_request(
    *,
    request: AgentRequest,
    sql_version: SQLVersion,
    db_id: str,
    dbms: str,
    mode: str,
) -> GetExplainPlanOutput | None:
    for container in (request.input_artifacts, request.runtime_state):
        raw_cache = (
            container.get("explain_plan_cache")
            or container.get("cached_explain_plan")
            or container.get("explain_plan_caches")
        )
        for cache in _iter_explain_plan_caches(raw_cache):
            if not _cache_matches_sql(cache, sql_version, db_id, dbms):
                continue
            raw_plan = cache.get("raw_plan") or cache.get("raw_explain_plan")
            if raw_plan is None:
                continue
            return GetExplainPlanOutput(
                dbms=dbms,
                mode=str(cache.get("mode") or mode),
                explain_sql=str(cache.get("explain_sql") or ""),
                raw_plan=raw_plan,
                warnings=["Reused Validator cached explain plan."],
            )
    return None


def _iter_explain_plan_caches(raw_cache: Any) -> list[dict]:
    if isinstance(raw_cache, dict):
        return [raw_cache]
    if isinstance(raw_cache, list):
        return [item for item in raw_cache if isinstance(item, dict)]
    return []


def _cache_matches_sql(
    cache: dict,
    sql_version: SQLVersion,
    db_id: str,
    dbms: str,
) -> bool:
    if cache.get("db_id") not in (None, db_id):
        return False
    if str(cache.get("dbms") or dbms).lower() != dbms:
        return False
    cache_version_id = cache.get("sql_version_id")
    if cache_version_id is not None and str(cache_version_id) == sql_version.version_id:
        return True
    cache_sql = cache.get("sql")
    return isinstance(cache_sql, str) and cache_sql.strip() == sql_version.sql.strip()


def _sql_structure_risk_tags(
    sql_structure: ParseSqlStructureOutput,
    *,
    sql: str,
    dbms: str,
    schema_stats: Any,
) -> list[str]:
    tags = [tag for tag in sql_structure.risky_patterns if tag != "select_star"]
    if sql and _has_top_level_select_star(sql, dbms):
        tags.append("select_star")
    if sql and _has_scalar_maxmin_subquery(sql, dbms):
        tags.append("scalar_maxmin_subquery")
    if sql and _has_nullable_sort_key(sql, dbms, schema_stats):
        tags.append("nullable_sort_key")
    return unique_preserve_order(tags)


def _parse_sql(sql: str, dbms: str) -> exp.Expression | None:
    try:
        dialect = _validate_dbms(dbms)
        return sqlglot.parse_one(sql, dialect=dialect)
    except Exception:
        return None


def _has_top_level_select_star(sql: str, dbms: str) -> bool:
    ast = _parse_sql(sql, dbms)
    if ast is None:
        return False
    for select in ast.find_all(exp.Select):
        for expression in select.expressions:
            if isinstance(expression, exp.Star):
                return True
            if isinstance(expression, exp.Column) and isinstance(expression.this, exp.Star):
                return True
    return False


def _has_scalar_maxmin_subquery(sql: str, dbms: str) -> bool:
    ast = _parse_sql(sql, dbms)
    if ast is None:
        return False
    for subquery in ast.find_all(exp.Subquery):
        if not (subquery.find(exp.Max) or subquery.find(exp.Min)):
            continue
        parent = subquery.parent
        while parent is not None:
            if isinstance(parent, exp.Where):
                return True
            parent = parent.parent
    return False


def _has_nullable_sort_key(
    sql: str,
    dbms: str,
    schema_stats: Any,
) -> bool:
    if not isinstance(schema_stats, CollectSchemaStatsOutput):
        return False
    ast = _parse_sql(sql, dbms)
    if ast is None:
        return False
    for order in ast.find_all(exp.Order):
        for ordered in order.expressions:
            for column in ordered.find_all(exp.Column):
                if _column_is_nullable(column, schema_stats):
                    return True
    return False


def _column_is_nullable(
    column: exp.Column,
    schema_stats: CollectSchemaStatsOutput,
) -> bool:
    table = column.table
    name = column.name
    if table:
        column_stats = schema_stats.columns.get(f"{table}.{name}")
        return bool(column_stats and column_stats.nullable)
    matches = [
        column_stats
        for column_stats in schema_stats.columns.values()
        if column_stats.column == name
    ]
    return any(bool(column_stats.nullable) for column_stats in matches)


def _full_scan_tables(plan_ir: PlanIR) -> list[str]:
    tables: list[str] = []
    for node in plan_ir.nodes:
        if node.table and (node.operation == "table_scan" or "full_table_scan" in node.flags):
            tables.append(node.table)
    return unique_preserve_order(tables)


def _row_estimate_skew_ratio(plan_ir: PlanIR) -> float | None:
    estimated_rows = sum(
        float(node.estimated_rows)
        for node in plan_ir.nodes
        if node.estimated_rows is not None
    )
    actual_rows = sum(
        float(node.actual_rows)
        for node in plan_ir.nodes
        if node.actual_rows is not None
    )
    ratio = _estimate_actual_ratio(estimated_rows or None, actual_rows or None)
    if ratio is not None and ratio >= 10:
        return ratio
    return None


def _estimate_actual_ratio(
    estimated_rows: float | None,
    actual_rows: float | None,
) -> float | None:
    if estimated_rows is None or actual_rows is None:
        return None
    if estimated_rows <= 0 or actual_rows <= 0:
        return None
    return max(actual_rows / estimated_rows, estimated_rows / actual_rows)


def _format_tables_with_counts(
    tables: list[str],
    schema_stats: Any,
) -> str:
    formatted: list[str] = []
    for table in tables:
        row_count = None
        if isinstance(schema_stats, CollectSchemaStatsOutput):
            table_stats = schema_stats.tables.get(table)
            row_count = table_stats.row_count if table_stats else None
        if row_count is None:
            formatted.append(table)
        else:
            formatted.append(f"{table} ({row_count} rows)")
    return ", ".join(formatted)


def _dbms_note(dbms: str) -> str | None:
    if dbms == "sqlite":
        return "SQLite plan evidence is based on EXPLAIN QUERY PLAN and has no full cost model."
    if dbms == "mysql":
        return "MySQL candidates should be validated with EXPLAIN FORMAT=JSON or EXPLAIN ANALYZE when allowed."
    return None


def _unique_rewrite_hints(hints: list[RewriteHint]) -> list[RewriteHint]:
    seen: set[tuple[str, str | None]] = set()
    result: list[RewriteHint] = []
    for hint in hints:
        key = (hint.strategy, hint.target_fragment)
        if key in seen:
            continue
        seen.add(key)
        result.append(hint)
    return result


def _has_redundant_same_table_lookup_join(sql: str) -> bool:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return False

    alias_to_table: dict[str, str] = {}
    aliases_by_table: dict[str, set[str]] = {}
    for table in ast.find_all(exp.Table):
        table_name = table.name
        alias = table.alias_or_name
        if not table_name or not alias:
            continue
        alias_to_table[alias] = table_name
        aliases_by_table.setdefault(table_name, set()).add(alias)

    if not any(len(aliases) >= 2 for aliases in aliases_by_table.values()):
        return False

    where_clause = ast.find(exp.Where)
    if where_clause is None or where_clause.this is None:
        return False

    literal_filtered_aliases: set[str] = set()
    for node in where_clause.this.walk():
        if not isinstance(node, exp.EQ):
            continue
        left = node.left
        right = node.right
        for candidate, other in ((left, right), (right, left)):
            if isinstance(candidate, exp.Column) and isinstance(other, exp.Literal):
                if candidate.table:
                    literal_filtered_aliases.add(candidate.table)

    if not literal_filtered_aliases:
        return False

    for join in ast.find_all(exp.Join):
        condition = join.args.get("on")
        if condition is None:
            continue
        for node in condition.walk():
            if not isinstance(node, exp.EQ):
                continue
            left = node.left
            right = node.right
            if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                continue
            if not left.table or not right.table or not left.name or not right.name:
                continue
            left_table = alias_to_table.get(left.table)
            right_table = alias_to_table.get(right.table)
            if left_table is None or right_table is None or left_table != right_table:
                continue
            if left.name != right.name:
                continue
            if left.table in literal_filtered_aliases or right.table in literal_filtered_aliases:
                return True
    return False


def _record_tool_call(
    tool_calls: list[dict],
    tool_name: str,
    input_summary: dict,
    action,
) -> Any:
    started_at = time.perf_counter()
    try:
        output = action()
    except Exception as exc:
        tool_calls.append(
            {
                "tool_name": tool_name,
                "input_summary": input_summary,
                "output_summary": {},
                "latency_ms": (time.perf_counter() - started_at) * 1000,
                "status": "failed",
                "error": str(exc),
            }
        )
        raise
    error = getattr(output, "error", None)
    tool_calls.append(
        {
            "tool_name": tool_name,
            "input_summary": input_summary,
            "output_summary": _summarize_tool_output(output),
            "latency_ms": (time.perf_counter() - started_at) * 1000,
            "status": "failed" if error else "success",
            "error": error,
        }
    )
    return output


def _summarize_tool_output(output: Any) -> dict:
    if isinstance(output, DetectDbCapabilitiesOutput):
        return {
            "dbms": output.dbms,
            "version": output.version,
            "default_explain_mode": output.default_explain_mode,
            "supports_explain_analyze": output.supports_explain_analyze,
            "notes": output.notes,
        }
    if isinstance(output, ParseSqlStructureOutput):
        return {
            "statement_type": output.statement_type,
            "tables": output.tables,
            "column_count": len(output.columns),
            "risky_patterns": output.risky_patterns,
            "parse_warnings": output.parse_warnings,
        }
    if isinstance(output, GetExplainPlanOutput):
        raw_plan = output.raw_plan
        if isinstance(raw_plan, list):
            raw_plan_size = len(raw_plan)
        elif isinstance(raw_plan, dict):
            raw_plan_size = len(raw_plan)
        elif raw_plan is None:
            raw_plan_size = 0
        else:
            raw_plan_size = 1
        return {
            "dbms": output.dbms,
            "mode": output.mode,
            "raw_plan_size": raw_plan_size,
            "warnings": output.warnings,
            "error": output.error,
        }
    if isinstance(output, PlanIR):
        return {
            "dbms": output.dbms,
            "node_count": len(output.nodes),
            "edge_count": len(output.edges),
            "global_flags": output.global_flags,
            "confidence": output.confidence,
            "warnings": output.warnings,
        }
    if isinstance(output, CollectSchemaStatsOutput):
        return {
            "tables": sorted(output.tables),
            "column_count": len(output.columns),
            "index_table_count": len(output.indexes),
            "foreign_key_count": len(output.foreign_keys),
            "warnings": output.warnings,
        }
    if isinstance(output, OptimizationDecision):
        return {
            "decision": output.decision,
            "next_action": output.next_action,
            "should_rewrite": output.should_rewrite,
            "risk_tags": output.risk_tags,
            "confidence": output.confidence,
        }
    return {"type": type(output).__name__}
