"""Deterministic decision layer for Explain Analyser.

This layer consumes normalized evidence from the tool chain and decides whether
the current SQL should enter SQL Rewriter. It is intentionally conservative:
only clearly low-risk plans are marked as skip_optimization so the Controller
can return or validate the current SQL directly.
"""

from __future__ import annotations

from agent.explainAnalyserAgent.utils.common import unique_preserve_order
from agent.explainAnalyserAgent.utils.models import CollectSchemaStatsOutput
from agent.explainAnalyserAgent.utils.models import OptimizationDecision
from agent.explainAnalyserAgent.utils.models import ParseSqlStructureOutput
from agent.explainAnalyserAgent.utils.models import PlanIR
from agent.explainAnalyserAgent.utils.models import PlanNode


SMALL_TABLE_ROW_THRESHOLD = 1_000
LARGE_TABLE_ROW_THRESHOLD = 100_000

HIGH_RISK_PLAN_FLAGS = {
    "temp_sort",
    "temp_group_by",
    "temp_distinct",
    "filesort",
    "correlated_subquery",
    "materialized_subquery",
}

HIGH_RISK_STATIC_PATTERNS = {
    "correlated_subquery",
    "implicit_cross_join",
    "join_without_condition",
    "scalar_maxmin_subquery",
    "nullable_sort_key",
    "function_on_column",
    "leading_wildcard_like",
    "non_sargable_predicate",
}

MEDIUM_RISK_STATIC_PATTERNS = {
    "or_predicate",
    "not_in_nullable",
    "select_star",
    "implicit_cast",
}

SELECTIVE_INDEX_OPERATIONS = {
    "index_lookup",
    "covering_index_lookup",
}


def decide_optimization_action(
    *,
    sql_structure: ParseSqlStructureOutput,
    plan_ir: PlanIR,
    schema_stats: CollectSchemaStatsOutput,
    previous_risk_tags: list[str] | None = None,
) -> OptimizationDecision:
    """Decide whether to call SQL Rewriter or stop the optimization loop early."""
    previous_risk_tags = previous_risk_tags or []
    plan_flags = set(plan_ir.global_flags)
    static_patterns = set(sql_structure.risky_patterns)
    evidence: list[str] = []

    hard_plan_risks = sorted(plan_flags & HIGH_RISK_PLAN_FLAGS)
    if hard_plan_risks:
        return OptimizationDecision(
            should_rewrite=True,
            decision="rewrite",
            confidence=0.88,
            reason="Execution plan contains high-risk physical operators.",
            evidence=[f"plan_flags={hard_plan_risks}"],
            risk_tags=hard_plan_risks,
            next_action="call_rewriter",
        )

    hard_static_risks = sorted(static_patterns & HIGH_RISK_STATIC_PATTERNS)
    if hard_static_risks:
        return OptimizationDecision(
            should_rewrite=True,
            decision="rewrite",
            confidence=0.82,
            reason="SQL structure contains rewriteable high-risk patterns.",
            evidence=[f"static_patterns={hard_static_risks}"],
            risk_tags=hard_static_risks,
            next_action="call_rewriter",
        )

    if _is_index_bound_full_scan(plan_ir, schema_stats, sql_structure):
        full_scan_tables = _full_scan_table_names(plan_ir)
        return OptimizationDecision(
            should_rewrite=False,
            decision="skip_optimization",
            confidence=0.9,
            reason=(
                "Full scan appears index-bound on a structurally simple query; "
                "SQL rewrite is unlikely to reduce scanned rows without adding an index."
            ),
            evidence=[
                f"full_scan_tables={full_scan_tables}",
                "rewrite_space=none_without_new_index",
            ],
            risk_tags=["full_table_scan"],
            next_action="return_current_sql",
        )

    large_scan_tables = _large_full_scan_tables(plan_ir, schema_stats)
    if large_scan_tables:
        return OptimizationDecision(
            should_rewrite=True,
            decision="rewrite",
            confidence=0.84,
            reason="Plan scans large tables without selective index access.",
            evidence=[f"large_full_scan_tables={large_scan_tables}"],
            risk_tags=["large_table_scan"],
            next_action="call_rewriter",
        )

    unknown_scan_tables = _unknown_full_scan_tables(plan_ir, schema_stats)
    if unknown_scan_tables and _has_filter_or_join(sql_structure):
        return OptimizationDecision(
            should_rewrite=True,
            decision="rewrite",
            confidence=0.62,
            reason="Plan contains full scans with unknown table sizes on a filtered or joined query.",
            evidence=[f"unknown_full_scan_tables={unknown_scan_tables}"],
            risk_tags=["full_table_scan"],
            next_action="call_rewriter",
        )

    medium_static_risks = sorted(static_patterns & MEDIUM_RISK_STATIC_PATTERNS)
    if medium_static_risks and _touches_large_or_unknown_tables(sql_structure, schema_stats):
        return OptimizationDecision(
            should_rewrite=True,
            decision="rewrite",
            confidence=0.7,
            reason="SQL has medium-risk patterns on large or unknown-size tables.",
            evidence=[f"static_patterns={medium_static_risks}"],
            risk_tags=medium_static_risks,
            next_action="call_rewriter",
        )

    if previous_risk_tags and set(previous_risk_tags) == plan_flags:
        return OptimizationDecision(
            should_rewrite=False,
            decision="need_validation_only",
            confidence=0.58,
            reason="No new risk tags were found compared with the previous analysis.",
            evidence=[f"previous_risk_tags={sorted(previous_risk_tags)}"],
            risk_tags=sorted(plan_flags),
            next_action="validate_current_sql",
        )

    if _only_small_table_scans(plan_ir, schema_stats):
        tables = [node.table for node in plan_ir.nodes if node.table]
        return OptimizationDecision(
            should_rewrite=False,
            decision="skip_optimization",
            confidence=0.82,
            reason="Only small-table scans were detected; SQL rewrite is unlikely to help.",
            evidence=[f"tables={tables}"],
            risk_tags=[],
            next_action="return_current_sql",
        )

    if _uses_selective_index_access(plan_ir) and not static_patterns:
        return OptimizationDecision(
            should_rewrite=False,
            decision="skip_optimization",
            confidence=0.8,
            reason="Plan already uses selective index access and no static SQL risks were detected.",
            evidence=_index_access_evidence(plan_ir),
            risk_tags=[],
            next_action="return_current_sql",
        )

    if not plan_flags and not static_patterns:
        return OptimizationDecision(
            should_rewrite=False,
            decision="skip_optimization",
            confidence=0.68,
            reason="No plan risks or static rewrite risks were detected.",
            evidence=["plan_flags=[]", "static_patterns=[]"],
            risk_tags=[],
            next_action="return_current_sql",
        )

    evidence.extend([f"plan_flags={sorted(plan_flags)}", f"static_patterns={sorted(static_patterns)}"])
    return OptimizationDecision(
        should_rewrite=False,
        decision="need_validation_only",
        confidence=0.55,
        reason="No deterministic rewrite opportunity was found, but evidence is not strong enough to skip validation.",
        evidence=evidence,
        risk_tags=unique_preserve_order(sorted(plan_flags | static_patterns)),
        next_action="validate_current_sql",
    )


def _large_full_scan_tables(
    plan_ir: PlanIR,
    schema_stats: CollectSchemaStatsOutput,
) -> list[str]:
    result: list[str] = []
    for node in _full_scan_nodes(plan_ir):
        if not node.table:
            continue
        table_stats = schema_stats.tables.get(node.table)
        if table_stats and table_stats.row_count is not None:
            if table_stats.row_count >= LARGE_TABLE_ROW_THRESHOLD:
                result.append(node.table)
    return unique_preserve_order(result)


def _unknown_full_scan_tables(
    plan_ir: PlanIR,
    schema_stats: CollectSchemaStatsOutput,
) -> list[str]:
    result: list[str] = []
    for node in _full_scan_nodes(plan_ir):
        if not node.table:
            continue
        table_stats = schema_stats.tables.get(node.table)
        if table_stats is None or table_stats.row_count is None:
            result.append(node.table)
    return unique_preserve_order(result)


def _only_small_table_scans(
    plan_ir: PlanIR,
    schema_stats: CollectSchemaStatsOutput,
) -> bool:
    table_nodes = [node for node in plan_ir.nodes if node.table]
    if not table_nodes:
        return False
    for node in table_nodes:
        if node.operation not in {"table_scan", "index_scan", "index_lookup", "covering_index_lookup"}:
            return False
        table_stats = schema_stats.tables.get(node.table or "")
        if table_stats is None or table_stats.row_count is None:
            return False
        if table_stats.row_count > SMALL_TABLE_ROW_THRESHOLD:
            return False
    return True


def _uses_selective_index_access(plan_ir: PlanIR) -> bool:
    table_nodes = [node for node in plan_ir.nodes if node.table]
    if not table_nodes:
        return False
    return all(node.operation in SELECTIVE_INDEX_OPERATIONS for node in table_nodes)


def _index_access_evidence(plan_ir: PlanIR) -> list[str]:
    evidence: list[str] = []
    for node in plan_ir.nodes:
        if node.table:
            evidence.append(
                f"{node.table}: operation={node.operation}, index={node.index}, access={node.access_type}"
            )
    return evidence


def _full_scan_nodes(plan_ir: PlanIR) -> list[PlanNode]:
    return [
        node
        for node in plan_ir.nodes
        if node.operation == "table_scan" or "full_table_scan" in node.flags
    ]


def _full_scan_table_names(plan_ir: PlanIR) -> list[str]:
    return unique_preserve_order(
        [node.table for node in _full_scan_nodes(plan_ir) if node.table]
    )


def _is_index_bound_full_scan(
    plan_ir: PlanIR,
    schema_stats: CollectSchemaStatsOutput,
    sql_structure: ParseSqlStructureOutput,
) -> bool:
    full_scan_tables = _full_scan_table_names(plan_ir)
    if len(full_scan_tables) != 1:
        return False
    if len(sql_structure.tables) != 1 or sql_structure.tables[0] != full_scan_tables[0]:
        return False
    if sql_structure.joins or sql_structure.subqueries:
        return False
    if sql_structure.group_by or sql_structure.order_by or sql_structure.limit is not None:
        return False
    if sql_structure.has_distinct or sql_structure.has_union or sql_structure.has_window_functions:
        return False
    if sql_structure.has_select_star or not sql_structure.predicates:
        return False
    if sql_structure.risky_patterns:
        return False
    if set(plan_ir.global_flags) & HIGH_RISK_PLAN_FLAGS:
        return False

    table = full_scan_tables[0]
    indexes = schema_stats.indexes.get(table) or []
    if indexes:
        return False

    table_stats = schema_stats.tables.get(table)
    if table_stats is None or table_stats.row_count is None:
        return False
    if table_stats.row_count < SMALL_TABLE_ROW_THRESHOLD:
        return False
    return True


def _has_filter_or_join(sql_structure: ParseSqlStructureOutput) -> bool:
    return bool(sql_structure.predicates or sql_structure.joins)


def _touches_large_or_unknown_tables(
    sql_structure: ParseSqlStructureOutput,
    schema_stats: CollectSchemaStatsOutput,
) -> bool:
    for table in sql_structure.tables:
        table_stats = schema_stats.tables.get(table)
        if table_stats is None or table_stats.row_count is None:
            return True
        if table_stats.row_count >= LARGE_TABLE_ROW_THRESHOLD:
            return True
    return False
