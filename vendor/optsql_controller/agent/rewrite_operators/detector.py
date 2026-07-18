"""Structured detection of deterministic rewrite-operator opportunities."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import sqlglot
import sqlglot.expressions as exp

from agent.rewrite_operators.models import OperatorOpportunity
from agent.rewrite_operators.shapes import argmax_aggregate_to_topk_shape
from agent.rewrite_operators.shapes import dimension_key_first_then_fact_probe_shape
from agent.rewrite_operators.shapes import distinct_join_to_semijoin_shape
from agent.rewrite_operators.shapes import distinct_extrema_to_grouped_having_shape
from agent.rewrite_operators.shapes import distinct_top1_to_grouped_extrema_shape
from agent.rewrite_operators.shapes import filter_dimension_before_top1_shape
from agent.rewrite_operators.shapes import grouped_max_top1_before_join_shape
from agent.rewrite_operators.shapes import prefer_summary_table_when_grain_matches_shape
from agent.rewrite_operators.shapes import reanchor_join_driver_shape
from agent.rewrite_operators.shapes import repeated_rescan_to_conditional_agg_shape
from agent.rewrite_operators.shapes import scalar_extrema_ladder_shape
from agent.rewrite_operators.shapes import scalar_extrema_anchor_then_lookup_tail_shape
from agent.rewrite_operators.shapes import symmetric_union_arm_pruning_shape
from agent.rewrite_operators.shapes import top1_anchor_then_lookup_tail_shape
from agent.rewrite_operators.shapes import topk_before_join_shape
from myTypes import BottleneckReport
from myTypes import RewriteHint
from myTypes import VerifiedContextBlueprint

_OPERATOR_TO_HINT_STRATEGY = {
    "projection_pruning": "reduce_select_columns",
    "redundant_distinct_elimination": "eliminate_redundant_distinct",
    "redundant_count_distinct_elimination": "eliminate_redundant_count_distinct",
    "scalar_maxmin_to_topk": "rewrite_scalar_maxmin_subquery",
    "scalar_extrema_anchor_then_lookup_tail": "align_order_by_with_index",
    "date_extraction_to_range": "avoid_function_on_column",
    "like_prefix_to_range": "avoid_function_on_column",
    "repeated_rescan_to_conditional_agg": "pre_aggregate_before_join",
    "dimension_key_first_then_fact_probe": "push_down_filter",
    "reanchor_join_driver": "push_down_filter",
    "prefer_summary_table_when_grain_matches": "push_down_filter",
    "topk_before_join": "align_order_by_with_index",
    "top1_anchor_then_lookup_tail": "align_order_by_with_index",
    "argmax_aggregate_to_topk": "align_order_by_with_index",
    "grouped_max_top1_before_join": "align_order_by_with_index",
    "filter_dimension_before_top1": "push_down_filter",
    "distinct_join_to_semijoin": "simplify_join_graph",
    "distinct_extrema_to_grouped_having": "align_order_by_with_index",
    "distinct_top1_to_grouped_extrema": "align_order_by_with_index",
    "redundant_bridge_join_elimination": "simplify_join_graph",
    "same_key_bridge_join_elimination": "eliminate_same_key_bridge_join",
    "unused_fk_join_elimination": "simplify_join_graph",
    "unused_fk_join_chain_elimination": "simplify_join_graph",
    "symmetric_union_arm_pruning": "simplify_join_graph",
    "same_table_lookup_to_scalar_subquery": "eliminate_redundant_self_join",
}


def detect_operator_opportunities(
    *,
    sql: str,
    report: BottleneckReport,
    blueprint: VerifiedContextBlueprint,
    physical_context: dict,
) -> list[OperatorOpportunity]:
    opportunities: list[OperatorOpportunity] = []
    hints_by_strategy = {hint.strategy: hint for hint in report.rewrite_hints}

    if _has_top_level_select_star(sql):
        opportunities.append(
            _build_opportunity(
                operator_name="projection_pruning",
                hint=hints_by_strategy.get("reduce_select_columns"),
                default_confidence=0.72,
                default_effect="Remove unnecessary projected columns and reduce row width.",
                semantic_risks=[],
            )
        )

    redundant_distinct_shape = _redundant_distinct_elimination_shape(sql, physical_context)
    if redundant_distinct_shape:
        opportunities.append(
            _build_opportunity(
                operator_name="redundant_distinct_elimination",
                hint=None,
                default_confidence=0.96,
                default_effect="Remove redundant DISTINCT when the projected columns already contain a unique key.",
                semantic_risks=["uniqueness proof depends on schema indexes"],
                target_fragment=redundant_distinct_shape.get("target_fragment"),
            )
        )

    redundant_count_distinct_shape = _redundant_count_distinct_elimination_shape(sql, physical_context)
    if redundant_count_distinct_shape:
        counted_column = redundant_count_distinct_shape["counted_column"]
        replacement = (
            "COUNT(*)"
            if redundant_count_distinct_shape.get("can_use_count_star")
            else f"COUNT({counted_column})"
        )
        opportunities.append(
            _build_opportunity(
                operator_name="redundant_count_distinct_elimination",
                hint=None,
                default_confidence=0.97,
                default_effect=(
                    "Remove redundant DISTINCT from COUNT over a unique key while preserving NULL-counting semantics."
                ),
                semantic_risks=["uniqueness proof depends on schema indexes", "count null semantics"],
                target_fragment=f"COUNT(DISTINCT {counted_column}) -> {replacement}",
            )
        )

    scalar_ladder_shape = scalar_extrema_ladder_shape(sql)
    if scalar_ladder_shape or _has_scalar_maxmin_subquery(sql):
        opportunities.append(
            _build_opportunity(
                operator_name="scalar_maxmin_to_topk",
                hint=hints_by_strategy.get("rewrite_scalar_maxmin_subquery"),
                default_confidence=0.82 if scalar_ladder_shape else 0.55,
                default_effect="Replace scalar extreme-value subquery with top-k on the same join graph.",
                semantic_risks=["tie semantics"],
                target_fragment=(
                    scalar_ladder_shape.target_fragment
                    if scalar_ladder_shape is not None
                    else None
                ),
            )
        )

    scalar_tail_shape = scalar_extrema_anchor_then_lookup_tail_shape(sql)
    if scalar_tail_shape and _shape_has_scan_pressure(
        report,
        physical_context,
        *scalar_tail_shape.prefix_tables,
        scalar_tail_shape.tail_table,
    ):
        opportunities.append(
            _build_opportunity(
                operator_name="scalar_extrema_anchor_then_lookup_tail",
                hint=(
                    hints_by_strategy.get("align_order_by_with_index")
                    or hints_by_strategy.get("rewrite_scalar_maxmin_subquery")
                ),
                default_confidence=0.91,
                default_effect="Replace the predecessor scalar extrema filter with one upstream top-1 anchor key, then probe the final lookup tail.",
                semantic_risks=["scalar extrema correlation semantics", "tail lookup uniqueness"],
                target_fragment=scalar_tail_shape.target_fragment,
            )
        )

    date_range_shape = _date_extraction_to_range_shape(sql)
    if date_range_shape and _has_scan_pressure(
        report,
        physical_context,
        date_range_shape.get("table"),
    ):
        opportunities.append(
            _build_opportunity(
                operator_name="date_extraction_to_range",
                hint=hints_by_strategy.get("avoid_function_on_column"),
                default_confidence=0.95,
                default_effect="Rewrite date extraction on the filtered column into a sargable range predicate.",
                semantic_risks=["date string ordering semantics"],
                target_fragment=date_range_shape.get("predicate_sql"),
            )
        )

    like_prefix_shape = _like_prefix_to_range_shape(sql)
    if like_prefix_shape and _has_scan_pressure(
        report,
        physical_context,
        like_prefix_shape.get("table"),
    ):
        opportunities.append(
            _build_opportunity(
                operator_name="like_prefix_to_range",
                hint=hints_by_strategy.get("avoid_function_on_column"),
                default_confidence=0.88,
                default_effect="Rewrite LIKE prefix into a half-open range on the raw column.",
                semantic_risks=["collation-sensitive prefix ordering"],
                target_fragment=like_prefix_shape.get("predicate_sql"),
            )
        )

    repeated_rescan_shape = repeated_rescan_to_conditional_agg_shape(sql)
    if repeated_rescan_shape and _has_scan_pressure(
        report,
        physical_context,
        repeated_rescan_shape.fact_table,
    ):
        opportunities.append(
            _build_opportunity(
                operator_name="repeated_rescan_to_conditional_agg",
                hint=hints_by_strategy.get("pre_aggregate_before_join"),
                default_confidence=0.94,
                default_effect="Collapse repeated grouped rescans into one conditional aggregation pass.",
                semantic_risks=["conditional aggregation null semantics"],
                target_fragment=repeated_rescan_shape.target_fragment,
            )
        )

    key_first_shape = dimension_key_first_then_fact_probe_shape(sql)
    if key_first_shape and _shape_has_scan_pressure(
        report,
        physical_context,
        key_first_shape.dimension_table,
        key_first_shape.fact_table,
    ):
        opportunities.append(
            _build_opportunity(
                operator_name="dimension_key_first_then_fact_probe",
                hint=hints_by_strategy.get("push_down_filter"),
                default_confidence=0.93,
                default_effect="Resolve the selective key first, then probe the fact table by that key.",
                semantic_risks=["key uniqueness / duplicate probe semantics"],
                target_fragment=key_first_shape.target_fragment,
            )
        )

    reanchor_shape = reanchor_join_driver_shape(sql)
    if reanchor_shape and _shape_has_scan_pressure(
        report,
        physical_context,
        reanchor_shape.driver_table,
        reanchor_shape.bridge_table,
        reanchor_shape.fact_table,
    ):
        opportunities.append(
            _build_opportunity(
                operator_name="reanchor_join_driver",
                hint=(
                    hints_by_strategy.get("push_down_filter")
                    or hints_by_strategy.get("align_order_by_with_index")
                    or hints_by_strategy.get("simplify_join_graph")
                ),
                default_confidence=0.95,
                default_effect="Resolve the selective driver key first, then probe the fact table and drop the bridge.",
                semantic_risks=["bridge-elimination equivalence", "top-k tie semantics"],
                target_fragment=reanchor_shape.target_fragment,
            )
        )

    summary_shape = prefer_summary_table_when_grain_matches_shape(sql)
    if summary_shape and _shape_has_scan_pressure(
        report,
        physical_context,
        summary_shape.detail_table,
        summary_shape.summary_table,
    ):
        opportunities.append(
            _build_opportunity(
                operator_name="prefer_summary_table_when_grain_matches",
                hint=hints_by_strategy.get("push_down_filter"),
                default_confidence=0.88,
                default_effect="Use a smaller summary-compatible table instead of the raw detail fact table.",
                semantic_risks=["summary/detail metric compatibility"],
                target_fragment=summary_shape.target_fragment,
            )
        )

    topk_shape = topk_before_join_shape(sql)
    if topk_shape and _has_scan_pressure(
        report,
        physical_context,
        topk_shape.base_table,
    ):
        opportunities.append(
            _build_opportunity(
                operator_name="topk_before_join",
                hint=hints_by_strategy.get("align_order_by_with_index"),
                default_confidence=0.86,
                default_effect="Apply top-k on the scan-driving base table before downstream joins.",
                semantic_risks=["join duplication after top-k"],
                target_fragment=(
                    f"ORDER BY {topk_shape.order_by_column} ... LIMIT 1 on "
                    f"{topk_shape.base_table}"
                ),
            )
        )

    anchor_tail_shape = top1_anchor_then_lookup_tail_shape(sql)
    if anchor_tail_shape and _shape_has_scan_pressure(
        report,
        physical_context,
        *anchor_tail_shape.prefix_tables,
        anchor_tail_shape.tail_table,
    ):
        opportunities.append(
            _build_opportunity(
                operator_name="top1_anchor_then_lookup_tail",
                hint=(
                    hints_by_strategy.get("align_order_by_with_index")
                    or hints_by_strategy.get("push_down_filter")
                ),
                default_confidence=0.92,
                default_effect="Resolve the top-1 anchor key upstream, then probe the final lookup tail by that key.",
                semantic_risks=["tail lookup uniqueness", "top-k tie semantics"],
                target_fragment=anchor_tail_shape.target_fragment,
            )
        )

    aggregate_argmax_shape = argmax_aggregate_to_topk_shape(sql)
    if aggregate_argmax_shape and _has_scan_pressure(
        report,
        physical_context,
        aggregate_argmax_shape.base_table,
    ):
        opportunities.append(
            _build_opportunity(
                operator_name="argmax_aggregate_to_topk",
                hint=hints_by_strategy.get("align_order_by_with_index"),
                default_confidence=0.9,
                default_effect="Replace repeated grouped aggregate scans with one grouped ORDER BY aggregate LIMIT.",
                semantic_risks=["top-k tie semantics"],
                target_fragment=aggregate_argmax_shape.having_sql,
            )
        )

    grouped_shape = grouped_max_top1_before_join_shape(sql)
    if grouped_shape and _shape_has_scan_pressure(
        report,
        physical_context,
        grouped_shape.base_table,
        *([grouped_shape.join_table] if grouped_shape.join_table else []),
    ):
        aggregate_sql = grouped_shape.aggregate_expression.sql(dialect="sqlite")
        group_sql = ", ".join(
            expression.sql(dialect="sqlite")
            for expression in grouped_shape.group_expressions
        )
        opportunities.append(
            _build_opportunity(
                operator_name="grouped_max_top1_before_join",
                hint=hints_by_strategy.get("align_order_by_with_index"),
                default_confidence=0.9,
                default_effect="Pre-aggregate the downstream join table before grouped top-k.",
                semantic_risks=["grouping grain", "top-k tie semantics"],
                target_fragment=f"GROUP BY {group_sql} ORDER BY {aggregate_sql} LIMIT 1",
            )
        )

    semijoin_shape = distinct_join_to_semijoin_shape(sql)
    if semijoin_shape and _shape_has_scan_pressure(
        report,
        physical_context,
        semijoin_shape.base_table,
        *list(semijoin_shape.inner_tables),
    ):
        opportunities.append(
            _build_opportunity(
                operator_name="distinct_join_to_semijoin",
                hint=hints_by_strategy.get("simplify_join_graph"),
                default_confidence=0.91,
                default_effect="Replace fanout-causing join plus DISTINCT with a correlated semi-join.",
                semantic_risks=["predicate correlation semantics"],
                target_fragment="DISTINCT join fanout boundary",
            )
        )

    distinct_top1_shape = distinct_top1_to_grouped_extrema_shape(sql)
    if distinct_top1_shape and _shape_has_scan_pressure(
        report,
        physical_context,
        distinct_top1_shape.base_table,
        distinct_top1_shape.join_table,
    ):
        extrema_fn = "MAX" if distinct_top1_shape.direction == "DESC" else "MIN"
        opportunities.append(
            _build_opportunity(
                operator_name="distinct_top1_to_grouped_extrema",
                hint=hints_by_strategy.get("align_order_by_with_index"),
                default_confidence=0.9,
                default_effect="Replace DISTINCT top-1 over joined rows with grouped extrema over the projected value.",
                semantic_risks=["distinct top-1 tie semantics"],
                target_fragment=(
                    f"SELECT DISTINCT {distinct_top1_shape.projection.sql(dialect='sqlite')} "
                    f"ORDER BY {extrema_fn}({distinct_top1_shape.metric.sql(dialect='sqlite')}) LIMIT 1"
                ),
            )
        )

    distinct_extrema_shape = distinct_extrema_to_grouped_having_shape(sql)
    if distinct_extrema_shape:
        agg_fn = "MAX" if distinct_extrema_shape.direction == "DESC" else "MIN"
        opportunities.append(
            _build_opportunity(
                operator_name="distinct_extrema_to_grouped_having",
                hint=(
                    hints_by_strategy.get("align_order_by_with_index")
                    or hints_by_strategy.get("rewrite_scalar_maxmin_subquery")
                ),
                default_confidence=0.9,
                default_effect="Replace DISTINCT plus scalar extrema equality with grouped HAVING on the same join graph.",
                semantic_risks=["distinct extrema tie semantics"],
                target_fragment=(
                    f"GROUP BY projected values HAVING {agg_fn}("
                    f"{distinct_extrema_shape.outer_metric.sql(dialect='sqlite')}) = extrema"
                ),
            )
        )

    redundant_bridge_shape = _redundant_bridge_join_elimination_shape(
        sql,
        blueprint,
        physical_context,
    )
    if redundant_bridge_shape and _shape_has_scan_pressure(
        report,
        physical_context,
        redundant_bridge_shape.get("left_table"),
        redundant_bridge_shape.get("bridge_table"),
        redundant_bridge_shape.get("right_table"),
    ):
        opportunities.append(
            _build_opportunity(
                operator_name="redundant_bridge_join_elimination",
                hint=hints_by_strategy.get("simplify_join_graph"),
                default_confidence=0.92,
                default_effect="Remove an unused bridge join when a direct narrower join exists.",
                semantic_risks=["direct join equivalence depends on schema topology"],
                target_fragment=redundant_bridge_shape.get("target_fragment"),
            )
        )

    same_key_bridge_shape = _same_key_bridge_join_elimination_shape(sql)
    if same_key_bridge_shape and _shape_has_scan_pressure(
        report,
        physical_context,
        same_key_bridge_shape.get("left_table"),
        same_key_bridge_shape.get("bridge_table"),
        same_key_bridge_shape.get("right_table"),
    ):
        opportunities.append(
            _build_opportunity(
                operator_name="same_key_bridge_join_elimination",
                hint=None,
                default_confidence=0.95,
                default_effect="Remove an unused bridge table that only relays the same join key between two tables.",
                semantic_risks=["bridge relay must not filter or duplicate rows beyond the direct key join"],
                target_fragment=same_key_bridge_shape.get("target_fragment"),
            )
        )

    unused_fk_join_shape = _unused_fk_join_elimination_shape(sql, physical_context)
    if unused_fk_join_shape and _shape_has_scan_pressure(
        report,
        physical_context,
        unused_fk_join_shape.get("base_table"),
        unused_fk_join_shape.get("joined_table"),
    ):
        opportunities.append(
            _build_opportunity(
                operator_name="unused_fk_join_elimination",
                hint=hints_by_strategy.get("simplify_join_graph"),
                default_confidence=0.96,
                default_effect="Remove an unused joined table when a foreign key already guarantees the matching referenced row exists.",
                semantic_risks=["foreign-key guarantee and referenced-key uniqueness must hold"],
                target_fragment=unused_fk_join_shape.get("target_fragment"),
            )
        )

    unused_fk_join_chain_shape = _unused_fk_join_chain_elimination_shape(sql, physical_context)
    if unused_fk_join_chain_shape and _shape_has_scan_pressure(
        report,
        physical_context,
        unused_fk_join_chain_shape.get("base_table"),
        *list(unused_fk_join_chain_shape.get("joined_tables") or ()),
    ):
        opportunities.append(
            _build_opportunity(
                operator_name="unused_fk_join_chain_elimination",
                hint=hints_by_strategy.get("simplify_join_graph"),
                default_confidence=0.97,
                default_effect="Remove a linear chain of unused joined tables when each hop is already guaranteed by foreign-key-to-unique-key existence.",
                semantic_risks=["every join in the chain must be pure existence checking"],
                target_fragment=unused_fk_join_chain_shape.get("target_fragment"),
            )
        )

    symmetric_union_shape = symmetric_union_arm_pruning_shape(sql)
    if symmetric_union_shape and _has_scan_pressure(
        report,
        physical_context,
        symmetric_union_shape.edge_table,
    ):
        opportunities.append(
            _build_opportunity(
                operator_name="symmetric_union_arm_pruning",
                hint=hints_by_strategy.get("simplify_join_graph"),
                default_confidence=0.83,
                default_effect="Drop the symmetric duplicate edge arm and keep one canonical orientation.",
                semantic_risks=["schema-specific canonical edge orientation"],
                target_fragment=symmetric_union_shape.target_fragment,
            )
        )

    filter_dim_shape = filter_dimension_before_top1_shape(sql)
    if filter_dim_shape and _shape_has_scan_pressure(
        report,
        physical_context,
        filter_dim_shape.fact_table,
        filter_dim_shape.dim_table,
    ):
        opportunities.append(
            _build_opportunity(
                operator_name="filter_dimension_before_top1",
                hint=hints_by_strategy.get("push_down_filter")
                or hints_by_strategy.get("align_order_by_with_index"),
                default_confidence=0.93,
                default_effect="Filter the dimension side first, then perform the top-k fact lookup.",
                semantic_risks=["dimension filter pushdown semantics"],
                target_fragment=(
                    f"{filter_dim_shape.dim_table} filtered before join to "
                    f"{filter_dim_shape.fact_table}"
                ),
            )
        )

    if _has_same_table_literal_lookup_join(sql):
        opportunities.append(
            _build_opportunity(
                operator_name="same_table_lookup_to_scalar_subquery",
                hint=hints_by_strategy.get("eliminate_redundant_self_join"),
                default_confidence=0.92,
                default_effect="Replace same-table literal lookup join with a direct key subquery.",
                semantic_risks=["scalar vs set lookup semantics"],
            )
        )

    return sorted(opportunities, key=lambda item: item.confidence, reverse=True)


def _build_opportunity(
    *,
    operator_name: str,
    hint: RewriteHint | None,
    default_confidence: float,
    default_effect: str,
    semantic_risks: list[str],
    target_fragment: str | None = None,
) -> OperatorOpportunity:
    return OperatorOpportunity(
        operator_name=operator_name,
        hint_strategy=(
            hint.strategy
            if hint is not None
            else _OPERATOR_TO_HINT_STRATEGY.get(operator_name, operator_name)
        ),
        matched=True,
        confidence=default_confidence,
        target_fragment=hint.target_fragment if hint is not None else target_fragment,
        expected_effect=hint.expected_effect if hint is not None else default_effect,
        semantic_risks=list(semantic_risks),
        requires_validation=hint.requires_validation if hint is not None else True,
        dbms_notes=hint.dbms_notes if hint is not None else None,
    )


def _has_scan_pressure(
    report: BottleneckReport,
    physical_context: dict,
    table: str | None,
) -> bool:
    if not table:
        return False
    cost_snapshot = physical_context.get("cost_snapshot") or {}
    full_scan_tables = {str(name) for name in (cost_snapshot.get("full_scan_tables") or [])}
    if table in full_scan_tables:
        return True
    risk_tags = {str(tag) for tag in (report.risk_tags or [])}
    if {"full_table_scan", "temp_sort", "temp_group"} & risk_tags:
        return True
    row_counts = physical_context.get("table_row_counts") or {}
    row_count = row_counts.get(table)
    return isinstance(row_count, int) and row_count >= 1000


def _shape_has_scan_pressure(
    report: BottleneckReport,
    physical_context: dict,
    *tables: str | None,
) -> bool:
    present_tables = [table for table in tables if table]
    if not present_tables:
        return False
    return any(_has_scan_pressure(report, physical_context, table) for table in present_tables)


def _has_top_level_select_star(sql: str) -> bool:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return False
    select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
    return bool(select and any(_is_star_expression(expr) for expr in select.expressions))


def _is_star_expression(expression: exp.Expression) -> bool:
    return isinstance(expression, exp.Star) or (
        isinstance(expression, exp.Column) and isinstance(expression.this, exp.Star)
    )


def _has_scalar_maxmin_subquery(sql: str) -> bool:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return False
    for subquery in ast.find_all(exp.Subquery):
        for func in subquery.find_all(exp.AggFunc):
            if func.key.upper() in {"MAX", "MIN"}:
                return True
    return False


def _has_same_table_literal_lookup_join(sql: str) -> bool:
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
            if isinstance(candidate, exp.Column) and isinstance(other, exp.Literal) and candidate.table:
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
            if left.name == right.name and (
                left.table in literal_filtered_aliases or right.table in literal_filtered_aliases
            ):
                return True
    return False


def _date_extraction_to_range_shape(sql: str) -> dict[str, Any] | None:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return None
    inferred_table = _single_from_table_name(ast)
    where_clause = ast.find(exp.Where)
    scope_expr = where_clause.this if where_clause is not None else None
    for node in ast.walk():
        shape = _date_range_match_shape(
            node,
            inferred_table=inferred_table,
            scope_expr=scope_expr,
        )
        if shape is not None:
            return shape
    return None


def _date_range_match_shape(
    node: exp.Expression,
    *,
    inferred_table: str | None,
    scope_expr: exp.Expression | None = None,
) -> dict[str, Any] | None:
    month_shape = _month_bucket_match_shape(
        node,
        inferred_table=inferred_table,
        scope_expr=scope_expr,
    )
    if month_shape is not None:
        return month_shape
    if isinstance(node, exp.Between):
        low = node.args.get("low")
        high = node.args.get("high")
        if not isinstance(low, exp.Literal) or not isinstance(high, exp.Literal):
            return None
        low_shape = _extract_date_bucket_function(node.this, literal_text=str(low.this))
        high_shape = _extract_date_bucket_function(node.this, literal_text=str(high.this))
        if low_shape is None or high_shape is None:
            return None
        return {
            "table": low_shape["column"].table or inferred_table,
            "predicate_sql": node.sql(dialect="sqlite"),
        }
    if not isinstance(node, (exp.EQ, exp.LT, exp.LTE, exp.GT, exp.GTE)):
        return None
    right = getattr(node, "right", None)
    if not isinstance(right, exp.Literal):
        return None
    left = getattr(node, "left", None)
    if isinstance(left, exp.Expression):
        extracted = _extract_date_bucket_function(left, literal_text=str(right.this))
        if extracted is not None:
            return {
                "table": extracted["column"].table or inferred_table,
                "predicate_sql": node.sql(dialect="sqlite"),
            }
        arithmetic = _year_difference_predicate_shape(left, literal_text=str(right.this))
        if arithmetic is not None:
            return {
                "table": arithmetic["column"].table or inferred_table,
                "predicate_sql": node.sql(dialect="sqlite"),
            }
    return None


def _month_bucket_match_shape(
    node: exp.Expression,
    *,
    inferred_table: str | None,
    scope_expr: exp.Expression | None,
) -> dict[str, Any] | None:
    if scope_expr is None:
        return None
    column: exp.Column | None = None
    months: list[int] | None = None
    if isinstance(node, exp.EQ) and isinstance(node.right, exp.Literal):
        column = _extract_month_bucket_column(node.left)
        months = _parse_month_literals([str(node.right.this)])
    elif isinstance(node, exp.In):
        column = _extract_month_bucket_column(node.this)
        literals = [str(item.this) for item in node.expressions if isinstance(item, exp.Literal)]
        if len(literals) != len(node.expressions):
            return None
        months = _parse_month_literals(literals)
    if column is None or not months:
        return None
    if _infer_fixed_year_for_column(scope_expr, column) is None:
        return None
    return {
        "table": column.table or inferred_table,
        "predicate_sql": node.sql(dialect="sqlite"),
    }


def _redundant_distinct_elimination_shape(
    sql: str,
    physical_context: dict,
) -> dict[str, Any] | None:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return None
    select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
    if select is None or select.args.get("distinct") is None:
        return None
    if any(select.args.get(key) is not None for key in ("group", "having", "with_", "qualify", "order", "limit")):
        return None
    if any(isinstance(node, exp.SetOperation) for node in ast.walk()):
        return None
    if any(True for _ in select.find_all(exp.AggFunc)):
        return None
    from_expr = select.args.get("from_")
    if from_expr is None or not isinstance(from_expr.this, exp.Table):
        return None
    base_table = from_expr.this
    base_alias = base_table.alias_or_name
    joins = select.args.get("joins") or []
    if len(joins) > 1:
        return None
    projected_columns: list[str] = []
    projected_columns_by_table: dict[str, set[str]] = {}
    allowed_tables = {base_alias, base_table.name}
    join_table: exp.Table | None = None
    join_alias: str | None = None
    if joins:
        join = joins[0]
        if join.side and str(join.side).upper() != "INNER":
            return None
        if not isinstance(join.this, exp.Table):
            return None
        join_table = join.this
        join_alias = join_table.alias_or_name
        allowed_tables.update({join_alias, join_table.name})
    for expression in select.expressions:
        if isinstance(expression, exp.Alias):
            expression = expression.this
        if not isinstance(expression, exp.Column):
            return None
        if expression.table and expression.table not in allowed_tables:
            return None
        projected_columns.append(expression.name)
        resolved_table = (
            base_table.name
            if expression.table in {"", None, base_alias, base_table.name}
            else join_table.name if join_table is not None else None
        )
        if resolved_table is None:
            return None
        projected_columns_by_table.setdefault(resolved_table, set()).add(expression.name)
    if not projected_columns:
        return None

    def _covered_unique_index(table_name: str, columns: set[str]) -> tuple[str, tuple[str, ...]] | None:
        indexes = physical_context.get("indexes", {}).get(table_name) or []
        for index in indexes:
            if not index.get("unique"):
                continue
            index_columns = tuple(str(column) for column in (index.get("columns") or []) if str(column))
            if index_columns and set(index_columns) <= columns:
                return str(index.get("name") or ""), index_columns
        return None

    projected_index = _covered_unique_index(base_table.name, projected_columns_by_table.get(base_table.name, set()))
    if not joins:
        if projected_index is None:
            return None
        return {
            "scope": "single_table",
            "table": base_table.name,
            "projected_columns": tuple(projected_columns),
            "projected_columns_by_table": {
                table_name: tuple(sorted(columns))
                for table_name, columns in projected_columns_by_table.items()
            },
            "unique_index_name": projected_index[0],
            "unique_index_columns": projected_index[1],
            "target_fragment": "SELECT DISTINCT",
        }

    assert join_table is not None
    assert join_alias is not None
    on_clause = joins[0].args.get("on")
    if on_clause is None:
        return None

    def _join_proof_for(
        preserved_table_name: str,
        preserved_aliases: set[str],
        other_table_name: str,
        other_aliases: set[str],
    ) -> dict[str, Any] | None:
        preserved_projection = projected_columns_by_table.get(preserved_table_name, set())
        preserved_unique_index = _covered_unique_index(preserved_table_name, preserved_projection)
        if preserved_unique_index is None:
            return None
        other_join_columns: set[str] = set()
        for predicate in _flatten_and_conditions(on_clause):
            if not isinstance(predicate, exp.EQ):
                return None
            left = predicate.left
            right = predicate.right
            if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                return None
            if left.table in preserved_aliases and right.table in other_aliases:
                other_join_columns.add(right.name)
                continue
            if right.table in preserved_aliases and left.table in other_aliases:
                other_join_columns.add(left.name)
                continue
            return None
        other_unique_index = _covered_unique_index(other_table_name, other_join_columns)
        if other_unique_index is None:
            return None
        return {
            "scope": "single_join",
            "table": base_table.name,
            "join_table": join_table.name,
            "preserved_table": preserved_table_name,
            "projected_columns": tuple(projected_columns),
            "projected_columns_by_table": {
                table_name: tuple(sorted(columns))
                for table_name, columns in projected_columns_by_table.items()
            },
            "preserved_unique_index_name": preserved_unique_index[0],
            "preserved_unique_index_columns": preserved_unique_index[1],
            "joined_unique_index_name": other_unique_index[0],
            "joined_unique_index_columns": other_unique_index[1],
            "target_fragment": "SELECT DISTINCT",
        }

    return _join_proof_for(
        base_table.name,
        {base_alias, base_table.name},
        join_table.name,
        {join_alias, join_table.name},
    ) or _join_proof_for(
        join_table.name,
        {join_alias, join_table.name},
        base_table.name,
        {base_alias, base_table.name},
    )
    return None


def _redundant_count_distinct_elimination_shape(
    sql: str,
    physical_context: dict,
) -> dict[str, Any] | None:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return None
    select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
    if select is None:
        return None
    if any(select.args.get(key) is not None for key in ("distinct", "group", "having", "with_", "qualify")):
        return None
    if any(isinstance(node, exp.SetOperation) for node in ast.walk()):
        return None
    if any(True for _ in select.find_all(exp.Join)):
        return None
    if len(select.expressions) != 1:
        return None

    expression = select.expressions[0]
    alias = expression.alias if isinstance(expression, exp.Alias) else None
    if isinstance(expression, exp.Alias):
        expression = expression.this
    if not isinstance(expression, exp.Count) or not isinstance(expression.this, exp.Distinct):
        return None
    distinct_expr = expression.this
    if len(distinct_expr.expressions) != 1:
        return None
    counted_expr = distinct_expr.expressions[0]
    if not isinstance(counted_expr, exp.Column):
        return None

    from_expr = select.args.get("from_")
    if from_expr is None or not isinstance(from_expr.this, exp.Table):
        return None
    base_table = from_expr.this
    base_alias = base_table.alias_or_name
    if counted_expr.table and counted_expr.table not in {base_alias, base_table.name}:
        return None

    indexes = physical_context.get("indexes", {}).get(base_table.name) or []
    column_meta = ((physical_context.get("columns") or {}).get(base_table.name) or {}).get(counted_expr.name) or {}
    for index in indexes:
        if not index.get("unique"):
            continue
        index_columns = [str(column) for column in (index.get("columns") or []) if str(column)]
        if index_columns != [counted_expr.name]:
            continue
        return {
            "table": base_table.name,
            "counted_column": counted_expr.name,
            "counted_column_sql": counted_expr.sql(dialect="sqlite"),
            "unique_index_name": str(index.get("name") or ""),
            "can_use_count_star": bool(column_meta.get("not_null")),
            "target_fragment": expression.sql(dialect="sqlite") if alias is None else alias,
        }
    return None


def _like_prefix_to_range_shape(sql: str) -> dict[str, Any] | None:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return None
    inferred_table = _single_from_table_name(ast)
    where_clause = ast.find(exp.Where)
    if where_clause is None:
        return None
    for node in where_clause.this.walk():
        if not isinstance(node, exp.Like):
            continue
        escape = node.args.get("escape")
        if escape is not None:
            continue
        if not isinstance(node.this, exp.Column) or not isinstance(node.expression, exp.Literal):
            continue
        pattern = str(node.expression.this)
        if not _is_safe_like_prefix_pattern(pattern):
            continue
        return {
            "table": node.this.table or inferred_table,
            "predicate_sql": node.sql(dialect="sqlite"),
        }
    return None


def _redundant_bridge_join_elimination_shape(
    sql: str,
    blueprint: VerifiedContextBlueprint,
    physical_context: dict,
) -> dict[str, Any] | None:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return None
    select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
    if select is None:
        return None
    joins = select.args.get("joins") or []
    from_expr = select.args.get("from_")
    if (
        from_expr is None
        or not isinstance(from_expr.this, exp.Table)
        or len(joins) != 2
        or not isinstance(joins[0].this, exp.Table)
        or not isinstance(joins[1].this, exp.Table)
    ):
        return None
    left = from_expr.this
    bridge = joins[0].this
    right = joins[1].this
    bridge_tables = {bridge.alias_or_name, bridge.name}
    for expression in list(select.expressions) + [select.args.get("order"), select.args.get("group"), select.args.get("having")]:
        if expression is None:
            continue
        for column in expression.find_all(exp.Column):
            if column.table in bridge_tables:
                return None
    direct_edge = _direct_blueprint_join_edge(blueprint, left.name, right.name)
    where_clause = select.args.get("where")
    if where_clause is not None:
        for predicate in _flatten_and_conditions(where_clause.this):
            referenced_tables = {
                str(column.table)
                for column in predicate.find_all(exp.Column)
                if column.table is not None
            }
            if not (referenced_tables & bridge_tables):
                continue
            if direct_edge is None or not _bridge_predicate_is_replaceable(
                predicate=predicate,
                left=left,
                bridge=bridge,
                right=right,
                joins=joins,
                direct_edge=direct_edge,
            ):
                return None
    if direct_edge is None and not _direct_join_evidence_from_schema(
        left_table=left.name,
        bridge_table=bridge.name,
        right_table=right.name,
        physical_context=physical_context,
    ):
        return None
    row_counts = physical_context.get("table_row_counts") or {}
    bridge_count = row_counts.get(bridge.name)
    left_count = row_counts.get(left.name)
    right_count = row_counts.get(right.name)
    if all(isinstance(value, int) for value in (bridge_count, left_count, right_count)):
        if bridge_count < max(left_count, right_count):
            return None
    return {
        "left_table": left.name,
        "bridge_table": bridge.name,
        "right_table": right.name,
        "target_fragment": bridge.name,
    }


def _same_key_bridge_join_elimination_shape(sql: str) -> Any | None:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return None
    select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
    if select is None:
        return None
    if any(select.args.get(key) is not None for key in ("group", "having", "with_", "qualify")):
        return None
    from_expr = select.args.get("from_")
    joins = select.args.get("joins") or []
    if (
        from_expr is None
        or not isinstance(from_expr.this, exp.Table)
        or len(joins) != 2
        or any(not isinstance(join.this, exp.Table) for join in joins)
    ):
        return None
    if any(join.side and str(join.side).upper() != "INNER" for join in joins):
        return None

    left = from_expr.this
    bridge = joins[0].this
    right = joins[1].this
    join_one = joins[0].args.get("on")
    join_two = joins[1].args.get("on")
    if not isinstance(join_one, exp.EQ) or not isinstance(join_two, exp.EQ):
        return None
    left_bridge = _column_pair_with_table(join_one, left.alias_or_name, bridge.alias_or_name)
    bridge_right = _column_pair_with_table(join_two, bridge.alias_or_name, right.alias_or_name)
    if left_bridge is None or bridge_right is None:
        return None
    left_col, bridge_col_one = left_bridge
    bridge_col_two, right_col = bridge_right
    if bridge_col_one.name != bridge_col_two.name:
        return None

    bridge_alias = bridge.alias_or_name
    for expression in select.expressions:
        if any(column.table == bridge_alias for column in expression.find_all(exp.Column)):
            return None
    for clause_name in ("where", "order", "limit"):
        clause = select.args.get(clause_name)
        if clause is None:
            continue
        for column in clause.find_all(exp.Column):
            if column.table == bridge_alias:
                return None

    return {
        "left_table": left.name,
        "left_alias": left.alias_or_name,
        "left_column": left_col.name,
        "bridge_table": bridge.name,
        "bridge_alias": bridge.alias_or_name,
        "bridge_column": bridge_col_one.name,
        "right_table": right.name,
        "right_alias": right.alias_or_name,
        "right_column": right_col.name,
        "target_fragment": bridge.name,
    }


def _unused_fk_join_elimination_shape(
    sql: str,
    physical_context: dict,
) -> dict[str, Any] | None:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return None
    select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
    if select is None:
        return None
    if any(select.args.get(key) is not None for key in ("group", "having", "with_", "qualify")):
        return None
    if any(isinstance(node, exp.SetOperation) for node in ast.walk()):
        return None
    from_expr = select.args.get("from_")
    joins = select.args.get("joins") or []
    if (
        from_expr is None
        or not isinstance(from_expr.this, exp.Table)
        or len(joins) != 1
        or not isinstance(joins[0].this, exp.Table)
    ):
        return None
    join = joins[0]
    if join.side and str(join.side).upper() != "INNER":
        return None

    base = from_expr.this
    joined = join.this
    base_aliases = {base.alias_or_name, base.name}
    joined_aliases = {joined.alias_or_name, joined.name}

    for expression in select.expressions:
        if any(column.table in joined_aliases for column in expression.find_all(exp.Column)):
            return None
    for clause_name in ("where", "order", "limit", "distinct"):
        clause = select.args.get(clause_name)
        if clause is None:
            continue
        for column in clause.find_all(exp.Column):
            if column.table in joined_aliases:
                return None

    on_clause = join.args.get("on")
    if on_clause is None:
        return None
    base_join_columns: set[str] = set()
    joined_join_columns: set[str] = set()
    for predicate in _flatten_and_conditions(on_clause):
        if not isinstance(predicate, exp.EQ):
            return None
        left = predicate.left
        right = predicate.right
        if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
            return None
        if left.table in base_aliases and right.table in joined_aliases:
            base_join_columns.add(left.name)
            joined_join_columns.add(right.name)
            continue
        if right.table in base_aliases and left.table in joined_aliases:
            base_join_columns.add(right.name)
            joined_join_columns.add(left.name)
            continue
        return None
    if len(base_join_columns) != 1 or len(joined_join_columns) != 1:
        return None

    foreign_keys = physical_context.get("foreign_keys") or []
    matching_fk = next(
        (
            fk for fk in foreign_keys
            if str(fk.get("from_table")) == base.name
            and str(fk.get("to_table")) == joined.name
            and str(fk.get("from_column")) in base_join_columns
            and str(fk.get("to_column")) in joined_join_columns
        ),
        None,
    )
    if matching_fk is None:
        return None

    joined_indexes = physical_context.get("indexes", {}).get(joined.name) or []
    joined_unique_ok = any(
        index.get("unique")
        and [str(column) for column in (index.get("columns") or []) if str(column)] == [str(matching_fk.get("to_column"))]
        for index in joined_indexes
    )
    if not joined_unique_ok:
        return None

    return {
        "base_table": base.name,
        "base_alias": base.alias_or_name,
        "joined_table": joined.name,
        "joined_alias": joined.alias_or_name,
        "base_join_column": str(matching_fk.get("from_column")),
        "joined_join_column": str(matching_fk.get("to_column")),
        "target_fragment": joined.name,
    }


def _unused_fk_join_chain_elimination_shape(
    sql: str,
    physical_context: dict,
) -> dict[str, Any] | None:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return None
    select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
    if select is None:
        return None
    if any(select.args.get(key) is not None for key in ("group", "having", "with_", "qualify")):
        return None
    if any(isinstance(node, exp.SetOperation) for node in ast.walk()):
        return None
    from_expr = select.args.get("from_")
    joins = select.args.get("joins") or []
    if (
        from_expr is None
        or not isinstance(from_expr.this, exp.Table)
        or len(joins) < 2
        or any(not isinstance(join.this, exp.Table) for join in joins)
    ):
        return None
    if any(join.side and str(join.side).upper() != "INNER" for join in joins):
        return None

    base = from_expr.this
    if any(_is_star_expression(expression) for expression in select.expressions):
        return None

    joined_tables = [join.this for join in joins]
    joined_aliases = {table.alias_or_name for table in joined_tables} | {table.name for table in joined_tables}
    base_aliases = {base.alias_or_name, base.name}

    for expression in select.expressions:
        if not _expression_uses_only_tables(expression, base_aliases):
            return None
    for clause_name in ("where", "order", "limit", "distinct"):
        clause = select.args.get(clause_name)
        if clause is None:
            continue
        for column in clause.find_all(exp.Column):
            if column.table in joined_aliases:
                return None

    foreign_keys = physical_context.get("foreign_keys") or []
    indexes_by_table = physical_context.get("indexes", {}) or {}
    current_table = base
    current_aliases = {base.alias_or_name, base.name}
    joined_names: list[str] = []
    join_steps: list[dict[str, str]] = []
    for join in joins:
        next_table = join.this
        next_aliases = {next_table.alias_or_name, next_table.name}
        on_clause = join.args.get("on")
        if on_clause is None:
            return None
        current_join_columns: set[str] = set()
        next_join_columns: set[str] = set()
        for predicate in _flatten_and_conditions(on_clause):
            if not isinstance(predicate, exp.EQ):
                return None
            left = predicate.left
            right = predicate.right
            if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                return None
            if left.table in current_aliases and right.table in next_aliases:
                current_join_columns.add(left.name)
                next_join_columns.add(right.name)
                continue
            if right.table in current_aliases and left.table in next_aliases:
                current_join_columns.add(right.name)
                next_join_columns.add(left.name)
                continue
            return None
        if len(current_join_columns) != 1 or len(next_join_columns) != 1:
            return None
        current_col = next(iter(current_join_columns))
        next_col = next(iter(next_join_columns))
        matching_fk = next(
            (
                fk for fk in foreign_keys
                if str(fk.get("from_table")) == current_table.name
                and str(fk.get("to_table")) == next_table.name
                and str(fk.get("from_column")) == current_col
                and str(fk.get("to_column")) == next_col
            ),
            None,
        )
        if matching_fk is None:
            return None
        if not any(
            index.get("unique")
            and [str(column) for column in (index.get("columns") or []) if str(column)] == [next_col]
            for index in (indexes_by_table.get(next_table.name) or [])
        ):
            return None
        joined_names.append(next_table.name)
        join_steps.append(
            {
                "from_table": current_table.name,
                "from_column": current_col,
                "to_table": next_table.name,
                "to_column": next_col,
            }
        )
        current_table = next_table
        current_aliases = next_aliases

    return {
        "base_table": base.name,
        "base_alias": base.alias_or_name,
        "joined_tables": tuple(joined_names),
        "join_steps": tuple(join_steps),
        "target_fragment": " -> ".join([base.name, *joined_names]),
    }


def _is_iso_month_literal(format_text: str, literal_text: str) -> bool:
    if format_text != "%Y-%m":
        return False
    if len(literal_text) != 7 or literal_text[4] != "-":
        return False
    year, month = literal_text.split("-", 1)
    return year.isdigit() and month.isdigit() and 1 <= int(month) <= 12


def _strftime_range_bounds(format_text: str, literal_text: str) -> tuple[str, str] | None:
    if format_text == "%Y" and len(literal_text) == 4 and literal_text.isdigit():
        return (f"{int(literal_text):04d}-01-01", f"{int(literal_text) + 1:04d}-01-01")
    if _is_iso_month_literal(format_text, literal_text):
        year_text, month_text = literal_text.split("-", 1)
        year = int(year_text)
        month = int(month_text)
        next_year = year + 1 if month == 12 else year
        next_month = 1 if month == 12 else month + 1
        return (f"{year:04d}-{month:02d}", f"{next_year:04d}-{next_month:02d}")
    if format_text == "%Y-%m-%d":
        return _date_wrapper_range_bounds(literal_text)
    return None


def _substring_range_bounds(length: int, literal_text: str) -> tuple[str, str] | None:
    if length == 4 and len(literal_text) == 4 and literal_text.isdigit():
        year = int(literal_text)
        return (f"{year:04d}", f"{year + 1:04d}")
    if length == 7 and _is_iso_month_literal("%Y-%m", literal_text):
        year_text, month_text = literal_text.split("-", 1)
        year = int(year_text)
        month = int(month_text)
        next_year = year + 1 if month == 12 else year
        next_month = 1 if month == 12 else month + 1
        return (f"{year:04d}-{month:02d}", f"{next_year:04d}-{next_month:02d}")
    if length == 6 and len(literal_text) == 6 and literal_text.isdigit():
        year = int(literal_text[:4])
        month = int(literal_text[4:])
        if not 1 <= month <= 12:
            return None
        next_year = year + 1 if month == 12 else year
        next_month = 1 if month == 12 else month + 1
        return (f"{year:04d}{month:02d}", f"{next_year:04d}{next_month:02d}")
    if length == 10:
        return _date_wrapper_range_bounds(literal_text)
    if length == 8 and len(literal_text) == 8 and literal_text.isdigit():
        iso_text = f"{literal_text[:4]}-{literal_text[4:6]}-{literal_text[6:8]}"
        bounds = _date_wrapper_range_bounds(iso_text)
        if bounds is None:
            return None
        return (literal_text, bounds[1].replace("-", ""))
    return None


def _date_wrapper_range_bounds(literal_text: str) -> tuple[str, str] | None:
    try:
        parsed = date.fromisoformat(literal_text)
    except ValueError:
        return None
    next_day = parsed + timedelta(days=1)
    return (parsed.isoformat(), next_day.isoformat())


def _extract_date_bucket_function(
    expr: exp.Expression,
    *,
    literal_text: str,
) -> dict[str, Any] | None:
    fmt: exp.Literal | None = None
    column: exp.Column | None = None
    bounds: tuple[str, str] | None = None
    if isinstance(expr, exp.Cast):
        inner = expr.this
        if isinstance(inner, exp.TimeToStr):
            format_arg = inner.args.get("format")
            if isinstance(format_arg, exp.Literal):
                fmt = format_arg
            source = inner.this
            source_inner = source.this if hasattr(source, "this") else source
            if isinstance(source_inner, exp.Column):
                column = source_inner
    elif isinstance(expr, exp.Anonymous) and expr.name.upper() == "STRFTIME":
        args = list(expr.expressions)
        if len(args) == 2 and isinstance(args[0], exp.Literal) and isinstance(args[1], exp.Column):
            fmt = args[0]
            column = args[1]
    elif isinstance(expr, exp.TimeToStr):
        format_arg = expr.args.get("format")
        if isinstance(format_arg, exp.Literal):
            fmt = format_arg
        source = expr.this
        if isinstance(source, exp.Cast) and isinstance(source.this, exp.Column):
            column = source.this
        elif hasattr(source, "this") and isinstance(source.this, exp.DPipe):
            synthesized = _extract_synthesized_yearmonth_source(
                source.this,
                format_text=str(format_arg.this) if isinstance(format_arg, exp.Literal) else None,
                literal_text=literal_text,
            )
            if synthesized is not None:
                return synthesized
        elif hasattr(source, "this") and isinstance(source.this, exp.Column):
            column = source.this
        elif isinstance(source, exp.Column):
            column = source
    elif isinstance(expr, exp.Substring):
        start = expr.args.get("start")
        length = expr.args.get("length")
        if (
            isinstance(expr.this, exp.Column)
            and isinstance(start, exp.Literal)
            and isinstance(length, exp.Literal)
            and str(start.this) == "1"
        ):
            column = expr.this
            bounds = _substring_range_bounds(int(str(length.this)), literal_text)
    elif isinstance(expr, exp.Date) and isinstance(expr.this, exp.Column):
        column = expr.this
        bounds = _date_wrapper_range_bounds(literal_text)
    if fmt is not None and column is not None:
        bounds = _strftime_range_bounds(str(fmt.this), literal_text)
    if column is None or bounds is None:
        return None
    return {"column": column, "range_start": bounds[0], "range_end": bounds[1]}


def _extract_month_bucket_column(expr: exp.Expression) -> exp.Column | None:
    if isinstance(expr, exp.TimeToStr):
        format_arg = expr.args.get("format")
        if not isinstance(format_arg, exp.Literal) or str(format_arg.this) != "%m":
            return None
        source = expr.this
        source_inner = source.this if hasattr(source, "this") else source
        if isinstance(source_inner, exp.Column):
            return source_inner
        if isinstance(source, exp.Column):
            return source
    return None


def _parse_month_literals(values: list[str]) -> list[int] | None:
    months: list[int] = []
    for value in values:
        if len(value) != 2 or not value.isdigit():
            return None
        month = int(value)
        if not 1 <= month <= 12:
            return None
        months.append(month)
    return sorted(set(months))


def _infer_fixed_year_for_column(scope_expr: exp.Expression, column: exp.Column) -> int | None:
    extracted_year = _infer_fixed_year_from_extracted_predicates(scope_expr, column)
    if extracted_year is not None:
        return extracted_year
    return _infer_fixed_year_from_raw_range(scope_expr, column)


def _infer_fixed_year_from_extracted_predicates(
    scope_expr: exp.Expression,
    column: exp.Column,
) -> int | None:
    target = column.sql(dialect="sqlite")
    for node in scope_expr.walk():
        if isinstance(node, exp.EQ) and isinstance(node.right, exp.Literal):
            extracted = _extract_year_bucket_column(node.left)
            if extracted and extracted.sql(dialect="sqlite") == target:
                year_text = str(node.right.this)
                if len(year_text) == 4 and year_text.isdigit():
                    return int(year_text)
        elif isinstance(node, exp.Between):
            low = node.args.get("low")
            high = node.args.get("high")
            extracted = _extract_year_bucket_column(node.this)
            if (
                extracted
                and extracted.sql(dialect="sqlite") == target
                and isinstance(low, exp.Literal)
                and isinstance(high, exp.Literal)
            ):
                low_text = str(low.this)
                high_text = str(high.this)
                if (
                    len(low_text) == 4
                    and len(high_text) == 4
                    and low_text.isdigit()
                    and high_text.isdigit()
                    and low_text == high_text
                ):
                    return int(low_text)
    return None


def _extract_year_bucket_column(expr: exp.Expression) -> exp.Column | None:
    if isinstance(expr, exp.Cast):
        expr = expr.this
    if isinstance(expr, exp.TimeToStr):
        format_arg = expr.args.get("format")
        if not isinstance(format_arg, exp.Literal) or str(format_arg.this) != "%Y":
            return None
        source = expr.this
        source_inner = source.this if hasattr(source, "this") else source
        if isinstance(source_inner, exp.Column):
            return source_inner
        if isinstance(source, exp.Column):
            return source
    return None


def _infer_fixed_year_from_raw_range(scope_expr: exp.Expression, column: exp.Column) -> int | None:
    target = column.sql(dialect="sqlite")
    lower_year: int | None = None
    upper_year: int | None = None
    between_year: int | None = None
    for node in scope_expr.walk():
        if isinstance(node, (exp.GTE, exp.GT, exp.LT, exp.LTE)):
            left = getattr(node, "left", None)
            right = getattr(node, "right", None)
            if not isinstance(left, exp.Column) or not isinstance(right, exp.Literal):
                continue
            if left.sql(dialect="sqlite") != target:
                continue
            literal_text = str(right.this)
            if isinstance(node, (exp.GTE, exp.GT)) and literal_text.endswith("-01-01"):
                year = _parse_year_start_text(literal_text)
                if year is not None:
                    lower_year = year
            elif isinstance(node, (exp.LT, exp.LTE)) and literal_text.endswith("-01-01"):
                year = _parse_year_start_text(literal_text)
                if year is not None:
                    upper_year = year - 1
        elif isinstance(node, exp.Between):
            low = node.args.get("low")
            high = node.args.get("high")
            if (
                isinstance(node.this, exp.Column)
                and node.this.sql(dialect="sqlite") == target
                and isinstance(low, exp.Literal)
                and isinstance(high, exp.Literal)
            ):
                low_year = _parse_year_start_text(str(low.this))
                high_year = _parse_year_end_text(str(high.this))
                if low_year is not None and high_year is not None and low_year == high_year:
                    between_year = low_year
    if between_year is not None:
        return between_year
    if lower_year is not None and upper_year is not None and lower_year == upper_year:
        return lower_year
    return None


def _parse_year_start_text(text: str) -> int | None:
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        return None
    if parsed.month == 1 and parsed.day == 1:
        return parsed.year
    return None


def _parse_year_end_text(text: str) -> int | None:
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        return None
    if parsed.month == 12 and parsed.day == 31:
        return parsed.year
    return None


def _extract_synthesized_yearmonth_source(
    expr: exp.Expression,
    *,
    format_text: str | None,
    literal_text: str,
) -> dict[str, Any] | None:
    if format_text not in {"%Y"}:
        return None
    shape = _synthesized_yearmonth_concat_shape(expr)
    if shape is None:
        return None
    if len(literal_text) != 4 or not literal_text.isdigit():
        return None
    year = int(literal_text)
    return {
        "column": shape["column"],
        "range_start": f"{year:04d}01",
        "range_end": f"{year + 1:04d}01",
    }


def _synthesized_yearmonth_concat_shape(expr: exp.Expression) -> dict[str, Any] | None:
    parts = _flatten_dpipe(expr)
    if len(parts) != 4:
        return None
    year_part, dash1, month_part, day_suffix = parts
    if not (
        isinstance(dash1, exp.Literal)
        and str(dash1.this) == "-"
        and isinstance(day_suffix, exp.Literal)
        and str(day_suffix.this) == "-01"
    ):
        return None
    year_col = _prefix_substring_shape(year_part, start="1", length="4")
    month_col = _prefix_substring_shape(month_part, start="5", length="2")
    if year_col is None or month_col is None:
        return None
    if year_col.sql(dialect="sqlite") != month_col.sql(dialect="sqlite"):
        return None
    return {"column": year_col}


def _flatten_dpipe(expr: exp.Expression) -> list[exp.Expression]:
    if isinstance(expr, exp.DPipe):
        return _flatten_dpipe(expr.left) + _flatten_dpipe(expr.right)
    return [expr]


def _prefix_substring_shape(
    expr: exp.Expression,
    *,
    start: str,
    length: str,
) -> exp.Column | None:
    if not isinstance(expr, exp.Substring):
        return None
    start_expr = expr.args.get("start")
    length_expr = expr.args.get("length")
    if not (
        isinstance(expr.this, exp.Column)
        and isinstance(start_expr, exp.Literal)
        and isinstance(length_expr, exp.Literal)
        and str(start_expr.this) == start
        and str(length_expr.this) == length
    ):
        return None
    return expr.this


def _year_difference_predicate_shape(expr: exp.Expression, *, literal_text: str) -> dict[str, Any] | None:
    if not literal_text.lstrip("-").isdigit() or not isinstance(expr, exp.Sub):
        return None
    if _current_year_now_expr(expr.left) is None:
        return None
    column = _year_cast_column_expr(expr.right)
    if column is None:
        return None
    return {"column": column}


def _current_year_now_expr(expr: exp.Expression) -> str | None:
    if not isinstance(expr, exp.Cast):
        return None
    inner = expr.this
    if not isinstance(inner, exp.TimeToStr):
        return None
    format_arg = inner.args.get("format")
    if not isinstance(format_arg, exp.Literal) or str(format_arg.this) != "%Y":
        return None
    source = inner.this
    source_inner = source.this if hasattr(source, "this") else source
    if not isinstance(source_inner, exp.Literal):
        return None
    if str(source_inner.this).lower() != "now":
        return None
    return "now"


def _year_cast_column_expr(expr: exp.Expression) -> exp.Column | None:
    if not isinstance(expr, exp.Cast):
        return None
    inner = expr.this
    if not isinstance(inner, exp.TimeToStr):
        return None
    format_arg = inner.args.get("format")
    if not isinstance(format_arg, exp.Literal) or str(format_arg.this) != "%Y":
        return None
    source = inner.this
    source_inner = source.this if hasattr(source, "this") else source
    if isinstance(source_inner, exp.Column):
        return source_inner
    if isinstance(source, exp.Column):
        return source
    return None


def _is_safe_like_prefix_pattern(pattern: str) -> bool:
    if not pattern.endswith("%") or pattern.count("%") != 1:
        return False
    if "_" in pattern:
        return False
    prefix = pattern[:-1]
    if not prefix:
        return False
    return all(ord(ch) < 128 and (ch.isalnum() or ch in "-_./:") for ch in prefix)


def _single_from_table_name(ast: exp.Expression) -> str | None:
    select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
    if select is None:
        return None
    from_expr = select.args.get("from_")
    if from_expr is None or from_expr.this is None or not isinstance(from_expr.this, exp.Table):
        return None
    return from_expr.this.name


def _flatten_and_conditions(expression: exp.Expression) -> list[exp.Expression]:
    if isinstance(expression, exp.And):
        return _flatten_and_conditions(expression.left) + _flatten_and_conditions(expression.right)
    return [expression]


def _expression_uses_only_tables(expression: exp.Expression, tables: set[str]) -> bool:
    referenced = {
        str(column.table)
        for column in expression.find_all(exp.Column)
        if column.table is not None
    }
    return referenced <= tables


def _column_pair_with_table(
    expression: exp.Expression,
    left_table: str,
    right_table: str,
) -> tuple[exp.Column, exp.Column] | None:
    if not isinstance(expression, exp.EQ):
        return None
    left = expression.left
    right = expression.right
    if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
        return None
    if left.table == left_table and right.table == right_table:
        return (left, right)
    if left.table == right_table and right.table == left_table:
        return (right, left)
    return None


def _direct_blueprint_join_edge(
    blueprint: VerifiedContextBlueprint,
    left_table: str,
    right_table: str,
) -> Any | None:
    for edge in blueprint.join_topology.edges:
        if {edge.source_table, edge.target_table} == {left_table, right_table}:
            return edge
    return None


def _direct_join_evidence_from_schema(
    *,
    left_table: str,
    bridge_table: str,
    right_table: str,
    physical_context: dict,
) -> bool:
    foreign_keys = physical_context.get("foreign_keys") or []
    indexes = physical_context.get("indexes") or {}
    direct_fk = any(
        {str(fk.get("from_table")), str(fk.get("to_table"))} == {left_table, right_table}
        for fk in foreign_keys
    )
    if direct_fk:
        return True
    left_bridge = [
        fk for fk in foreign_keys if {str(fk.get("from_table")), str(fk.get("to_table"))} == {left_table, bridge_table}
    ]
    bridge_right = [
        fk for fk in foreign_keys if {str(fk.get("from_table")), str(fk.get("to_table"))} == {bridge_table, right_table}
    ]
    if not left_bridge or not bridge_right:
        return False
    right_unique_columns = {
        tuple(index.get("columns") or [])
        for index in indexes.get(right_table, [])
        if index.get("unique")
    }
    left_unique_columns = {
        tuple(index.get("columns") or [])
        for index in indexes.get(left_table, [])
        if index.get("unique")
    }
    shared_bridge_keys = {
        str(fk.get("from_column") or fk.get("to_column"))
        for fk in [*left_bridge, *bridge_right]
    }
    return any((column,) in right_unique_columns or (column,) in left_unique_columns for column in shared_bridge_keys)


def _bridge_predicate_is_replaceable(
    *,
    predicate: exp.Expression,
    left: exp.Table,
    bridge: exp.Table,
    right: exp.Table,
    joins: list[exp.Join],
    direct_edge: Any,
) -> bool:
    bridge_aliases = {bridge.alias_or_name, bridge.name}
    bridge_col: exp.Column | None = None
    other: exp.Expression | None = None
    if isinstance(predicate, (exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
        if isinstance(predicate.left, exp.Column) and predicate.left.table in bridge_aliases and isinstance(predicate.right, exp.Literal):
            bridge_col = predicate.left
            other = predicate.right
        elif isinstance(predicate.right, exp.Column) and predicate.right.table in bridge_aliases and isinstance(predicate.left, exp.Literal):
            bridge_col = predicate.right
            other = predicate.left
    elif isinstance(predicate, exp.In):
        if isinstance(predicate.this, exp.Column) and predicate.this.table in bridge_aliases:
            if _in_expression_has_only_literals(predicate):
                bridge_col = predicate.this
                other = exp.Tuple(expressions=[expr.copy() for expr in predicate.args.get("expressions") or []])
    elif isinstance(predicate, exp.Is):
        if isinstance(predicate.left, exp.Column) and predicate.left.table in bridge_aliases and isinstance(predicate.right, exp.Null):
            bridge_col = predicate.left
            other = predicate.right
    elif isinstance(predicate, exp.Not) and isinstance(predicate.this, exp.Is):
        inner = predicate.this
        if isinstance(inner.left, exp.Column) and inner.left.table in bridge_aliases and isinstance(inner.right, exp.Null):
            bridge_col = inner.left
            other = exp.Not(this=exp.Null())
    if bridge_col is None or other is None:
        return False
    return _bridge_filter_replacement_column_name(
        bridge_column=bridge_col.name,
        left=left,
        bridge=bridge,
        right=right,
        joins=joins,
        direct_edge=direct_edge,
    ) is not None


def _bridge_filter_replacement_column_name(
    *,
    bridge_column: str,
    left: exp.Table,
    bridge: exp.Table,
    right: exp.Table,
    joins: list[exp.Join],
    direct_edge: Any,
) -> tuple[str, str] | None:
    left_aliases = {left.alias_or_name, left.name}
    right_aliases = {right.alias_or_name, right.name}
    bridge_aliases = {bridge.alias_or_name, bridge.name}
    mapped_to_left: str | None = None
    mapped_to_right: str | None = None
    for join in joins:
        on_clause = join.args.get("on")
        if on_clause is None:
            continue
        for predicate in _flatten_and_conditions(on_clause):
            if not isinstance(predicate, exp.EQ):
                continue
            if not isinstance(predicate.left, exp.Column) or not isinstance(predicate.right, exp.Column):
                continue
            for bridge_side, other_side in ((predicate.left, predicate.right), (predicate.right, predicate.left)):
                if bridge_side.table not in bridge_aliases or bridge_side.name != bridge_column:
                    continue
                if other_side.table in left_aliases:
                    mapped_to_left = other_side.name
                if other_side.table in right_aliases:
                    mapped_to_right = other_side.name
    if direct_edge.source_table == left.name and direct_edge.source_column == (mapped_to_left or direct_edge.source_column):
        if direct_edge.target_table == right.name:
            return (right.alias_or_name, direct_edge.target_column)
    if direct_edge.target_table == left.name and direct_edge.target_column == (mapped_to_left or direct_edge.target_column):
        if direct_edge.source_table == right.name:
            return (right.alias_or_name, direct_edge.source_column)
    if direct_edge.source_table == right.name and direct_edge.source_column == (mapped_to_right or direct_edge.source_column):
        if direct_edge.target_table == left.name:
            return (left.alias_or_name, direct_edge.target_column)
    if direct_edge.target_table == right.name and direct_edge.target_column == (mapped_to_right or direct_edge.target_column):
        if direct_edge.source_table == left.name:
            return (left.alias_or_name, direct_edge.source_column)
    return None


def _is_safe_conditional_aggregate_argument(expression: exp.Expression | None) -> bool:
    if expression is None:
        return False
    for node in expression.walk():
        if isinstance(node, (exp.AggFunc, exp.Window, exp.Subquery, exp.Exists)):
            return False
    return True


def _in_expression_has_only_literals(predicate: exp.In) -> bool:
    expressions = predicate.args.get("expressions") or []
    return bool(expressions) and all(isinstance(item, exp.Literal) for item in expressions)
