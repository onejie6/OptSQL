"""Shared structural matchers for deterministic rewrite operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sqlglot
import sqlglot.expressions as exp


@dataclass(frozen=True)
class SummaryTableSubstitution:
    """Explicitly registered summary/detail substitutions."""

    substitution_id: str
    detail_table: str
    summary_table: str
    detail_metric_column: str
    summary_metric_column: str
    join_key_columns: tuple[str, ...]
    required_join_tables: frozenset[str]
    optional_join_tables: frozenset[str]
    allowed_detail_columns: frozenset[str]
    aggregate_function: str
    target_fragment: str

    @property
    def column_mapping(self) -> dict[str, str]:
        mapping = {column: column for column in self.join_key_columns}
        mapping[self.detail_metric_column] = self.summary_metric_column
        return mapping

    @property
    def target_allowed_columns(self) -> frozenset[str]:
        return frozenset(self.column_mapping.values())


@dataclass(frozen=True)
class ReanchorJoinDriverMatch:
    driver_table: str
    driver_alias: str
    driver_key: str
    bridge_table: str
    bridge_alias: str
    bridge_key: str
    fact_table: str
    fact_alias: str
    fact_key: str
    order: exp.Order
    limit: exp.Limit
    projections: tuple[exp.Expression, ...]
    driver_predicates: tuple[exp.Expression, ...]
    target_fragment: str


@dataclass(frozen=True)
class DimensionKeyFirstThenFactProbeMatch:
    dimension_table: str
    dimension_alias: str
    fact_table: str
    fact_alias: str
    dimension_key: str
    fact_key: str
    order: exp.Order
    limit: exp.Limit
    projections: tuple[exp.Expression, ...]
    target_fragment: str


@dataclass(frozen=True)
class PreferSummaryTableWhenGrainMatchesMatch:
    substitution_id: str
    substitution: SummaryTableSubstitution
    detail_table: str
    detail_alias: str
    summary_table: str
    summary_metric_column: str
    column_mapping: dict[str, str]
    target_allowed_columns: frozenset[str]
    mode: str
    target_fragment: str


@dataclass(frozen=True)
class ScalarExtremaOrderSpec:
    column: exp.Column
    direction: str


@dataclass(frozen=True)
class ScalarExtremaLadderMatch:
    base_predicates: tuple[exp.Expression, ...]
    ladder_predicate_norms: tuple[str, ...]
    order_specs: tuple[ScalarExtremaOrderSpec, ...]
    target_fragment: str


@dataclass(frozen=True)
class SymmetricUnionArmPruningMatch:
    shape_type: str
    edge_table: str
    target_sql: str
    target_fragment: str
    canonical_predicate: exp.Expression | None = None
    canonical_select: exp.Select | None = None


@dataclass(frozen=True)
class _SymmetricConnectedArm:
    edge_alias: str
    literals: tuple[str, str]


@dataclass(frozen=True)
class TopkBeforeJoinMatch:
    join_count: int
    base_table: str
    base_alias: str
    order_by_column: str
    base_columns: tuple[str, ...]
    where_clause: exp.Where | None


@dataclass(frozen=True)
class Top1AnchorThenLookupTailMatch:
    prefix_tables: tuple[str, ...]
    tail_table: str
    tail_alias: str
    predecessor_table: str
    predecessor_alias: str
    predecessor_tail_key: str
    tail_key: str
    projections: tuple[exp.Expression, ...]
    where_clause: exp.Where | None
    order: exp.Order
    limit: exp.Limit
    target_fragment: str


@dataclass(frozen=True)
class DistinctTop1ToGroupedExtremaMatch:
    base_table: str
    base_alias: str
    join_table: str
    join_alias: str
    projection: exp.Column
    metric: exp.Column
    direction: str
    where_predicates: tuple[exp.Expression, ...]
    limit_expr: exp.Expression


@dataclass(frozen=True)
class ScalarExtremaAnchorThenLookupTailMatch:
    prefix_tables: tuple[str, ...]
    tail_table: str
    tail_alias: str
    predecessor_table: str
    predecessor_alias: str
    predecessor_tail_key: str
    tail_key: str
    metric_column: str
    direction: str
    projections: tuple[exp.Expression, ...]
    outer_predicates: tuple[exp.Expression, ...]
    target_fragment: str


@dataclass(frozen=True)
class DistinctExtremaToGroupedHavingMatch:
    projections: tuple[exp.Expression, ...]
    outer_predicates: tuple[exp.Expression, ...]
    outer_metric: exp.Column
    inner_metric: exp.Column
    direction: str
    inner_select: exp.Select
    target_fragment: str


@dataclass(frozen=True)
class GroupedAggregateJoinHop:
    left_table: str
    left_alias: str
    left_column: str
    right_table: str
    right_alias: str
    right_column: str


@dataclass(frozen=True)
class GroupedMaxTop1BeforeJoinMatch:
    join_count: int
    base_table: str
    base_alias: str
    join_hops: tuple[GroupedAggregateJoinHop, ...]
    projection_expression: exp.Expression
    group_expressions: tuple[exp.Expression, ...]
    join_table: str | None
    join_alias: str | None
    base_join_key: str | None
    join_key: str | None
    aggregate_expression: exp.Expression
    aggregate_function: str
    where_clause: exp.Where | None
    limit_expr: exp.Expression
    ordered_expr: exp.Ordered


@dataclass(frozen=True)
class FilterDimensionBeforeTop1Match:
    fact_table: str
    fact_alias: str
    dim_table: str
    dim_alias: str
    fact_key: str
    dim_key: str
    dim_columns: tuple[str, ...]
    where_clause: exp.Where


@dataclass(frozen=True)
class ArgmaxAggregateToTopkMatch:
    base_table: str
    aggregate_expression: exp.Expression
    best_direction: str
    having_sql: str


@dataclass(frozen=True)
class DistinctJoinToSemijoinMatch:
    base_table: str
    base_alias: str
    join_table: str
    join_alias: str
    inner_tables: tuple[str, ...]
    inner_aliases: tuple[str, ...]
    inner_join_ons: tuple[exp.Expression, ...]
    base_predicates: tuple[exp.Expression, ...]
    inner_predicates: tuple[exp.Expression, ...]
    correlated_predicates: tuple[exp.Expression, ...]


@dataclass(frozen=True)
class SameKeyBridgeJoinEliminationMatch:
    left_table: str
    left_alias: str
    left_column: str
    bridge_table: str
    bridge_alias: str
    bridge_column: str
    right_table: str
    right_alias: str
    right_column: str
    target_fragment: str


@dataclass(frozen=True)
class CTEGroupedFactScanMatch:
    cte_name: str
    fact_table: str
    group_key: str
    group_alias: str
    aggregate_alias: str
    aggregate_expr: exp.Expression
    where: exp.Expression | None


@dataclass(frozen=True)
class RepeatedRescanToConditionalAggMatch:
    fact_table: str
    group_key: str
    group_alias: str
    scan_specs: tuple[CTEGroupedFactScanMatch, ...]
    outer_select: exp.Select
    cte_alias_to_output_alias: dict[str, str]
    target_fragment: str


SUMMARY_TABLE_SUBSTITUTIONS: tuple[SummaryTableSubstitution, ...] = (
    SummaryTableSubstitution(
        substitution_id="formula_1_fastest_lap_time_text",
        detail_table="lapTimes",
        summary_table="results",
        detail_metric_column="time",
        summary_metric_column="FastestLapTime",
        join_key_columns=("raceId",),
        required_join_tables=frozenset({"races"}),
        optional_join_tables=frozenset({"circuits"}),
        allowed_detail_columns=frozenset({"time", "raceId"}),
        aggregate_function="min",
        target_fragment="lapTimes -> results.FastestLapTime",
    ),
)


def reanchor_join_driver_shape(sql: str) -> ReanchorJoinDriverMatch | None:
    select = _parse_select(sql)
    if select is None:
        return None
    if any(
        select.args.get(key) is not None
        for key in ("group", "having", "distinct")
    ):
        return None
    order = select.args.get("order")
    limit = select.args.get("limit")
    from_expr = select.args.get("from_")
    joins = select.args.get("joins") or []
    if (
        order is None
        or limit is None
        or len(order.expressions) != 1
        or from_expr is None
        or not isinstance(from_expr.this, exp.Table)
        or len(joins) != 2
        or any(not isinstance(join.this, exp.Table) for join in joins)
    ):
        return None
    if not isinstance(limit.expression, exp.Literal) or str(limit.expression.this) != "1":
        return None
    if limit.args.get("offset") is not None:
        return None
    if any(join.side and str(join.side).upper() != "INNER" for join in joins):
        return None

    ordered = order.expressions[0]
    order_expr = ordered.this if isinstance(ordered, exp.Ordered) else ordered
    if not isinstance(order_expr, exp.Column) or not order_expr.table:
        return None

    tables = [from_expr.this] + [join.this for join in joins if isinstance(join.this, exp.Table)]
    alias_to_table = {table.alias_or_name: table for table in tables}
    projected_aliases = {
        str(column.table)
        for expression in select.expressions
        for column in expression.find_all(exp.Column)
        if column.table is not None
    }
    if len(projected_aliases) != 1:
        return None
    fact_alias = next(iter(projected_aliases))
    if fact_alias not in alias_to_table:
        return None
    driver_alias = str(order_expr.table)
    if driver_alias not in alias_to_table or driver_alias == fact_alias:
        return None
    bridge_aliases = set(alias_to_table) - {driver_alias, fact_alias}
    if len(bridge_aliases) != 1:
        return None
    bridge_alias = next(iter(bridge_aliases))

    if any(
        not _expression_uses_only_tables(expression, {fact_alias, alias_to_table[fact_alias].name})
        for expression in select.expressions
    ):
        return None

    where_clause = select.args.get("where")
    driver_predicates: list[exp.Expression] = []
    if where_clause is not None and where_clause.this is not None:
        for predicate in _flatten_and_conditions(where_clause.this):
            referenced = {
                str(column.table)
                for column in predicate.find_all(exp.Column)
                if column.table is not None
            }
            if not referenced:
                driver_predicates.append(predicate.copy())
                continue
            if referenced <= {driver_alias, alias_to_table[driver_alias].name}:
                driver_predicates.append(predicate.copy())
                continue
            return None

    edge_map: dict[frozenset[str], tuple[exp.Column, exp.Column]] = {}
    for left_alias, right_alias, left_col, right_col in _join_column_pairs(joins):
        edge_map[frozenset({left_alias, right_alias})] = (left_col, right_col)

    driver_bridge_edge = edge_map.get(frozenset({driver_alias, bridge_alias}))
    bridge_fact_edge = edge_map.get(frozenset({bridge_alias, fact_alias}))
    if driver_bridge_edge is None or bridge_fact_edge is None:
        return None
    if frozenset({driver_alias, fact_alias}) in edge_map:
        return None

    driver_bridge_pair = _orient_pair(driver_bridge_edge, driver_alias, bridge_alias)
    bridge_fact_pair = _orient_pair(bridge_fact_edge, bridge_alias, fact_alias)
    if driver_bridge_pair is None or bridge_fact_pair is None:
        return None
    driver_key_col, bridge_driver_col = driver_bridge_pair
    bridge_fact_col, fact_key_col = bridge_fact_pair
    if bridge_driver_col.name != bridge_fact_col.name:
        return None

    return ReanchorJoinDriverMatch(
        driver_table=alias_to_table[driver_alias].name,
        driver_alias=driver_alias,
        driver_key=driver_key_col.name,
        bridge_table=alias_to_table[bridge_alias].name,
        bridge_alias=bridge_alias,
        bridge_key=bridge_driver_col.name,
        fact_table=alias_to_table[fact_alias].name,
        fact_alias=fact_alias,
        fact_key=fact_key_col.name,
        order=order.copy(),
        limit=limit.copy(),
        projections=tuple(expression.copy() for expression in select.expressions),
        driver_predicates=tuple(driver_predicates),
        target_fragment=(
            f"re-anchor on {alias_to_table[driver_alias].name} and probe "
            f"{alias_to_table[fact_alias].name} by resolved key"
        ),
    )


def top1_anchor_then_lookup_tail_shape(sql: str) -> Top1AnchorThenLookupTailMatch | None:
    select = _parse_select(sql)
    if select is None:
        return None
    if any(select.args.get(key) is not None for key in ("group", "having", "distinct", "with_")):
        return None
    order = select.args.get("order")
    limit = select.args.get("limit")
    from_expr = select.args.get("from_")
    joins = select.args.get("joins") or []
    if (
        order is None
        or limit is None
        or len(order.expressions) != 1
        or from_expr is None
        or not isinstance(from_expr.this, exp.Table)
        or len(joins) < 2
        or any(not isinstance(join.this, exp.Table) for join in joins)
    ):
        return None
    if not isinstance(limit.expression, exp.Literal) or str(limit.expression.this) != "1":
        return None
    if limit.args.get("offset") is not None:
        return None
    if any(join.side and str(join.side).upper() != "INNER" for join in joins):
        return None

    ordered = order.expressions[0]
    order_expr = ordered.this if isinstance(ordered, exp.Ordered) else ordered
    if not isinstance(order_expr, exp.Column) or not order_expr.table:
        return None

    tables = [from_expr.this] + [join.this for join in joins if isinstance(join.this, exp.Table)]
    aliases = [table.alias_or_name for table in tables]
    if not _joins_are_linear_binary_chain(joins, tables, aliases):
        return None

    tail_table = tables[-1]
    tail_alias = aliases[-1]
    allowed_tail_tables = {tail_alias, tail_table.name}
    if any(not _expression_uses_only_tables(expr, allowed_tail_tables) for expr in select.expressions):
        return None
    if str(order_expr.table) in allowed_tail_tables:
        return None

    where_clause = select.args.get("where")
    if where_clause is not None:
        for predicate in _flatten_and_conditions(where_clause.this):
            if _referenced_tables(predicate) & allowed_tail_tables:
                return None

    tail_join_pairs = _join_column_pairs([joins[-1]])
    if len(tail_join_pairs) != 1:
        return None
    _, _, left_col, right_col = tail_join_pairs[0]
    oriented = _orient_pair((left_col, right_col), aliases[-2], tail_alias)
    if oriented is None:
        return None
    predecessor_tail_col, tail_key_col = oriented

    return Top1AnchorThenLookupTailMatch(
        prefix_tables=tuple(table.name for table in tables[:-1]),
        tail_table=tail_table.name,
        tail_alias=tail_alias,
        predecessor_table=tables[-2].name,
        predecessor_alias=aliases[-2],
        predecessor_tail_key=predecessor_tail_col.name,
        tail_key=tail_key_col.name,
        projections=tuple(expression.copy() for expression in select.expressions),
        where_clause=where_clause.copy() if where_clause is not None else None,
        order=order.copy(),
        limit=limit.copy(),
        target_fragment=(
            f"resolve top-1 {predecessor_tail_col.name} before looking up "
            f"{tail_table.name}"
        ),
    )


def dimension_key_first_then_fact_probe_shape(sql: str) -> DimensionKeyFirstThenFactProbeMatch | None:
    select = _parse_select(sql)
    if select is None:
        return None
    if any(
        select.args.get(key) is not None
        for key in ("where", "group", "having", "distinct")
    ):
        return None
    if len(select.expressions) < 1:
        return None
    order = select.args.get("order")
    limit = select.args.get("limit")
    from_expr = select.args.get("from_")
    joins = select.args.get("joins") or []
    if (
        order is None
        or limit is None
        or len(order.expressions) != 1
        or from_expr is None
        or not isinstance(from_expr.this, exp.Table)
        or len(joins) != 1
        or not isinstance(joins[0].this, exp.Table)
    ):
        return None
    if not isinstance(limit.expression, exp.Literal) or str(limit.expression.this) != "1":
        return None
    if limit.args.get("offset") is not None:
        return None

    ordered = order.expressions[0]
    order_expr = ordered.this if isinstance(ordered, exp.Ordered) else ordered
    if not isinstance(order_expr, exp.Column) or not order_expr.table:
        return None

    dimension_table = from_expr.this
    fact_table = joins[0].this
    dimension_alias = dimension_table.alias_or_name
    fact_alias = fact_table.alias_or_name
    if order_expr.table not in {dimension_alias, dimension_table.name}:
        return None
    if any(
        not _expression_uses_only_tables(expression, {fact_alias, fact_table.name})
        for expression in select.expressions
    ):
        return None

    on_clause = joins[0].args.get("on")
    if on_clause is None:
        return None
    fact_key: str | None = None
    dimension_key: str | None = None
    for predicate in _flatten_and_conditions(on_clause):
        if not isinstance(predicate, exp.EQ):
            continue
        left = predicate.left
        right = predicate.right
        if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
            continue
        if left.table in {dimension_alias, dimension_table.name} and right.table in {fact_alias, fact_table.name}:
            dimension_key, fact_key = left.name, right.name
            break
        if right.table in {dimension_alias, dimension_table.name} and left.table in {fact_alias, fact_table.name}:
            dimension_key, fact_key = right.name, left.name
            break
    if dimension_key is None or fact_key is None:
        return None

    return DimensionKeyFirstThenFactProbeMatch(
        dimension_table=dimension_table.name,
        dimension_alias=dimension_alias,
        fact_table=fact_table.name,
        fact_alias=fact_alias,
        dimension_key=dimension_key,
        fact_key=fact_key,
        order=order.copy(),
        limit=limit.copy(),
        projections=tuple(expression.copy() for expression in select.expressions),
        target_fragment=f"{dimension_table.name} key-first probe into {fact_table.name}",
    )


def prefer_summary_table_when_grain_matches_shape(sql: str) -> PreferSummaryTableWhenGrainMatchesMatch | None:
    select = _parse_select(sql)
    if select is None:
        return None
    if any(select.args.get(key) is not None for key in ("distinct", "group", "having")):
        return None
    from_expr = select.args.get("from_")
    if from_expr is None or not isinstance(from_expr.this, exp.Table):
        return None
    detail_table = from_expr.this
    detail_alias = detail_table.alias_or_name
    joins = select.args.get("joins") or []
    join_tables = {
        join.this.name.lower()
        for join in joins
        if isinstance(join.this, exp.Table)
    }

    for substitution in SUMMARY_TABLE_SUBSTITUTIONS:
        if detail_table.name.lower() != substitution.detail_table.lower():
            continue
        allowed_join_tables = substitution.required_join_tables | substitution.optional_join_tables
        if not substitution.required_join_tables <= join_tables:
            continue
        if join_tables - allowed_join_tables:
            continue
        if not _summary_substitution_columns_match(
            select=select,
            detail_alias=detail_alias,
            detail_table=detail_table.name,
            substitution=substitution,
        ):
            continue
        mode = _summary_substitution_mode(
            select=select,
            detail_alias=detail_alias,
            detail_table=detail_table.name,
            substitution=substitution,
        )
        if mode is None:
            continue
        return PreferSummaryTableWhenGrainMatchesMatch(
            substitution_id=substitution.substitution_id,
            substitution=substitution,
            detail_table=substitution.detail_table,
            detail_alias=detail_alias,
            summary_table=substitution.summary_table,
            summary_metric_column=substitution.summary_metric_column,
            column_mapping=substitution.column_mapping,
            target_allowed_columns=substitution.target_allowed_columns,
            mode=mode,
            target_fragment=substitution.target_fragment,
        )
    return None


def symmetric_union_arm_pruning_shape(sql: str) -> SymmetricUnionArmPruningMatch | None:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return None
    if isinstance(ast, exp.Union):
        return _symmetric_union_shape(ast)
    select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
    if select is None:
        return None
    return _symmetric_or_shape(select)


def _parse_select(sql: str) -> exp.Select | None:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return None
    return ast if isinstance(ast, exp.Select) else ast.find(exp.Select)


def scalar_extrema_ladder_shape(sql: str) -> ScalarExtremaLadderMatch | None:
    select = _parse_select(sql)
    if select is None:
        return None
    if any(select.args.get(key) is not None for key in ("group", "having", "distinct", "order", "limit")):
        return None
    where_clause = select.args.get("where")
    if where_clause is None or where_clause.this is None:
        return None
    predicates = _flatten_and_conditions(where_clause.this)
    base_predicates: list[exp.Expression] = []
    candidate_predicates: list[exp.Expression] = []
    for predicate in predicates:
        if any(isinstance(node, exp.Subquery) for node in predicate.walk()):
            candidate_predicates.append(predicate)
        else:
            base_predicates.append(predicate.copy())
    if not candidate_predicates:
        return None

    outer_graph = _select_graph_signature(select)
    if outer_graph is None:
        return None
    normalized_base = [_normalize_sql(predicate.sql(dialect="sqlite")) for predicate in base_predicates]

    ladder_specs: list[dict[str, Any]] = []
    for predicate in candidate_predicates:
        spec = _scalar_extrema_predicate_spec(predicate, outer_graph)
        if spec is None:
            return None
        ladder_specs.append(spec)

    remaining = list(ladder_specs)
    order_specs: list[ScalarExtremaOrderSpec] = []
    accumulated_required = list(normalized_base)
    used_norms: list[str] = []
    while remaining:
        matched_index: int | None = None
        for index, spec in enumerate(remaining):
            if sorted(spec["subquery_where_norms"]) != sorted(accumulated_required):
                continue
            matched_index = index
            break
        if matched_index is None:
            return None
        spec = remaining.pop(matched_index)
        order_specs.append(
            ScalarExtremaOrderSpec(
                column=spec["outer_column"].copy(),
                direction=spec["direction"],
            )
        )
        used_norms.append(spec["predicate_norm"])
        accumulated_required.append(spec["predicate_norm"])

    if not order_specs:
        return None
    return ScalarExtremaLadderMatch(
        base_predicates=tuple(base_predicates),
        ladder_predicate_norms=tuple(used_norms),
        order_specs=tuple(order_specs),
        target_fragment="nested scalar MIN/MAX ladder",
    )


def topk_before_join_shape(sql: str) -> TopkBeforeJoinMatch | None:
    select = _parse_select(sql)
    if select is None:
        return None
    if select.args.get("group") is not None or select.args.get("having") is not None:
        return None
    if select.args.get("distinct") is not None:
        return None
    order = select.args.get("order")
    limit = select.args.get("limit")
    if order is None or limit is None or len(order.expressions) != 1:
        return None
    ordered = order.expressions[0]
    order_expr = ordered.this if isinstance(ordered, exp.Ordered) else ordered
    if not isinstance(order_expr, exp.Column) or not order_expr.table:
        return None
    if not isinstance(limit.expression, exp.Literal) or str(limit.expression.this) != "1":
        return None
    if limit.args.get("offset") is not None:
        return None
    from_expr = select.args.get("from_")
    if from_expr is None or not isinstance(from_expr.this, exp.Table):
        return None
    base_table = from_expr.this
    base_alias = base_table.alias_or_name
    if order_expr.table not in {base_alias, base_table.name}:
        return None
    where_clause = select.args.get("where")
    if where_clause is not None:
        for column in where_clause.find_all(exp.Column):
            if column.table and column.table not in {base_alias, base_table.name}:
                return None
    joins = list(select.find_all(exp.Join))
    if len(joins) != 1:
        return None
    base_columns: set[str] = set()
    for join in joins:
        on = join.args.get("on")
        if on is None:
            return None
        for node in on.walk():
            if not isinstance(node, exp.EQ):
                continue
            for side in (node.left, node.right):
                if isinstance(side, exp.Column) and side.table in {base_alias, base_table.name}:
                    base_columns.add(side.name)
    if not base_columns:
        return None
    return TopkBeforeJoinMatch(
        join_count=len(joins),
        base_table=base_table.name,
        base_alias=base_alias,
        order_by_column=order_expr.name,
        base_columns=tuple(sorted(base_columns)),
        where_clause=where_clause.copy() if where_clause is not None else None,
    )


def distinct_top1_to_grouped_extrema_shape(sql: str) -> DistinctTop1ToGroupedExtremaMatch | None:
    select = _parse_select(sql)
    if select is None:
        return None
    if select.args.get("group") is not None or select.args.get("having") is not None:
        return None
    if select.args.get("distinct") is None:
        return None
    if len(select.expressions) != 1:
        return None
    projection_expr = select.expressions[0]
    projection_col = projection_expr.this if isinstance(projection_expr, exp.Alias) else projection_expr
    if not isinstance(projection_col, exp.Column) or not projection_col.table:
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
    base_table = from_expr.this
    base_alias = base_table.alias_or_name
    join_table = join.this
    join_alias = join_table.alias_or_name
    where_clause = select.args.get("where")
    base_predicates: list[exp.Expression] = []
    metric_col: exp.Column | None = None
    direction: str | None = None
    limit_expr: exp.Expression | None = None

    order = select.args.get("order")
    limit = select.args.get("limit")
    if order is not None or limit is not None:
        if order is None or limit is None or len(order.expressions) != 1:
            return None
        if not isinstance(limit.expression, exp.Literal) or str(limit.expression.this) != "1":
            return None
        if limit.args.get("offset") is not None:
            return None
        ordered = order.expressions[0]
        order_expr = ordered.this if isinstance(ordered, exp.Ordered) else ordered
        if not isinstance(order_expr, exp.Column) or not order_expr.table:
            return None
        metric_col = order_expr
        direction = "DESC" if getattr(ordered, "args", {}).get("desc") else "ASC"
        limit_expr = limit.expression.copy()
        if where_clause is not None and where_clause.this is not None:
            base_predicates = [predicate.copy() for predicate in _flatten_and_conditions(where_clause.this)]
    else:
        if where_clause is None or where_clause.this is None:
            return None
        predicates = _flatten_and_conditions(where_clause.this)
        scalar_specs: list[tuple[int, exp.Column, str]] = []
        for idx, predicate in enumerate(predicates):
            if not isinstance(predicate, exp.EQ):
                continue
            if isinstance(predicate.left, exp.Column) and isinstance(predicate.right, exp.Subquery):
                outer_col = predicate.left
                subquery = predicate.right
            elif isinstance(predicate.right, exp.Column) and isinstance(predicate.left, exp.Subquery):
                outer_col = predicate.right
                subquery = predicate.left
            else:
                continue
            if not isinstance(subquery.this, exp.Select):
                continue
            inner_select = subquery.this
            if inner_select.args.get("group") is not None or inner_select.args.get("having") is not None:
                continue
            inner_order = inner_select.args.get("order")
            inner_limit = inner_select.args.get("limit")
            if inner_order is None or inner_limit is None or len(inner_order.expressions) != 1:
                continue
            if not isinstance(inner_limit.expression, exp.Literal) or str(inner_limit.expression.this) != "1":
                continue
            inner_expr = inner_select.expressions[0]
            inner_value = inner_expr.this if isinstance(inner_expr, exp.Alias) else inner_expr
            inner_ordered = inner_order.expressions[0]
            inner_order_expr = inner_ordered.this if isinstance(inner_ordered, exp.Ordered) else inner_ordered
            if not isinstance(inner_value, exp.Column) or not isinstance(inner_order_expr, exp.Column):
                continue
            if inner_value.name != inner_order_expr.name:
                continue
            if outer_col.name != inner_value.name:
                continue
            outer_metric_side_norms = []
            metric_side_tables = {str(outer_col.table)}
            for j, p in enumerate(predicates):
                if j == idx:
                    continue
                referenced = _referenced_tables(p)
                if referenced and not referenced <= metric_side_tables:
                    continue
                outer_metric_side_norms.append(_normalize_sql_ignoring_tables(p))
            inner_where = inner_select.args.get("where")
            inner_where_norms = []
            if inner_where is not None and inner_where.this is not None:
                inner_where_norms = [
                    _normalize_sql_ignoring_tables(p)
                    for p in _flatten_and_conditions(inner_where.this)
                ]
            if sorted(outer_metric_side_norms) != sorted(inner_where_norms):
                continue
            scalar_specs.append(
                (
                    idx,
                    outer_col.copy(),
                    "DESC" if getattr(inner_ordered, "args", {}).get("desc") else "ASC",
                )
            )
        if len(scalar_specs) != 1:
            return None
        scalar_index, metric_col, direction = scalar_specs[0]
        base_predicates = [predicate.copy() for idx, predicate in enumerate(predicates) if idx != scalar_index]
        limit_expr = exp.Literal.number(1)

    assert metric_col is not None
    assert direction is not None
    assert limit_expr is not None
    projection_tables = {projection_col.table}
    metric_tables = {metric_col.table}
    valid_tables = {base_alias, base_table.name, join_alias, join_table.name}
    if not projection_tables <= valid_tables or not metric_tables <= valid_tables:
        return None
    if projection_tables == metric_tables:
        return None
    return DistinctTop1ToGroupedExtremaMatch(
        base_table=base_table.name,
        base_alias=base_alias,
        join_table=join_table.name,
        join_alias=join_alias,
        projection=projection_col.copy(),
        metric=metric_col.copy(),
        direction=direction,
        where_predicates=tuple(base_predicates),
        limit_expr=limit_expr,
    )


def scalar_extrema_anchor_then_lookup_tail_shape(sql: str) -> ScalarExtremaAnchorThenLookupTailMatch | None:
    select = _parse_select(sql)
    if select is None:
        return None
    if any(select.args.get(key) is not None for key in ("group", "having", "distinct", "with_")):
        return None
    if select.args.get("order") is not None or select.args.get("limit") is not None:
        return None
    from_expr = select.args.get("from_")
    joins = select.args.get("joins") or []
    where_clause = select.args.get("where")
    if (
        from_expr is None
        or not isinstance(from_expr.this, exp.Table)
        or len(joins) < 2
        or any(not isinstance(join.this, exp.Table) for join in joins)
        or where_clause is None
        or where_clause.this is None
    ):
        return None
    if any(join.side and str(join.side).upper() != "INNER" for join in joins):
        return None

    tables = [from_expr.this] + [join.this for join in joins if isinstance(join.this, exp.Table)]
    aliases = [table.alias_or_name for table in tables]
    if not _joins_are_linear_binary_chain(joins, tables, aliases):
        return None

    tail_table = tables[-1]
    tail_alias = aliases[-1]
    predecessor_table = tables[-2]
    predecessor_alias = aliases[-2]
    allowed_tail_tables = {tail_alias, tail_table.name}
    if any(not _expression_uses_only_tables(expr, allowed_tail_tables) for expr in select.expressions):
        return None

    predicates = _flatten_and_conditions(where_clause.this)
    scalar_spec: tuple[int, str, str] | None = None
    outer_predicates: list[exp.Expression] = []
    predecessor_predicate_norms: list[str] = []
    for idx, predicate in enumerate(predicates):
        if (
            isinstance(predicate, exp.EQ)
            and isinstance(predicate.left, exp.Column)
            and isinstance(predicate.right, exp.Subquery)
            and predicate.left.table in {predecessor_alias, predecessor_table.name}
        ):
            outer_col = predicate.left
            subquery = predicate.right
        elif (
            isinstance(predicate, exp.EQ)
            and isinstance(predicate.right, exp.Column)
            and isinstance(predicate.left, exp.Subquery)
            and predicate.right.table in {predecessor_alias, predecessor_table.name}
        ):
            outer_col = predicate.right
            subquery = predicate.left
        else:
            referenced = _referenced_tables(predicate)
            if referenced & allowed_tail_tables:
                return None
            outer_predicates.append(predicate.copy())
            if referenced and referenced <= {predecessor_alias, predecessor_table.name}:
                predecessor_predicate_norms.append(_normalize_sql_ignoring_tables(predicate))
            continue

        if scalar_spec is not None or not isinstance(subquery.this, exp.Select):
            return None
        inner_select = subquery.this
        if inner_select.args.get("group") is not None or inner_select.args.get("having") is not None:
            return None
        inner_from = inner_select.args.get("from_")
        inner_joins = inner_select.args.get("joins") or []
        inner_order = inner_select.args.get("order")
        inner_limit = inner_select.args.get("limit")
        if (
            inner_from is None
            or not isinstance(inner_from.this, exp.Table)
            or inner_joins
            or inner_order is None
            or inner_limit is None
            or len(inner_order.expressions) != 1
            or not isinstance(inner_limit.expression, exp.Literal)
            or str(inner_limit.expression.this) != "1"
            or inner_limit.args.get("offset") is not None
        ):
            return None
        inner_expr = inner_select.expressions[0]
        inner_value = inner_expr.this if isinstance(inner_expr, exp.Alias) else inner_expr
        inner_ordered = inner_order.expressions[0]
        inner_order_expr = inner_ordered.this if isinstance(inner_ordered, exp.Ordered) else inner_ordered
        if not isinstance(inner_value, exp.Column) or not isinstance(inner_order_expr, exp.Column):
            return None
        if (
            inner_from.this.name != predecessor_table.name
            or inner_value.name != inner_order_expr.name
            or outer_col.name != inner_value.name
        ):
            return None
        inner_where = inner_select.args.get("where")
        if inner_where is None or inner_where.this is None:
            return None
        inner_metric_norms: list[str] = []
        saw_correlation = False
        for inner_predicate in _flatten_and_conditions(inner_where.this):
            inner_refs = _referenced_tables(inner_predicate)
            if inner_refs <= {inner_from.this.alias_or_name, inner_from.this.name}:
                inner_metric_norms.append(_normalize_sql_ignoring_tables(inner_predicate))
                continue
            if isinstance(inner_predicate, exp.EQ):
                left = inner_predicate.left
                right = inner_predicate.right
                if (
                    isinstance(left, exp.Column)
                    and isinstance(right, exp.Column)
                    and left.name == right.name
                    and (
                        left.table in {inner_from.this.alias_or_name, inner_from.this.name}
                        or right.table in {inner_from.this.alias_or_name, inner_from.this.name}
                    )
                ):
                    saw_correlation = True
                    continue
            return None
        if sorted(inner_metric_norms) != sorted(predecessor_predicate_norms):
            return None
        if not saw_correlation:
            return None
        scalar_spec = (
            idx,
            outer_col.name,
            "DESC" if getattr(inner_ordered, "args", {}).get("desc") else "ASC",
        )

    if scalar_spec is None:
        return None

    tail_join_pairs = _join_column_pairs([joins[-1]])
    if len(tail_join_pairs) != 1:
        return None
    _, _, left_col, right_col = tail_join_pairs[0]
    oriented = _orient_pair((left_col, right_col), predecessor_alias, tail_alias)
    if oriented is None:
        return None
    predecessor_tail_col, tail_key_col = oriented
    _, metric_column, direction = scalar_spec

    return ScalarExtremaAnchorThenLookupTailMatch(
        prefix_tables=tuple(table.name for table in tables[:-1]),
        tail_table=tail_table.name,
        tail_alias=tail_alias,
        predecessor_table=predecessor_table.name,
        predecessor_alias=predecessor_alias,
        predecessor_tail_key=predecessor_tail_col.name,
        tail_key=tail_key_col.name,
        metric_column=metric_column,
        direction=direction,
        projections=tuple(expression.copy() for expression in select.expressions),
        outer_predicates=tuple(outer_predicates),
        target_fragment=(
            f"normalize scalar extrema on {predecessor_table.name}.{metric_column} "
            f"into a tail lookup by {predecessor_tail_col.name}"
        ),
    )


def distinct_extrema_to_grouped_having_shape(sql: str) -> DistinctExtremaToGroupedHavingMatch | None:
    select = _parse_select(sql)
    if select is None:
        return None
    if select.args.get("distinct") is None:
        return None
    if any(select.args.get(key) is not None for key in ("group", "having", "order", "limit")):
        return None
    if not select.expressions:
        return None
    projection_exprs: list[exp.Expression] = []
    for expression in select.expressions:
        expr = expression.this if isinstance(expression, exp.Alias) else expression
        if not isinstance(expr, exp.Column):
            return None
        projection_exprs.append(expr.copy())
    where_clause = select.args.get("where")
    if where_clause is None or where_clause.this is None:
        return None
    predicates = _flatten_and_conditions(where_clause.this)
    scalar_match: tuple[exp.Column, exp.Column, str, exp.Select] | None = None
    outer_predicates: list[exp.Expression] = []
    for predicate in predicates:
        matched = _match_scalar_extrema_equality(predicate)
        if matched is None:
            outer_predicates.append(predicate.copy())
            continue
        if scalar_match is not None:
            return None
        scalar_match = matched
    if scalar_match is None:
        return None
    outer_metric, inner_metric, direction, inner_select = scalar_match
    return DistinctExtremaToGroupedHavingMatch(
        projections=tuple(projection_exprs),
        outer_predicates=tuple(outer_predicates),
        outer_metric=outer_metric.copy(),
        inner_metric=inner_metric.copy(),
        direction=direction,
        inner_select=inner_select.copy(),
        target_fragment=(
            f"SELECT DISTINCT ... WHERE {outer_metric.name} = "
            f"(SELECT {inner_metric.name} ... ORDER BY {inner_metric.name} {direction} LIMIT 1)"
        ),
    )


def grouped_max_top1_before_join_shape(sql: str) -> GroupedMaxTop1BeforeJoinMatch | None:
    select = _parse_select(sql)
    if select is None:
        return None
    if select.args.get("distinct") is not None or select.args.get("having") is not None:
        return None
    if len(select.expressions) != 1:
        return None
    selected_expr = select.expressions[0]
    selected_value = selected_expr.this.copy() if isinstance(selected_expr, exp.Alias) else selected_expr.copy()
    group = select.args.get("group")
    if group is None or not group.expressions:
        return None
    group_expressions = tuple(expression.copy() for expression in group.expressions)
    group_sql = {
        expression.sql(dialect="sqlite")
        for expression in group.expressions
    }
    if selected_value.sql(dialect="sqlite") not in group_sql:
        return None
    order = select.args.get("order")
    limit = select.args.get("limit")
    if order is None or limit is None or len(order.expressions) != 1:
        return None
    if not isinstance(limit.expression, exp.Literal) or str(limit.expression.this) != "1":
        return None
    ordered_expr = order.expressions[0]
    order_expr = ordered_expr.this if isinstance(ordered_expr, exp.Ordered) else ordered_expr
    if not isinstance(order_expr, (exp.Count, exp.Sum, exp.Max, exp.Min, exp.Avg)):
        return None
    aggregate_function = order_expr.key.upper()
    from_expr = select.args.get("from_")
    if from_expr is None or not isinstance(from_expr.this, exp.Table):
        return None
    base_table = from_expr.this
    base_alias = base_table.alias_or_name
    joins = list(select.args.get("joins") or [])
    if len(joins) > 2:
        return None
    join_hops: list[GroupedAggregateJoinHop] = []
    join_table: exp.Table | None = None
    join_alias: str | None = None
    base_join_key: str | None = None
    join_key: str | None = None
    current_left_table = base_table
    current_left_alias = base_alias
    for idx, join in enumerate(joins):
        if join.side and str(join.side).upper() != "INNER":
            return None
        if not isinstance(join.this, exp.Table):
            return None
        current_right_table = join.this
        current_right_alias = current_right_table.alias_or_name
        on_clause = join.args.get("on")
        if on_clause is None:
            return None
        left_key: str | None = None
        right_key: str | None = None
        for node in on_clause.walk():
            if not isinstance(node, exp.EQ):
                continue
            left = node.left
            right = node.right
            if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                continue
            if left.table in {current_left_alias, current_left_table.name} and right.table in {current_right_alias, current_right_table.name}:
                left_key = left.name
                right_key = right.name
                break
            if right.table in {current_left_alias, current_left_table.name} and left.table in {current_right_alias, current_right_table.name}:
                left_key = right.name
                right_key = left.name
                break
        if left_key is None or right_key is None:
            return None
        join_hops.append(
            GroupedAggregateJoinHop(
                left_table=current_left_table.name,
                left_alias=current_left_alias,
                left_column=left_key,
                right_table=current_right_table.name,
                right_alias=current_right_alias,
                right_column=right_key,
            )
        )
        if idx == 0:
            join_table = current_right_table
            join_alias = current_right_alias
            base_join_key = left_key
            join_key = right_key
        current_left_table = current_right_table
        current_left_alias = current_right_alias
    where_clause = select.args.get("where")
    return GroupedMaxTop1BeforeJoinMatch(
        join_count=len(joins),
        base_table=base_table.name,
        base_alias=base_alias,
        join_hops=tuple(join_hops),
        projection_expression=selected_value,
        group_expressions=group_expressions,
        join_table=join_table.name if join_table is not None else None,
        join_alias=join_alias,
        base_join_key=base_join_key,
        join_key=join_key,
        aggregate_expression=order_expr.copy(),
        aggregate_function=aggregate_function,
        where_clause=where_clause.copy() if where_clause is not None else None,
        limit_expr=limit.expression.copy(),
        ordered_expr=ordered_expr.copy(),
    )


def filter_dimension_before_top1_shape(sql: str) -> FilterDimensionBeforeTop1Match | None:
    select = _parse_select(sql)
    if select is None:
        return None
    if select.args.get("group") is not None or select.args.get("having") is not None:
        return None
    if select.args.get("distinct") is not None:
        return None
    if len(select.expressions) != 1 or not isinstance(select.expressions[0], exp.Column):
        return None
    selected_col = select.expressions[0]
    order = select.args.get("order")
    limit = select.args.get("limit")
    if order is None or limit is None or len(order.expressions) != 1:
        return None
    if not isinstance(limit.expression, exp.Literal) or str(limit.expression.this) != "1":
        return None
    ordered = order.expressions[0]
    order_expr = ordered.this if isinstance(ordered, exp.Ordered) else ordered
    if not isinstance(order_expr, exp.Column) or not order_expr.table:
        return None
    from_expr = select.args.get("from_")
    if from_expr is None or not isinstance(from_expr.this, exp.Table):
        return None
    fact_table = from_expr.this
    fact_alias = fact_table.alias_or_name
    joins = list(select.find_all(exp.Join))
    if len(joins) != 1:
        return None
    join = joins[0]
    if join.side and str(join.side).upper() != "INNER":
        return None
    if not isinstance(join.this, exp.Table):
        return None
    dim_table = join.this
    dim_alias = dim_table.alias_or_name
    if selected_col.table not in {dim_alias, dim_table.name}:
        return None
    if order_expr.table not in {fact_alias, fact_table.name}:
        return None
    where_clause = select.args.get("where")
    if where_clause is None or where_clause.this is None:
        return None
    outer_in: exp.In | None = None
    for node in where_clause.this.walk():
        if isinstance(node, exp.In) and isinstance(node.this, exp.Column) and isinstance(node.args.get("query"), exp.Subquery):
            outer_in = node
            break
    if outer_in is None or outer_in.this.table not in {dim_alias, dim_table.name}:
        return None
    if not isinstance(outer_in.args.get("query"), exp.Subquery) or not isinstance(outer_in.args["query"].this, exp.Select):
        return None
    for column in where_clause.find_all(exp.Column):
        if column.table and column.table not in {dim_alias, dim_table.name}:
            return None
    on_clause = join.args.get("on")
    if on_clause is None:
        return None
    fact_key: str | None = None
    dim_key: str | None = None
    for node in on_clause.walk():
        if not isinstance(node, exp.EQ):
            continue
        left = node.left
        right = node.right
        if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
            continue
        if left.table in {fact_alias, fact_table.name} and right.table in {dim_alias, dim_table.name}:
            fact_key, dim_key = left.name, right.name
            break
        if right.table in {fact_alias, fact_table.name} and left.table in {dim_alias, dim_table.name}:
            fact_key, dim_key = right.name, left.name
            break
    if fact_key is None or dim_key is None:
        return None
    return FilterDimensionBeforeTop1Match(
        fact_table=fact_table.name,
        fact_alias=fact_alias,
        dim_table=dim_table.name,
        dim_alias=dim_alias,
        fact_key=fact_key,
        dim_key=dim_key,
        dim_columns=tuple(sorted({selected_col.name, dim_key})),
        where_clause=where_clause.copy(),
    )


def argmax_aggregate_to_topk_shape(sql: str) -> ArgmaxAggregateToTopkMatch | None:
    select = _parse_select(sql)
    if select is None:
        return None
    if select.args.get("order") is not None or select.args.get("limit") is not None:
        return None
    group = select.args.get("group")
    having = select.args.get("having")
    from_expr = select.args.get("from_")
    if group is None or having is None or from_expr is None or not isinstance(from_expr.this, exp.Table):
        return None
    if len(select.expressions) != 1 or len(group.expressions) != 1:
        return None
    selected = select.expressions[0]
    group_expr = group.expressions[0]
    if not isinstance(selected, exp.Column) or not isinstance(group_expr, exp.Column):
        return None
    if selected.sql(dialect="sqlite") != group_expr.sql(dialect="sqlite"):
        return None
    comparison = having.this
    if not isinstance(comparison, exp.EQ):
        return None
    if isinstance(comparison.left, exp.AggFunc) and isinstance(comparison.right, exp.Subquery):
        aggregate_expression = comparison.left
        nested_subquery = comparison.right
    elif isinstance(comparison.right, exp.AggFunc) and isinstance(comparison.left, exp.Subquery):
        aggregate_expression = comparison.right
        nested_subquery = comparison.left
    else:
        return None
    inner_max_select = nested_subquery.this
    if not isinstance(inner_max_select, exp.Select) or len(inner_max_select.expressions) != 1:
        return None
    best_aggregate = inner_max_select.expressions[0]
    if isinstance(best_aggregate, exp.Max):
        best_direction = "DESC"
    elif isinstance(best_aggregate, exp.Min):
        best_direction = "ASC"
    else:
        return None
    nested_from = inner_max_select.args.get("from_")
    if nested_from is None or not isinstance(nested_from.this, exp.Subquery):
        return None
    grouped_subquery = nested_from.this.this
    if not isinstance(grouped_subquery, exp.Select):
        return None
    if grouped_subquery.args.get("group") is None or len(grouped_subquery.args["group"].expressions) != 1:
        return None
    grouped_from = grouped_subquery.args.get("from_")
    if grouped_from is None or not isinstance(grouped_from.this, exp.Table):
        return None
    if grouped_from.this.name != from_expr.this.name:
        return None
    grouped_expr = grouped_subquery.expressions[0] if grouped_subquery.expressions else None
    if not isinstance(grouped_expr, exp.Alias) or not isinstance(grouped_expr.this, exp.AggFunc):
        return None
    max_arg = best_aggregate.this
    if not isinstance(max_arg, exp.Column) or grouped_expr.alias_or_name.lower() != max_arg.name.lower():
        return None
    if grouped_subquery.args.get("where") is not None:
        outer_where = select.args.get("where")
        if outer_where is None:
            return None
        if _normalize_sql(grouped_subquery.args["where"].sql(dialect="sqlite")) != _normalize_sql(outer_where.sql(dialect="sqlite")):
            return None
    return ArgmaxAggregateToTopkMatch(
        base_table=from_expr.this.name,
        aggregate_expression=aggregate_expression.copy(),
        best_direction=best_direction,
        having_sql=having.sql(dialect="sqlite"),
    )


def distinct_join_to_semijoin_shape(sql: str) -> DistinctJoinToSemijoinMatch | None:
    select = _parse_select(sql)
    if select is None:
        return None
    if select.args.get("distinct") is None:
        return None
    if select.args.get("group") is not None or select.args.get("having") is not None:
        return None
    if select.args.get("order") is not None or select.args.get("limit") is not None:
        return None
    from_expr = select.args.get("from_")
    joins = select.args.get("joins") or []
    if from_expr is None or not isinstance(from_expr.this, exp.Table):
        return None
    if len(joins) < 1 or any(not isinstance(join.this, exp.Table) for join in joins):
        return None
    if any(join.side and str(join.side).upper() != "INNER" for join in joins):
        return None
    tables = [from_expr.this] + [join.this for join in joins if isinstance(join.this, exp.Table)]
    aliases = [table.alias_or_name for table in tables]
    projection_anchor_idx = _distinct_projection_anchor_index(select, tables)
    if projection_anchor_idx is None or projection_anchor_idx not in {0, len(tables) - 1}:
        return None
    anchor_table = tables[projection_anchor_idx]
    anchor_alias = aliases[projection_anchor_idx]
    allowed_anchor_tables = {anchor_alias, anchor_table.name}
    if any(not _expression_uses_only_tables(expr, allowed_anchor_tables) for expr in select.expressions):
        return None
    if not _joins_are_linear_binary_chain(joins, tables, aliases):
        return None
    where_clause = select.args.get("where")
    base_predicates: list[exp.Expression] = []
    inner_predicates: list[exp.Expression] = []
    correlated_predicates: list[exp.Expression] = []
    allowed_all_tables = set(aliases) | {table.name for table in tables}
    non_anchor_aliases = {
        alias
        for idx, alias in enumerate(aliases)
        if idx != projection_anchor_idx
    }
    non_anchor_table_names = {
        table.name
        for idx, table in enumerate(tables)
        if idx != projection_anchor_idx
    }
    allowed_non_anchor_tables = non_anchor_aliases | non_anchor_table_names
    if where_clause is not None:
        for predicate in _flatten_and_conditions(where_clause.this):
            referenced_tables = _referenced_tables(predicate)
            if not referenced_tables or referenced_tables <= allowed_anchor_tables:
                base_predicates.append(predicate.copy())
            elif referenced_tables <= allowed_non_anchor_tables:
                inner_predicates.append(predicate.copy())
            elif referenced_tables <= allowed_all_tables:
                correlated_predicates.append(predicate.copy())
            else:
                return None

    edge_ons = [_combine_conditions(_flatten_and_conditions(join.args.get("on"))) for join in joins]
    if any(on is None for on in edge_ons):
        return None

    if projection_anchor_idx == 0:
        ordered_inner_tables = tables[1:]
        ordered_inner_aliases = aliases[1:]
        correlated_predicates.append(edge_ons[0].copy())
        inner_join_ons = [edge.copy() for edge in edge_ons[1:]]
    else:
        ordered_inner_tables = list(reversed(tables[:-1]))
        ordered_inner_aliases = list(reversed(aliases[:-1]))
        correlated_predicates.append(edge_ons[-1].copy())
        inner_join_ons = [edge.copy() for edge in reversed(edge_ons[:-1])]

    if not correlated_predicates:
        return None
    return DistinctJoinToSemijoinMatch(
        base_table=anchor_table.name,
        base_alias=anchor_alias,
        join_table=ordered_inner_tables[0].name,
        join_alias=ordered_inner_aliases[0],
        inner_tables=tuple(table.name for table in ordered_inner_tables),
        inner_aliases=tuple(ordered_inner_aliases),
        inner_join_ons=tuple(inner_join_ons),
        base_predicates=tuple(base_predicates),
        inner_predicates=tuple(inner_predicates),
        correlated_predicates=tuple(correlated_predicates),
    )


def repeated_rescan_to_conditional_agg_shape(sql: str) -> RepeatedRescanToConditionalAggMatch | None:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return None
    with_clause = ast.args.get("with_")
    if with_clause is None or len(with_clause.expressions) < 2:
        return None
    scans = [_cte_grouped_fact_scan_match(cte) for cte in with_clause.expressions]
    if any(scan is None for scan in scans):
        return None
    first_scan = scans[0]
    assert first_scan is not None
    for scan in scans[1:]:
        assert scan is not None
        if scan.fact_table != first_scan.fact_table or scan.group_key != first_scan.group_key:
            return None
    outer_select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
    if outer_select is None:
        return None
    from_expr = outer_select.args.get("from_")
    joins = outer_select.args.get("joins") or []
    if from_expr is None or not isinstance(from_expr.this, exp.Table):
        return None
    outer_tables = [from_expr.this] + [join.this for join in joins if isinstance(join.this, exp.Table)]
    if len(outer_tables) != len(scans):
        return None
    cte_aliases = {scan.cte_name for scan in scans if scan is not None}
    outer_aliases = {table.alias_or_name for table in outer_tables}
    if outer_aliases != cte_aliases:
        return None
    for join in joins:
        on_clause = join.args.get("on")
        if on_clause is None:
            return None
        join_matches = False
        for predicate in _flatten_and_conditions(on_clause):
            if not isinstance(predicate, exp.EQ):
                continue
            if not isinstance(predicate.left, exp.Column) or not isinstance(predicate.right, exp.Column):
                continue
            if predicate.left.name == first_scan.group_alias and predicate.right.name == first_scan.group_alias:
                join_matches = True
                break
        if not join_matches:
            return None
    for expression in outer_select.expressions:
        for column in expression.find_all(exp.Column):
            if column.table not in cte_aliases:
                return None
    alias_to_scan = {scan.cte_name: scan for scan in scans if scan is not None}
    ordered_scans: list[CTEGroupedFactScanMatch] = []
    for table in outer_tables:
        scan = alias_to_scan.get(table.alias_or_name)
        if scan is None:
            return None
        ordered_scans.append(scan)
    return RepeatedRescanToConditionalAggMatch(
        fact_table=first_scan.fact_table,
        group_key=first_scan.group_key,
        group_alias=first_scan.group_alias,
        scan_specs=tuple(ordered_scans),
        outer_select=outer_select.copy(),
        cte_alias_to_output_alias={scan.cte_name: scan.aggregate_alias for scan in ordered_scans},
        target_fragment="sibling grouped CTE rescans",
    )


def _cte_grouped_fact_scan_match(cte: exp.CTE) -> CTEGroupedFactScanMatch | None:
    if not isinstance(cte.this, exp.Select):
        return None
    select = cte.this
    from_expr = select.args.get("from_")
    group = select.args.get("group")
    if from_expr is None or not isinstance(from_expr.this, exp.Table) or group is None or len(group.expressions) != 1 or len(select.expressions) != 2:
        return None
    projected_group = select.expressions[0]
    group_expr = group.expressions[0]
    aggregate = select.expressions[1]
    if not isinstance(projected_group, exp.Column) or not isinstance(group_expr, exp.Column) or projected_group.name != group_expr.name or not isinstance(aggregate, exp.Alias) or not isinstance(aggregate.this, (exp.Sum, exp.Count, exp.Avg, exp.Min, exp.Max)):
        return None
    aggregate_expr = aggregate.this
    if isinstance(aggregate_expr, exp.Count):
        if not (isinstance(aggregate_expr.this, exp.Star) or _is_safe_conditional_aggregate_argument(aggregate_expr.this)):
            return None
    elif not _is_safe_conditional_aggregate_argument(aggregate_expr.this):
        return None
    return CTEGroupedFactScanMatch(
        cte_name=cte.alias_or_name,
        fact_table=from_expr.this.name,
        group_key=group_expr.name,
        group_alias=projected_group.alias_or_name or projected_group.name,
        aggregate_alias=aggregate.alias_or_name,
        aggregate_expr=aggregate_expr.copy(),
        where=select.args.get("where").this.copy() if select.args.get("where") is not None else None,
    )


def _summary_substitution_columns_match(
    *,
    select: exp.Select,
    detail_alias: str,
    detail_table: str,
    substitution: SummaryTableSubstitution,
) -> bool:
    allowed_tables = {detail_alias, detail_table}
    for column in select.find_all(exp.Column):
        if column.table not in allowed_tables:
            continue
        if column.name not in substitution.allowed_detail_columns:
            return False
    return True


def _select_graph_signature(select: exp.Select) -> str | None:
    from_expr = select.args.get("from_")
    if from_expr is None or not isinstance(from_expr.this, exp.Table):
        return None
    parts = [from_expr.sql(dialect="sqlite")]
    for join in select.args.get("joins") or []:
        if not isinstance(join.this, exp.Table):
            return None
        parts.append(join.sql(dialect="sqlite"))
    return _normalize_sql(" ".join(parts))


def _summary_substitution_mode(
    *,
    select: exp.Select,
    detail_alias: str,
    detail_table: str,
    substitution: SummaryTableSubstitution,
) -> str | None:
    allowed_tables = {detail_alias, detail_table}
    limit = select.args.get("limit")
    if limit is not None:
        order = select.args.get("order")
        if order is None or len(order.expressions) != 1:
            return None
        ordered = order.expressions[0]
        order_expr = ordered.this if isinstance(ordered, exp.Ordered) else ordered
        if not (
            isinstance(order_expr, exp.Column)
            and order_expr.table in allowed_tables
            and order_expr.name == substitution.detail_metric_column
        ):
            return None
        return "order_limit"

    if len(select.expressions) != 1:
        return None
    projection = select.expressions[0]
    aggregate = projection.this if isinstance(projection, exp.Alias) else projection
    if substitution.aggregate_function == "min":
        if not isinstance(aggregate, exp.Min):
            return None
    else:
        return None
    aggregate_arg = aggregate.this
    if not (
        isinstance(aggregate_arg, exp.Column)
        and aggregate_arg.table in allowed_tables
        and aggregate_arg.name == substitution.detail_metric_column
    ):
        return None
    return "aggregate_min"


def _scalar_extrema_predicate_spec(
    predicate: exp.Expression,
    outer_graph: str,
) -> dict[str, Any] | None:
    if not isinstance(predicate, exp.EQ):
        return None
    if isinstance(predicate.left, exp.Column) and isinstance(predicate.right, exp.Subquery):
        outer_column = predicate.left
        subquery = predicate.right
    elif isinstance(predicate.right, exp.Column) and isinstance(predicate.left, exp.Subquery):
        outer_column = predicate.right
        subquery = predicate.left
    else:
        return None
    if not isinstance(subquery.this, exp.Select):
        return None
    inner_select = subquery.this
    if inner_select.args.get("group") is not None or inner_select.args.get("having") is not None:
        return None
    if len(inner_select.expressions) != 1:
        return None
    inner_graph = _select_graph_signature(inner_select)
    if inner_graph != outer_graph:
        return None
    aggregate = inner_select.expressions[0]
    if isinstance(aggregate, exp.Max):
        direction = "DESC"
    elif isinstance(aggregate, exp.Min):
        direction = "ASC"
    else:
        return None
    aggregate_arg = aggregate.this
    if not isinstance(aggregate_arg, exp.Column):
        return None
    if _normalize_sql(outer_column.sql(dialect="sqlite")) != _normalize_sql(
        aggregate_arg.sql(dialect="sqlite")
    ):
        return None
    inner_where = inner_select.args.get("where")
    where_norms = []
    if inner_where is not None and inner_where.this is not None:
        where_norms = [
            _normalize_sql(item.sql(dialect="sqlite"))
            for item in _flatten_and_conditions(inner_where.this)
        ]
    return {
        "outer_column": outer_column,
        "direction": direction,
        "subquery_where_norms": where_norms,
        "predicate_norm": _normalize_sql(predicate.sql(dialect="sqlite")),
    }


def _symmetric_union_shape(ast: exp.Union) -> SymmetricUnionArmPruningMatch | None:
    left = ast.left
    right = ast.right
    if not isinstance(left, exp.Select) or not isinstance(right, exp.Select):
        return None
    left_shape = _symmetric_select_arm(left)
    right_shape = _symmetric_select_arm(right)
    if left_shape is None or right_shape is None:
        return None
    if left_shape.edge_alias != right_shape.edge_alias:
        return None
    if left_shape.literals != tuple(reversed(right_shape.literals)):
        return None
    left_without_where = left.copy()
    right_without_where = right.copy()
    left_without_where.set("where", None)
    right_without_where.set("where", None)
    if _normalize_sql(left_without_where.sql(dialect="sqlite")) != _normalize_sql(
        right_without_where.sql(dialect="sqlite")
    ):
        return None
    canonical_predicate = _canonical_connected_pair_predicate(
        left_shape.edge_alias,
        left_shape.literals[0],
        left_shape.literals[1],
    )
    canonical_select = left.copy()
    canonical_select.set("where", exp.Where(this=canonical_predicate))
    return SymmetricUnionArmPruningMatch(
        shape_type="union",
        edge_table="connected",
        canonical_select=canonical_select,
        target_sql=ast.sql(dialect="sqlite"),
        target_fragment="symmetric UNION arms on connected",
    )


def _symmetric_or_shape(select: exp.Select) -> SymmetricUnionArmPruningMatch | None:
    if not _select_mentions_connected(select):
        return None
    where_clause = select.args.get("where")
    if where_clause is None or where_clause.this is None:
        return None
    or_node = where_clause.this
    if not isinstance(or_node, exp.Or):
        return None
    left_arm = _connected_literal_pair_from_and(or_node.left)
    right_arm = _connected_literal_pair_from_and(or_node.right)
    if left_arm is None or right_arm is None:
        return None
    if left_arm.edge_alias != right_arm.edge_alias:
        return None
    if left_arm.literals != tuple(reversed(right_arm.literals)):
        return None
    return SymmetricUnionArmPruningMatch(
        shape_type="or",
        edge_table="connected",
        target_sql=or_node.sql(dialect="sqlite"),
        canonical_predicate=_canonical_connected_pair_predicate(
            left_arm.edge_alias,
            left_arm.literals[0],
            left_arm.literals[1],
        ),
        target_fragment="symmetric OR arms on connected",
    )


def _symmetric_select_arm(select: exp.Select) -> _SymmetricConnectedArm | None:
    if not _select_mentions_connected(select):
        return None
    where_clause = select.args.get("where")
    if where_clause is None or where_clause.this is None:
        return None
    return _connected_literal_pair_from_and(where_clause.this)


def _select_mentions_connected(select: exp.Select) -> bool:
    from_expr = select.args.get("from_")
    joins = select.args.get("joins") or []
    tables = []
    if from_expr is not None and isinstance(from_expr.this, exp.Table):
        tables.append(from_expr.this)
    tables.extend(join.this for join in joins if isinstance(join.this, exp.Table))
    return any(table.name.lower() == "connected" for table in tables)


def _connected_literal_pair_from_and(expression: exp.Expression) -> _SymmetricConnectedArm | None:
    predicates = _flatten_and_conditions(expression)
    pairs: dict[str, str] = {}
    edge_alias: str | None = None
    for predicate in predicates:
        if not isinstance(predicate, exp.EQ):
            return None
        left = predicate.left
        right = predicate.right
        if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
            column = left
            literal = str(right.this)
        elif isinstance(right, exp.Column) and isinstance(left, exp.Literal):
            column = right
            literal = str(left.this)
        else:
            return None
        if column.name not in {"atom_id", "atom_id2"} or not column.table:
            return None
        if edge_alias is None:
            edge_alias = column.table
        elif edge_alias != column.table:
            return None
        pairs[column.name] = literal
    if edge_alias is None or set(pairs) != {"atom_id", "atom_id2"}:
        return None
    return _SymmetricConnectedArm(
        edge_alias=edge_alias,
        literals=(pairs["atom_id"], pairs["atom_id2"]),
    )


def _canonical_connected_pair_predicate(
    edge_alias: str,
    left_literal: str,
    right_literal: str,
) -> exp.Expression:
    canonical_left, canonical_right = sorted((left_literal, right_literal))
    return exp.and_(
        exp.EQ(
            this=exp.column("atom_id", table=edge_alias),
            expression=exp.Literal.string(canonical_left),
        ),
        exp.EQ(
            this=exp.column("atom_id2", table=edge_alias),
            expression=exp.Literal.string(canonical_right),
        ),
    )


def _flatten_and_conditions(expression: exp.Expression) -> list[exp.Expression]:
    if isinstance(expression, exp.And):
        return _flatten_and_conditions(expression.left) + _flatten_and_conditions(expression.right)
    return [expression]


def _combine_conditions(conditions: list[exp.Expression]) -> exp.Expression | None:
    if not conditions:
        return None
    combined = conditions[0].copy()
    for condition in conditions[1:]:
        combined = exp.and_(combined, condition.copy())
    return combined


def _expression_uses_only_tables(expression: exp.Expression, tables: set[str]) -> bool:
    referenced = {
        str(column.table)
        for column in expression.find_all(exp.Column)
        if column.table is not None
    }
    return referenced <= tables


def _referenced_tables(expression: exp.Expression) -> set[str]:
    return {
        str(column.table)
        for column in expression.find_all(exp.Column)
        if column.table is not None
    }


def _normalize_sql_ignoring_tables(expression: exp.Expression) -> str:
    rewritten = expression.copy()
    for column in rewritten.find_all(exp.Column):
        column.set("table", None)
    return _normalize_sql(rewritten.sql(dialect="sqlite"))


def _match_scalar_extrema_equality(
    predicate: exp.Expression,
) -> tuple[exp.Column, exp.Column, str, exp.Select] | None:
    if not isinstance(predicate, exp.EQ):
        return None
    if isinstance(predicate.left, exp.Column) and isinstance(predicate.right, exp.Subquery):
        outer_col = predicate.left
        subquery = predicate.right
    elif isinstance(predicate.right, exp.Column) and isinstance(predicate.left, exp.Subquery):
        outer_col = predicate.right
        subquery = predicate.left
    else:
        return None
    if not isinstance(subquery.this, exp.Select):
        return None
    inner_select = subquery.this
    if inner_select.args.get("group") is not None or inner_select.args.get("having") is not None:
        return None
    inner_order = inner_select.args.get("order")
    inner_limit = inner_select.args.get("limit")
    if inner_order is None or inner_limit is None or len(inner_order.expressions) != 1:
        return None
    if not isinstance(inner_limit.expression, exp.Literal) or str(inner_limit.expression.this) != "1":
        return None
    if inner_limit.args.get("offset") is not None:
        return None
    inner_expr = inner_select.expressions[0]
    inner_value = inner_expr.this if isinstance(inner_expr, exp.Alias) else inner_expr
    inner_ordered = inner_order.expressions[0]
    inner_order_expr = inner_ordered.this if isinstance(inner_ordered, exp.Ordered) else inner_ordered
    if not isinstance(inner_value, exp.Column) or not isinstance(inner_order_expr, exp.Column):
        return None
    if inner_value.name != inner_order_expr.name or outer_col.name != inner_value.name:
        return None
    direction = "DESC" if getattr(inner_ordered, "args", {}).get("desc") else "ASC"
    return outer_col.copy(), inner_value.copy(), direction, inner_select.copy()


def _distinct_projection_anchor_index(
    select: exp.Select,
    tables: list[exp.Table],
) -> int | None:
    aliases = [table.alias_or_name for table in tables]
    projection_anchor_idx: int | None = None
    for expression in select.expressions:
        referenced_tables = _referenced_tables(expression)
        if not referenced_tables:
            return None
        candidate_indexes = [
            idx
            for idx, table in enumerate(tables)
            if referenced_tables <= {aliases[idx], table.name}
        ]
        if len(candidate_indexes) != 1:
            return None
        candidate_idx = candidate_indexes[0]
        if projection_anchor_idx is None:
            projection_anchor_idx = candidate_idx
        elif projection_anchor_idx != candidate_idx:
            return None
    return projection_anchor_idx


def _joins_are_linear_binary_chain(
    joins: list[exp.Join],
    tables: list[exp.Table],
    aliases: list[str],
) -> bool:
    if len(joins) != len(tables) - 1:
        return False
    for idx, join in enumerate(joins):
        on_clause = join.args.get("on")
        if on_clause is None:
            return False
        predicates = _flatten_and_conditions(on_clause)
        if not predicates:
            return False
        left_refs = {aliases[idx], tables[idx].name}
        right_refs = {aliases[idx + 1], tables[idx + 1].name}
        saw_link_predicate = False
        for predicate in predicates:
            referenced_tables = _referenced_tables(predicate)
            if not referenced_tables:
                return False
            if not referenced_tables <= (left_refs | right_refs):
                return False
            if referenced_tables & left_refs and referenced_tables & right_refs:
                saw_link_predicate = True
        if not saw_link_predicate:
            return False
    return True


def _join_column_pairs(
    joins: list[exp.Join],
) -> list[tuple[str, str, exp.Column, exp.Column]]:
    pairs: list[tuple[str, str, exp.Column, exp.Column]] = []
    for join in joins:
        on_clause = join.args.get("on")
        if on_clause is None:
            continue
        eq_pairs = []
        for predicate in _flatten_and_conditions(on_clause):
            if not isinstance(predicate, exp.EQ):
                continue
            left = predicate.left
            right = predicate.right
            if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                continue
            if not left.table or not right.table:
                continue
            eq_pairs.append((str(left.table), str(right.table), left.copy(), right.copy()))
        if len(eq_pairs) == 1:
            pairs.append(eq_pairs[0])
    return pairs


def _orient_pair(
    pair: tuple[exp.Column, exp.Column],
    left_alias: str,
    right_alias: str,
) -> tuple[exp.Column, exp.Column] | None:
    left_col, right_col = pair
    if str(left_col.table) == left_alias and str(right_col.table) == right_alias:
        return left_col, right_col
    if str(right_col.table) == left_alias and str(left_col.table) == right_alias:
        return right_col, left_col
    return None


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.lower().split())


def _is_safe_conditional_aggregate_argument(expression: exp.Expression | None) -> bool:
    if expression is None:
        return False
    if isinstance(expression, exp.Column):
        return True
    if isinstance(expression, exp.Literal):
        return True
    if isinstance(expression, exp.Cast):
        return _is_safe_conditional_aggregate_argument(expression.this)
    return False
