"""Tool: parse_sql_structure.

Parses a SQL query into a compact structural summary: referenced tables and
columns, joins, predicates, grouping, ordering, subqueries, and static risk
patterns that are useful before looking at DBMS execution plans.
"""

from __future__ import annotations

from typing import Iterable

import sqlglot
import sqlglot.expressions as exp

from agent.explainAnalyserAgent.utils.common import dialect_for_dbms
from agent.explainAnalyserAgent.utils.common import unique_preserve_order
from agent.explainAnalyserAgent.utils.models import JoinInfo
from agent.explainAnalyserAgent.utils.models import ParseSqlStructureInput
from agent.explainAnalyserAgent.utils.models import ParseSqlStructureOutput
from agent.explainAnalyserAgent.utils.models import PredicateInfo
from agent.explainAnalyserAgent.utils.models import SubqueryInfo


def parse_sql_structure(input_data: ParseSqlStructureInput) -> ParseSqlStructureOutput:
    """Parse SQL with sqlglot and return a DBMS-neutral structure summary."""
    warnings: list[str] = []
    try:
        ast = sqlglot.parse_one(
            input_data.sql,
            dialect=dialect_for_dbms(input_data.dbms),
        )
    except Exception as exc:
        return ParseSqlStructureOutput(
            statement_type="unknown",
            parse_warnings=[f"SQL parse failed: {exc}"],
        )

    risky_patterns: list[str] = []
    tables = unique_preserve_order([table.name for table in ast.find_all(exp.Table)])
    columns = unique_preserve_order([_column_name(col) for col in ast.find_all(exp.Column)])
    joins = _extract_joins(ast)
    predicates = _extract_predicates(ast)
    subqueries = _extract_subqueries(ast)
    group_by = _expressions_to_sql(_group_expressions(ast), input_data.dbms)
    order_by = _expressions_to_sql(_order_expressions(ast), input_data.dbms)
    limit = _extract_limit(ast)

    if any(subquery.correlated for subquery in subqueries):
        risky_patterns.append("correlated_subquery")
    if any(join.condition is None for join in joins):
        risky_patterns.append("join_without_condition")
    if any(pattern in pred.risky_patterns for pred in predicates for pattern in pred.risky_patterns):
        risky_patterns.extend(
            pattern
            for pred in predicates
            for pattern in pred.risky_patterns
            if pattern not in risky_patterns
        )

    has_select_star = any(isinstance(node, exp.Star) for node in ast.find_all(exp.Star))
    if has_select_star:
        risky_patterns.append("select_star")

    statement_type = "select"
    if ast.args.get("with"):
        statement_type = "with"
    if isinstance(ast, exp.Union):
        statement_type = "select"

    return ParseSqlStructureOutput(
        statement_type=statement_type,
        tables=tables,
        columns=columns,
        joins=joins,
        predicates=predicates,
        group_by=group_by,
        order_by=order_by,
        limit=limit,
        subqueries=subqueries,
        has_distinct=bool(ast.find(exp.Distinct)),
        has_union=bool(ast.find(exp.Union)),
        has_window_functions=bool(ast.find(exp.Window)),
        has_select_star=has_select_star,
        risky_patterns=unique_preserve_order(risky_patterns),
        parse_warnings=warnings,
    )


def _extract_joins(ast: exp.Expression) -> list[JoinInfo]:
    joins: list[JoinInfo] = []
    for join in ast.find_all(exp.Join):
        right_table = None
        if isinstance(join.this, exp.Table):
            right_table = join.this.name
        condition = join.args.get("on")
        condition_sql = condition.sql() if condition is not None else None
        joins.append(
            JoinInfo(
                join_type=(join.args.get("kind") or "inner").upper(),
                left_table=None,
                right_table=right_table,
                condition=condition_sql,
                columns=[_column_name(col) for col in condition.find_all(exp.Column)]
                if condition is not None
                else [],
            )
        )
    return joins


def _extract_predicates(ast: exp.Expression) -> list[PredicateInfo]:
    predicates: list[PredicateInfo] = []
    for clause_name, expression_type in (
        ("where", exp.Where),
        ("having", exp.Having),
    ):
        clause = ast.find(expression_type)
        if clause is not None and clause.this is not None:
            predicates.append(_predicate_from_expression(clause_name, clause.this))

    for join in ast.find_all(exp.Join):
        condition = join.args.get("on")
        if condition is not None:
            predicates.append(_predicate_from_expression("join", condition))
    return predicates


def _predicate_from_expression(clause: str, expression: exp.Expression) -> PredicateInfo:
    columns = [_column_name(col) for col in expression.find_all(exp.Column)]
    tables = unique_preserve_order([col.table for col in expression.find_all(exp.Column) if col.table])
    operators = unique_preserve_order(type(node).__name__ for node in expression.walk())
    risky_patterns = _detect_predicate_risks(expression)
    return PredicateInfo(
        clause=clause if clause in {"where", "having", "join", "case"} else "unknown",
        expression=expression.sql(),
        columns=unique_preserve_order(columns),
        tables=tables,
        operators=operators,
        risky_patterns=risky_patterns,
    )


def _detect_predicate_risks(expression: exp.Expression) -> list[str]:
    risks: list[str] = []
    for like in expression.find_all(exp.Like):
        pattern = like.expression
        if isinstance(pattern, exp.Literal) and str(pattern.this).startswith("%"):
            risks.append("leading_wildcard_like")
    for func in expression.find_all(exp.Func):
        if any(isinstance(child, exp.Column) for child in func.find_all(exp.Column)):
            risks.append("function_on_column")
    if expression.find(exp.Or):
        risks.append("or_predicate")
    if expression.find(exp.Not) and " IN " in expression.sql().upper():
        risks.append("not_in_nullable")
    return unique_preserve_order(risks)


def _extract_subqueries(ast: exp.Expression) -> list[SubqueryInfo]:
    outer_aliases = _table_aliases(ast)
    subqueries: list[SubqueryInfo] = []
    for subquery in ast.find_all(exp.Subquery):
        inner_aliases = _table_aliases(subquery)
        outer_refs = [
            _column_name(col)
            for col in subquery.find_all(exp.Column)
            if col.table and col.table in outer_aliases and col.table not in inner_aliases
        ]
        subqueries.append(
            SubqueryInfo(
                location=_subquery_location(subquery),
                query=subquery.sql(),
                correlated=bool(outer_refs),
                referenced_outer_columns=unique_preserve_order(outer_refs),
            )
        )
    return subqueries


def _table_aliases(expression: exp.Expression) -> set[str]:
    aliases: set[str] = set()
    for table in expression.find_all(exp.Table):
        aliases.add(table.alias_or_name)
        aliases.add(table.name)
    return aliases


def _subquery_location(subquery: exp.Subquery) -> str:
    parent = subquery.parent
    while parent is not None:
        if isinstance(parent, exp.From):
            return "from"
        if isinstance(parent, exp.Where):
            return "where"
        if isinstance(parent, exp.Having):
            return "having"
        if isinstance(parent, exp.Select):
            return "select"
        parent = parent.parent
    return "unknown"


def _group_expressions(ast: exp.Expression) -> Iterable[exp.Expression]:
    group = ast.args.get("group")
    return group.expressions if group is not None else []


def _order_expressions(ast: exp.Expression) -> Iterable[exp.Expression]:
    order = ast.args.get("order")
    return order.expressions if order is not None else []


def _expressions_to_sql(expressions: Iterable[exp.Expression], dbms: str) -> list[str]:
    return [expression.sql(dialect=dialect_for_dbms(dbms)) for expression in expressions]


def _extract_limit(ast: exp.Expression) -> int | None:
    limit = ast.args.get("limit")
    if limit is None or limit.expression is None:
        return None
    try:
        return int(str(limit.expression.this))
    except Exception:
        return None


def _column_name(column: exp.Column) -> str:
    table = column.table
    name = column.name
    return f"{table}.{name}" if table else name
