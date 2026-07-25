"""Conservative, gold-free checks for LLM-proposed SQL replacements."""

from __future__ import annotations

import re

from sqlglot import exp, parse_one


def _parse(sql: str) -> exp.Expression | None:
    try:
        return parse_one(sql, dialect="sqlite")
    except Exception:
        return None


def _projection_width(tree: exp.Expression) -> int | None:
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    return len(select.expressions) if select is not None else None


def _has_wildcard_projection(tree: exp.Expression) -> bool:
    return any(
        isinstance(expression, exp.Star)
        or any(expression.find_all(exp.Star))
        for select in tree.find_all(exp.Select)
        for expression in select.expressions
    )


def _literal_predicates(tree: exp.Expression) -> set[tuple[str, str]]:
    predicates: set[tuple[str, str]] = set()
    for equality in tree.find_all(exp.EQ):
        pairs = ((equality.this, equality.expression), (equality.expression, equality.this))
        for column, literal in pairs:
            if isinstance(column, exp.Column) and isinstance(literal, exp.Literal):
                predicates.add((column.name.lower(), str(literal.this).lower()))
                break
    return predicates


def _literal_values_by_column(
    tree: exp.Expression,
) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    for column, literal in _column_literal_pairs(tree, exp.EQ):
        values.setdefault(column.name.lower(), set()).add(
            str(literal.this).lower()
        )
    return values


def _columns_by_literal(
    tree: exp.Expression,
) -> dict[str, set[str]]:
    columns: dict[str, set[str]] = {}
    for column, literal in _column_literal_pairs(tree, exp.EQ):
        columns.setdefault(str(literal.this).lower(), set()).add(
            column.name.lower()
        )
    return columns


def _column_literal_pairs(
    tree: exp.Expression,
    expression_type: type[exp.Expression],
) -> list[tuple[exp.Column, exp.Literal]]:
    pairs: list[tuple[exp.Column, exp.Literal]] = []
    for predicate in tree.find_all(expression_type):
        sides = (
            (predicate.this, predicate.expression),
            (predicate.expression, predicate.this),
        )
        for column, literal in sides:
            if isinstance(column, exp.Column) and isinstance(literal, exp.Literal):
                pairs.append((column, literal))
                break
    return pairs


def _normalized_pattern(value: str) -> str:
    return value.lower().strip().strip("%").strip("_")


def _pattern_values_by_column(
    tree: exp.Expression,
) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    for column, literal in _column_literal_pairs(tree, exp.Like):
        values.setdefault(column.name.lower(), set()).add(
            _normalized_pattern(str(literal.this))
        )
    return values


def _table_aliases(tree: exp.Expression) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for table in tree.find_all(exp.Table):
        if not table.name:
            continue
        table_name = table.name.lower()
        aliases[table_name] = table_name
        if table.alias:
            aliases[table.alias.lower()] = table_name
    return aliases


def _qualified_literal_predicates(
    tree: exp.Expression,
) -> dict[tuple[str, str], set[str]]:
    aliases = _table_aliases(tree)
    table_names = set(aliases.values())
    sole_table = next(iter(table_names)) if len(table_names) == 1 else ""
    predicates: dict[tuple[str, str], set[str]] = {}
    for equality in tree.find_all(exp.EQ):
        pairs = ((equality.this, equality.expression), (equality.expression, equality.this))
        for column, literal in pairs:
            if not isinstance(column, exp.Column) or not isinstance(literal, exp.Literal):
                continue
            qualifier = column.table.lower() if column.table else ""
            table_name = aliases.get(qualifier, qualifier) if qualifier else sole_table
            key = (column.name.lower(), str(literal.this).lower())
            predicates.setdefault(key, set()).add(table_name)
            break
    return predicates


def _has_distinct_aggregate(tree: exp.Expression) -> bool:
    return any(
        isinstance(function.this, exp.Distinct)
        for function in tree.find_all(exp.AggFunc)
    )


def _contains(tree: exp.Expression, expression_type: type[exp.Expression]) -> bool:
    return any(tree.find_all(expression_type))


def _has_non_window_aggregate(tree: exp.Expression) -> bool:
    return any(
        function.find_ancestor(exp.Window) is None
        for function in tree.find_all(exp.AggFunc)
    )


def _has_select_distinct(tree: exp.Expression) -> bool:
    return any(
        select.args.get("distinct") is not None
        for select in tree.find_all(exp.Select)
    )


def _outer_join_count(tree: exp.Expression) -> int:
    return sum(
        str(join.side or "").upper() in {"LEFT", "RIGHT", "FULL"}
        for join in tree.find_all(exp.Join)
    )


def _ordering_columns(tree: exp.Expression) -> set[str]:
    return {
        column.name.lower()
        for order in tree.find_all(exp.Order)
        for ordered in order.expressions
        for column in ordered.find_all(exp.Column)
    }


def _comparison_operators(
    tree: exp.Expression,
) -> dict[tuple[str, str], str]:
    operators: dict[tuple[str, str], str] = {}
    for operator_type in (exp.GT, exp.GTE, exp.LT, exp.LTE):
        for predicate in tree.find_all(operator_type):
            if not isinstance(predicate.expression, exp.Literal):
                continue
            key = (
                predicate.this.sql(dialect="sqlite").lower(),
                str(predicate.expression.this).lower(),
            )
            operators[key] = operator_type.__name__.lower()
    return operators


def _predicate_cast_count(tree: exp.Expression) -> int:
    return sum(
        cast.find_ancestor(exp.Where, exp.Having) is not None
        for cast in tree.find_all(exp.Cast)
    )


def _filtered_column_names(tree: exp.Expression) -> set[str]:
    return {
        column.name.lower()
        for clause_type in (exp.Where, exp.Having)
        for clause in tree.find_all(clause_type)
        for column in clause.find_all(exp.Column)
    }


def _predicate_function_names(tree: exp.Expression) -> set[str]:
    return {
        function.key.lower()
        for clause_type in (exp.Where, exp.Having)
        for clause in tree.find_all(clause_type)
        for function in clause.find_all(exp.Func)
    }


def _aggregate_function_names(tree: exp.Expression) -> set[str]:
    return {
        function.key.lower()
        for function in tree.find_all(exp.AggFunc)
        if function.find_ancestor(exp.Window) is None
    }


def _projection_comparison_count(tree: exp.Expression) -> int:
    comparison_types = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)
    return sum(
        1
        for select in tree.find_all(exp.Select)
        for expression in select.expressions
        for comparison_type in comparison_types
        for _ in expression.find_all(comparison_type)
    )


def _projected_column_names(tree: exp.Expression) -> set[str]:
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if select is None:
        return set()
    return {
        column.name.lower()
        for expression in select.expressions
        for column in expression.find_all(exp.Column)
    }


def _filtered_in_columns(tree: exp.Expression) -> set[str]:
    return {
        expression.this.name.lower()
        for expression in tree.find_all(exp.In)
        if isinstance(expression.this, exp.Column)
    }


def _intent_mentions_literal(intent: str, literal: str) -> bool:
    normalized_intent = intent.lower()
    normalized_literal = literal.lower().strip()
    if normalized_literal in {"0", "1"}:
        return True
    return normalized_literal in normalized_intent


def validate_semantic_change(
    original_sql: str,
    proposed_sql: str,
    question: str,
    evidence: str,
) -> tuple[str, ...]:
    """Return rejection reasons for an insufficiently grounded SQL change."""

    original = _parse(original_sql)
    proposed = _parse(proposed_sql)
    if original is None or proposed is None:
        return ("unparseable_sql",)

    reasons: list[str] = []
    original_width = _projection_width(original)
    proposed_width = _projection_width(proposed)
    if (
        original_width is not None
        and proposed_width is not None
        and proposed_width < original_width
    ):
        reasons.append("projection_arity_reduction")
    if (
        original_width is not None
        and proposed_width is not None
        and proposed_width > original_width
    ):
        reasons.append("projection_arity_increase")

    if _has_wildcard_projection(proposed) and not _has_wildcard_projection(original):
        reasons.append("new_wildcard_projection")

    intent = f"{question}\n{evidence}"
    added_predicates = _literal_predicates(proposed) - _literal_predicates(original)
    if any(
        not _intent_mentions_literal(intent, literal)
        for _, literal in added_predicates
    ):
        reasons.append("ungrounded_new_literal_filter")

    original_has_multiplication = any(original.find_all(exp.Mul))
    proposed_has_multiplication = any(proposed.find_all(exp.Mul))
    multiplication_is_explicit = bool(
        re.search(
            r"\*|\bmultiply\b|\bmultiplied\b|\bproduct\s+of\b|\btimes\b|相乘|乘以",
            intent,
            flags=re.IGNORECASE,
        )
    )
    if (
        proposed_has_multiplication
        and not original_has_multiplication
        and not multiplication_is_explicit
    ):
        reasons.append("ungrounded_new_multiplication")

    if _has_distinct_aggregate(proposed) and not _has_distinct_aggregate(original):
        reasons.append("new_distinct_aggregate")

    if _has_non_window_aggregate(proposed) and not _has_non_window_aggregate(
        original
    ):
        reasons.append("new_aggregate_operation")

    if _contains(proposed, exp.Or) and not _contains(original, exp.Or):
        reasons.append("new_boolean_disjunction")

    if _contains(proposed, exp.SetOperation) and not _contains(
        original, exp.SetOperation
    ):
        reasons.append("new_set_operation")

    if _contains(proposed, exp.Exists) and not _contains(original, exp.Exists):
        reasons.append("new_exists_semantics")

    if _contains(original, exp.Limit) and not _contains(proposed, exp.Limit):
        reasons.append("removed_row_limit")
    if _contains(proposed, exp.Limit) and not _contains(original, exp.Limit):
        reasons.append("new_row_limit")

    if _outer_join_count(proposed) > _outer_join_count(original):
        reasons.append("new_outer_join")

    if _has_select_distinct(proposed) and not _has_select_distinct(original):
        reasons.append("new_select_distinct")

    original_literals = _literal_values_by_column(original)
    proposed_literals = _literal_values_by_column(proposed)
    if any(
        column in proposed_literals
        and original_values
        and proposed_literals[column]
        and original_values.isdisjoint(proposed_literals[column])
        for column, original_values in original_literals.items()
    ):
        reasons.append("replaced_equality_literal")

    original_columns_by_literal = _columns_by_literal(original)
    proposed_columns_by_literal = _columns_by_literal(proposed)
    if any(
        literal in proposed_columns_by_literal
        and original_columns
        and proposed_columns_by_literal[literal]
        and original_columns.isdisjoint(proposed_columns_by_literal[literal])
        for literal, original_columns in original_columns_by_literal.items()
    ):
        reasons.append("relocated_literal_value")

    original_patterns = _pattern_values_by_column(original)
    proposed_equalities = {
        column: {_normalized_pattern(value) for value in values}
        for column, values in proposed_literals.items()
    }
    if any(
        column in proposed_equalities
        and pattern_values
        and proposed_equalities[column]
        and pattern_values.isdisjoint(proposed_equalities[column])
        for column, pattern_values in original_patterns.items()
    ):
        reasons.append("changed_pattern_literal")

    original_ordering = _ordering_columns(original)
    proposed_ordering = _ordering_columns(proposed)
    if (
        original_ordering
        and proposed_ordering
        and original_ordering != proposed_ordering
    ):
        reasons.append("changed_ordering_column")

    original_operators = _comparison_operators(original)
    proposed_operators = _comparison_operators(proposed)
    if any(
        key in proposed_operators and proposed_operators[key] != operator
        for key, operator in original_operators.items()
    ):
        reasons.append("changed_comparison_strictness")
    if (
        _contains(original, exp.GTE)
        and _contains(proposed, exp.GT)
        and not _contains(proposed, exp.GTE)
    ) or (
        _contains(original, exp.LTE)
        and _contains(proposed, exp.LT)
        and not _contains(proposed, exp.LTE)
    ):
        reasons.append("tightened_comparison_operator")

    if (
        _predicate_cast_count(proposed) > _predicate_cast_count(original)
        and set(_table_aliases(proposed).values())
        == set(_table_aliases(original).values())
    ):
        reasons.append("new_predicate_cast")

    if (
        _filtered_column_names(proposed) - _filtered_column_names(original)
        and set(_table_aliases(proposed).values())
        == set(_table_aliases(original).values())
    ):
        reasons.append("new_filtered_column")

    if (
        _predicate_function_names(proposed) - _predicate_function_names(original)
        and set(_table_aliases(proposed).values())
        == set(_table_aliases(original).values())
    ):
        reasons.append("new_predicate_function")

    original_aggregates = _aggregate_function_names(original)
    proposed_aggregates = _aggregate_function_names(proposed)
    if (
        "count" in original_aggregates
        and "count" not in proposed_aggregates
        and "sum" in proposed_aggregates
    ):
        reasons.append("replaced_count_with_sum")

    if _projection_comparison_count(proposed) > _projection_comparison_count(
        original
    ):
        reasons.append("new_projection_comparison")

    original_qualified = _qualified_literal_predicates(original)
    proposed_qualified = _qualified_literal_predicates(proposed)
    if any(
        key in original_qualified
        and original_qualified[key]
        and proposed_tables
        and original_qualified[key].isdisjoint(proposed_tables)
        for key, proposed_tables in proposed_qualified.items()
    ):
        reasons.append("relocated_literal_filter")

    added_in_filters = _filtered_in_columns(proposed) - _filtered_in_columns(original)
    if added_in_filters & _projected_column_names(original):
        reasons.append("new_filter_on_projected_column")

    return tuple(dict.fromkeys(reasons))
