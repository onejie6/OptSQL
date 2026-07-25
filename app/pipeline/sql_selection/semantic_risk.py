from __future__ import annotations

import re
from typing import Iterable

from sqlglot import exp, parse_one


_NULL_INTENT_TERMS = (
    "not null",
    "non-null",
    "nonnull",
    "known value",
    "available value",
    "missing value",
)
_TIE_INTENT_TERMS = ("tie", "ties", "tied", "dense rank", "without gaps")
_LATEST_INTENT_TERMS = (
    "latest",
    "most recent",
    "newest",
    "current",
    "last recorded",
)
_AVERAGE_INTENT_TERMS = ("average", "avg", "mean")
_MAX_INTENT_TERMS = ("maximum", "max", "highest", "largest", "top", *_LATEST_INTENT_TERMS)
_MIN_INTENT_TERMS = ("minimum", "min", "lowest", "smallest", "bottom")
_FORMULA_OPERATORS = (exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Mod, exp.Pow)


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    normalized = re.sub(r"\s+", " ", text.lower())
    return any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized) for term in terms)


def _has_not_null_predicate(tree: exp.Expression) -> bool:
    return any(
        isinstance(node.parent, exp.Not) and isinstance(node.expression, exp.Null)
        for node in tree.find_all(exp.Is)
    )


def _window_aliases(select: exp.Select) -> set[str]:
    aliases = set()
    for projection in select.expressions:
        if projection.find(exp.Window) is not None and projection.alias:
            aliases.add(projection.alias.lower())
    return aliases


def _cte_map(tree: exp.Expression) -> dict[str, exp.Select]:
    result = {}
    for cte in tree.find_all(exp.CTE):
        inner_select = cte.this if isinstance(cte.this, exp.Select) else cte.this.find(exp.Select)
        if inner_select is not None:
            result[cte.alias_or_name.lower()] = inner_select
    return result


def _derived_window_aliases(select: exp.Select, ctes: dict[str, exp.Select]) -> set[str]:
    from_clause = select.args.get("from_")
    if from_clause is None:
        return set()
    source = from_clause.this
    if isinstance(source, exp.Subquery):
        inner_select = source.this if isinstance(source.this, exp.Select) else source.this.find(exp.Select)
        return _window_aliases(inner_select) if inner_select is not None else set()
    if isinstance(source, exp.Table):
        inner_select = ctes.get(source.name.lower())
        return _window_aliases(inner_select) if inner_select is not None else set()
    return set()


def _has_late_population_filter(tree: exp.Expression) -> bool:
    ctes = _cte_map(tree)
    for select in tree.find_all(exp.Select):
        window_aliases = _derived_window_aliases(select, ctes)
        where = select.args.get("where")
        if not window_aliases or where is None:
            continue
        direct_columns = {
            column.name.lower()
            for column in where.find_all(exp.Column)
            if column.find_ancestor(exp.Subquery) is None
        }
        if direct_columns - window_aliases:
            return True
    return False


def _has_unsupported_latest_row(tree: exp.Expression, intent: str) -> bool:
    if _contains_any(intent, (*_LATEST_INTENT_TERMS, *_MAX_INTENT_TERMS, *_MIN_INTENT_TERMS)):
        return False
    for select in tree.find_all(exp.Select):
        limit = select.args.get("limit")
        order = select.args.get("order")
        if limit is None or order is None:
            continue
        limit_value = limit.expression
        if not isinstance(limit_value, exp.Literal) or limit_value.this != "1":
            continue
        order_columns = " ".join(column.name.lower() for column in order.find_all(exp.Column))
        if any(term in order_columns for term in ("date", "time", "year", "created", "updated")):
            return True
    return False


def _counted_table_names(intent: str, table_names: Iterable[str]) -> set[str]:
    counted_terms = {
        match.group(1).rstrip("s")
        for match in re.finditer(
            r"count\s*[\[(]\s*(?:distinct\s+)?(?:female\s+|male\s+)?([a-z_]+)",
            intent.lower(),
        )
    }
    return {name for name in table_names if name.lower().rstrip("s") in counted_terms}


def _has_redundant_bridge_over_direct_fk(
    tree: exp.Expression,
    intent: str,
    schema: dict | None,
) -> bool:
    tables = schema.get("tables", {}) if isinstance(schema, dict) else {}
    if not tables:
        return False
    sql_tables = {table.name.lower() for table in tree.find_all(exp.Table)}
    counted_tables = _counted_table_names(intent, tables)
    for source_name in counted_tables:
        source = tables[source_name]
        for column in source.get("columns", {}).values():
            for target_name, _ in column.get("foreign_keys", []):
                if (
                    source_name.lower() in sql_tables
                    and target_name.lower() in sql_tables
                    and len(sql_tables) > 2
                ):
                    return True
    return False


def _unsupported_aggregate_findings(tree: exp.Expression, intent: str) -> list[str]:
    findings = []
    if tree.find(exp.Avg) is not None and not _contains_any(intent, _AVERAGE_INTENT_TERMS):
        findings.append("unsupported_average")
    if tree.find(exp.Max) is not None and not _contains_any(intent, _MAX_INTENT_TERMS):
        findings.append("unsupported_maximum")
    if tree.find(exp.Min) is not None and not _contains_any(intent, _MIN_INTENT_TERMS):
        findings.append("unsupported_minimum")
    return findings


def _has_evidence_formula_source_mismatch(tree: exp.Expression, evidence: str) -> bool:
    source_columns = {
        reference.rsplit(".", 1)[-1].strip().lower()
        for reference in re.findall(r"`([^`]+)`", evidence)
        if reference.strip()
    }
    function_sources = {
        reference.strip().lower()
        for reference in re.findall(
            r"\b(?:sum|avg|average|min|max)\s*\(\s*(?:distinct\s+)?"
            r"[`\"]?([A-Za-z_][A-Za-z0-9_]*)[`\"]?\s*\)",
            evidence,
            re.IGNORECASE,
        )
    }
    source_columns.update(function_sources)
    if not source_columns:
        return False

    sql_columns = {column.name.lower() for column in tree.find_all(exp.Column)}
    if not source_columns.issubset(sql_columns):
        return True

    evidence_has_arithmetic = re.search(
        r"(?:[+*/-]|\b(?:add|subtract|multiply|divide)\s*\()",
        evidence,
        re.IGNORECASE,
    )
    has_arithmetic = any(tree.find(operator) is not None for operator in _FORMULA_OPERATORS)
    return bool(evidence_has_arithmetic and not has_arithmetic)


def _has_optional_join_for_required_projection(tree: exp.Expression) -> bool:
    for select in tree.find_all(exp.Select):
        projected_tables = {
            column.table.lower()
            for projection in select.expressions
            for column in projection.find_all(exp.Column)
            if column.table
        }
        if not projected_tables:
            continue
        for join in select.args.get("joins") or []:
            if str(join.args.get("side") or "").upper() != "LEFT":
                continue
            joined_alias = join.this.alias_or_name.lower()
            if joined_alias and joined_alias in projected_tables:
                return True
    return False


def _has_missing_account_owner_role(tree: exp.Expression, question: str) -> bool:
    question_lower = question.lower()
    if not (
        "loan" in question_lower
        and "account" in question_lower
        and "client" in question_lower
    ):
        return False
    tables = {table.name.lower() for table in tree.find_all(exp.Table)}
    if not {"loan", "disp", "client"}.issubset(tables):
        return False
    for equality in tree.find_all(exp.EQ):
        left, right = equality.this, equality.expression
        pairs = ((left, right), (right, left))
        for column, literal in pairs:
            if (
                isinstance(column, exp.Column)
                and column.name.lower() == "type"
                and isinstance(literal, exp.Literal)
                and literal.is_string
                and str(literal.this).upper() == "OWNER"
            ):
                return False
    return True


def _has_denominator_cohort_mismatch(
    tree: exp.Expression,
    question: str,
    evidence: str,
) -> bool:
    cohort_match = re.search(
        r"\bpercentage\s+of\s+(?:the\s+)?(.+?)\s+(?:who|that|which|whose|with)\b",
        question,
        re.IGNORECASE,
    )
    if not cohort_match or tree.find(exp.Count) is None:
        return False

    cohort = cohort_match.group(1).lower().rstrip("s")
    expected = {
        (column.lower(), value.lower())
        for column, value in re.findall(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*['\"]([^'\"]+)['\"]",
            evidence,
        )
        if value.lower().rstrip("s") in cohort
    }
    if not expected:
        return False

    def equality_binding(node: exp.Expression) -> tuple[str, str] | None:
        if not isinstance(node, exp.EQ):
            return None
        left, right = node.this, node.expression
        if isinstance(left, exp.Column) and isinstance(right, exp.Literal) and right.is_string:
            return left.name.lower(), str(right.this).lower()
        if isinstance(right, exp.Column) and isinstance(left, exp.Literal) and left.is_string:
            return right.name.lower(), str(left.this).lower()
        return None

    for equality in tree.find_all(exp.EQ):
        binding = equality_binding(equality)
        if binding not in expected:
            continue
        if equality.find_ancestor(exp.Where) is not None or equality.find_ancestor(exp.Filter) is not None:
            return False
        for division in tree.find_all(exp.Div):
            denominator = division.expression
            if any(equality is nested for nested in denominator.walk()):
                return False
    return True


def _has_unsupported_type_coercion(
    tree: exp.Expression,
    evidence: str,
    schema: dict | None,
) -> bool:
    if _contains_any(evidence, ("cast", "convert", "conversion", "numeric")):
        return False
    tables = schema.get("tables", {}) if isinstance(schema, dict) else {}
    text_columns = {
        column_name.lower()
        for table in tables.values()
        for column_name, column in table.get("columns", {}).items()
        if str(column.get("column_type", "")).upper() in {"TEXT", "CHAR", "VARCHAR"}
    }
    for cast in tree.find_all(exp.Cast):
        column = cast.this if isinstance(cast.this, exp.Column) else None
        if column is not None and column.name.lower() in text_columns:
            return True
    return False


def _has_structured_relationship_decode_mismatch(
    sql: str,
    tree: exp.Expression,
    question: str,
    evidence: str,
    schema: dict | None,
) -> bool:
    structured_ids = re.findall(
        r"\b([A-Za-z0-9]+(?:_[A-Za-z0-9]+){2,})\b",
        f"{question} {evidence}",
    )
    asks_for_endpoints = re.search(
        r"\b(?:what|which)\s+(?:are\s+the\s+)?(?:two\s+)?"
        r"(?:atoms?|nodes?|vertices?|endpoints?)\b",
        question,
        re.IGNORECASE,
    )
    if not structured_ids or asks_for_endpoints is None:
        return False

    relation_match = re.search(
        r"\bis\s+the\s+([A-Za-z_][A-Za-z0-9_]*)\s+id\b",
        evidence,
        re.IGNORECASE,
    )
    tables = schema.get("tables", {}) if isinstance(schema, dict) else {}
    if relation_match:
        relation_name = relation_match.group(1).lower().rstrip("s")
        relation_table = next(
            (
                table
                for table_name, table in tables.items()
                if table_name.lower().rstrip("s") == relation_name
            ),
            None,
        )
        if relation_table is not None:
            endpoint_ids = [
                name
                for name in relation_table.get("columns", {})
                if name.lower().endswith("_id")
                and name.lower().removesuffix("_id") not in {relation_name, "molecule"}
            ]
            if len(endpoint_ids) >= 2:
                return False

    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    projection_count = len(select.expressions) if select is not None else 0
    decodes_identifier = bool(re.search(r"\b(?:substr|substring|instr)\s*\(", sql, re.IGNORECASE))
    return tree.find(exp.Union) is not None or projection_count < 2 or not decodes_identifier


def _has_formula_output_arity_mismatch(
    tree: exp.Expression,
    question: str,
    evidence: str,
) -> bool:
    formula_outputs = [
        match.group(1).strip().lower()
        for match in re.finditer(
            r"(?:^|;)\s*([^=;]+?)\s*=\s*(?:add|subtract|multiply|divide)\s*\(",
            evidence,
            re.IGNORECASE,
        )
    ]
    if not any(output in question.lower() for output in formula_outputs):
        return False
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    return select is not None and len(select.expressions) != 1


def _has_duplicate_attribute_projection(tree: exp.Expression, question: str) -> bool:
    if re.search(r"\b(?:and|along with|together with)\b", question, re.IGNORECASE):
        return False
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if select is None or len(select.expressions) <= 1:
        return False
    projected_columns = []
    for projection in select.expressions:
        expression = projection.this if isinstance(projection, exp.Alias) else projection
        if not isinstance(expression, exp.Column):
            return False
        projected_columns.append({expression.name.lower()})
    return bool(
        projected_columns
        and all(len(columns) == 1 for columns in projected_columns)
        and len(set().union(*projected_columns)) == 1
    )


def _outer_projection_names(tree: exp.Expression) -> set[str]:
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if select is None:
        return set()
    names = set()
    for projection in select.expressions:
        for column in projection.find_all(exp.Column):
            names.add(column.name.lower())
    return names


def _has_projection_mapping_mismatch(
    tree: exp.Expression,
    question: str,
    evidence: str,
    schema: dict | None,
) -> bool:
    expected = set()
    question_lower = question.lower()
    for clause in evidence.split(";"):
        mapping = re.match(
            r"\s*(.+?)\s+refers\s+to\s+([A-Za-z_][A-Za-z0-9_.]*)\s*$",
            clause,
            re.IGNORECASE,
        )
        if mapping and mapping.group(1).strip().lower() in question_lower:
            expected.add(mapping.group(2).split(".")[-1].lower())

    tables = schema.get("tables", {}) if isinstance(schema, dict) else {}
    for table_name, table in tables.items():
        phrase = table_name.replace("_", " ").lower()
        if not re.search(
            rf"(?<!\w)names?\s+of(?:\s+\w+){{0,4}}\s+{re.escape(phrase)}(?!\w)",
            question_lower,
        ):
            continue
        singular = phrase.rstrip("s")
        for column_name in table.get("columns", {}):
            if column_name.lower().replace("_", " ") in {"name", singular}:
                expected.add(column_name.lower())

    return bool(expected and not expected.issubset(_outer_projection_names(tree)))


def audit_semantic_risks(
    sql: str,
    question: str,
    evidence: str,
    *,
    dialect: str = "sqlite",
    schema: dict | None = None,
) -> tuple[str, ...]:
    """Return conservative, gold-free semantic-risk findings for candidate selection."""
    try:
        tree = parse_one(sql, dialect=dialect)
    except Exception:
        return ("unparseable_sql",)

    intent = f"{question}\n{evidence}"
    findings = []
    if _has_not_null_predicate(tree) and not _contains_any(intent, _NULL_INTENT_TERMS):
        findings.append("unsupported_not_null_filter")
    if (
        tree.find(exp.Distinct) is not None
        and tree.find(exp.Join) is None
        and not _contains_any(intent, ("distinct", "unique"))
    ):
        findings.append("unsupported_distinct")
    if tree.find(exp.DenseRank) is not None and not _contains_any(intent, _TIE_INTENT_TERMS):
        findings.append("unsupported_dense_rank")
    if _has_late_population_filter(tree):
        findings.append("population_filter_after_window")
    if _has_unsupported_latest_row(tree, intent):
        findings.append("unsupported_latest_row")
    findings.extend(_unsupported_aggregate_findings(tree, intent))
    if _has_redundant_bridge_over_direct_fk(tree, intent, schema):
        findings.append("redundant_bridge_over_direct_fk")
    if _has_projection_mapping_mismatch(tree, question, evidence, schema):
        findings.append("projection_mapping_mismatch")
    if _has_evidence_formula_source_mismatch(tree, evidence):
        findings.append("evidence_formula_source_mismatch")
    if _has_optional_join_for_required_projection(tree):
        findings.append("optional_join_for_required_projection")
    if _has_missing_account_owner_role(tree, question):
        findings.append("missing_account_owner_role")
    if _has_denominator_cohort_mismatch(tree, question, evidence):
        findings.append("denominator_cohort_filter_missing")
    if _has_unsupported_type_coercion(tree, evidence, schema):
        findings.append("unsupported_type_coercion")
    if _has_structured_relationship_decode_mismatch(sql, tree, question, evidence, schema):
        findings.append("structured_relationship_decode_mismatch")
    if _has_formula_output_arity_mismatch(tree, question, evidence):
        findings.append("formula_output_arity_mismatch")
    if _has_duplicate_attribute_projection(tree, question):
        findings.append("duplicate_attribute_projection")
    return tuple(findings)
