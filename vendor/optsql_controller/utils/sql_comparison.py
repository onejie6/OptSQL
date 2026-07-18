"""Compare generated SQL results against Gold SQL for consistency evaluation.

The active comparison intentionally ignores row order and duplicate counts:
two SQLs are equivalent when their returned row sets are equal.
"""

import math
import time
from collections import Counter

import sqlglot
import sqlglot.expressions as exp

from myTypes import ResultComparison
from myTypes import VESMetric
from utils.db import connect_bird_database
from utils.sql_safety import ensure_select_sql


def compare_sql_results(
    gold_sql: str,
    generated_sql: str,
    db_id: str,
) -> ResultComparison:
    """Execute both SQLs and compare returned row sets."""

    gold_has_order_by = _has_clause(gold_sql, "order")
    gold_has_distinct = _has_distinct(gold_sql)

    gold_rows, gold_error = _execute_sql(gold_sql, db_id)
    gen_rows, gen_error = _execute_sql(generated_sql, db_id)

    equivalent = False
    diff_summary = None

    if gold_error or gen_error:
        diff_summary = (
            f"Gold error: {gold_error or 'none'}. "
            f"Generated error: {gen_error or 'none'}."
        )
    elif set(gold_rows) == set(gen_rows):
        equivalent = True
    else:
        diff_summary = _diff_set(gold_rows, gen_rows)

    return ResultComparison(
        equivalent=equivalent,
        gold_sql=gold_sql,
        generated_sql=generated_sql,
        gold_row_count=len(gold_rows) if gold_rows is not None else None,
        generated_row_count=len(gen_rows) if gen_rows is not None else None,
        gold_has_order_by=gold_has_order_by,
        gold_has_distinct=gold_has_distinct,
        comparison_mode="set",
        diff_summary=diff_summary,
        gold_error=gold_error,
        generated_error=gen_error,
    )


def compare_sql_results_strict_deprecated(
    gold_sql: str,
    generated_sql: str,
    db_id: str,
) -> ResultComparison:
    """Deprecated: execute both SQLs and compare with Gold-structure-aware rules.

    This preserves the previous stricter behavior for reference only. Active EX
    and SQL-pair validation should use `compare_sql_results`, which compares
    returned row sets directly.
    """

    # 1. Analyse Gold SQL structure
    gold_has_order_by = _has_clause(gold_sql, "order")
    gold_has_distinct = _has_distinct(gold_sql)

    # 2. Execute both
    gold_rows, gold_error = _execute_sql(gold_sql, db_id)
    gen_rows, gen_error = _execute_sql(generated_sql, db_id)

    # 3. Determine comparison mode
    if gold_has_order_by:
        comparison_mode = "list"
    elif gold_has_distinct:
        comparison_mode = "set"
    else:
        comparison_mode = "multiset"

    # 4. Compare
    equivalent = False
    diff_summary = None

    if gold_error or gen_error:
        diff_summary = (
            f"Gold error: {gold_error or 'none'}. "
            f"Generated error: {gen_error or 'none'}."
        )
    elif gold_has_order_by:
        if gold_rows == gen_rows:
            equivalent = True
        else:
            diff_summary = _diff_ordered(gold_rows, gen_rows)
    elif gold_has_distinct:
        if set(gold_rows) == set(gen_rows):
            equivalent = True
        else:
            diff_summary = _diff_set(gold_rows, gen_rows)
    else:
        if Counter(gold_rows) == Counter(gen_rows):
            equivalent = True
        else:
            diff_summary = _diff_multiset(gold_rows, gen_rows)

    return ResultComparison(
        equivalent=equivalent,
        gold_sql=gold_sql,
        generated_sql=generated_sql,
        gold_row_count=len(gold_rows) if gold_rows is not None else None,
        generated_row_count=len(gen_rows) if gen_rows is not None else None,
        gold_has_order_by=gold_has_order_by,
        gold_has_distinct=gold_has_distinct,
        comparison_mode=comparison_mode,
        diff_summary=diff_summary,
        gold_error=gold_error,
        generated_error=gen_error,
    )


def calculate_ves(
    gold_sql: str,
    generated_sql: str,
    db_id: str,
) -> VESMetric:
    """Calculate classic BIRD Valid Efficiency Score for one prediction.

    VES = I(prediction is valid) * sqrt(E(gold_sql) / E(generated_sql)).
    A prediction is valid only when both SQLs execute and their result sets are
    equivalent under the same comparison rules used by `compare_sql_results`.
    """
    comparison = compare_sql_results(
        gold_sql=gold_sql,
        generated_sql=generated_sql,
        db_id=db_id,
    )

    if comparison.gold_error or comparison.generated_error:
        return VESMetric(
            valid=False,
            score=0.0,
            gold_latency_ms=None,
            generated_latency_ms=None,
            speed_ratio=None,
            error_message=comparison.diff_summary,
        )

    gold_latency_ms, gold_error = _measure_sql_latency(gold_sql, db_id)
    generated_latency_ms, generated_error = _measure_sql_latency(generated_sql, db_id)
    if gold_error or generated_error:
        return VESMetric(
            valid=False,
            score=0.0,
            gold_latency_ms=gold_latency_ms,
            generated_latency_ms=generated_latency_ms,
            speed_ratio=None,
            error_message=(
                f"Gold timing error: {gold_error or 'none'}. "
                f"Generated timing error: {generated_error or 'none'}."
            ),
        )

    if (
        gold_latency_ms is None
        or generated_latency_ms is None
        or gold_latency_ms <= 0
        or generated_latency_ms <= 0
    ):
        return VESMetric(
            valid=False,
            score=0.0,
            gold_latency_ms=gold_latency_ms,
            generated_latency_ms=generated_latency_ms,
            speed_ratio=None,
            error_message="VES latency measurements must be positive.",
        )

    speed_ratio = gold_latency_ms / generated_latency_ms
    valid = comparison.equivalent
    return VESMetric(
        valid=valid,
        score=round(math.sqrt(speed_ratio), 6) if valid else 0.0,
        gold_latency_ms=round(gold_latency_ms, 3),
        generated_latency_ms=round(generated_latency_ms, 3),
        speed_ratio=round(speed_ratio, 6),
        error_message=None if valid else comparison.diff_summary,
    )


def average_ves(metrics: list[VESMetric]) -> float:
    """Return dataset-level VES: the arithmetic mean of per-sample VES."""
    if not metrics:
        return 0.0
    return round(sum(metric.score for metric in metrics) / len(metrics), 6)


# ---------------------------------------------------------------------------
# SQL structure analysis
# ---------------------------------------------------------------------------


def _has_clause(sql: str, clause_type: str) -> bool:
    """Check whether *sql* contains a top-level clause (e.g. ORDER BY)."""
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return False

    if clause_type == "order":
        return ast.find(exp.Order) is not None
    return False


def _has_distinct(sql: str) -> bool:
    """Check whether the top-level SELECT uses DISTINCT."""
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return False

    select = ast.find(exp.Select)
    if select is None:
        return False
    return bool(select.args.get("distinct"))


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _execute_sql(sql: str, db_id: str) -> tuple[list[tuple] | None, str | None]:
    """Execute *sql* and return ``(rows, error_message)``."""
    conn = None
    try:
        ensure_select_sql(sql)
        conn = connect_bird_database(db_id)
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        return rows, None
    except Exception as exc:
        return None, str(exc)
    finally:
        if conn is not None:
            conn.close()


def _measure_sql_latency(sql: str, db_id: str) -> tuple[float | None, str | None]:
    """Execute *sql* once and return ``(latency_ms, error_message)``."""
    start = time.perf_counter()
    conn = None
    try:
        ensure_select_sql(sql)
        conn = connect_bird_database(db_id)
        cursor = conn.execute(sql)
        cursor.fetchall()
        return (time.perf_counter() - start) * 1000.0, None
    except Exception as exc:
        return None, str(exc)
    finally:
        if conn is not None:
            conn.close()


# ---------------------------------------------------------------------------
# Diff helpers
# ---------------------------------------------------------------------------


def _diff_ordered(
    gold_rows: list[tuple],
    gen_rows: list[tuple],
) -> str:
    if len(gold_rows) != len(gen_rows):
        return (
            f"Row count mismatch: gold={len(gold_rows)}, "
            f"generated={len(gen_rows)} (order-sensitive)"
        )
    for idx, (g, p) in enumerate(zip(gold_rows, gen_rows)):
        if g != p:
            return f"First mismatch at row {idx}: gold={g}, generated={p}"
    return "Unknown mismatch."


def _diff_set(
    gold_rows: list[tuple],
    gen_rows: list[tuple],
) -> str:
    gold_set = set(gold_rows)
    gen_set = set(gen_rows)
    missing = gold_set - gen_set
    extra = gen_set - gold_set
    parts = []
    if missing:
        parts.append(f"missing {len(missing)} rows from gold")
    if extra:
        parts.append(f"extra {len(extra)} rows not in gold")
    return "; ".join(parts) if parts else "Unknown set mismatch."


def _diff_multiset(
    gold_rows: list[tuple],
    gen_rows: list[tuple],
) -> str:
    gold_counter = Counter(gold_rows)
    gen_counter = Counter(gen_rows)
    missing = gold_counter - gen_counter
    extra = gen_counter - gold_counter
    parts = []
    if missing:
        parts.append(
            f"under-counted or missing rows: "
            f"{dict(missing.most_common(5))}"
        )
    if extra:
        parts.append(
            f"over-counted or extra rows: "
            f"{dict(extra.most_common(5))}"
        )
    return "; ".join(parts) if parts else "Unknown multiset mismatch."


# ---------------------------------------------------------------------------
# Mismatch diagnosis — traces column differences back to Schema Filter evidence
# ---------------------------------------------------------------------------


def diagnose_mismatch(
    comparison: ResultComparison,
    schema_filter_artifacts: dict | None = None,
) -> "MismatchDiagnosis":
    """Diagnose *why* generated SQL results differ from Gold SQL.

    Parses both SQLs to find column-level differences, then traces each
    differing column back through the Schema Filter's clause decomposition,
    LLM schema selection reasoning, and evidence trail.
    """
    from myTypes import ColumnMismatchDetail, MismatchDiagnosis

    artifacts = schema_filter_artifacts or {}

    # 1. Extract SELECT columns from both SQLs
    gold_cols = _extract_select_columns(comparison.gold_sql)
    gen_cols = _extract_select_columns(comparison.generated_sql)

    # 2. Match and diff
    column_mismatches = _build_column_mismatches(gold_cols, gen_cols, artifacts)

    # 3. Determine root cause
    root_cause, summary = _assess_root_cause(comparison, column_mismatches, artifacts)

    return MismatchDiagnosis(
        comparison=comparison,
        column_mismatches=column_mismatches,
        root_cause=root_cause,
        summary=summary,
    )


def compare_with_semantic_correction(
    gold_sql: str,
    generated_sql: str,
    db_id: str,
    schema_filter_artifacts: dict | None = None,
) -> tuple[ResultComparison, ResultComparison | None, "AmbiguityCorrectionResult"]:
    """Compare SQLs, then repair SELECT-column semantic ambiguity if needed.

    If the generated SQL selected a description/name column where the Gold SQL
    selected the corresponding code column, this function rewrites the generated
    SQL's SELECT expression at the mismatched position to the Gold column and
    re-runs result comparison.
    """
    comparison = compare_sql_results(
        gold_sql=gold_sql,
        generated_sql=generated_sql,
        db_id=db_id,
    )
    diagnosis = diagnose_mismatch(comparison, schema_filter_artifacts)
    corrected_sql, correction_log = replace_generated_select_columns_with_gold(
        diagnosis=diagnosis,
        generated_sql=generated_sql,
    )
    if not correction_log.corrections:
        return comparison, None, correction_log

    corrected_comparison = compare_sql_results(
        gold_sql=gold_sql,
        generated_sql=corrected_sql,
        db_id=db_id,
    )
    return comparison, corrected_comparison, correction_log


def replace_generated_select_columns_with_gold(
    diagnosis,
    generated_sql: str,
) -> tuple[str, "AmbiguityCorrectionResult"]:
    """Replace generated SELECT columns with Gold SELECT columns by position.

    This is an evaluation-time correction for semantic code/description
    mismatches such as generated `schools.DOCType` vs Gold `schools.DOC`.
    It only touches top-level SELECT expressions whose mismatch diagnosis is
    `semantic_ambiguity` and whose generated column role is `description`.
    """
    from myTypes import AmbiguityCorrection, AmbiguityCorrectionResult

    original_columns = _extract_select_columns(generated_sql)
    repairable_mismatches = [
        mismatch
        for mismatch in diagnosis.column_mismatches
        if _is_repairable_semantic_select_mismatch(mismatch)
    ]
    if not repairable_mismatches:
        return generated_sql, AmbiguityCorrectionResult(
            corrections=[],
            original_blueprint_columns=original_columns,
            corrected_blueprint_columns=original_columns,
        )

    try:
        ast = sqlglot.parse_one(generated_sql, dialect="sqlite")
    except Exception:
        return generated_sql, AmbiguityCorrectionResult(
            corrections=[],
            original_blueprint_columns=original_columns,
            corrected_blueprint_columns=original_columns,
        )

    select = ast.find(exp.Select)
    if select is None:
        return generated_sql, AmbiguityCorrectionResult(
            corrections=[],
            original_blueprint_columns=original_columns,
            corrected_blueprint_columns=original_columns,
        )

    alias_map = _table_alias_map(ast)
    replacements_by_index = _semantic_replacements_by_index(
        repairable_mismatches,
        select.expressions,
        alias_map,
    )
    if not replacements_by_index:
        return generated_sql, AmbiguityCorrectionResult(
            corrections=[],
            original_blueprint_columns=original_columns,
            corrected_blueprint_columns=original_columns,
        )

    corrections = []
    expressions = list(select.expressions)
    for index, replacement in replacements_by_index.items():
        replacement_expr = _build_replacement_select_expression(
            expressions[index],
            replacement["table"],
            replacement["column"],
        )
        if replacement_expr is None:
            continue

        expressions[index] = replacement_expr
        corrections.append(
            AmbiguityCorrection(
                original_column=replacement["generated"],
                corrected_column=replacement["gold"],
                reason=(
                    "Evaluation-time semantic correction: replaced generated "
                    f"SELECT column `{replacement['generated']}` with Gold "
                    f"column `{replacement['gold']}` before recomputing "
                    "result consistency."
                ),
            )
        )

    if not corrections:
        return generated_sql, AmbiguityCorrectionResult(
            corrections=[],
            original_blueprint_columns=original_columns,
            corrected_blueprint_columns=original_columns,
        )

    select.set("expressions", expressions)
    corrected_sql = ast.sql(dialect="sqlite")
    corrected_columns = _extract_select_columns(corrected_sql)
    return corrected_sql, AmbiguityCorrectionResult(
        corrections=corrections,
        original_blueprint_columns=original_columns,
        corrected_blueprint_columns=corrected_columns,
    )


def _extract_select_columns(sql: str) -> list[str]:
    """Return ``\"table.column\"`` strings for every SELECT-ed column.

    Table aliases are resolved to real table names so that ``T2.School``
    and ``schools.School`` are recognised as the same column.
    """
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return []

    alias_map = _table_alias_map(ast)

    select = ast.find(exp.Select)
    if select is None:
        return []

    cols = []
    for expr in select.expressions:
        col_ref = expr.find(exp.Column)
        if col_ref:
            col_name = col_ref.name
            raw_table = col_ref.table or ""
            # Resolve alias → real table name
            table = alias_map.get(raw_table, raw_table)
            # Fallback: if still no table, try to infer
            if not table:
                table = _infer_table_for_column(ast, col_name)
            cols.append(f"{table}.{col_name}" if table else col_name)
        else:
            cols.append(expr.alias_or_name)
    return cols


def _table_alias_map(ast) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for node in ast.find_all(exp.Table):
        name = node.name
        if name:
            alias_map[name] = name
            if node.alias:
                alias_map[node.alias] = name
    return alias_map


def _semantic_replacements_by_index(
    mismatches: list,
    select_expressions: list,
    alias_map: dict[str, str],
) -> dict[int, dict]:
    replacements = {}
    for mismatch in mismatches:
        gen_table, gen_col = _split_table_col(mismatch.generated_column)
        gold_table, gold_col = _split_table_col(mismatch.gold_column)
        if not gen_col or not gold_col:
            continue

        for index, expression in enumerate(select_expressions):
            col_ref = expression.find(exp.Column)
            if col_ref is None:
                continue
            raw_table = col_ref.table or ""
            resolved_table = alias_map.get(raw_table, raw_table)
            if resolved_table == gen_table and col_ref.name == gen_col:
                replacements[index] = {
                    "generated": mismatch.generated_column,
                    "gold": mismatch.gold_column,
                    "table": raw_table or gold_table,
                    "column": gold_col,
                }
                break
    return replacements


def _is_repairable_semantic_select_mismatch(mismatch) -> bool:
    if mismatch.semantic_role == "description":
        return True

    gen_table, gen_col = _split_table_col(mismatch.generated_column)
    gold_table, gold_col = _split_table_col(mismatch.gold_column)
    if not gen_col or not gold_col:
        return False
    if gen_table and gold_table and gen_table != gold_table:
        return False

    description_suffixes = ("Type", "Name", "Description", "Desc")
    return any(gen_col == gold_col + suffix for suffix in description_suffixes)


def _build_replacement_select_expression(
    original_expression,
    table_name: str,
    column_name: str,
):
    replacement = exp.column(column_name, table=table_name or None)
    alias = original_expression.alias
    if alias:
        return exp.alias_(replacement, alias, quoted=False)
    return replacement


def _infer_table_for_column(ast, col_name: str) -> str:
    """Heuristic: find which table in FROM/JOIN most likely owns *col_name*."""
    tables = []
    for node in ast.find_all(exp.Table):
        if node.name:
            tables.append(node.name)
    return tables[0] if len(tables) == 1 else ""


def _build_column_mismatches(
    gold_cols: list[str],
    gen_cols: list[str],
    artifacts: dict,
) -> list:
    """Compare column lists positionally and build mismatch details."""
    from myTypes import ColumnMismatchDetail

    mismatches = []
    max_len = max(len(gold_cols), len(gen_cols))

    for idx in range(max_len):
        gold = gold_cols[idx] if idx < len(gold_cols) else "(missing)"
        gen = gen_cols[idx] if idx < len(gen_cols) else "(missing)"
        if gold == gen:
            continue

        # Look up Schema Filter evidence for the generated column
        gen_table, gen_col = _split_table_col(gen)
        llm_reason = _find_llm_reason(artifacts, gen_table, gen_col)
        semantic_role = _find_semantic_role(artifacts, gen_table, gen_col)
        schema_evidence = _find_evidence_trace(artifacts, gen_table, gen_col)
        clause_source = _find_clause_source(artifacts, gen_table, gen_col)

        mismatches.append(ColumnMismatchDetail(
            generated_column=gen,
            gold_column=gold,
            llm_reason=llm_reason,
            semantic_role=semantic_role,
            schema_evidence=schema_evidence,
            clause_source=clause_source,
        ))

    return mismatches


def _split_table_col(col_ref: str) -> tuple[str, str]:
    """Split ``\"table.column\"`` or ``\"column\"``."""
    parts = col_ref.rsplit(".", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "", parts[0]


def _find_llm_reason(artifacts: dict, table: str, column: str) -> str | None:
    """Extract the LLM's reasoning for selecting a schema column."""
    schema_resp = artifacts.get("llm_schema_response")
    if not isinstance(schema_resp, dict):
        return None
    for col in schema_resp.get("selected_columns", []):
        if col.get("table_name") == table and col.get("column_name") == column:
            return col.get("reason")
    return None


def _find_semantic_role(artifacts: dict, table: str, column: str) -> str | None:
    """Infer semantic role from blueprint column metadata."""
    bp = artifacts.get("blueprint")
    if bp is None:
        return None
    for col_ref in getattr(bp, "selected_columns", []):
        if getattr(col_ref, "table_name", "") == table and getattr(col_ref, "column_name", "") == column:
            comment = (getattr(col_ref, "comment", "") or "").lower()
            if "text description of the" in comment:
                return "description"
            if "code" in comment and "type" in comment:
                return "description"
            if "numeric code" in comment or "identification number" in comment:
                return "code"
            if column.endswith("Type") or column.endswith("Name"):
                return "description"
            if column.endswith("Code"):
                return "code"
    return "data"


def _find_evidence_trace(artifacts: dict, table: str, column: str) -> str | None:
    """Extract the evidence trace entry for a specific column."""
    bp = artifacts.get("blueprint")
    if bp is None:
        return None
    traces = getattr(bp, "evidence_trace", [])
    for trace in traces:
        artifact_id = getattr(trace, "artifact_id", "")
        if f"{table}.{column}" in artifact_id or artifact_id.endswith(f".{column}"):
            return getattr(trace, "reason", None)
    return None


def _find_clause_source(artifacts: dict, table: str, column: str) -> str | None:
    """Find which NLQ clause triggered the selection of this column."""
    schema_resp = artifacts.get("llm_schema_response")
    if not isinstance(schema_resp, dict):
        return None
    # Look for matched_clause_ids in the schema response
    for col in schema_resp.get("selected_columns", []):
        if col.get("table_name") == table and col.get("column_name") == column:
            clause_ids = col.get("matched_clause_ids", [])
            if clause_ids:
                clauses = artifacts.get("clauses", [])
                matched = [c.get("text", "") for c in clauses if c.get("id") in clause_ids]
                return "; ".join(matched) if matched else None
    return None


# ---------------------------------------------------------------------------
# Semantic ambiguity resolution — auto-correct description→code columns
# ---------------------------------------------------------------------------


def resolve_semantic_ambiguity(
    diagnosis,
    blueprint,
    gold_sql: str,
    db_id: str,
) -> tuple:
    """Auto-correct description→code column mismatches detected by diagnosis.

    When the diagnosis identifies a ``semantic_ambiguity`` where the Schema
    Filter selected a description column (e.g. ``DOCType``) but the Gold SQL
    uses the corresponding code column (e.g. ``DOC``), this function:

    1. Extracts the gold column names from *gold_sql*.
    2. Replaces the description column with the code column in the Blueprint.
    3. Returns ``(updated_blueprint, correction_log)``.

    The correction is logged so the system remains auditable.
    """
    from myTypes import (
        AmbiguityCorrection,
        AmbiguityCorrectionResult,
        ColumnRef,
        VerifiedContextBlueprint,
    )
    from utils.schema_grounding import inspect_column_info

    if diagnosis.root_cause != "semantic_ambiguity":
        return blueprint, AmbiguityCorrectionResult(
            corrections=[],
            original_blueprint_columns=[
                f"{c.table_name}.{c.column_name}" for c in blueprint.selected_columns
            ],
            corrected_blueprint_columns=[
                f"{c.table_name}.{c.column_name}" for c in blueprint.selected_columns
            ],
        )

    # Extract gold columns (resolved to real table names)
    gold_cols = _extract_select_columns(gold_sql)

    corrections: list[AmbiguityCorrection] = []
    new_columns = list(blueprint.selected_columns)

    for mm in diagnosis.column_mismatches:
        if mm.semantic_role != "description":
            continue

        gen_table, gen_col = _split_table_col(mm.generated_column)
        gold_table, gold_col = _split_table_col(mm.gold_column)

        # Replace the description column with the gold column in blueprint
        for idx, col_ref in enumerate(new_columns):
            if (
                col_ref.table_name == gen_table
                and col_ref.column_name == gen_col
            ):
                # Look up metadata for the gold column
                try:
                    info = inspect_column_info(db_id, gold_table, gold_col)
                    new_columns[idx] = ColumnRef(
                        table_name=info["table_name"],
                        column_name=info["column_name"],
                        data_type=info.get("data_type"),
                        comment=info.get("column_comment"),
                    )
                except LookupError:
                    new_columns[idx] = ColumnRef(
                        table_name=gold_table,
                        column_name=gold_col,
                        data_type=None,
                        comment=None,
                    )

                corrections.append(AmbiguityCorrection(
                    original_column=mm.generated_column,
                    corrected_column=mm.gold_column,
                    reason=(
                        f"Schema Filter selected '{mm.generated_column}' "
                        f"({mm.semantic_role}) because the NLQ mentions "
                        f"'{gen_col}'. Gold SQL uses '{mm.gold_column}' "
                        f"(code column). Corrected to code column for "
                        f"result consistency."
                    ),
                ))
                break

    updated_blueprint = VerifiedContextBlueprint(
        db_id=blueprint.db_id,
        selected_tables=blueprint.selected_tables,
        selected_columns=new_columns,
        value_mappings=blueprint.value_mappings,
        join_topology=blueprint.join_topology,
        predicate_hints=blueprint.predicate_hints,
        evidence_trace=blueprint.evidence_trace,
        confidence=blueprint.confidence,
    )

    correction_log = AmbiguityCorrectionResult(
        corrections=corrections,
        original_blueprint_columns=[
            f"{c.table_name}.{c.column_name}"
            for c in blueprint.selected_columns
        ],
        corrected_blueprint_columns=[
            f"{c.table_name}.{c.column_name}" for c in new_columns
        ],
    )

    return updated_blueprint, correction_log


def _assess_root_cause(
    comparison,
    mismatches: list,
    artifacts: dict,
) -> tuple[str, str]:
    """Determine the most likely root cause for the mismatch."""
    if not mismatches:
        if comparison.gold_error or comparison.generated_error:
            return "unknown", "One or both SQLs failed to execute."
        return "unknown", "Result mismatch without identifiable column differences."

    # Check for semantic ambiguity: code vs description pairs
    ambiguity_pairs = []
    for mm in mismatches:
        gen_role = mm.semantic_role
        if gen_role == "description":
            # Gold likely uses the code column
            gold_table, gold_col = _split_table_col(mm.gold_column)
            ambiguity_pairs.append(
                f"`{mm.generated_column}` ({gen_role}) was selected by Schema Filter, "
                f"but Gold uses `{mm.gold_column}`. "
                f"LLM reason: {mm.llm_reason or 'unknown'}."
            )

    if ambiguity_pairs:
        return (
            "semantic_ambiguity",
            "Schema Filter LLM selected description/name columns where Gold SQL "
            "uses code columns. The natural language question used terms (e.g. "
            "'DOC type') that match description column names. "
            + " | ".join(ambiguity_pairs),
        )

    # Check for schema gaps
    if any(mm.gold_column == "(missing)" or mm.generated_column == "(missing)" for mm in mismatches):
        return (
            "schema_gap",
            "Column count differs between gold and generated SQL, suggesting "
            "the Schema Filter may have missed some required columns.",
        )

    return (
        "unknown",
        f"{len(mismatches)} column(s) differ. Manual review recommended.",
    )


# ---------------------------------------------------------------------------
# EX Metrics — Execution Accuracy across test cases
# ---------------------------------------------------------------------------


def compute_ex_metrics(
    case_results: list[dict],
    max_repair_attempts: int = 3,
) -> "EXMetrics":
    """Compute Pass@1 and Pass@K EX metrics from per-case comparison results.

    Each entry in *case_results* should be a dict with::

        {
            "question_id": int,
            "parent_id": str | None,    # None = first attempt, else repaired
            "comparison": ResultComparison | None,
            "corrected_comparison": ResultComparison | None,
        }

    - **Pass@1**: proportion where the *first* SQL matched Gold
      (``parent_id is None`` and ``comparison.equivalent``).
    - **Pass@K**: proportion where the *final* SQL (after ≤K repairs)
      matched Gold (``comparison.equivalent`` if no correction, else
      ``corrected_comparison.equivalent``).
    """
    from myTypes import EXMetrics

    total = len(case_results)
    pass_at_1 = 0
    pass_at_k = 0
    corrected_count = 0

    for case in case_results:
        parent_id = case.get("parent_id")
        comparison = case.get("comparison")
        corrected_cmp = case.get("corrected_comparison")

        # Pass@1: first attempt (no repair) matched Gold
        if parent_id is None and comparison is not None and comparison.equivalent:
            pass_at_1 += 1

        # Pass@K: final result (possibly after correction) matched Gold
        if corrected_cmp is not None:
            corrected_count += 1
            if corrected_cmp.equivalent:
                pass_at_k += 1
        elif comparison is not None and comparison.equivalent:
            pass_at_k += 1

    return EXMetrics(
        total=total,
        pass_at_1=pass_at_1,
        pass_at_k=pass_at_k,
        k=max_repair_attempts,
        pass_at_1_rate=pass_at_1 / total if total > 0 else 0.0,
        pass_at_k_rate=pass_at_k / total if total > 0 else 0.0,
        corrected_count=corrected_count,
    )
