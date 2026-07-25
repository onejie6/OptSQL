"""Run a candidate-level Qwen Meta-Controller without using gold SQL."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .runtime_utils import execute_rows as _execute_rows
    from .runtime_utils import iter_snapshot_items as _iter_snapshot_items
except ImportError:
    from runtime_utils import execute_rows as _execute_rows
    from runtime_utils import iter_snapshot_items as _iter_snapshot_items
from app.config.config import LLMConfig
from app.llm import LLM


LABELS = "ABCDE"


def _load_items(snapshot_path: Path, wanted_ids: set[int]) -> dict[int, dict[str, Any]]:
    return {
        int(item["question_id"]): item
        for item in _iter_snapshot_items(snapshot_path)
        if int(item["question_id"]) in wanted_ids
    }


def _load_llm(config_path: Path, model: str) -> LLM:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    profile_name = config["run"]["default_llm_profile"]
    profile = dict(config["llm_profiles"][profile_name])
    profile.update(
        model=model,
        temperature=0.2,
        max_tokens=16384,
        max_request_n=1,
        n_call_strategy="split",
    )
    return LLM(LLMConfig(**profile))


def _candidate_source(index: int, total: int) -> str:
    if total and total % 3 == 0:
        width = total // 3
        return ("dc", "skeleton", "icl")[min(index // width, 2)]
    return "unknown"


def _cluster_candidates(item: dict[str, Any], timeout_seconds: float) -> list[dict[str, Any]]:
    candidates = [
        candidate.strip()
        for candidate in (item.get("sql_candidates_after_revision") or [])
        if isinstance(candidate, str) and candidate.strip()
    ]
    cache: dict[str, tuple[set[tuple[Any, ...]] | None, str | None]] = {}
    for sql in dict.fromkeys(candidates):
        cache[sql] = _execute_rows(item["database_path"], sql, timeout_seconds)
    executable = [sql for sql in candidates if cache[sql][0] is not None]
    nonempty = [sql for sql in executable if cache[sql][0]]
    pool = nonempty or executable
    clusters: dict[frozenset[tuple[Any, ...]], dict[str, Any]] = {}
    for index, sql in enumerate(candidates):
        rows, _ = cache[sql]
        if sql not in pool or rows is None:
            continue
        key = frozenset(rows)
        cluster = clusters.setdefault(
            key,
            {
                "sql": sql,
                "rows": rows,
                "support": 0,
                "sources": set(),
                "first_index": index,
            },
        )
        cluster["support"] += 1
        cluster["sources"].add(_candidate_source(index, len(candidates)))
    denominator = len(pool)
    ranked = sorted(
        clusters.values(),
        key=lambda cluster: (-cluster["support"], cluster["first_index"]),
    )
    for cluster in ranked:
        cluster["score"] = cluster["support"] / denominator if denominator else 0.0

    final_sql = (item.get("final_selected_sql") or "").strip()
    final_rows, _ = cache.get(
        final_sql,
        _execute_rows(item["database_path"], final_sql, timeout_seconds),
    )
    final_key = frozenset(final_rows) if final_rows is not None else None
    chosen = ranked[:5]
    if final_key is not None and all(frozenset(cluster["rows"]) != final_key for cluster in chosen):
        final_cluster = next(
            (cluster for cluster in ranked if frozenset(cluster["rows"]) == final_key),
            None,
        )
        if final_cluster is not None:
            chosen = [*chosen[:4], final_cluster]
    # Do not let support ranking become an accidental positional prior for the
    # semantic judge. The order is stable across reruns but independent of vote
    # count and generator identity.
    question_id = int(item["question_id"])
    return sorted(
        chosen,
        key=lambda cluster: hashlib.sha256(
            f"meta-controller-v2:{question_id}:{cluster['sql']}".encode()
        ).hexdigest(),
    )


def _schema_context(item: dict[str, Any]) -> str:
    schema = item.get("database_schema_after_schema_linking") or item.get("database_schema") or {}
    tables = schema.get("tables") or {}
    linked = item.get("final_linked_tables_and_columns") or {}
    lines: list[str] = []
    selected_tables = list(linked) if linked else list(tables)
    for table_name in selected_tables:
        table = tables.get(table_name) or {}
        columns = table.get("columns") or {}
        selected_columns = linked.get(table_name) or list(columns)
        rendered = []
        for column_name in selected_columns:
            column = columns.get(column_name) or {}
            details = [str(column.get("column_type") or "")]
            if column.get("description"):
                details.append(str(column["description"])[:160])
            rendered.append(f"{column_name} ({'; '.join(x for x in details if x)})")
        lines.append(f"{table_name}: " + ", ".join(rendered))
    return "\n".join(lines)[:18000]


def _value_context(item: dict[str, Any]) -> str:
    linked = item.get("final_linked_tables_and_columns") or {}
    values: list[str] = []
    for table, columns in (item.get("retrieved_values") or {}).items():
        if linked and table not in linked:
            continue
        for column, candidates in (columns or {}).items():
            if linked and column not in (linked.get(table) or []):
                continue
            rendered = []
            for candidate in candidates or []:
                if isinstance(candidate, dict):
                    value = candidate.get("value")
                    distance = candidate.get("distance")
                    suffix = f" (distance={float(distance):.4f})" if distance is not None else ""
                else:
                    value = candidate
                    suffix = ""
                rendered.append(repr(value)[:100] + suffix)
            if rendered:
                values.append(f"{table}.{column}: {', '.join(rendered[:5])}")
    return "\n".join(values[:40])


def _rows_preview(rows: set[tuple[Any, ...]]) -> str:
    rendered = sorted((repr(row) for row in rows), key=str)[:5]
    return "[" + ", ".join(text[:300] for text in rendered) + "]"


def _few_shot_context(item: dict[str, Any], limit: int = 3) -> str:
    blocks: list[str] = []
    for index, example in enumerate((item.get("few_shot_examples") or [])[:limit], 1):
        blocks.append(
            f"Example {index}\n"
            f"Question: {example.get('question') or ''}\n"
            f"Evidence: {example.get('evidence') or '(none)'}\n"
            f"SQL: {example.get('sql') or ''}"
        )
    return "\n\n".join(blocks) or "(none)"


def _cluster_prompt(item: dict[str, Any], clusters: list[dict[str, Any]]) -> str:
    candidate_blocks = []
    for index, cluster in enumerate(clusters):
        candidate_blocks.append(
            f"Candidate {LABELS[index]}\n"
            f"generator_families={','.join(sorted(cluster['sources']))}\n"
            f"row_count={len(cluster['rows'])}, result_preview={_rows_preview(cluster['rows'])}\n"
            f"SQL:\n{cluster['sql']}"
        )
    return f"""You are the meta-controller for a Text-to-SQL system. Select the SQL that best answers the user question.

Important rules:
- Candidate support is correlated because generators share prompts. A majority can be confidently wrong.
- Candidate order is randomized and vote counts are intentionally hidden. Do not infer quality from labels or provenance.
- Verify tables, joins, filters, aggregation grain, DISTINCT, NULL handling, ordering, LIMIT, and evidence values.
- Prefer the smallest sufficient join graph. An unnecessary INNER JOIN can silently remove otherwise valid rows.
- Return exactly the attributes requested. Never add SELECT columns merely to provide more detail.
- Phrases such as "list all" or "identify all" request all matching rows, not every column of the entity or relationship.
- Treat NULL filtering as a semantic choice, not a generic robustness improvement. Add IS NOT NULL only when the wording, evidence, or a candidate's demonstrated failure requires it.
- For top-k within groups, use RANK semantics when ties should remain; ROW_NUMBER is valid only when the question requires exactly k rows or specifies a tie-breaker.
- Execution success and plausible-looking rows do not prove semantic correctness.
- Grounded values are approximate retrieval neighbors, not a command to include every listed variant.
- A literal named by the question/evidence normally requires exact equality. LIKE with '%' broadens the answer and needs explicit wording such as contains, starts with, or any variant.
- Compare SQL clauses directly to the wording. Never broaden a filter merely because nearby database values exist.
- If none is defensible, set needs_recovery=true instead of guessing.
- Do not optimize runtime in this phase; judge answer correctness only.

Question:
{item.get('question') or ''}

Evidence:
{item.get('evidence') or '(none)'}

Linked schema:
{_schema_context(item)}

Grounded values:
{_value_context(item) or '(none)'}

Candidates:
{chr(10).join(candidate_blocks)}

Return JSON only:
{{"decision":"A|B|C|D|E|RECOVER","confidence":0.0,"needs_recovery":false,"reason":"concise clause-level justification","risk_checks":["..."]}}
"""


def _parse_json_object(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def _ask_json_votes(
    llm: LLM,
    prompt: str,
    *,
    n: int,
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    messages, usage = llm.ask(
        [{"role": "user", "content": prompt}],
        system_message={
            "role": "system",
            "content": "You are a rigorous SQLite and Text-to-SQL reviewer. Return valid JSON only.",
        },
        n=n,
        timeout=300,
    )
    raw = [message.content or "" for message in messages]
    parsed = [value for text in raw if (value := _parse_json_object(text)) is not None]
    return parsed, raw, usage


def _majority_decision(votes: list[dict[str, Any]], cluster_count: int) -> dict[str, Any]:
    normalized = []
    for vote in votes:
        decision = str(vote.get("decision") or "").strip().upper()
        needs_recovery = bool(vote.get("needs_recovery")) or decision == "RECOVER"
        if needs_recovery:
            decision = "RECOVER"
        if decision in LABELS[:cluster_count] or decision == "RECOVER":
            normalized.append((decision, float(vote.get("confidence") or 0.0)))
    counts = collections.Counter(decision for decision, _ in normalized)
    if not counts:
        return {"decision": None, "confidence": 0.0, "agreement": 0, "valid_votes": 0}
    decision, agreement = counts.most_common(1)[0]
    confidences = [confidence for current, confidence in normalized if current == decision]
    return {
        "decision": decision,
        "confidence": sum(confidences) / len(confidences),
        "agreement": agreement,
        "valid_votes": len(normalized),
    }


def _critic_prompt(item: dict[str, Any], original_sql: str, proposed_sql: str) -> str:
    return f"""Decide whether the proposed SQL is more semantically faithful to the question than the current SQL.
Do not prefer novelty or speed. Check every changed clause against the schema and evidence.

Grounding rules:
- Retrieved database values are approximate nearest neighbors. Similar spellings are not aliases and do not redefine the question.
- When the question/evidence names a literal value, exact equality is the default benchmark interpretation.
- LIKE with '%' is semantically broader and is justified only by explicit wording such as contains, starts with, partial match, or variants.
- In particular, "X refers to COLUMN" grounds the literal X to COLUMN; it does not authorize matching every value containing X.
- The current SQL is an unverified candidate, not a trusted baseline. KEEP_ORIGINAL only when the proposed change lacks textual support.
- Judge result semantics only. Readability, maintainability, query style, and shorter SQL are never reasons to keep a semantically weaker query.
- For top-k within groups, RANK semantics keeps boundary ties; ROW_NUMBER arbitrarily removes them. Treat them as equivalent only when ties are impossible or exactly k rows is explicitly required.
- Return exactly the requested attributes. Reject unrequested output-column expansion even when the extra columns seem useful.

Question: {item.get('question') or ''}
Evidence: {item.get('evidence') or '(none)'}
Linked schema:\n{_schema_context(item)}
Approximate retrieved values (diagnostic only):\n{_value_context(item) or '(none)'}

CURRENT SQL:\n{original_sql}

PROPOSED SQL:\n{proposed_sql}

Return JSON only:
{{"decision":"ACCEPT_PROPOSED|KEEP_ORIGINAL","confidence":0.0,"reason":"clause-level comparison"}}
"""


def _recovery_prompt(item: dict[str, Any], clusters: list[dict[str, Any]]) -> str:
    candidates = "\n\n".join(
        f"Candidate {LABELS[index]}:\n{cluster['sql']}"
        for index, cluster in enumerate(clusters)
    )
    return f"""Synthesize one SQLite SELECT query that correctly answers the question. Existing candidates are inconclusive.
Diagnose their likely semantic mistakes, then produce a corrected query using only the provided schema. Do not use markdown.

Question: {item.get('question') or ''}
Evidence: {item.get('evidence') or '(none)'}
Linked schema:\n{_schema_context(item)}
Grounded values:\n{_value_context(item) or '(none)'}

Existing candidates:\n{candidates}

Return JSON only: {{"sql":"SELECT ...","reason":"what was corrected"}}
"""


def _semantic_audit_prompt(
    item: dict[str, Any], current_sql: str, clusters: list[dict[str, Any]]
) -> str:
    alternatives = "\n\n".join(
        f"Candidate {LABELS[index]}"
        f"{' [CURRENT SQL - not a repair]' if cluster['sql'].strip() == current_sql.strip() else ''}:\n"
        f"{cluster['sql']}"
        for index, cluster in enumerate(clusters)
    )
    return f"""Audit the current SQLite query against the question clause by clause. This is a correctness recovery phase, not runtime optimization.

Rules:
- PASS if the query already returns exactly the requested rows and columns, even if its style differs from examples.
- CANDIDATE selects an existing candidate that repairs a concrete semantic defect.
- REWRITE is allowed only when no existing candidate repairs that defect.
- If you identify a defect in CURRENT SQL, never select the candidate marked CURRENT as the repair. That is internally inconsistent.
- Never add output columns merely to be helpful. "List all" means all matching rows, not all table columns.
- Prefer the smallest sufficient join graph; unnecessary INNER JOINs can remove rows.
- Retrieved values are approximate neighbors, not aliases or instructions to broaden filters.
- A question/evidence literal normally uses exact equality unless partial matching is explicit.
- Do not add generic defensive conditions such as IS NOT NULL without textual or schema-specific justification.
- For top-k within groups, "top k" normally keeps ties via RANK semantics. ROW_NUMBER arbitrarily drops tied rows unless exactly k rows or a tie-breaker is requested.
- Few-shot examples show benchmark conventions but may come from other schemas. Never copy their identifiers.
- If uncertain, PASS. The current SQL must remain unchanged unless the defect and repair are both specific.

Question:
{item.get('question') or ''}

Evidence:
{item.get('evidence') or '(none)'}

Linked schema:
{_schema_context(item)}

Approximate retrieved values:
{_value_context(item) or '(none)'}

Retrieved few-shot examples:
{_few_shot_context(item)}

CURRENT SQL:
{current_sql}

Existing executable candidates (labels are randomized; vote counts are hidden):
{alternatives}

Return JSON only. Use PASS when CURRENT is correct; do not return CANDIDATE with the CURRENT label. For CANDIDATE, set candidate to the repairing label. For REWRITE, provide one complete corrected query:
{{"decision":"PASS|CANDIDATE|REWRITE","candidate":"A|B|C|D|E|null","confidence":0.0,"issues":["specific clause defect"],"sql":"SELECT ... or null"}}
"""


def _semantic_recovery(
    llm: LLM,
    item: dict[str, Any],
    current_sql: str,
    clusters: list[dict[str, Any]],
    timeout_seconds: float,
    allow_free_rewrite: bool,
) -> tuple[str | None, dict[str, Any]]:
    votes, raw, usage = _ask_json_votes(
        llm, _semantic_audit_prompt(item, current_sql, clusters), n=3
    )
    rewrite_votes: list[dict[str, Any]] = []
    execution_groups: dict[frozenset[tuple[Any, ...]], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    execution_errors: list[dict[str, str]] = []
    for vote in votes:
        decision = str(vote.get("decision") or "").strip().upper()
        if decision == "CANDIDATE":
            label = str(vote.get("candidate") or "").strip().upper()
            if label not in LABELS[: len(clusters)]:
                continue
            sql = clusters[LABELS.index(label)]["sql"]
        elif decision == "REWRITE":
            if not allow_free_rewrite:
                continue
            sql = str(vote.get("sql") or "").strip()
        else:
            continue
        if float(vote.get("confidence") or 0) < 0.75:
            continue
        if not _safe_select_sql(sql):
            continue
        rewrite_votes.append(vote)
        rows, error = _execute_rows(item["database_path"], sql, timeout_seconds)
        if rows is None or error:
            execution_errors.append({"sql": sql, "error": error or "unknown"})
            continue
        execution_groups[frozenset(rows)].append(
            {"sql": sql, "vote": vote, "kind": decision}
        )

    current_rows, current_error = _execute_rows(
        item["database_path"], current_sql, timeout_seconds
    )
    ranked = sorted(execution_groups.items(), key=lambda pair: len(pair[1]), reverse=True)
    proposed_sql: str | None = None
    agreement = 0
    selected_repair_kinds: list[str] = []
    if ranked:
        rows_key, group = ranked[0]
        agreement = len(group)
        if agreement >= 2 and (current_rows is None or rows_key != frozenset(current_rows)):
            proposed_sql = group[0]["sql"]
            selected_repair_kinds = [str(entry["kind"]) for entry in group]
    info = {
        "votes": votes,
        "raw": raw,
        "usage": usage,
        "qualified_rewrite_votes": len(rewrite_votes),
        "rewrite_result_agreement": agreement,
        "selected_repair_kinds": selected_repair_kinds,
        "current_error": current_error,
        "execution_errors": execution_errors,
    }
    return proposed_sql, info


def _safe_select_sql(sql: str) -> bool:
    if not sql.strip():
        return False
    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return False
    blocked = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter, exp.Command)
    return not isinstance(tree, blocked) and tree.find(exp.Select) is not None


def _normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _projection_expansion_is_grounded(
    item: dict[str, Any], original_sql: str, proposed_sql: str
) -> bool:
    """Reject unrequested output expansion while allowing named attributes."""
    try:
        original = sqlglot.parse_one(original_sql, dialect="sqlite").find(exp.Select)
        proposed = sqlglot.parse_one(proposed_sql, dialect="sqlite").find(exp.Select)
    except Exception:
        return False
    if original is None or proposed is None:
        return False
    original_expressions = list(original.expressions)
    proposed_expressions = list(proposed.expressions)
    if len(proposed_expressions) <= len(original_expressions):
        return True

    original_columns = {
        _normalize_identifier(column.name)
        for expression in original_expressions
        for column in expression.find_all(exp.Column)
        if column.name
    }
    request_text = _normalize_identifier(
        f"{item.get('question') or ''} {item.get('evidence') or ''}"
    )
    for expression in proposed_expressions:
        expression_columns = {
            _normalize_identifier(column.name)
            for column in expression.find_all(exp.Column)
            if column.name
        }
        new_columns = expression_columns - original_columns
        if new_columns and not all(column in request_text for column in new_columns):
            return False
    return True


def _unsafe_semantic_change(original_sql: str, proposed_sql: str) -> str | None:
    """Catch small SQL rewrites with deterministic, known-bad semantics."""
    try:
        original = sqlglot.parse_one(original_sql, dialect="sqlite")
        proposed = sqlglot.parse_one(proposed_sql, dialect="sqlite")
    except Exception:
        return "parse_failure"

    def count_case_with_nonnull_else(tree: exp.Expression) -> bool:
        return any(
            isinstance(count.this, exp.Case)
            and count.this.args.get("default") is not None
            and not isinstance(count.this.args.get("default"), exp.Null)
            for count in tree.find_all(exp.Count)
        )

    if count_case_with_nonnull_else(proposed) and not count_case_with_nonnull_else(original):
        return "count_case_nonnull_else_counts_false_rows"

    def like_patterns(tree: exp.Expression) -> dict[str, str]:
        patterns: dict[str, str] = {}
        for like in tree.find_all(exp.Like):
            if isinstance(like.this, exp.Column) and isinstance(like.expression, exp.Literal):
                patterns[_normalize_identifier(like.this.name)] = str(like.expression.this)
        return patterns

    original_likes = like_patterns(original)
    for column, proposed_pattern in like_patterns(proposed).items():
        original_pattern = original_likes.get(column)
        if not original_pattern or original_pattern == proposed_pattern:
            continue
        if original_pattern.endswith("%") and proposed_pattern.endswith("%"):
            original_prefix = original_pattern[:-1]
            proposed_prefix = proposed_pattern[:-1]
            added = proposed_prefix[len(original_prefix) :] if proposed_prefix.startswith(original_prefix) else ""
            if added and not any(character.isalnum() for character in added):
                return "unjustified_like_delimiter_narrowing"
    return None


def _critic_accepts(
    llm: LLM,
    item: dict[str, Any],
    original_sql: str,
    proposed_sql: str,
) -> tuple[bool, dict[str, Any]]:
    votes, raw, usage = _ask_json_votes(
        llm, _critic_prompt(item, original_sql, proposed_sql), n=2
    )
    accepts = [
        vote
        for vote in votes
        if str(vote.get("decision") or "").upper() == "ACCEPT_PROPOSED"
        and float(vote.get("confidence") or 0) >= 0.7
    ]
    return len(accepts) == 2, {"votes": votes, "raw": raw, "usage": usage}


def _run_one(
    llm: LLM,
    item: dict[str, Any],
    manifest_row: dict[str, Any],
    timeout_seconds: float,
    enable_semantic_recovery: bool,
    allow_free_rewrite: bool,
) -> dict[str, Any]:
    base_sql = (item.get("final_selected_sql") or "").strip()
    clusters = _cluster_candidates(item, timeout_seconds)
    if not clusters:
        return {
            "question_id": item["question_id"],
            "stratum": manifest_row["stratum"],
            "status": "fallback_no_executable_clusters",
            "base_sql": base_sql,
            "final_sql": base_sql,
            "changed": False,
        }
    votes, raw, usage = _ask_json_votes(llm, _cluster_prompt(item, clusters), n=3)
    majority = _majority_decision(votes, len(clusters))
    proposed_sql: str | None = None
    route = "fallback_low_consensus"
    recovery_info: dict[str, Any] | None = None
    if (
        majority["decision"] in LABELS[: len(clusters)]
        and majority["agreement"] >= 2
        and majority["confidence"] >= 0.65
    ):
        proposed_sql = clusters[LABELS.index(majority["decision"])]["sql"]
        route = "candidate_adjudication"
    elif (
        allow_free_rewrite
        and majority["decision"] == "RECOVER"
        and majority["agreement"] >= 2
    ):
        recovery_votes, recovery_raw, recovery_usage = _ask_json_votes(
            llm, _recovery_prompt(item, clusters), n=1
        )
        recovery_info = {
            "votes": recovery_votes,
            "raw": recovery_raw,
            "usage": recovery_usage,
        }
        if recovery_votes:
            candidate_sql = str(recovery_votes[0].get("sql") or "").strip()
            if _safe_select_sql(candidate_sql):
                rows, error = _execute_rows(item["database_path"], candidate_sql, timeout_seconds)
                if rows is not None and error is None:
                    proposed_sql = candidate_sql
                    route = "recovery"

    critic_info: dict[str, Any] | None = None
    final_sql = base_sql
    if proposed_sql and proposed_sql.strip() != base_sql:
        proposed_cluster = next(
            (
                cluster
                for cluster in clusters
                if cluster["sql"].strip() == proposed_sql.strip()
            ),
            None,
        )
        safety_rejection = _unsafe_semantic_change(base_sql, proposed_sql)
        if safety_rejection:
            route += f"_safety_rejected_{safety_rejection}"
        elif proposed_cluster is not None and len(proposed_cluster["sources"]) < 2:
            route += "_generator_diversity_rejected"
        elif not _projection_expansion_is_grounded(item, base_sql, proposed_sql):
            route += "_projection_expansion_rejected"
        elif majority["agreement"] == 3 and majority["confidence"] >= 0.8:
            final_sql = proposed_sql.strip()
            critic_info = {"bypassed": "three_vote_candidate_consensus"}
        else:
            accepted, critic_info = _critic_accepts(llm, item, base_sql, proposed_sql)
            if accepted:
                final_sql = proposed_sql.strip()
            else:
                route += "_critic_rejected"
    elif proposed_sql:
        route += "_same_sql"

    semantic_recovery_info: dict[str, Any] | None = None
    if enable_semantic_recovery and final_sql == base_sql:
        semantic_sql, semantic_recovery_info = _semantic_recovery(
            llm,
            item,
            base_sql,
            clusters,
            timeout_seconds,
            allow_free_rewrite,
        )
        if semantic_sql:
            base_cluster = next(
                (cluster for cluster in clusters if cluster["sql"].strip() == base_sql),
                None,
            )
            semantic_cluster = next(
                (
                    cluster
                    for cluster in clusters
                    if cluster["sql"].strip() == semantic_sql.strip()
                ),
                None,
            )
            repair_kinds = set(semantic_recovery_info["selected_repair_kinds"])
            evidence_supported = (
                semantic_cluster is not None
                and base_cluster is not None
                and repair_kinds == {"CANDIDATE"}
                and semantic_cluster["support"] > base_cluster["support"]
                and len(semantic_cluster["sources"]) >= 2
            )
            safety_rejection = _unsafe_semantic_change(base_sql, semantic_sql)
            if safety_rejection:
                route += f"__semantic_safety_rejected_{safety_rejection}"
            elif not evidence_supported:
                route += "__semantic_evidence_rejected"
            elif not _projection_expansion_is_grounded(item, base_sql, semantic_sql):
                route += "__semantic_projection_expansion_rejected"
            elif (
                semantic_recovery_info["rewrite_result_agreement"] == 3
                and set(semantic_recovery_info["selected_repair_kinds"]) == {"CANDIDATE"}
            ):
                final_sql = semantic_sql.strip()
                semantic_recovery_info["critic"] = {
                    "bypassed": "three_vote_existing_candidate_consensus"
                }
                semantic_recovery_info["proposed_sql"] = semantic_sql
                route += "__semantic_recovery"
            else:
                accepted, semantic_critic = _critic_accepts(
                    llm, item, base_sql, semantic_sql
                )
                semantic_recovery_info["critic"] = semantic_critic
                semantic_recovery_info["proposed_sql"] = semantic_sql
                if accepted:
                    final_sql = semantic_sql.strip()
                    route += "__semantic_recovery"
                else:
                    route += "__semantic_critic_rejected"
        else:
            route += "__semantic_no_consensus"

    cluster_summary = [
        {
            "label": LABELS[index],
            "support": cluster["support"],
            "score": cluster["score"],
            "sources": sorted(cluster["sources"]),
            "row_count": len(cluster["rows"]),
            "sql": cluster["sql"],
        }
        for index, cluster in enumerate(clusters)
    ]
    return {
        "question_id": item["question_id"],
        "db_id": item["database_id"],
        "difficulty": item.get("difficulty"),
        "stratum": manifest_row["stratum"],
        "status": "success",
        "route": route,
        "base_sql": base_sql,
        "proposed_sql": proposed_sql,
        "final_sql": final_sql,
        "changed": final_sql != base_sql,
        "clusters": cluster_summary,
        "meta_votes": votes,
        "meta_raw": raw,
        "meta_usage": usage,
        "majority": majority,
        "recovery": recovery_info,
        "critic": critic_info,
        "semantic_recovery": semantic_recovery_info,
    }


def _completed_ids(checkpoint: Path) -> set[int]:
    if not checkpoint.exists():
        return set()
    latest: dict[int, dict[str, Any]] = {}
    for line in checkpoint.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[int(row["question_id"])] = row
    return {
        question_id
        for question_id, row in latest.items()
        if row.get("status") != "error"
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen3-coder-plus")
    parser.add_argument("--sql-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--stratum", action="append")
    parser.add_argument("--question-id", type=int, action="append")
    parser.add_argument("--enable-semantic-recovery", action="store_true")
    parser.add_argument("--allow-free-rewrite", action="store_true")
    args = parser.parse_args()

    if not os.getenv("DS_API_KEY") and not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("No model API key is set in the process environment")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = manifest["items"]
    if args.stratum:
        requested_strata = set(args.stratum)
        rows = [row for row in rows if row["stratum"] in requested_strata]
    if args.question_id:
        requested_ids = set(args.question_id)
        rows = [row for row in rows if int(row["question_id"]) in requested_ids]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "meta_controller_results.jsonl"
    completed = _completed_ids(checkpoint)
    scheduled = [row for row in rows if int(row["question_id"]) not in completed]
    if args.max_items is not None:
        scheduled = scheduled[: args.max_items]
    item_map = _load_items(args.snapshot, {int(row["question_id"]) for row in scheduled})
    llm = _load_llm(args.config, args.model)
    print(
        json.dumps(
            {"completed": len(completed), "scheduled": len(scheduled), "model": args.model},
            ensure_ascii=False,
        ),
        flush=True,
    )
    for index, row in enumerate(scheduled, 1):
        question_id = int(row["question_id"])
        try:
            result = _run_one(
                llm,
                item_map[question_id],
                row,
                args.sql_timeout_seconds,
                args.enable_semantic_recovery,
                args.allow_free_rewrite,
            )
        except Exception as exc:
            item = item_map[question_id]
            result = {
                "question_id": question_id,
                "db_id": item.get("database_id"),
                "difficulty": item.get("difficulty"),
                "stratum": row["stratum"],
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "base_sql": item.get("final_selected_sql") or "",
                "final_sql": item.get("final_selected_sql") or "",
                "changed": False,
            }
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
        print(
            json.dumps(
                {
                    "finished": index,
                    "scheduled": len(scheduled),
                    "question_id": question_id,
                    "status": result["status"],
                    "route": result.get("route"),
                    "changed": result["changed"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
