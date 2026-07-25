"""Generate independent SQL repairs and accept only execution-consensus changes.

The repair path never reads gold SQL. Gold is consumed only by the separate
post-run evaluator.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.config import LLMConfig
from app.dataset import load_dataset, save_dataset
from app.llm import LLM
from app.pipeline.sql_selection.semantic_risk import audit_semantic_risks
from app.pipeline.sql_selection.semantic_change_guard import (
    validate_semantic_change,
)
try:
    from .run_candidate_meta_controller import (
        _cluster_candidates,
        _few_shot_context,
        _parse_json_object,
        _rows_preview,
        _safe_select_sql,
        _schema_context,
        _value_context,
    )
    from .runtime_utils import execute_rows as _execute_rows
    from .runtime_utils import iter_snapshot_items as _iter_snapshot_items
except ImportError:
    from run_candidate_meta_controller import (
        _cluster_candidates,
        _few_shot_context,
        _parse_json_object,
        _rows_preview,
        _safe_select_sql,
        _schema_context,
        _value_context,
    )
    from runtime_utils import execute_rows as _execute_rows
    from runtime_utils import iter_snapshot_items as _iter_snapshot_items


GENERATOR_PERSPECTIVES = (
    (
        "requirements",
        "Decompose every requested output, filter, aggregation, ordering, limit, "
        "tie, and NULL requirement before writing SQL.",
    ),
    (
        "grounding",
        "Rebuild the minimal valid join graph from the schema and ground every "
        "literal and column reference in the question or evidence.",
    ),
    (
        "adversarial",
        "Treat all existing SQL as potentially wrong. Find the most likely shared "
        "semantic mistake, then independently derive the corrected answer.",
    ),
)

JUDGE_FOCUSES = (
    "Check complete requirement coverage and requested projection.",
    "Check joins, filters, value grounding, and NULL behavior.",
    "Check aggregation, grouping, ordering, limits, and ties.",
    "Check schema validity and whether the execution result is semantically plausible.",
)

LABEL_PATTERN = re.compile(r"<result>\s*([AB])\s*</result>", re.IGNORECASE)


def _load_llms(
    config_path: Path,
    model: str,
    profile_name: str | None = None,
) -> tuple[LLM, LLM]:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    profile_name = profile_name or config["run"]["default_llm_profile"]
    base_profile = dict(config["llm_profiles"][profile_name])

    generator_profile = dict(base_profile)
    generator_profile.update(
        model=model,
        max_tokens=16384,
        temperature=0.0,
        n_call_strategy="split",
        max_request_n=1,
        thinking="enabled",
        reasoning_effort="high",
    )
    judge_profile = dict(base_profile)
    judge_profile.update(
        model=model,
        max_tokens=2048,
        temperature=0.2,
        n_call_strategy="split",
        max_request_n=1,
        thinking="disabled",
        reasoning_effort=None,
    )
    return LLM(LLMConfig(**generator_profile)), LLM(LLMConfig(**judge_profile))


def _full_schema_context(item: dict[str, Any]) -> str:
    schema = item.get("database_schema") or {}
    tables = schema.get("tables") or {}
    lines: list[str] = []
    for table_name, table in tables.items():
        columns = table.get("columns") or {}
        rendered = []
        for column_name, column in columns.items():
            details = [str(column.get("column_type") or "")]
            if column.get("description"):
                details.append(str(column["description"])[:180])
            rendered.append(
                f"{column_name} ({'; '.join(value for value in details if value)})"
            )
        lines.append(f"{table_name}: " + ", ".join(rendered))
    return "\n".join(lines)[:32000] or _schema_context(item)


def _candidate_context(clusters: list[dict[str, Any]]) -> str:
    blocks = []
    for index, cluster in enumerate(clusters[:3], 1):
        blocks.append(
            f"Existing candidate {index}\n"
            f"Result preview: {_rows_preview(cluster['rows'])}\n"
            f"SQL:\n{cluster['sql']}"
        )
    return "\n\n".join(blocks) or "(none)"


def _generation_prompt(
    item: dict[str, Any],
    base_sql: str,
    clusters: list[dict[str, Any]],
    perspective: str,
) -> str:
    return f"""You are an independent Text-to-SQL repair agent. Produce one SQLite SELECT query that exactly answers the question.

Review perspective:
{perspective}

Rules:
- Derive the answer from the question, evidence, and schema. Existing SQL is diagnostic only.
- Return exactly the requested columns and rows.
- Prefer the smallest sufficient join graph. Avoid joins that can silently remove rows.
- Retrieved values are approximate neighbors, not aliases. Exact question literals normally use equality.
- Preserve top-k ties with RANK unless the question requires exactly k rows or gives a tie-breaker.
- Do not add generic IS NOT NULL filters or broaden equality to LIKE without textual support.
- Execution success does not prove semantic correctness.
- Use only the supplied schema. Return one complete, read-only query.

Question:
{item.get('question') or ''}

Evidence:
{item.get('evidence') or '(none)'}

Full database schema:
{_full_schema_context(item)}

Linked-schema focus:
{_schema_context(item)}

Retrieved values:
{_value_context(item) or '(none)'}

Few-shot conventions:
{_few_shot_context(item)}

CURRENT SQL:
{base_sql}

Existing executable alternatives:
{_candidate_context(clusters)}

Return JSON only:
{{"sql":"SELECT ...","diagnosis":"specific semantic issue repaired"}}
"""


def _ask_repair(
    llm: LLM,
    prompt: str,
) -> tuple[str | None, str, dict[str, int]]:
    responses, usage = llm.ask(
        [{"role": "user", "content": prompt}],
        system_message={
            "role": "system",
            "content": "You are a rigorous SQLite expert. Return valid JSON only.",
        },
        n=1,
        timeout=600,
    )
    raw = responses[0].content or "" if responses else ""
    parsed = _parse_json_object(raw)
    sql = str((parsed or {}).get("sql") or "").strip()
    return (sql if _safe_select_sql(sql) else None), raw, usage


def _judge_prompt(
    item: dict[str, Any],
    sql_a: str,
    rows_a: set[tuple[Any, ...]],
    sql_b: str,
    rows_b: set[tuple[Any, ...]],
    focus: str,
) -> str:
    return f"""Select the SQL that more exactly answers the question. Candidate order is randomized; do not prefer the first or newer query.

Review emphasis:
{focus}

Check all requested columns, joins, filters, literals, aggregation grain,
DISTINCT, NULL behavior, ordering, limits, and ties. Execution rows are
diagnostic evidence, not ground truth. Audit warnings are advisory only.

Question:
{item.get('question') or ''}

Evidence:
{item.get('evidence') or '(none)'}

Schema:
{_schema_context(item)}

Candidate A audit warnings:
{'; '.join(audit_semantic_risks(sql_a, item.get('question') or '', item.get('evidence') or '', schema=item.get('database_schema_after_schema_linking'))) or 'none'}
Candidate A SQL:
{sql_a}
Candidate A result:
{_rows_preview(rows_a)}

Candidate B audit warnings:
{'; '.join(audit_semantic_risks(sql_b, item.get('question') or '', item.get('evidence') or '', schema=item.get('database_schema_after_schema_linking'))) or 'none'}
Candidate B SQL:
{sql_b}
Candidate B result:
{_rows_preview(rows_b)}

Return exactly one XML result tag containing A or B, with no explanation.
"""


def _parse_label(content: str) -> str | None:
    text = content.strip()
    match = LABEL_PATTERN.search(text)
    if match:
        return match.group(1).upper()
    if text.upper() in {"A", "B"}:
        return text.upper()
    labels = re.findall(r"\b(?:CANDIDATE\s+)?([AB])\b", text.upper())
    return labels[0] if labels and len(set(labels)) == 1 else None


def _ask_judge(
    llm: LLM,
    prompt: str,
    roles: dict[str, str],
) -> tuple[str | None, str, dict[str, int]]:
    responses, usage = llm.ask(
        [{"role": "user", "content": prompt}],
        n=1,
        timeout=300,
    )
    raw = responses[0].content or "" if responses else ""
    label = _parse_label(raw)
    return roles.get(label or ""), raw, usage


def _run_one(
    generator: LLM,
    judge: LLM,
    item: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    question_id = int(item["question_id"])
    base_sql = str(item.get("final_selected_sql") or "").strip()
    base_rows, base_error = _execute_rows(
        item["database_path"], base_sql, timeout_seconds
    )
    if base_rows is None:
        base_rows = set()

    clusters = _cluster_candidates(item, timeout_seconds)
    proposals: list[dict[str, Any]] = []
    usage = collections.Counter()
    for name, perspective in GENERATOR_PERSPECTIVES:
        sql, raw, call_usage = _ask_repair(
            generator,
            _generation_prompt(item, base_sql, clusters, perspective),
        )
        usage.update(call_usage)
        rows = None
        error = "invalid_sql"
        if sql:
            rows, error = _execute_rows(item["database_path"], sql, timeout_seconds)
        proposals.append(
            {
                "perspective": name,
                "sql": sql,
                "rows": rows,
                "error": error,
                "raw": raw[:2000],
            }
        )

    result_groups: dict[frozenset[tuple[Any, ...]], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for proposal in proposals:
        if proposal["rows"] is not None:
            result_groups[frozenset(proposal["rows"])].append(proposal)
    ranked = sorted(result_groups.values(), key=len, reverse=True)
    consensus = ranked[0] if ranked and len(ranked[0]) >= 2 else []
    proposed_sql = consensus[0]["sql"] if consensus else None
    proposed_rows = consensus[0]["rows"] if consensus else None

    status = "keep_no_generation_consensus"
    final_sql = base_sql
    judge_votes: list[str | None] = []
    judge_raw: list[str] = []
    if proposed_sql and proposed_rows is not None:
        if frozenset(proposed_rows) == frozenset(base_rows):
            status = "keep_execution_equivalent"
        else:
            for index, focus in enumerate(JUDGE_FOCUSES):
                incumbent_first = index < len(JUDGE_FOCUSES) // 2
                if incumbent_first:
                    sql_a, rows_a = base_sql, base_rows
                    sql_b, rows_b = proposed_sql, proposed_rows
                    roles = {"A": "incumbent", "B": "proposed"}
                else:
                    sql_a, rows_a = proposed_sql, proposed_rows
                    sql_b, rows_b = base_sql, base_rows
                    roles = {"A": "proposed", "B": "incumbent"}
                vote, raw, call_usage = _ask_judge(
                    judge,
                    _judge_prompt(item, sql_a, rows_a, sql_b, rows_b, focus),
                    roles,
                )
                usage.update(call_usage)
                judge_votes.append(vote)
                judge_raw.append(raw[:1000])
            proposed_votes = collections.Counter(judge_votes)["proposed"]
            guard_reasons = validate_semantic_change(
                base_sql,
                proposed_sql,
                item.get("question") or "",
                item.get("evidence") or "",
            )
            if proposed_votes >= 3 and not guard_reasons:
                final_sql = proposed_sql
                status = "accepted_consensus_repair"
            elif guard_reasons:
                status = "keep_semantic_guard_rejected"
            else:
                status = "keep_judge_rejected"

    return {
        "question_id": question_id,
        "db_id": item.get("database_id"),
        "status": status,
        "base_sql": base_sql,
        "base_error": base_error,
        "final_sql": final_sql,
        "changed": final_sql.strip() != base_sql.strip(),
        "generation_consensus": len(consensus),
        "proposals": [
            {
                key: value
                for key, value in proposal.items()
                if key != "rows"
            }
            for proposal in proposals
        ],
        "judge_votes": judge_votes,
        "judge_raw": judge_raw,
        "token_usage": dict(usage),
    }


def _load_checkpoint(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                if row.get("status") != "error":
                    rows[int(row["question_id"])] = row
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument(
        "--llm-profile",
        default=None,
        help="LLM profile from the TOML config; defaults to run.default_llm_profile",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sql-timeout-seconds", type=float, default=5.0)
    args = parser.parse_args()

    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    completed = _load_checkpoint(args.checkpoint)
    all_items = list(_iter_snapshot_items(args.snapshot))
    scheduled = [
        item for item in all_items if int(item["question_id"]) not in completed
    ]
    generator, judge = _load_llms(args.config, args.model, args.llm_profile)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "total": len(all_items),
                "completed": len(completed),
                "scheduled": len(scheduled),
                "workers": args.workers,
            }
        ),
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _run_one,
                generator,
                judge,
                item,
                args.sql_timeout_seconds,
            ): int(item["question_id"])
            for item in scheduled
        }
        for index, future in enumerate(as_completed(futures), 1):
            question_id = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                item = next(
                    value
                    for value in scheduled
                    if int(value["question_id"]) == question_id
                )
                base_sql = str(item.get("final_selected_sql") or "")
                row = {
                    "question_id": question_id,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "base_sql": base_sql,
                    "final_sql": base_sql,
                    "changed": False,
                }
            with args.checkpoint.open("a", encoding="utf-8") as target:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
                target.flush()
            if row["status"] != "error":
                completed[question_id] = row
            print(
                f"repair {len(completed)}/{len(all_items)} "
                f"qid={question_id} status={row['status']} "
                f"changed={row['changed']}",
                flush=True,
            )

    dataset = load_dataset(str(args.snapshot))
    for item in dataset:
        row = completed.get(int(item.question_id))
        if row:
            item.final_selected_sql = row["final_sql"]
    save_dataset(dataset, str(args.output))
    print(f"Saved consensus-repaired snapshot to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
