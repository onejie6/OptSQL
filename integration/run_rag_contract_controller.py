"""Rejudge disputed SQL repairs using a candidate-blind answer contract."""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.dataset import load_dataset, save_dataset
from app.pipeline.sql_selection.semantic_change_guard import (
    validate_semantic_change,
)
try:
    from .run_candidate_meta_controller import (
        _few_shot_context,
        _parse_json_object,
        _rows_preview,
        _schema_context,
        _value_context,
    )
    from .run_consensus_repair import _full_schema_context, _load_llms
    from .runtime_utils import execute_rows as _execute_rows
    from .runtime_utils import iter_snapshot_items as _iter_snapshot_items
except ImportError:
    from run_candidate_meta_controller import (
        _few_shot_context,
        _parse_json_object,
        _rows_preview,
        _schema_context,
        _value_context,
    )
    from run_consensus_repair import _full_schema_context, _load_llms
    from runtime_utils import execute_rows as _execute_rows
    from runtime_utils import iter_snapshot_items as _iter_snapshot_items


ELIGIBLE_STATUSES = {
    "accepted_consensus_repair",
    "keep_judge_rejected",
}


def _load_latest(path: Path) -> dict[int, dict[str, Any]]:
    latest = {}
    if not path.exists():
        return latest
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("status") != "error":
                latest[int(row["question_id"])] = row
    return latest


def _consensus_proposal(
    item: dict[str, Any],
    source_row: dict[str, Any],
    timeout_seconds: float,
) -> tuple[str | None, set[tuple[Any, ...]] | None]:
    groups: dict[frozenset[tuple[Any, ...]], list[str]] = collections.defaultdict(list)
    for proposal in source_row.get("proposals") or []:
        sql = str(proposal.get("sql") or "").strip()
        if not sql:
            continue
        rows, error = _execute_rows(item["database_path"], sql, timeout_seconds)
        if rows is not None and error is None:
            groups[frozenset(rows)].append(sql)
    ranked = sorted(groups.items(), key=lambda pair: len(pair[1]), reverse=True)
    if not ranked or len(ranked[0][1]) < 2:
        return None, None
    rows, sqls = ranked[0]
    return sqls[0], set(rows)


def _contract_prompt(item: dict[str, Any]) -> str:
    return f"""Create a benchmark answer contract for this Text-to-SQL question before seeing any candidate SQL.

The contract must describe the exact result expected by BIRD, not a generally
helpful business interpretation. The question and evidence have priority.
Retrieved examples are advisory conventions from other schemas; use them only
when their wording matches.

Resolve:
- exact output columns and output column count;
- row grain and entity being counted;
- aggregation, DISTINCT, grouping, numerator, and denominator;
- filters and literal matching;
- ordering, LIMIT, and whether boundary ties remain;
- date, age, NULL, and boolean representation;
- operations that would be plausible but unsupported.

Question:
{item.get('question') or ''}

Evidence:
{item.get('evidence') or '(none)'}

Linked schema:
{_schema_context(item)}

Full schema fallback:
{_full_schema_context(item)}

Grounded values:
{_value_context(item) or '(none)'}

Retrieved BIRD-train examples:
{_few_shot_context(item, limit=7)}

Return JSON only:
{{
  "projection_count": 1,
  "projection_items": ["..."],
  "row_grain": "...",
  "aggregation": "...",
  "distinct_policy": "required|forbidden|unspecified",
  "filters": ["..."],
  "ordering": "...",
  "limit": "integer|null|unspecified",
  "tie_policy": "single_row|keep_ties|unspecified",
  "date_age_policy": "...",
  "unsupported_changes": ["..."],
  "uncertainties": ["..."]
}}
"""


def _judge_prompt(
    item: dict[str, Any],
    contract: dict[str, Any],
    sql_a: str,
    rows_a: set[tuple[Any, ...]],
    role_a: str,
    sql_b: str,
    rows_b: set[tuple[Any, ...]],
    role_b: str,
) -> str:
    return f"""Compare two SQLite queries against a candidate-blind BIRD answer contract.

This is a regression gate. Select a replacement only if it fixes a specific
contract violation in the current behavior without introducing any unsupported
semantic refinement. BIRD often expects literal SQL conventions rather than a
broader real-world interpretation. Do not change LIMIT into tie-preserving
RANK, alter output shape, refine age/date calculations, normalize raw values,
or reinterpret a formula unless the contract explicitly requires it. The
incumbent_defect field must describe a defect in the incumbent query, regardless
of whether it appears as Candidate A or Candidate B.

Question:
{item.get('question') or ''}

Evidence:
{item.get('evidence') or '(none)'}

Answer contract:
{json.dumps(contract, ensure_ascii=False)}

Candidate A role: {role_a}
Candidate A SQL:
{sql_a}
Candidate A execution preview:
{_rows_preview(rows_a)}

Candidate B role: {role_b}
Candidate B SQL:
{sql_b}
Candidate B execution preview:
{_rows_preview(rows_b)}

Return JSON only:
{{
  "decision": "A|B",
  "confidence": 0.0,
  "incumbent_defect": "specific contract violation or none",
  "proposal_support": "specific contract field or none",
  "proposal_unsupported_change": "specific added semantic change or none"
}}
"""


def _ask_json(llm, prompt: str) -> tuple[dict[str, Any] | None, str, dict[str, int]]:
    responses, usage = llm.ask(
        [{"role": "user", "content": prompt}],
        system_message={
            "role": "system",
            "content": "You are a conservative BIRD Text-to-SQL evaluator. Return valid JSON only.",
        },
        n=1,
        timeout=600,
    )
    raw = responses[0].content or "" if responses else ""
    return _parse_json_object(raw), raw, usage


def _run_one(
    planner_llm,
    judge_llm,
    item: dict[str, Any],
    source_row: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    question_id = int(item["question_id"])
    base_sql = str(item.get("final_selected_sql") or "")
    base_rows, base_error = _execute_rows(
        item["database_path"], base_sql, timeout_seconds
    )
    proposed_sql, proposed_rows = _consensus_proposal(
        item, source_row, timeout_seconds
    )
    if (
        base_rows is None
        or proposed_sql is None
        or proposed_rows is None
        or frozenset(base_rows) == frozenset(proposed_rows)
    ):
        return {
            "question_id": question_id,
            "status": "keep_no_distinct_proposal",
            "base_sql": base_sql,
            "final_sql": base_sql,
            "changed": False,
            "error": base_error,
        }

    guard_reasons = validate_semantic_change(
        base_sql,
        proposed_sql,
        item.get("question") or "",
        item.get("evidence") or "",
    )
    if guard_reasons:
        return {
            "question_id": question_id,
            "status": "keep_semantic_guard_rejected",
            "base_sql": base_sql,
            "proposed_sql": proposed_sql,
            "final_sql": base_sql,
            "changed": False,
            "guard_reasons": list(guard_reasons),
        }

    contract, contract_raw, contract_usage = _ask_json(
        planner_llm, _contract_prompt(item)
    )
    if contract is None:
        return {
            "question_id": question_id,
            "status": "keep_invalid_contract",
            "base_sql": base_sql,
            "proposed_sql": proposed_sql,
            "final_sql": base_sql,
            "changed": False,
            "contract_raw": contract_raw[:3000],
            "token_usage": contract_usage,
        }

    votes = []
    raw_votes = []
    usage = collections.Counter(contract_usage)
    orders = (
        (
            base_sql,
            base_rows,
            "incumbent/current",
            proposed_sql,
            proposed_rows,
            "proposed replacement",
            {"A": "base", "B": "proposed"},
        ),
        (
            proposed_sql,
            proposed_rows,
            "proposed replacement",
            base_sql,
            base_rows,
            "incumbent/current",
            {"A": "proposed", "B": "base"},
        ),
    )
    for sql_a, rows_a, role_a, sql_b, rows_b, role_b, roles in orders:
        vote, raw, call_usage = _ask_json(
            judge_llm,
            _judge_prompt(
                item,
                contract,
                sql_a,
                rows_a,
                role_a,
                sql_b,
                rows_b,
                role_b,
            ),
        )
        usage.update(call_usage)
        decision = str((vote or {}).get("decision") or "").strip().upper()
        confidence = float((vote or {}).get("confidence") or 0)
        role = roles.get(decision)
        votes.append(
            {
                "role": role,
                "confidence": confidence,
                "incumbent_defect": (vote or {}).get("incumbent_defect"),
                "proposal_support": (vote or {}).get("proposal_support"),
                "proposal_unsupported_change": (vote or {}).get(
                    "proposal_unsupported_change"
                ),
            }
        )
        raw_votes.append(raw[:3000])

    accepts = [
        vote
        for vote in votes
        if vote["role"] == "proposed"
        and vote["confidence"] >= 0.85
        and str(vote.get("incumbent_defect") or "").strip().lower()
        not in {"", "none"}
        and str(vote.get("proposal_support") or "").strip().lower()
        not in {"", "none"}
        and str(vote.get("proposal_unsupported_change") or "").strip().lower()
        in {"", "none"}
    ]
    accepted = len(accepts) == 2
    return {
        "question_id": question_id,
        "db_id": item.get("database_id"),
        "status": (
            "accepted_contract_repair"
            if accepted
            else "keep_contract_rejected"
        ),
        "base_sql": base_sql,
        "proposed_sql": proposed_sql,
        "final_sql": proposed_sql if accepted else base_sql,
        "changed": accepted,
        "contract": contract,
        "contract_raw": contract_raw[:3000],
        "votes": votes,
        "raw_votes": raw_votes,
        "token_usage": dict(usage),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
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
    source_rows = _load_latest(args.source_checkpoint)
    completed = _load_latest(args.checkpoint)
    items = list(_iter_snapshot_items(args.snapshot))
    scheduled = [
        item
        for item in items
        if int(item["question_id"]) not in completed
        and source_rows.get(int(item["question_id"]), {}).get("status")
        in ELIGIBLE_STATUSES
    ]
    planner, judge = _load_llms(args.config, args.model, args.llm_profile)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "total_items": len(items),
                "eligible": len(scheduled) + len(completed),
                "completed": len(completed),
                "scheduled": len(scheduled),
            }
        ),
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _run_one,
                planner,
                judge,
                item,
                source_rows[int(item["question_id"])],
                args.sql_timeout_seconds,
            ): int(item["question_id"])
            for item in scheduled
        }
        for future in as_completed(futures):
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
                f"contract {len(completed)}/{len(items)} qid={question_id} "
                f"status={row['status']} changed={row['changed']}",
                flush=True,
            )

    dataset = load_dataset(str(args.snapshot))
    for item in dataset:
        decision = completed.get(int(item.question_id))
        if decision:
            item.final_selected_sql = decision["final_sql"]
    save_dataset(dataset, str(args.output))
    print(f"Saved contract-controller snapshot to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
