"""Syntax-only SQL revision for generated SQL candidates."""

from __future__ import annotations

import time
from typing import Any, Callable

from agent.sqlBuilderAgent.generation_methods.models import SQLCandidate
from agent.sqlBuilderAgent.generation_methods.prompts import build_syntax_revision_prompt
from agent.sqlBuilderAgent.generation_methods.utils import hash_rows
from agent.sqlBuilderAgent.generation_methods.utils import normalize_sql
from agent.sqlBuilderAgent.generation_methods.utils import parse_xml_result
from agent.sqlBuilderAgent.generation_methods.utils import rows_to_table_string
from utils.db import connect_bird_database
from utils.sql_safety import ensure_select_sql


LLMTextFn = Callable[[str], str]


class SyntaxOnlyReviser:
    """Run only syntax/execution repair over unique SQL candidates."""

    def revise(
        self,
        *,
        candidates: list[SQLCandidate],
        db_id: str,
        schema_profile: str,
        question: str,
        evidence: str | None,
        llm_text: LLMTextFn,
    ) -> tuple[list[SQLCandidate], dict[str, str], dict[str, str]]:
        revised: list[SQLCandidate] = []
        prompts: dict[str, str] = {}
        raw_responses: dict[str, str] = {}
        seen: dict[str, SQLCandidate] = {}
        for candidate in candidates:
            normalized = normalize_sql(candidate.sql)
            if normalized and normalized not in seen:
                seen[normalized] = candidate

        for index, candidate in enumerate(seen.values()):
            execution = execute_select(candidate.sql, db_id)
            if execution["executable"]:
                revised.append(_candidate_with_execution(candidate, execution))
                continue

            prompt = build_syntax_revision_prompt(
                schema_profile=schema_profile,
                question=question,
                evidence=evidence,
                sql=candidate.sql,
                error=str(execution["error_message"] or ""),
            )
            prompt_key = f"syntax_revision_{index}"
            prompts[prompt_key] = prompt
            response = llm_text(prompt)
            raw_responses[prompt_key] = response
            revised_sql = parse_xml_result(response)
            if not revised_sql:
                revised.append(
                    SQLCandidate(
                        sql=candidate.sql,
                        source=f"{candidate.source}:syntax_failed",
                        revised_from=None,
                        error_message=str(execution["error_message"] or "syntax revision returned no SQL"),
                    )
                )
                continue

            revised_execution = execute_select(revised_sql, db_id)
            revised.append(
                _candidate_with_execution(
                    SQLCandidate(
                        sql=revised_sql,
                        source=f"{candidate.source}:syntax_revision",
                        revised_from=candidate.sql,
                        error_message=None if revised_execution["executable"] else revised_execution["error_message"],
                    ),
                    revised_execution,
                )
            )
        return revised, prompts, raw_responses


def execute_select(sql: str, db_id: str) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        ensure_select_sql(sql)
        with connect_bird_database(db_id) as conn:
            rows = conn.execute(sql).fetchall()
        return {
            "executable": True,
            "rows": rows,
            "row_count": len(rows),
            "latency_ms": round((time.perf_counter() - start) * 1000.0, 3),
            "result_hash": hash_rows(rows),
            "result_table_str": rows_to_table_string(rows),
            "error_message": None,
        }
    except Exception as exc:
        return {
            "executable": False,
            "rows": None,
            "row_count": None,
            "latency_ms": round((time.perf_counter() - start) * 1000.0, 3),
            "result_hash": None,
            "result_table_str": str(exc),
            "error_message": str(exc),
        }


def _candidate_with_execution(candidate: SQLCandidate, execution: dict[str, Any]) -> SQLCandidate:
    return SQLCandidate(
        sql=candidate.sql,
        source=candidate.source,
        revised_from=candidate.revised_from,
        error_message=None if execution["executable"] else str(execution["error_message"]),
        latency_ms=execution["latency_ms"],
        row_count=execution["row_count"],
        result_hash=execution["result_hash"],
        consistency_score=candidate.consistency_score,
        selection_reason=candidate.selection_reason,
    )
