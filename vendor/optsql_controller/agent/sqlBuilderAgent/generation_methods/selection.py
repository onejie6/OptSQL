"""Deterministic SQL candidate selection."""

from __future__ import annotations

from collections import Counter
from typing import Any

from agent.sqlBuilderAgent.generation_methods.models import SQLCandidate
from agent.sqlBuilderAgent.generation_methods.revision import execute_select


class ConsistencySelector:
    """Select by executable result consistency, then lower latency."""

    def __init__(self, *, shortcut_consistency_threshold: float = 0.75) -> None:
        self.shortcut_consistency_threshold = shortcut_consistency_threshold

    def select(
        self,
        *,
        candidates: list[SQLCandidate],
        db_id: str,
    ) -> tuple[SQLCandidate, list[dict[str, Any]]]:
        if not candidates:
            raise ValueError("SQL selection requires at least one candidate.")

        evaluated = [self._ensure_evaluated(candidate, db_id) for candidate in candidates]
        executable = [candidate for candidate in evaluated if candidate.result_hash]
        trace: list[dict[str, Any]] = [
            {
                "sql": candidate.sql,
                "source": candidate.source,
                "row_count": candidate.row_count,
                "latency_ms": candidate.latency_ms,
                "result_hash": candidate.result_hash,
                "error_message": candidate.error_message,
            }
            for candidate in evaluated
        ]
        if not executable:
            fallback = evaluated[0]
            return SQLCandidate(
                **{**fallback.__dict__, "selection_reason": "fallback_first_non_executable"}
            ), trace

        counts = Counter(candidate.result_hash for candidate in executable)
        scored: list[SQLCandidate] = []
        for candidate in executable:
            consistency = counts[candidate.result_hash] / len(executable)
            scored.append(
                SQLCandidate(
                    sql=candidate.sql,
                    source=candidate.source,
                    revised_from=candidate.revised_from,
                    error_message=candidate.error_message,
                    latency_ms=candidate.latency_ms,
                    row_count=candidate.row_count,
                    result_hash=candidate.result_hash,
                    consistency_score=round(consistency, 4),
                    selection_reason=None,
                )
            )

        scored.sort(key=lambda item: (item.consistency_score, -(item.latency_ms or float("inf"))), reverse=True)
        best = scored[0]
        reason = (
            "shortcut_consistency"
            if best.consistency_score >= self.shortcut_consistency_threshold
            else "consistency_then_latency"
        )
        return SQLCandidate(
            sql=best.sql,
            source=best.source,
            revised_from=best.revised_from,
            error_message=best.error_message,
            latency_ms=best.latency_ms,
            row_count=best.row_count,
            result_hash=best.result_hash,
            consistency_score=best.consistency_score,
            selection_reason=reason,
        ), trace

    @staticmethod
    def _ensure_evaluated(candidate: SQLCandidate, db_id: str) -> SQLCandidate:
        if candidate.result_hash or candidate.error_message:
            return candidate
        execution = execute_select(candidate.sql, db_id)
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
