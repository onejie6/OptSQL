"""Data contracts for embedded SQL generation methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SQLCandidate:
    sql: str
    source: str
    revised_from: str | None = None
    error_message: str | None = None
    latency_ms: float | None = None
    row_count: int | None = None
    result_hash: str | None = None
    consistency_score: float = 0.0
    selection_reason: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    selected_sql: str
    raw_candidates: list[SQLCandidate]
    revised_candidates: list[SQLCandidate]
    selected_candidate: SQLCandidate
    prompts: dict[str, str]
    raw_responses: dict[str, str]
    selection_trace: list[dict[str, Any]]
