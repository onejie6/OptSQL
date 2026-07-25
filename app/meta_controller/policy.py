from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations
from typing import Any, Iterable

from .models import ControllerAction, ControllerDecision, ControllerStage


def _schema_refs(linked: dict[str, list[str]] | None) -> set[str]:
    refs: set[str] = set()
    for table_name, columns in (linked or {}).items():
        refs.add(str(table_name).lower())
        refs.update(f"{str(table_name).lower()}.{str(column).lower()}" for column in columns or [])
    return refs


def _mean_pairwise_jaccard(ref_sets: Iterable[set[str]]) -> float:
    sets = list(ref_sets)
    scores: list[float] = []
    for left, right in combinations(sets, 2):
        union = left | right
        scores.append(len(left & right) / len(union) if union else 1.0)
    return sum(scores) / len(scores) if scores else 1.0


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


class EvidencePolicy:
    """Deterministic, gold-blind policy for allocating extra pipeline work."""

    def __init__(self, config: Any):
        self._config = config

    def assess_schema(
        self,
        item: Any,
        results: dict[str, dict[str, list[str]] | None],
        current_budget: int,
    ) -> ControllerDecision:
        refs_by_source = {name: _schema_refs(value) for name, value in results.items()}
        union_refs = set().union(*refs_by_source.values()) if refs_by_source else set()
        failed_sources = sorted(name for name, value in results.items() if value is None)
        agreement = _mean_pairwise_jaccard(refs_by_source.values())
        max_budget = self._config.max_schema_sampling_budget
        needs_more_evidence = (
            bool(failed_sources)
            or len(union_refs) < self._config.min_schema_references
            or agreement < self._config.schema_agreement_threshold
        )
        can_escalate = self._config.mode == "adaptive" and current_budget < max_budget
        if needs_more_evidence and can_escalate:
            action = ControllerAction.ESCALATE
            reason = "schema evidence is missing or disagrees across independent linkers"
            next_budget = max_budget
            confidence = max(0.0, min(1.0, agreement))
        elif not union_refs:
            action = ControllerAction.FALLBACK
            reason = "no schema reference survived; preserve full schema as a safe fallback"
            next_budget = current_budget
            confidence = 0.0
        else:
            action = ControllerAction.CONTINUE
            reason = "schema evidence is sufficient for the current budget"
            next_budget = current_budget
            confidence = max(0.0, min(1.0, agreement))
        return ControllerDecision(
            question_id=int(item.question_id),
            database_id=str(item.database_id),
            stage=ControllerStage.SCHEMA_LINKING,
            action=action,
            reason=reason,
            confidence=confidence,
            signals={
                "agreement": agreement,
                "union_reference_count": len(union_refs),
                "failed_sources": failed_sources,
                "source_reference_counts": {name: len(refs) for name, refs in refs_by_source.items()},
            },
            budget_before={"per_llm_linker": current_budget},
            budget_after={"per_llm_linker": next_budget},
        )

    def assess_generation(
        self,
        item: Any,
        candidates_by_source: dict[str, list[str] | None],
        current_budget: int,
    ) -> ControllerDecision:
        normalized_sources: dict[str, set[str]] = {
            source: {normalize_sql(sql) for sql in (candidates or []) if sql and sql.strip()}
            for source, candidates in candidates_by_source.items()
        }
        support: dict[str, set[str]] = defaultdict(set)
        for source, candidates in normalized_sources.items():
            for candidate in candidates:
                support[candidate].add(source)
        unique_count = len(support)
        max_support = max((len(sources) for sources in support.values()), default=0)
        failed_sources = sorted(source for source, candidates in candidates_by_source.items() if candidates is None)
        needs_more_evidence = (
            bool(failed_sources)
            or unique_count < self._config.min_unique_candidates
            or max_support < self._config.min_cross_source_support
        )
        max_budget = self._config.max_generation_sampling_budget
        can_escalate = self._config.mode == "adaptive" and current_budget < max_budget
        if needs_more_evidence and can_escalate:
            action = ControllerAction.ESCALATE
            reason = "candidate set lacks diversity or independent cross-route support"
            next_budget = max_budget
        elif not support:
            action = ControllerAction.FALLBACK
            reason = "no valid candidate was generated"
            next_budget = current_budget
        else:
            action = ControllerAction.CONTINUE
            reason = "candidate evidence is sufficient for execution-based selection"
            next_budget = current_budget
        confidence = min(1.0, max_support / max(1, len(normalized_sources)))
        return ControllerDecision(
            question_id=int(item.question_id),
            database_id=str(item.database_id),
            stage=ControllerStage.SQL_GENERATION,
            action=action,
            reason=reason,
            confidence=confidence,
            signals={
                "unique_candidate_count": unique_count,
                "max_cross_source_support": max_support,
                "failed_sources": failed_sources,
                "source_candidate_counts": {
                    source: len(candidates) for source, candidates in normalized_sources.items()
                },
            },
            budget_before={"per_generator": current_budget},
            budget_after={"per_generator": next_budget},
        )

    def assess_selection(
        self,
        item: Any,
        candidates: list[tuple[Any, ...]],
        shortcut_threshold: float,
        evaluator_budget: int,
    ) -> ControllerDecision:
        scores = [float(candidate[2]) for candidate in candidates]
        top_score = scores[0] if scores else 0.0
        score_margin = top_score - scores[1] if len(scores) > 1 else top_score
        if not candidates:
            action = ControllerAction.FALLBACK
            reason = "no executable result cluster is available"
            confidence = 0.0
            votes = 0
        elif len(candidates) == 1 or top_score >= shortcut_threshold:
            action = ControllerAction.CONTINUE
            reason = "execution-result consensus is sufficient for shortcut selection"
            confidence = min(1.0, top_score)
            votes = 0
        else:
            action = ControllerAction.REFLECT
            reason = "execution-result clusters disagree; run independent pairwise judgments"
            confidence = min(1.0, max(0.0, top_score))
            votes = evaluator_budget
        return ControllerDecision(
            question_id=int(item.question_id),
            database_id=str(item.database_id),
            stage=ControllerStage.SQL_SELECTION,
            action=action,
            reason=reason,
            confidence=confidence,
            signals={
                "executable_cluster_count": len(candidates),
                "top_consistency_score": top_score,
                "top_score_margin": score_margin,
            },
            budget_before={"pairwise_votes": 0},
            budget_after={"pairwise_votes": votes},
        )

    def assess_optimization(
        self,
        item: Any,
        *,
        risk_score: int,
        min_risk_score: int,
        max_rewrites: int,
        trigger_mode: str,
    ) -> ControllerDecision:
        should_optimize = max_rewrites > 0 and (
            trigger_mode == "all" or risk_score >= min_risk_score
        )
        return ControllerDecision(
            question_id=int(item.question_id),
            database_id=str(item.database_id),
            stage=ControllerStage.OPTIMIZATION,
            action=ControllerAction.OPTIMIZE if should_optimize else ControllerAction.CONTINUE,
            reason=(
                "execution-plan risk justifies a semantics-preserving rewrite"
                if should_optimize
                else "no optimization risk exceeds the configured trigger"
            ),
            confidence=min(1.0, risk_score / max(1, min_risk_score + 2)),
            signals={"risk_score": risk_score, "trigger_mode": trigger_mode},
            budget_before={"rewrite_attempts": 0},
            budget_after={"rewrite_attempts": max_rewrites if should_optimize else 0},
        )
