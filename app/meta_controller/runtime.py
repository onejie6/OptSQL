from __future__ import annotations

from typing import Any

from .models import ControllerDecision
from .policy import EvidencePolicy
from .trace import ControllerTraceStore


class MetaControllerRuntime:
    """Gold-blind stage supervisor with an append-only decision trace."""

    def __init__(self, config: Any):
        self.config = config
        self.policy = EvidencePolicy(config)
        self.trace = ControllerTraceStore(config.trace_path)

    def assess_schema(
        self,
        item: Any,
        results: dict[str, dict[str, list[str]] | None],
        current_budget: int,
    ) -> ControllerDecision:
        decision = self.policy.assess_schema(item, results, current_budget)
        self.trace.append_decision(decision)
        return decision

    def assess_generation(
        self,
        item: Any,
        candidates_by_source: dict[str, list[str] | None],
        current_budget: int,
    ) -> ControllerDecision:
        decision = self.policy.assess_generation(item, candidates_by_source, current_budget)
        self.trace.append_decision(decision)
        return decision

    def assess_selection(
        self,
        item: Any,
        candidates: list[tuple[Any, ...]],
        shortcut_threshold: float,
        evaluator_budget: int,
    ) -> ControllerDecision:
        decision = self.policy.assess_selection(
            item,
            candidates,
            shortcut_threshold,
            evaluator_budget,
        )
        self.trace.append_decision(decision)
        return decision

    def assess_optimization(
        self,
        item: Any,
        *,
        risk_score: int,
        min_risk_score: int,
        max_rewrites: int,
        trigger_mode: str,
    ) -> ControllerDecision:
        decision = self.policy.assess_optimization(
            item,
            risk_score=risk_score,
            min_risk_score=min_risk_score,
            max_rewrites=max_rewrites,
            trigger_mode=trigger_mode,
        )
        self.trace.append_decision(decision)
        return decision

    def record_stage_outcome(
        self,
        decision: ControllerDecision,
        *,
        status: str,
        selected_sql: str | None,
        token_usage: dict[str, int],
    ) -> None:
        self.trace.append_decision(
            decision,
            event="stage_outcome",
            status=status,
            selected_sql=selected_sql,
            token_usage=token_usage,
        )

    def record_reflection_result(
        self,
        decision: ControllerDecision,
        *,
        signals_after: dict[str, Any],
        token_usage: dict[str, int],
    ) -> None:
        self.trace.append_decision(
            decision,
            event="reflection_result",
            signals_after=signals_after,
            token_usage=token_usage,
        )
