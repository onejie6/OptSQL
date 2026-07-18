"""Unified optimization knowledge base interface."""

from myTypes import OptimizationCase
from myTypes import RetrievedStrategy
from myTypes import VerifiedContextBlueprint


class UnifiedOptimizationKnowledgeBase:
    """Retrieve and evolve optimization rules, cases, and negative examples."""

    def retrieve_hybrid_strategies(
        self,
        question: str,
        sql: str,
        blueprint: VerifiedContextBlueprint,
        bottleneck_tags: list[str],
        top_k: int,
    ) -> list[RetrievedStrategy]:
        raise NotImplementedError

    def retrieve_negative_cases(
        self,
        rule_ids: list[str],
        blueprint: VerifiedContextBlueprint,
        sql: str,
    ) -> list[OptimizationCase]:
        raise NotImplementedError

    def upsert_if_novel_case(self, optimization_case: OptimizationCase) -> bool:
        raise NotImplementedError

    def list_expert_rules(self) -> list[dict]:
        raise NotImplementedError
