"""SQL Rewriter Agent implementation.

The rewriter proposes one guarded candidate SQL. It does not prove
equivalence, measure performance, accept candidates, or modify schema.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import date, timedelta
import inspect
from typing import Any, Callable

import sqlglot
import sqlglot.expressions as exp

from agent.base import BaseAgent
from agent.validator import ValidatorAgent
from agent.rewrite_operators import build_operator_strategies
from agent.rewrite_operators import build_operator_strategies_from_opportunities
from agent.rewrite_operators import detect_operator_opportunities
from agent.rewrite_operators.registry import get_operator_strategy_metadata
from agent.rewrite_operators.shapes import argmax_aggregate_to_topk_shape
from agent.rewrite_operators.shapes import dimension_key_first_then_fact_probe_shape
from agent.rewrite_operators.shapes import distinct_join_to_semijoin_shape
from agent.rewrite_operators.shapes import distinct_extrema_to_grouped_having_shape
from agent.rewrite_operators.shapes import distinct_top1_to_grouped_extrema_shape
from agent.rewrite_operators.shapes import filter_dimension_before_top1_shape
from agent.rewrite_operators.shapes import grouped_max_top1_before_join_shape
from agent.rewrite_operators.shapes import prefer_summary_table_when_grain_matches_shape
from agent.rewrite_operators.shapes import reanchor_join_driver_shape
from agent.rewrite_operators.shapes import repeated_rescan_to_conditional_agg_shape
from agent.rewrite_operators.shapes import scalar_extrema_anchor_then_lookup_tail_shape
from agent.rewrite_operators.shapes import scalar_extrema_ladder_shape
from agent.rewrite_operators.shapes import symmetric_union_arm_pruning_shape
from agent.rewrite_operators.shapes import top1_anchor_then_lookup_tail_shape
from agent.rewrite_operators.shapes import topk_before_join_shape
from myTypes import (
    AgentRequest,
    AgentResponse,
    BottleneckReport,
    ColumnRef,
    GenericStrategyRewritePlan,
    JoinEdge,
    JoinGraph,
    NoOpRewritePlan,
    OperatorDeterministicRewritePlan,
    RejectedRewritePlan,
    RetrievedStrategy,
    RewritePlan,
    RewriteHint,
    SQLVersion,
    VerifiedContextBlueprint,
)
from utils.db import connect_bird_database
from utils.bird_table_stats_cache import get_cached_bird_db_table_row_counts


LLMRewriteFn = Callable[..., str]
LLMFreeExploreFn = Callable[[dict], str]
FREE_EXPLORATION_PROMPT_PROFILES = {"coder", "strong_llm"}
@dataclass(frozen=True)
class ApplicabilityResult:
    matched: bool
    confidence: float
    matched_conditions: list[str]
    failed_conditions: list[str]
    semantic_risks: list[str]
    required_fragments: dict[str, Any]


class SQLRewriterAgent(BaseAgent):
    """Rewrite SQL with retrieved optimization strategies under guardrails."""

    name = "sql_rewriter"

    def __init__(
        self,
        *,
        rag_engine: Any | None = None,
        llm_rewriter: LLMRewriteFn | None = None,
        llm_free_explorer: LLMFreeExploreFn | None = None,
        free_exploration_prompt_profile: str = "strong_llm",
        top_k: int = 5,
        expert_top_k: int = 3,
        hist_top_k: int = 3,
        max_projection_columns: int = 16,
        rag_confidence_threshold: float = 0.7,
    ) -> None:
        self.rag_engine = rag_engine
        self.llm_rewriter = llm_rewriter
        self.llm_free_explorer = llm_free_explorer
        self.free_exploration_prompt_profile = _normalize_free_exploration_prompt_profile(
            free_exploration_prompt_profile
        )
        self.top_k = top_k
        self.expert_top_k = expert_top_k
        self.hist_top_k = hist_top_k
        self.max_projection_columns = max_projection_columns
        self.rag_confidence_threshold = rag_confidence_threshold

    # ------------------------------------------------------------------
    # Input selection
    # ------------------------------------------------------------------

    def select_sql_version(self, request: AgentRequest) -> SQLVersion:
        for container in (request.input_artifacts, request.runtime_state):
            for key in ("sql_version", "current_sql_version"):
                sql_version = _coerce_sql_version(container.get(key))
                if sql_version is not None:
                    return sql_version
        raise ValueError(
            "SQLRewriterAgent requires input_artifacts['sql_version'] "
            "or runtime_state['current_sql_version']."
        )

    def select_blueprint(self, request: AgentRequest) -> VerifiedContextBlueprint:
        for container in (request.input_artifacts, request.runtime_state):
            blueprint = _coerce_blueprint(container.get("blueprint"))
            if blueprint is not None:
                return blueprint
        raise ValueError("SQLRewriterAgent requires input_artifacts['blueprint'].")

    def select_bottleneck_report(self, request: AgentRequest) -> BottleneckReport:
        for container in (request.input_artifacts, request.runtime_state):
            report = _coerce_bottleneck_report(container.get("bottleneck_report"))
            if report is not None:
                return report
        raise ValueError("SQLRewriterAgent requires input_artifacts['bottleneck_report'].")

    def select_retrieved_strategies(self, request: AgentRequest) -> list[RetrievedStrategy]:
        raw = request.input_artifacts.get("retrieved_strategies")
        if raw is None:
            return []
        if not isinstance(raw, list):
            raise ValueError("input_artifacts['retrieved_strategies'] must be a list.")
        return [_coerce_retrieved_strategy(item) for item in raw]

    def select_free_exploration_history(self, request: AgentRequest) -> list[dict]:
        history = request.input_artifacts.get("free_exploration_history")
        reflection_context = request.input_artifacts.get("reflection_context") or {}
        if history is None:
            history = reflection_context.get("failed_free_exploration_directions")
        if history is None:
            return []
        if not isinstance(history, list):
            raise ValueError("free_exploration_history must be a list.")
        result: list[dict] = []
        for item in history:
            if isinstance(item, dict):
                result.append(dict(item))
            else:
                result.append({"direction": str(item)})
        return result

    # ------------------------------------------------------------------
    # Retrieval and planning
    # ------------------------------------------------------------------

    def retrieve_strategies(
        self,
        question: str,
        sql_version: SQLVersion,
        blueprint: VerifiedContextBlueprint,
        bottleneck_report: BottleneckReport,
    ) -> list[RetrievedStrategy]:
        """Retrieve concrete strategies, falling back to deterministic MVP rules."""
        strategies: list[RetrievedStrategy] = []
        if self.rag_engine is not None:
            tags = _unique(
                list(bottleneck_report.risk_tags)
                + [hint.strategy for hint in bottleneck_report.rewrite_hints]
            )
            strategies = list(
                self.rag_engine.retrieve_hybrid_strategies(
                    question=question,
                    sql=sql_version.sql,
                    blueprint=blueprint,
                    bottleneck_tags=tags,
                    top_k=self.top_k,
                )
            )
        return strategies or self._fallback_strategies(bottleneck_report)

    def retrieve_rag_strategies(
        self,
        question: str,
        sql_version: SQLVersion,
        blueprint: VerifiedContextBlueprint,
        bottleneck_report: BottleneckReport,
    ) -> list[RetrievedStrategy]:
        if self.rag_engine is None:
            return []
        tags = _unique(
            list(bottleneck_report.risk_tags)
            + [hint.strategy for hint in bottleneck_report.rewrite_hints]
        )
        return list(
            self.rag_engine.retrieve_hybrid_strategies(
                question=question,
                sql=sql_version.sql,
                blueprint=blueprint,
                bottleneck_tags=tags,
                top_k=self.top_k,
            )
        )

    def partition_strategy_sources(
        self,
        strategies: list[RetrievedStrategy],
    ) -> tuple[list[RetrievedStrategy], list[RetrievedStrategy], list[RetrievedStrategy]]:
        operators: list[RetrievedStrategy] = []
        experts: list[RetrievedStrategy] = []
        hist: list[RetrievedStrategy] = []
        for strategy in strategies:
            source_type = _strategy_source_type(strategy)
            if source_type == "operator":
                operators.append(strategy)
            elif source_type == "expert":
                experts.append(strategy)
            else:
                hist.append(strategy)
        return operators, experts, hist

    def rank_source_strategies(
        self,
        sql: str,
        strategies: list[RetrievedStrategy],
        *,
        top_k: int,
    ) -> list[RetrievedStrategy]:
        ranked = sorted(
            strategies,
            key=lambda strategy: _source_strategy_priority_score(sql, strategy),
            reverse=True,
        )
        return ranked[: max(0, int(top_k))]

    def high_confidence_strategies(
        self,
        strategies: list[RetrievedStrategy],
    ) -> list[RetrievedStrategy]:
        return [
            strategy
            for strategy in strategies
            if strategy.confidence >= self.rag_confidence_threshold
        ]

    def plan_rewrite(
        self,
        sql_version: SQLVersion,
        strategies: list[RetrievedStrategy],
        bottleneck_report: BottleneckReport,
        blueprint: VerifiedContextBlueprint | None = None,
    ) -> RewritePlan | NoOpRewritePlan:
        """Choose one safe rewrite plan. Returns NoOpRewritePlan when nothing is safe."""
        plans = self.plan_rewrite_candidates(
            sql_version,
            strategies,
            bottleneck_report,
            blueprint,
        )
        return plans[0] if plans else NoOpRewritePlan(reason="no safe rewrite candidate")

    def plan_rewrite_candidates(
        self,
        sql_version: SQLVersion,
        strategies: list[RetrievedStrategy],
        bottleneck_report: BottleneckReport,
        blueprint: VerifiedContextBlueprint | None = None,
    ) -> list[RewritePlan]:
        """Return candidate rewrite plans after explicit operator/generic planning."""
        if blueprint is None:
            raise ValueError("plan_rewrite_candidates requires a VerifiedContextBlueprint.")
        operator_strategies, expert_strategies, hist_strategies = self.partition_strategy_sources(
            strategies
        )
        operator_plans = self.plan_operator_candidates(
            sql_version=sql_version,
            strategies=operator_strategies,
            bottleneck_report=bottleneck_report,
            blueprint=blueprint,
        )
        generic_plans = self.plan_generic_retrieval_candidates(
            sql_version=sql_version,
            strategies=[*expert_strategies, *hist_strategies],
            bottleneck_report=bottleneck_report,
            blueprint=blueprint,
        )
        return self.merge_planned_candidates(operator_plans, generic_plans)

    def plan_operator_candidates(
        self,
        *,
        sql_version: SQLVersion,
        strategies: list[RetrievedStrategy],
        bottleneck_report: BottleneckReport,
        blueprint: VerifiedContextBlueprint,
    ) -> list[RewritePlan]:
        return self._plan_candidates_for_source(
            sql_version=sql_version,
            strategies=strategies,
            bottleneck_report=bottleneck_report,
            blueprint=blueprint,
            allowed_source_types={"operator"},
        )

    def plan_generic_retrieval_candidates(
        self,
        *,
        sql_version: SQLVersion,
        strategies: list[RetrievedStrategy],
        bottleneck_report: BottleneckReport,
        blueprint: VerifiedContextBlueprint,
    ) -> list[RewritePlan]:
        return self._plan_candidates_for_source(
            sql_version=sql_version,
            strategies=strategies,
            bottleneck_report=bottleneck_report,
            blueprint=blueprint,
            allowed_source_types={"expert", "hist", "unknown"},
        )

    def merge_planned_candidates(
        self,
        operator_plans: list[RewritePlan],
        generic_plans: list[RewritePlan],
    ) -> list[RewritePlan]:
        return sorted(
            [*operator_plans, *generic_plans],
            key=self._plan_sort_key,
            reverse=True,
        )

    def _plan_candidates_for_source(
        self,
        *,
        sql_version: SQLVersion,
        strategies: list[RetrievedStrategy],
        bottleneck_report: BottleneckReport,
        blueprint: VerifiedContextBlueprint,
        allowed_source_types: set[str],
    ) -> list[RewritePlan]:
        plans: list[RewritePlan] = []
        for hint in bottleneck_report.rewrite_hints:
            if hint.strategy in {"no_rewrite", "add_index_candidate"}:
                continue
            for strategy in strategies:
                if _strategy_source_type(strategy) not in allowed_source_types:
                    continue
                if not self._align_hint_strategy(hint, strategy):
                    # When no deterministic checker exists, skip alignment and
                    # delegate to the LLM rewriter directly — but only if one is configured.
                    if _has_mvp_checker(hint.strategy) or self.llm_rewriter is None:
                        continue
                applicability = self._check_applicability(
                    sql_version=sql_version,
                    blueprint=blueprint,
                    hint=hint,
                    strategy=strategy,
                )
                if not self._is_plan_allowed(hint, strategy, applicability):
                    continue
                plans.append(
                    self._build_rewrite_plan(
                        sql_version=sql_version,
                        hint=hint,
                        strategy=strategy,
                        applicability=applicability,
                    )
                )
        return sorted(plans, key=self._plan_sort_key, reverse=True)

    def _fallback_strategies(self, report: BottleneckReport) -> list[RetrievedStrategy]:
        hint_names = {hint.strategy for hint in report.rewrite_hints}
        return build_operator_strategies(hint_names)

    def _align_hint_strategy(self, hint: RewriteHint, strategy: RetrievedStrategy) -> bool:
        explicit_hint_strategies = set(strategy.hint_strategies or [])
        if explicit_hint_strategies:
            return hint.strategy in explicit_hint_strategies
        return bool(_strategy_families(strategy) & _HINT_TO_RULE_FAMILIES.get(hint.strategy, set()))

    def _check_applicability(
        self,
        *,
        sql_version: SQLVersion,
        blueprint: VerifiedContextBlueprint,
        hint: RewriteHint,
        strategy: RetrievedStrategy,
    ) -> ApplicabilityResult:
        if strategy.rule_id == "builtin_filter_dimension_before_top1":
            return self._check_filter_dimension_before_top1(sql_version.sql)
        if strategy.rule_id == "builtin_grouped_max_top1_before_join":
            return self._check_grouped_max_top1_before_join(
                sql_version.sql,
                blueprint,
                strategy,
                _physical_schema_context(
                    db_id=blueprint.db_id,
                    dbms="sqlite",
                    blueprint=blueprint,
                    cost_snapshot={},
                ),
            )
        if strategy.rule_id == "builtin_date_extraction_to_range":
            return self._check_date_extraction_to_range(sql_version.sql)
        if strategy.rule_id == "builtin_like_prefix_to_range":
            return self._check_like_prefix_to_range(sql_version.sql)
        if strategy.rule_id == "builtin_redundant_distinct_elimination":
            return self._check_redundant_distinct_elimination(
                sql_version.sql,
                _physical_schema_context(
                    db_id=blueprint.db_id,
                    dbms="sqlite",
                    blueprint=blueprint,
                    cost_snapshot={},
                ),
            )
        if strategy.rule_id == "builtin_redundant_count_distinct_elimination":
            return self._check_redundant_count_distinct_elimination(
                sql_version.sql,
                _physical_schema_context(
                    db_id=blueprint.db_id,
                    dbms="sqlite",
                    blueprint=blueprint,
                    cost_snapshot={},
                ),
            )
        if strategy.rule_id == "builtin_argmax_aggregate_to_topk":
            return self._check_argmax_aggregate_to_topk(
                sql_version.sql,
                blueprint,
                strategy,
            )
        if strategy.rule_id == "builtin_distinct_join_to_semijoin":
            return self._check_distinct_join_to_semijoin(sql_version.sql)
        if strategy.rule_id == "builtin_distinct_extrema_to_grouped_having":
            return self._check_distinct_extrema_to_grouped_having(
                sql_version.sql,
                blueprint,
                strategy,
            )
        if strategy.rule_id == "builtin_distinct_top1_to_grouped_extrema":
            return self._check_distinct_top1_to_grouped_extrema(
                sql_version.sql,
                blueprint,
                strategy,
            )
        if strategy.rule_id == "builtin_scalar_extrema_anchor_then_lookup_tail":
            return self._check_scalar_extrema_anchor_then_lookup_tail(
                sql_version.sql,
                _physical_schema_context(
                    db_id=blueprint.db_id,
                    dbms="sqlite",
                    blueprint=blueprint,
                    cost_snapshot={},
                ),
            )
        if strategy.rule_id == "builtin_repeated_rescan_to_conditional_agg":
            return self._check_repeated_rescan_to_conditional_agg(sql_version.sql)
        if strategy.rule_id == "builtin_redundant_bridge_join_elimination":
            return self._check_redundant_bridge_join_elimination(sql_version.sql, blueprint)
        if strategy.rule_id == "builtin_same_key_bridge_join_elimination":
            return self._check_same_key_bridge_join_elimination(sql_version.sql)
        if strategy.rule_id == "builtin_unused_fk_join_elimination":
            return self._check_unused_fk_join_elimination(
                sql_version.sql,
                _physical_schema_context(
                    db_id=blueprint.db_id,
                    dbms="sqlite",
                    blueprint=blueprint,
                    cost_snapshot={},
                ),
            )
        if strategy.rule_id == "builtin_unused_fk_join_chain_elimination":
            return self._check_unused_fk_join_chain_elimination(
                sql_version.sql,
                _physical_schema_context(
                    db_id=blueprint.db_id,
                    dbms="sqlite",
                    blueprint=blueprint,
                    cost_snapshot={},
                ),
            )
        if strategy.rule_id == "builtin_dimension_key_first_then_fact_probe":
            return self._check_dimension_key_first_then_fact_probe(
                sql_version.sql,
                _physical_schema_context(
                    db_id=blueprint.db_id,
                    dbms="sqlite",
                    blueprint=blueprint,
                    cost_snapshot={},
                ),
            )
        if strategy.rule_id == "builtin_reanchor_join_driver":
            return self._check_reanchor_join_driver(
                sql_version.sql,
                _physical_schema_context(
                    db_id=blueprint.db_id,
                    dbms="sqlite",
                    blueprint=blueprint,
                    cost_snapshot={},
                ),
            )
        if strategy.rule_id == "builtin_top1_anchor_then_lookup_tail":
            return self._check_top1_anchor_then_lookup_tail(
                sql_version.sql,
                _physical_schema_context(
                    db_id=blueprint.db_id,
                    dbms="sqlite",
                    blueprint=blueprint,
                    cost_snapshot={},
                ),
            )
        if strategy.rule_id == "builtin_prefer_summary_table_when_grain_matches":
            return self._check_prefer_summary_table_when_grain_matches(sql_version.sql)
        if strategy.rule_id == "builtin_symmetric_union_arm_pruning":
            return self._check_symmetric_union_arm_pruning(sql_version.sql)
        if hint.strategy == "reduce_select_columns":
            return self._check_reduce_select_columns(sql_version.sql, blueprint)
        if hint.strategy == "rewrite_scalar_maxmin_subquery":
            return self._check_scalar_maxmin(sql_version.sql, strategy)
        if hint.strategy == "align_order_by_with_index":
            return self._check_topk_before_join(
                sql_version.sql,
                strategy,
                _physical_schema_context(
                    db_id=blueprint.db_id,
                    dbms="sqlite",
                    blueprint=blueprint,
                    cost_snapshot={},
                ),
            )
        if hint.strategy == "eliminate_redundant_self_join":
            return self._check_redundant_self_join_lookup(sql_version.sql, strategy)
        # No deterministic MVP checker — delegate to LLM rewriter
        if hint.strategy in ("no_rewrite", "add_index_candidate"):
            return ApplicabilityResult(
                matched=False,
                confidence=0.0,
                matched_conditions=[],
                failed_conditions=[f"No MVP checker for {hint.strategy}."],
                semantic_risks=[],
                required_fragments={},
            )
        if _is_hist_template_strategy(strategy):
            return self._check_hist_template_compatibility(sql_version.sql, strategy)
        return ApplicabilityResult(
            matched=True,
            confidence=0.65,
            matched_conditions=[f"Delegating {hint.strategy} to LLM rewriter (no deterministic checker)."],
            failed_conditions=[],
            semantic_risks=["LLM-based rewrite requires validator confirmation."],
            required_fragments={},
        )

    def _check_reduce_select_columns(
        self,
        sql: str,
        blueprint: VerifiedContextBlueprint,
    ) -> ApplicabilityResult:
        failed: list[str] = []
        matched: list[str] = []
        if _has_top_level_select_star(sql):
            matched.append("SQL contains top-level SELECT *.")
        else:
            failed.append("SQL does not contain top-level SELECT *.")
        projection_columns = self._projection_columns(sql, blueprint)
        if projection_columns:
            matched.append("Blueprint has candidate projection columns.")
        else:
            failed.append("Blueprint has no candidate projection columns.")
        if len(projection_columns) > self.max_projection_columns:
            failed.append(
                f"Projection has {len(projection_columns)} columns, above MVP limit "
                f"{self.max_projection_columns}."
            )
        return ApplicabilityResult(
            matched=not failed,
            confidence=0.85 if not failed else 0.0,
            matched_conditions=matched,
            failed_conditions=failed,
            semantic_risks=[],
            required_fragments={"projection_columns": projection_columns},
        )

    def _check_scalar_maxmin(self, sql: str, strategy: RetrievedStrategy) -> ApplicabilityResult:
        failed: list[str] = []
        matched: list[str] = []
        risks = ["Tie rows may make ORDER BY ... LIMIT 1 non-equivalent."]
        shape = scalar_extrema_ladder_shape(sql)
        if shape:
            matched.append("SQL contains a nested scalar MIN/MAX ladder on one join graph.")
            matched.append("The scalar extrema can be replaced with one ORDER BY ... LIMIT 1.")
        elif _has_scalar_maxmin_subquery(sql):
            matched.append("SQL contains a scalar MAX/MIN subquery.")
        else:
            failed.append("SQL does not contain a scalar MAX/MIN subquery.")
        if strategy.confidence < 0.75:
            risks.append("Strategy confidence is below high-risk MVP threshold.")
        return ApplicabilityResult(
            matched=bool(matched) and not failed,
            confidence=(0.84 if shape else 0.58) if matched and not failed else 0.0,
            matched_conditions=matched,
            failed_conditions=failed,
            semantic_risks=risks,
            required_fragments={"operator_match": shape} if shape else {},
        )

    def _check_topk_before_join(
        self,
        sql: str,
        strategy: RetrievedStrategy,
        physical_context: dict,
    ) -> ApplicabilityResult:
        failed: list[str] = []
        matched: list[str] = []
        risks = [
            "Downstream joins may duplicate the top-k base row after early LIMIT pushdown."
        ]
        shape = topk_before_join_shape(sql)
        if not shape:
            failed.append("SQL does not match the top-k-before-join shape.")
        else:
            matched.append("SQL has a base-table ORDER BY/LIMIT shape with downstream joins.")
            if shape.join_count != 1:
                failed.append("SQL must use exactly one downstream inner join for this built-in rewrite.")
            else:
                matched.append("SQL joins downstream tables after the scan-driving base table.")
                joined_table, joined_join_column = _single_join_joined_side_key(sql)
                if joined_table is None or joined_join_column is None:
                    failed.append("Joined-side key could not be resolved for uniqueness proof.")
                elif not _table_has_single_column_uniqueness(physical_context, joined_table, joined_join_column):
                    failed.append("Joined-side join key is not provably unique, so early top-k pushdown may change multiplicity.")
                else:
                    matched.append("Schema metadata proves the joined-side join key is unique.")
        return ApplicabilityResult(
            matched=bool(matched) and not failed,
            confidence=0.82 if matched and not failed else 0.0,
            matched_conditions=matched,
            failed_conditions=failed,
            semantic_risks=risks,
            required_fragments={"operator_match": shape} if shape else {},
        )

    def _check_grouped_max_top1_before_join(
        self,
        sql: str,
        blueprint: VerifiedContextBlueprint,
        strategy: RetrievedStrategy,
        physical_context: dict,
    ) -> ApplicabilityResult:
        failed: list[str] = []
        matched: list[str] = []
        risks = [
            "Only safe when grouped aggregate state can be combined after early pre-aggregation without changing tie or multiplicity semantics."
        ]
        shape = grouped_max_top1_before_join_shape(sql)
        if not shape:
            failed.append("SQL does not match the grouped aggregate top-1 shape.")
        else:
            matched.append("SQL is a grouped aggregate ORDER BY ... LIMIT 1 query.")
            if shape.join_count == 0:
                matched.append("Single-table grouped top-1 can be normalized through an aggregate subquery.")
            elif shape.join_count == 1:
                matched.append("Single-join grouped top-1 can be reduced by pre-aggregating before the final top-1.")
                aggregate_tables = {
                    str(column.table)
                    for column in shape.aggregate_expression.find_all(exp.Column)
                    if column.table is not None
                }
                group_is_base_join_key = (
                    len(shape.group_expressions) == 1
                    and isinstance(shape.group_expressions[0], exp.Column)
                    and shape.base_join_key is not None
                    and shape.group_expressions[0].table in {shape.base_alias, shape.base_table}
                    and shape.group_expressions[0].name == shape.base_join_key
                )
                join_only_legacy_path = (
                    bool(shape.join_alias and shape.join_table)
                    and aggregate_tables <= {shape.join_alias, shape.join_table}
                    and group_is_base_join_key
                )
                if not shape.join_table or not shape.join_key:
                    failed.append("Joined-side key could not be resolved for the grouped top-1 rewrite.")
                elif join_only_legacy_path:
                    matched.append("Join-side aggregate can be pre-computed by join key before reattaching filtered base keys.")
                elif not _table_has_single_column_uniqueness(physical_context, shape.join_table, shape.join_key):
                    failed.append("Joined-side join key is not provably unique, so regrouping after pre-aggregation may change multiplicity.")
                else:
                    matched.append("Schema metadata proves the joined-side join key is unique.")
            elif shape.join_count == 2:
                matched.append("Two-hop grouped top-1 is eligible only for a guarded linear fact-to-dimension chain rewrite.")
                if len(shape.join_hops) != 2:
                    failed.append("SQL must expose exactly two linear join hops for the guarded two-hop rewrite.")
                else:
                    first_hop, second_hop = shape.join_hops
                    group_tables = {
                        str(column.table)
                        for expression in shape.group_expressions
                        for column in expression.find_all(exp.Column)
                        if column.table is not None
                    }
                    projection_tables = {
                        str(column.table)
                        for column in shape.projection_expression.find_all(exp.Column)
                        if column.table is not None
                    }
                    aggregate_tables = {
                        str(column.table)
                        for column in shape.aggregate_expression.find_all(exp.Column)
                        if column.table is not None
                    }
                    final_tables = {second_hop.right_alias, second_hop.right_table}
                    non_final_tables = {
                        shape.base_alias,
                        shape.base_table,
                        first_hop.right_alias,
                        first_hop.right_table,
                    }
                    if group_tables and not group_tables <= final_tables:
                        failed.append("Guardrail: two-hop rewrite only handles final grouping expressions that come entirely from the last dimension table.")
                    elif projection_tables and not projection_tables <= final_tables:
                        failed.append("Guardrail: two-hop rewrite only handles final projections that come entirely from the last dimension table.")
                    elif aggregate_tables and not aggregate_tables <= non_final_tables:
                        failed.append("Guardrail: two-hop rewrite only handles aggregates computed from the base table and middle table.")
                    elif not _table_has_single_column_uniqueness(physical_context, first_hop.right_table, first_hop.right_column):
                        failed.append("Middle-table join key is not provably unique, so the first hop may fan out fact rows.")
                    elif not _table_has_single_column_uniqueness(physical_context, second_hop.right_table, second_hop.right_column):
                        failed.append("Final dimension join key is not provably unique, so regrouping by displayed dimension columns may change multiplicity.")
                    else:
                        matched.append("Schema metadata proves both joined-side keys are unique along the two-hop chain.")
            else:
                failed.append("SQL must use at most two inner joins for this built-in rewrite.")
            if not failed:
                candidate_sql = self._rewrite_grouped_max_top1_before_join(sql, shape)
                preflight_failure = self._preflight_performance_guardrail_failure(
                    strategy=strategy,
                    source_sql=sql,
                    candidate_sql=candidate_sql,
                    db_id=blueprint.db_id,
                )
                if preflight_failure is not None:
                    failed.append(preflight_failure)
        return ApplicabilityResult(
            matched=bool(matched) and not failed,
            confidence=0.9 if matched and not failed else 0.0,
            matched_conditions=matched,
            failed_conditions=failed,
            semantic_risks=risks,
            required_fragments={"operator_match": shape} if shape else {},
        )

    def _check_filter_dimension_before_top1(self, sql: str) -> ApplicabilityResult:
        failed: list[str] = []
        matched: list[str] = []
        risks = [
            "Only safe when the pushed predicate filters the joined dimension table without changing join multiplicity."
        ]
        shape = filter_dimension_before_top1_shape(sql)
        if not shape:
            failed.append("SQL does not match the dimension-filter-before-top1 shape.")
        else:
            matched.append("SQL selects from a joined dimension table while ordering by a fact-table top-1 key.")
            matched.append("Dimension-side filter can be isolated before the fact join.")
        return ApplicabilityResult(
            matched=bool(matched) and not failed,
            confidence=0.93 if matched and not failed else 0.0,
            matched_conditions=matched,
            failed_conditions=failed,
            semantic_risks=risks,
            required_fragments={"operator_match": shape} if shape else {},
        )

    def _check_redundant_self_join_lookup(
        self,
        sql: str,
        strategy: RetrievedStrategy,
    ) -> ApplicabilityResult:
        failed: list[str] = []
        matched: list[str] = []
        if _has_same_table_literal_lookup_join(sql):
            matched.append("SQL contains a same-table alias lookup join with a literal filter.")
        else:
            failed.append("SQL does not contain the targeted same-table alias lookup join pattern.")

        source_join_count, target_join_count = _rewrite_template_join_counts(
            strategy.rewrite_template
        )
        sql_join_count = _sql_join_count(sql)
        if source_join_count is None or target_join_count is None:
            failed.append("Strategy rewrite template has no analyzable join-count reduction.")
        else:
            if source_join_count != sql_join_count:
                matched.append(
                    "Strategy source template does not exactly match SQL join count; "
                    "allowing exploratory applicability."
                )
                risks = ["Template join count differs from SQL join count; treat as exploratory."]
            else:
                risks = []
                matched.append("Strategy source template join count matches the SQL structure.")
            if source_join_count != sql_join_count:
                pass
            else:
                matched.append("Strategy source template join count matches the SQL structure.")
            if target_join_count >= source_join_count:
                failed.append("Strategy does not reduce join count.")
            else:
                matched.append("Strategy reduces join count after removing the lookup alias.")

        if "where text_column = literal" in strategy.rewrite_template.lower():
            matched.append("Strategy template preserves the filtered lookup shape.")
        else:
            failed.append("Strategy template does not preserve the filtered lookup shape.")

        semantic_risks = ["LLM-based rewrite requires validator confirmation."]
        if source_join_count is not None and target_join_count is not None and source_join_count != sql_join_count:
            semantic_risks.append(
                f"Template join count {source_join_count} differs from SQL join count {sql_join_count}."
            )

        return ApplicabilityResult(
            matched=bool(matched) and not failed,
            confidence=0.9 if not failed and source_join_count == sql_join_count else 0.68 if not failed else 0.0,
            matched_conditions=matched,
            failed_conditions=failed,
            semantic_risks=semantic_risks,
            required_fragments={},
        )

    def _check_date_extraction_to_range(self, sql: str) -> ApplicabilityResult:
        shape = _date_extraction_to_range_shape(sql)
        failed: list[str] = []
        matched: list[str] = []
        if not shape:
            failed.append("SQL does not match a supported date-extraction equality shape.")
        else:
            matched.append("SQL filters a raw date column through STRFTIME or SUBSTR extraction.")
            matched.append("A half-open range can replace the extraction predicate.")
        return ApplicabilityResult(
            matched=bool(matched) and not failed,
            confidence=0.95 if matched and not failed else 0.0,
            matched_conditions=matched,
            failed_conditions=failed,
            semantic_risks=["Only safe for ISO-like date values with comparable ordering semantics."],
            required_fragments=shape or {},
        )

    def _check_redundant_distinct_elimination(
        self,
        sql: str,
        physical_context: dict,
    ) -> ApplicabilityResult:
        shape = _redundant_distinct_elimination_shape(
            sql,
            require_unique_index=True,
            physical_context=physical_context,
        )
        failed: list[str] = []
        matched: list[str] = []
        if not shape:
            failed.append("SQL does not match the supported top-level redundant DISTINCT shape.")
        else:
            if shape.get("scope") == "single_join":
                matched.append("SQL uses top-level SELECT DISTINCT over a single inner join.")
                matched.append("Projected columns keep one side unique while the joined side is many-to-one on the join key.")
            else:
                matched.append("SQL uses top-level SELECT DISTINCT over plain columns from one base table.")
                matched.append("Schema indexes prove the projected columns already contain a unique key.")
        return ApplicabilityResult(
            matched=bool(matched) and not failed,
            confidence=0.86 if matched and not failed else 0.0,
            matched_conditions=matched,
            failed_conditions=failed,
            semantic_risks=["Only safe when projected columns already contain a unique key."],
            required_fragments=shape or {},
        )

    def _check_redundant_count_distinct_elimination(
        self,
        sql: str,
        physical_context: dict,
    ) -> ApplicabilityResult:
        shape = _redundant_count_distinct_elimination_shape(sql, physical_context)
        failed: list[str] = []
        matched: list[str] = []
        if not shape:
            failed.append("SQL does not match the supported top-level redundant COUNT DISTINCT shape.")
        else:
            matched.append("SQL uses top-level COUNT(DISTINCT column) over one base table.")
            matched.append("Schema metadata proves the counted column is unique.")
            if shape.get("can_use_count_star"):
                matched.append("The counted unique column is non-null, so COUNT(*) preserves semantics.")
            else:
                matched.append("The counted unique column may be nullable, so COUNT(column) preserves semantics.")
        return ApplicabilityResult(
            matched=bool(matched) and not failed,
            confidence=0.85 if matched and not failed else 0.0,
            matched_conditions=matched,
            failed_conditions=failed,
            semantic_risks=[
                "Only safe when uniqueness is established from schema indexes and NULL-counting semantics are preserved."
            ],
            required_fragments=shape or {},
        )

    def _check_like_prefix_to_range(self, sql: str) -> ApplicabilityResult:
        shape = _like_prefix_to_range_shape(sql)
        failed: list[str] = []
        matched: list[str] = []
        if not shape:
            failed.append("SQL does not match a conservative LIKE prefix pattern eligible for range rewrite.")
        else:
            matched.append("SQL has a single trailing-percent LIKE predicate on one column.")
            matched.append("Prefix is conservative enough for lexical half-open range rewrite.")
        return ApplicabilityResult(
            matched=bool(matched) and not failed,
            confidence=0.88 if matched and not failed else 0.0,
            matched_conditions=matched,
            failed_conditions=failed,
            semantic_risks=["Only safe under compatible collation and prefix-order semantics."],
            required_fragments=shape or {},
        )

    def _check_argmax_aggregate_to_topk(
        self,
        sql: str,
        blueprint: VerifiedContextBlueprint,
        strategy: RetrievedStrategy,
    ) -> ApplicabilityResult:
        shape = argmax_aggregate_to_topk_shape(sql)
        failed: list[str] = []
        matched: list[str] = []
        if not shape:
            failed.append("SQL does not match the repeated aggregate argmax shape.")
        else:
            matched.append("SQL computes the same grouped aggregate twice to keep only the best group.")
            matched.append("The query can be reduced to one grouped ORDER BY aggregate LIMIT pass.")
            candidate_sql = self._rewrite_argmax_aggregate_to_topk(sql, shape)
            preflight_failure = self._preflight_performance_guardrail_failure(
                strategy=strategy,
                source_sql=sql,
                candidate_sql=candidate_sql,
                db_id=blueprint.db_id,
            )
            if preflight_failure is not None:
                failed.append(preflight_failure)
        return ApplicabilityResult(
            matched=bool(matched) and not failed,
            confidence=0.9 if matched and not failed else 0.0,
            matched_conditions=matched,
            failed_conditions=failed,
            semantic_risks=["Ties may require validator confirmation."],
            required_fragments={"operator_match": shape} if shape else {},
        )

    def _check_distinct_join_to_semijoin(self, sql: str) -> ApplicabilityResult:
        shape = distinct_join_to_semijoin_shape(sql)
        failed: list[str] = []
        matched: list[str] = []
        if not shape:
            failed.append("SQL does not match DISTINCT-over-fanout semi-join simplification shape.")
        else:
            matched.append("SELECT DISTINCT projects only outer-table columns.")
            matched.append("Joined table can be converted into an existence filter.")
        return ApplicabilityResult(
            matched=bool(matched) and not failed,
            confidence=0.91 if matched and not failed else 0.0,
            matched_conditions=matched,
            failed_conditions=failed,
            semantic_risks=["Predicate split between outer and inner side requires validator confirmation."],
            required_fragments={"operator_match": shape} if shape else {},
        )

    def _check_distinct_extrema_to_grouped_having(
        self,
        sql: str,
        blueprint: VerifiedContextBlueprint,
        strategy: RetrievedStrategy,
    ) -> ApplicabilityResult:
        shape = distinct_extrema_to_grouped_having_shape(sql)
        failed: list[str] = []
        matched: list[str] = []
        if not shape:
            failed.append("SQL does not match the guarded DISTINCT extrema-filter shape.")
        else:
            matched.append("SQL uses SELECT DISTINCT with one scalar extrema equality in the WHERE clause.")
            matched.append("The duplicate-sensitive semantics can be expressed as grouping on the projected values plus HAVING over the same metric.")
            candidate_sql = self._rewrite_distinct_extrema_to_grouped_having(sql, shape)
            preflight_failure = self._preflight_performance_guardrail_failure(
                strategy=strategy,
                source_sql=sql,
                candidate_sql=candidate_sql,
                db_id=blueprint.db_id,
            )
            if preflight_failure is not None:
                failed.append(preflight_failure)
        return ApplicabilityResult(
            matched=bool(matched) and not failed,
            confidence=0.89 if matched and not failed else 0.0,
            matched_conditions=matched,
            failed_conditions=failed,
            semantic_risks=["Ties may require validator confirmation when multiple projected groups share the same extrema."],
            required_fragments={"operator_match": shape} if shape else {},
        )

    def _preflight_candidate_performance(
        self,
        *,
        source_sql: str,
        candidate_sql: str,
        db_id: str,
    ) -> dict[str, Any] | None:
        validator = ValidatorAgent()
        source_version = SQLVersion(
            version_id="preflight-source",
            parent_id=None,
            sql=source_sql,
            source_agent=self.name,
            rewrite_rule_ids=[],
            explanation="preflight source",
            created_at="preflight",
        )
        candidate_version = SQLVersion(
            version_id="preflight-candidate",
            parent_id="preflight-source",
            sql=candidate_sql,
            source_agent=self.name,
            rewrite_rule_ids=[],
            explanation="preflight candidate",
            created_at="preflight",
        )
        source_metrics = validator.validate_syntax(source_version, db_id, "sqlite")
        candidate_metrics = validator.validate_syntax(candidate_version, db_id, "sqlite")
        if not source_metrics.executable or not candidate_metrics.executable:
            return None
        return validator.measure_performance_delta(source_metrics, candidate_metrics)

    def _preflight_performance_guardrail_failure(
        self,
        *,
        strategy: RetrievedStrategy,
        source_sql: str,
        candidate_sql: str | None,
        db_id: str,
    ) -> str | None:
        if not candidate_sql:
            return None
        metadata = None
        if strategy.source_type == "operator":
            metadata = get_operator_strategy_metadata(strategy.rule_id)
        preflight_policy = (
            metadata.preflight_policy
            if metadata is not None and metadata.preflight_policy is not None
            else strategy.preflight_policy
        )
        if preflight_policy is None:
            return None
        failure_message = (
            metadata.preflight_failure_message
            if metadata is not None and metadata.preflight_failure_message is not None
            else strategy.preflight_failure_message
            or "Performance guardrail: candidate shows no measured improvement in preflight."
        )
        preflight = self._preflight_candidate_performance(
            source_sql=source_sql,
            candidate_sql=candidate_sql,
            db_id=db_id,
        )
        if preflight is None:
            return None
        if not self._preflight_policy_satisfied(preflight_policy, preflight):
            return failure_message
        return None

    def _preflight_policy_satisfied(
        self,
        preflight_policy: str,
        preflight: dict[str, Any],
    ) -> bool:
        if preflight_policy == "must_improve_any_metric":
            return bool(preflight.get("performance_better", False))
        if preflight_policy == "must_reduce_scan_rows":
            return bool(preflight.get("scan_rows_better_without_latency_regression", False))
        return False

    def _check_distinct_top1_to_grouped_extrema(
        self,
        sql: str,
        blueprint: VerifiedContextBlueprint,
        strategy: RetrievedStrategy,
    ) -> ApplicabilityResult:
        shape = distinct_top1_to_grouped_extrema_shape(sql)
        failed: list[str] = []
        matched: list[str] = []
        if not shape:
            failed.append("SQL does not match the supported DISTINCT top-1 shape.")
        else:
            matched.append("SQL uses SELECT DISTINCT with LIMIT 1 over a single inner join.")
            matched.append("Projection column and ordering metric come from opposite sides of the join.")
            matched.append("DISTINCT top-1 can be canonicalized as grouped extrema over the projected value.")
            candidate_sql = self._rewrite_distinct_top1_to_grouped_extrema(sql, shape)
            preflight_failure = self._preflight_performance_guardrail_failure(
                strategy=strategy,
                source_sql=sql,
                candidate_sql=candidate_sql,
                db_id=blueprint.db_id,
            )
            if preflight_failure is not None:
                failed.append(preflight_failure)
        return ApplicabilityResult(
            matched=bool(matched) and not failed,
            confidence=0.9 if matched and not failed else 0.0,
            matched_conditions=matched,
            failed_conditions=failed,
            semantic_risks=["Ties may require validator confirmation when multiple projected values share the same extrema."],
            required_fragments={"operator_match": shape} if shape else {},
        )

    def _check_scalar_extrema_anchor_then_lookup_tail(
        self,
        sql: str,
        physical_context: dict,
    ) -> ApplicabilityResult:
        shape = scalar_extrema_anchor_then_lookup_tail_shape(sql)
        failed: list[str] = []
        matched: list[str] = []
        if not shape:
            failed.append("SQL does not match the guarded scalar-extrema anchor-tail shape.")
        else:
            matched.append("SQL filters a predecessor metric by a scalar extrema subquery on a linear join chain.")
            matched.append("The final projection comes entirely from a lookup tail that can be probed after resolving one upstream top-1 key.")
            if not _table_has_single_column_uniqueness(physical_context, shape.tail_table, shape.tail_key):
                failed.append("Final lookup tail key is not provably unique, so probing it after scalar-extrema normalization may fan out the chosen row.")
            else:
                matched.append("Schema metadata proves the final lookup tail key is unique.")
        return ApplicabilityResult(
            matched=bool(matched) and not failed,
            confidence=0.89 if matched and not failed else 0.0,
            matched_conditions=matched,
            failed_conditions=failed,
            semantic_risks=[
                "Only safe when the scalar extrema subquery ranges over the predecessor table alone and its predecessor-side predicates match the outer query."
            ],
            required_fragments={"operator_match": shape} if shape else {},
        )

    def _check_repeated_rescan_to_conditional_agg(self, sql: str) -> ApplicabilityResult:
        shape = repeated_rescan_to_conditional_agg_shape(sql)
        failed: list[str] = []
        matched: list[str] = []
        if not shape:
            failed.append("SQL does not match the repeated grouped rescan-to-conditional-aggregation shape.")
        else:
            matched.append("Two sibling grouped scans read the same fact table with different predicates.")
            matched.append("The outer query only rejoins those siblings on the shared grouping key.")
        return ApplicabilityResult(
            matched=bool(matched) and not failed,
            confidence=0.94 if matched and not failed else 0.0,
            matched_conditions=matched,
            failed_conditions=failed,
            semantic_risks=["Conditional aggregation rewrite must preserve NULL and inner-join key semantics."],
            required_fragments={"operator_match": shape} if shape else {},
        )

    def _check_redundant_bridge_join_elimination(
        self,
        sql: str,
        blueprint: VerifiedContextBlueprint,
    ) -> ApplicabilityResult:
        shape = _redundant_bridge_join_elimination_shape(sql, blueprint)
        failed: list[str] = []
        matched: list[str] = []
        if not shape:
            failed.append("SQL does not match the unused bridge-join elimination shape.")
        else:
            matched.append("Middle bridge table contributes no projected or filtered columns.")
            matched.append("Blueprint provides a direct join edge between the remaining endpoint tables.")
        return ApplicabilityResult(
            matched=bool(matched) and not failed,
            confidence=0.92 if matched and not failed else 0.0,
            matched_conditions=matched,
            failed_conditions=failed,
            semantic_risks=["Direct endpoint join must preserve the original answer grain."],
            required_fragments=shape or {},
        )

    def _check_same_key_bridge_join_elimination(self, sql: str) -> ApplicabilityResult:
        shape = _same_key_bridge_join_elimination_shape(sql)
        failed: list[str] = []
        matched: list[str] = []
        if not shape:
            failed.append("SQL does not match the same-key bridge join elimination shape.")
        else:
            matched.append("SQL joins through a bridge table using the same bridge key column on both sides.")
            matched.append("The bridge table contributes no projected columns or non-join predicates.")
        return ApplicabilityResult(
            matched=bool(matched) and not failed,
            confidence=0.95 if matched and not failed else 0.0,
            matched_conditions=matched,
            failed_conditions=failed,
            semantic_risks=["Only safe when the bridge table is a pure key relay."],
            required_fragments={"operator_match": shape} if shape else {},
        )

    def _check_unused_fk_join_elimination(
        self,
        sql: str,
        physical_context: dict,
    ) -> ApplicabilityResult:
        shape = _unused_fk_join_elimination_shape(sql, physical_context)
        failed: list[str] = []
        matched: list[str] = []
        if not shape:
            failed.append("SQL does not match the unused foreign-key join elimination shape.")
        else:
            matched.append("SQL has exactly one inner join whose joined table is otherwise unused.")
            matched.append("A foreign key from the preserved table to a unique key on the joined table guarantees existence.")
        return ApplicabilityResult(
            matched=bool(matched) and not failed,
            confidence=0.96 if matched and not failed else 0.0,
            matched_conditions=matched,
            failed_conditions=failed,
            semantic_risks=["Only safe when the foreign key and unique referenced key prove the join is pure existence checking."],
            required_fragments={"operator_match": shape} if shape else {},
        )

    def _check_unused_fk_join_chain_elimination(
        self,
        sql: str,
        physical_context: dict,
    ) -> ApplicabilityResult:
        shape = _unused_fk_join_chain_elimination_shape(sql, physical_context)
        failed: list[str] = []
        matched: list[str] = []
        if not shape:
            failed.append("SQL does not match the unused foreign-key join-chain elimination shape.")
        else:
            matched.append("SQL has a linear inner-join chain whose joined tables are otherwise unused.")
            matched.append("Each hop is guaranteed by a foreign key to a unique referenced key, so the whole chain is pure existence checking.")
        return ApplicabilityResult(
            matched=bool(matched) and not failed,
            confidence=0.97 if matched and not failed else 0.0,
            matched_conditions=matched,
            failed_conditions=failed,
            semantic_risks=["Only safe when every join in the chain is a pure foreign-key existence check."],
            required_fragments={"operator_match": shape} if shape else {},
        )

    def _check_dimension_key_first_then_fact_probe(self, sql: str, physical_context: dict) -> ApplicabilityResult:
        shape = dimension_key_first_then_fact_probe_shape(sql)
        failed: list[str] = []
        matched: list[str] = []
        if not shape:
            failed.append("SQL does not match the dimension-key-first fact-probe shape.")
        else:
            matched.append("SQL orders a selective dimension table before touching the fact table.")
            matched.append("The outer query only projects fact-side columns for the resolved key.")
            if not _table_has_single_column_uniqueness(physical_context, shape.fact_table, shape.fact_key):
                failed.append("Fact-side probe key is not provably unique, so key probing may expand one top-1 row into multiple fact rows.")
            else:
                matched.append("Schema metadata proves the fact-side probe key is unique.")
        return ApplicabilityResult(
            matched=bool(matched) and not failed,
            confidence=0.93 if matched and not failed else 0.0,
            matched_conditions=matched,
            failed_conditions=failed,
            semantic_risks=["Only safe when probing the fact table by the resolved dimension key preserves row semantics."],
            required_fragments={"operator_match": shape} if shape else {},
        )

    def _check_reanchor_join_driver(self, sql: str, physical_context: dict) -> ApplicabilityResult:
        shape = reanchor_join_driver_shape(sql)
        failed: list[str] = []
        matched: list[str] = []
        if not shape:
            failed.append("SQL does not match the re-anchor-join-driver shape.")
        else:
            matched.append("SQL orders or filters on one driver table but projects only fact-side columns.")
            matched.append("A bridge key can be eliminated after resolving the driver key first.")
            if not _table_has_single_column_uniqueness(physical_context, shape.fact_table, shape.fact_key):
                failed.append("Fact-side probe key is not provably unique, so re-anchoring may expand or collapse the original top-1 row set.")
            else:
                matched.append("Schema metadata proves the fact-side probe key is unique.")
        return ApplicabilityResult(
            matched=bool(matched) and not failed,
            confidence=0.95 if matched and not failed else 0.0,
            matched_conditions=matched,
            failed_conditions=failed,
            semantic_risks=[
                "Only safe when the bridge contributes no required output/filter columns and the resolved key preserves top-1 semantics."
            ],
            required_fragments={"operator_match": shape} if shape else {},
        )

    def _check_top1_anchor_then_lookup_tail(
        self,
        sql: str,
        physical_context: dict,
    ) -> ApplicabilityResult:
        shape = top1_anchor_then_lookup_tail_shape(sql)
        failed: list[str] = []
        matched: list[str] = []
        if not shape:
            failed.append("SQL does not match the top1-anchor-then-lookup-tail shape.")
        else:
            matched.append("SQL is a linear multi-join top-1 that projects only the final lookup tail.")
            matched.append("Tail-side predicates are absent, so the top-1 anchor can be resolved upstream.")
            if not _table_has_single_column_uniqueness(physical_context, shape.tail_table, shape.tail_key):
                failed.append("Final lookup tail key is not provably unique, so probing it after top-1 may fan out the chosen row.")
            else:
                matched.append("Schema metadata proves the final lookup tail key is unique.")
        return ApplicabilityResult(
            matched=bool(matched) and not failed,
            confidence=0.9 if matched and not failed else 0.0,
            matched_conditions=matched,
            failed_conditions=failed,
            semantic_risks=[
                "Only safe when the final tail is a uniqueness-backed lookup and all duplicate-sensitive semantics stay inside the upstream top-1 subquery."
            ],
            required_fragments={"operator_match": shape} if shape else {},
        )

    def _check_prefer_summary_table_when_grain_matches(self, sql: str) -> ApplicabilityResult:
        shape = prefer_summary_table_when_grain_matches_shape(sql)
        failed: list[str] = []
        matched: list[str] = []
        if not shape:
            failed.append("SQL does not match any guarded summary/detail substitution shape.")
        else:
            matched.append(
                "SQL only uses the detail table for a metric carried by an explicit summary/detail substitution."
            )
            matched.append("The join graph can be preserved while swapping the detail table for the summary table.")
        return ApplicabilityResult(
            matched=bool(matched) and not failed,
            confidence=0.88 if matched and not failed else 0.0,
            matched_conditions=matched,
            failed_conditions=failed,
            semantic_risks=["Only safe for explicitly registered summary/detail metric substitutions."],
            required_fragments={"operator_match": shape} if shape else {},
        )

    def _check_symmetric_union_arm_pruning(self, sql: str) -> ApplicabilityResult:
        shape = symmetric_union_arm_pruning_shape(sql)
        failed: list[str] = []
        matched: list[str] = []
        if not shape:
            failed.append("SQL does not match the guarded symmetric-edge duplication shape.")
        else:
            matched.append("SQL duplicates the same connected-edge lookup through swapped endpoint arms.")
            matched.append("A single canonical endpoint orientation can answer the query under the guarded schema rule.")
        return ApplicabilityResult(
            matched=bool(matched) and not failed,
            confidence=0.83 if matched and not failed else 0.0,
            matched_conditions=matched,
            failed_conditions=failed,
            semantic_risks=["Only safe for the guarded connected-edge canonical orientation rule."],
            required_fragments={"operator_match": shape} if shape else {},
        )

    def _check_hist_template_compatibility(
        self,
        sql: str,
        strategy: RetrievedStrategy,
    ) -> ApplicabilityResult:
        template_shape = _template_source_shape(strategy.rewrite_template)
        sql_shape = _sql_shape(sql)
        matched: list[str] = []
        failed: list[str] = []
        risks = ["LLM-based rewrite requires validator confirmation."]

        if template_shape["join_count"] <= sql_shape["join_count"]:
            matched.append("Template source join count does not exceed SQL join count.")
        else:
            failed.append(
                f"Template source join count {template_shape['join_count']} exceeds SQL join count {sql_shape['join_count']}."
            )

        if template_shape["has_where"] and not sql_shape["has_where"]:
            failed.append("Template requires a WHERE clause but SQL has none.")
        elif not template_shape["has_where"] or sql_shape["has_where"]:
            matched.append("WHERE-clause shape is compatible.")

        if template_shape["has_literal_filter"] and not sql_shape["has_literal_filter"]:
            failed.append("Template requires a literal filter but SQL does not expose one.")
        elif not template_shape["has_literal_filter"] or sql_shape["has_literal_filter"]:
            matched.append("Literal-filter shape is compatible.")

        if template_shape["select_expression_count"] > sql_shape["select_expression_count"]:
            failed.append(
                "Template projects more source expressions than the SQL exposes."
            )
        else:
            matched.append("Projection width is compatible with the template source shape.")

        for flag, label in (
            ("has_cast", "CAST expression"),
            ("has_division", "division/rate expression"),
            ("has_group_by", "GROUP BY"),
            ("has_distinct", "DISTINCT"),
            ("has_in_subquery", "IN subquery"),
            ("has_scalar_subquery", "scalar subquery"),
        ):
            if template_shape[flag] and not sql_shape[flag]:
                failed.append(f"Template requires {label} shape that SQL does not contain.")

        if template_shape["has_order_by"] and sql_shape["has_order_by"]:
            matched.append("ORDER BY shape is compatible.")
        if template_shape["has_limit"] and sql_shape["has_limit"]:
            matched.append("LIMIT shape is compatible.")

        confidence = _hist_template_shape_score(template_shape, sql_shape)
        if confidence < 0.45:
            failed.append("Template source shape is too weakly aligned with the SQL.")
        elif confidence < 0.7:
            risks.append("Template source shape only partially aligns with the SQL.")
            matched.append("Template source shape is exploratory but potentially usable.")
        else:
            matched.append("Template source shape is strongly aligned with the SQL.")

        return ApplicabilityResult(
            matched=bool(matched) and not failed,
            confidence=confidence if not failed else 0.0,
            matched_conditions=matched,
            failed_conditions=failed,
            semantic_risks=risks,
            required_fragments={
                "template_shape_score": confidence,
                "template_join_count": template_shape["join_count"],
                "sql_join_count": sql_shape["join_count"],
            },
        )

    def _projection_columns(
        self,
        sql: str,
        blueprint: VerifiedContextBlueprint,
    ) -> list[ColumnRef]:
        tables, _alias_for_table = _extract_table_refs(sql)
        candidates = [
            col for col in blueprint.selected_columns if not tables or col.table_name in tables
        ] or list(blueprint.selected_columns)
        result: list[ColumnRef] = []
        seen: set[tuple[str, str]] = set()
        for col in candidates:
            key = (col.table_name, col.column_name)
            if key not in seen:
                seen.add(key)
                result.append(col)
        return result

    def _is_plan_allowed(
        self,
        hint: RewriteHint,
        strategy: RetrievedStrategy,
        applicability: ApplicabilityResult,
    ) -> bool:
        if not applicability.matched:
            return False
        if not strategy.rule_id or not strategy.rewrite_template:
            return False
        if hint.risk == "high":
            return strategy.confidence >= 0.6
        return True

    def _build_rewrite_plan(
        self,
        *,
        sql_version: SQLVersion,
        hint: RewriteHint,
        strategy: RetrievedStrategy,
        applicability: ApplicabilityResult,
    ) -> RewritePlan:
        source_type = _strategy_source_type(strategy)
        common_kwargs = dict(
            plan_type="operator_deterministic" if source_type == "operator" else "generic_strategy",
            source_type=source_type,
            rule_id=strategy.rule_id,
            rule_name=strategy.rule_name,
            hint_strategy=hint.strategy,
            source_sql_version_id=sql_version.version_id,
            target_fragment=hint.target_fragment,
            rewrite_template=strategy.rewrite_template,
            risk=hint.risk,
            expected_effect=hint.expected_effect,
            requires_validation=hint.requires_validation,
            dbms_notes=hint.dbms_notes,
            matched_conditions=tuple(applicability.matched_conditions),
            semantic_risks=tuple(applicability.semantic_risks),
            required_fragments=dict(applicability.required_fragments),
            strategy_confidence=strategy.confidence,
            applicability_confidence=applicability.confidence,
            retrieval_rerank_score=_generic_retrieval_rerank_score(
                sql_version.sql,
                hint,
                strategy,
                applicability,
            ),
            hist_template=_is_hist_template_strategy(strategy),
        )
        if source_type == "operator":
            return OperatorDeterministicRewritePlan(
                operator_match=applicability.required_fragments.get("operator_match"),
                **common_kwargs,
            )
        return GenericStrategyRewritePlan(
            llm_strategy=not str(strategy.rule_id).startswith("builtin_"),
            **common_kwargs,
        )

    def _score_plan(self, plan: RewritePlan) -> float:
        risk_penalty = {"low": 0.0, "medium": 0.2, "high": 0.45}
        strategy_bonus = {
            "eliminate_redundant_self_join": 0.2,
            "push_down_filter": 0.14,
            "simplify_join_graph": 0.12,
            "pre_aggregate_before_join": 0.1,
            "rewrite_scalar_maxmin_subquery": 0.08,
            "rewrite_or_to_union": 0.08,
            "avoid_function_on_column": 0.06,
            "align_order_by_with_index": 0.05,
            "add_null_guard_for_sort_key": 0.03,
            "reduce_select_columns": 0.02,
        }
        source_type = str(plan.get("source_type") or "unknown").lower()
        rerank_weight = 0.45 if source_type != "operator" else 0.0
        operator_source_bonus = 0.14 if source_type == "operator" else 0.0
        operator_match_bonus = 0.08 if source_type == "operator" and _operator_match_from_plan(plan) is not None else 0.0
        return (
            float(plan.get("strategy_confidence") or 0.0) * 0.5
            + float(plan.get("applicability_confidence") or 0.0) * 0.35
            + float(plan.get("retrieval_rerank_score") or 0.0) * rerank_weight
            + 0.4
            + operator_source_bonus
            + operator_match_bonus
            + strategy_bonus.get(str(plan.get("hint_strategy")), 0.0)
            - risk_penalty.get(str(plan.get("risk")), 0.3)
            - 0.1 * len(plan.get("semantic_risks") or [])
            - (0.08 if plan.get("hist_template") else 0.0)
        )

    def _plan_sort_key(self, plan: RewritePlan) -> tuple[int, float]:
        source_type = str(plan.get("source_type") or "unknown").lower()
        source_priority = {
            "operator": 3,
            "expert": 2,
            "hist": 1,
        }.get(source_type, 0)
        return (source_priority, self._score_plan(plan))

    # ------------------------------------------------------------------
    # Rewrite and guardrails
    # ------------------------------------------------------------------

    def rewrite(
        self,
        sql_version: SQLVersion,
        rewrite_plan: RewritePlan | NoOpRewritePlan,
        blueprint: VerifiedContextBlueprint,
        reflection_context: dict | None = None,
    ) -> tuple[SQLVersion | None, str | None]:
        if not rewrite_plan:
            raise ValueError("rewrite requires a non-empty rewrite_plan.")
        deterministic_sql = self._deterministic_rewrite(sql_version.sql, rewrite_plan, blueprint)
        llm_invoked = False
        candidate_sql = ""
        if deterministic_sql:
            candidate_sql = deterministic_sql.strip()
        else:
            llm_result = self._llm_rewrite(
                sql_version,
                rewrite_plan,
                blueprint,
                reflection_context=reflection_context,
            )
            if self.llm_rewriter is not None:
                llm_invoked = True
            candidate_sql = (llm_result or "").strip()
        if _is_no_optimization_response(candidate_sql):
            return None, "LLM rewriter reported no safe optimization space."
        if not candidate_sql:
            if llm_invoked:
                return None, "LLM rewriter returned empty SQL."
            return None, f"No rewrite implementation for {rewrite_plan.get('hint_strategy')}."
        rule_id = str(rewrite_plan["rule_id"])
        return (
            SQLVersion(
                version_id=uuid.uuid4().hex[:12],
                parent_id=sql_version.version_id,
                sql=candidate_sql,
                source_agent=self.name,
                rewrite_rule_ids=[rule_id],
                explanation=(
                    f"Applied {rule_id}: {rewrite_plan.get('expected_effect')}. "
                    "Candidate requires Validator checks."
                ),
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ),
            None,
        )

    def free_explore_rewrite(
        self,
        sql_version: SQLVersion,
        context: dict,
    ) -> tuple[SQLVersion | None, str | None]:
        if self.llm_free_explorer is None:
            raise ValueError("llm_free_explorer is required for low-confidence RAG exploration.")
        candidate_sql = (self.llm_free_explorer(context) or "").strip()
        if not candidate_sql:
            return None, "LLM free exploration returned empty SQL."
        if _is_no_optimization_response(candidate_sql):
            return None, "LLM free exploration reported no safe optimization space."
        direction = str(context.get("current_exploration_direction") or "free exploration")
        return (
            SQLVersion(
                version_id=uuid.uuid4().hex[:12],
                parent_id=sql_version.version_id,
                sql=candidate_sql,
                source_agent=self.name,
                rewrite_rule_ids=["llm_free_exploration"],
                explanation=(
                    f"Applied LLM free exploration because RAG confidence was below "
                    f"{self.rag_confidence_threshold:.2f}: {direction}. "
                    "Candidate requires Validator checks."
                ),
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ),
            None,
        )

    def build_free_exploration_context(
        self,
        *,
        request: AgentRequest,
        sql_version: SQLVersion,
        blueprint: VerifiedContextBlueprint,
        bottleneck_report: BottleneckReport,
        operator_strategies: list[RetrievedStrategy],
        expert_strategies: list[RetrievedStrategy],
        hist_strategies: list[RetrievedStrategy],
        failed_directions: list[dict],
    ) -> dict:
        allowed_scope = _allowed_schema_scope(blueprint)
        allowed_tables = ", ".join(allowed_scope["tables"]) or "<none>"
        allowed_columns = ", ".join(allowed_scope["columns"]) or "<none>"
        physical_context = _physical_schema_context(
            db_id=request.task.db_id,
            dbms=request.task.dbms,
            blueprint=blueprint,
            cost_snapshot=bottleneck_report.cost_snapshot,
        )
        available_indexes = _available_index_summary(physical_context)
        available_indexes_text = (
            "\n".join(f"- {index}" for index in available_indexes)
            if available_indexes
            else "- <none>"
        )
        analyser_improvement_notes = _compact_analyser_improvement_notes(
            bottleneck_report
        )
        evidence = request.task.evidence or "<none>"
        analyser_notes_text = (
            "\n".join(f"- {note}" for note in analyser_improvement_notes)
            if analyser_improvement_notes
            else "- No concrete analyser rewrite suggestion was provided."
        )
        bottleneck_summary_lines = _compact_bottleneck_summary_lines(bottleneck_report)
        bottleneck_summary_text = (
            "\n".join(f"- {line}" for line in bottleneck_summary_lines)
            if bottleneck_summary_lines
            else "- <none>"
        )
        operator_strategy_lines = _compact_strategy_lines(operator_strategies)
        operator_strategy_text = (
            "\n".join(f"- {line}" for line in operator_strategy_lines)
            if operator_strategy_lines
            else "- <none>"
        )
        expert_strategy_lines = _compact_strategy_lines(expert_strategies)
        expert_strategy_text = (
            "\n".join(f"- {line}" for line in expert_strategy_lines)
            if expert_strategy_lines
            else "- <none>"
        )
        hist_strategy_lines = _compact_strategy_lines(hist_strategies)
        hist_strategy_text = (
            "\n".join(f"- {line}" for line in hist_strategy_lines)
            if hist_strategy_lines
            else "- <none>"
        )
        failed_direction_lines = _compact_failed_direction_lines(failed_directions)
        failed_direction_text = (
            "\n".join(f"- {line}" for line in failed_direction_lines)
            if failed_direction_lines
            else "- <none>"
        )
        prompt_profile = self.free_exploration_prompt_profile
        return {
            "mode": "llm_free_exploration",
            "prompt_profile": prompt_profile,
            "question": request.task.question,
            "evidence": request.task.evidence,
            "analyser_improvement_notes": analyser_improvement_notes,
            "db_id": request.task.db_id,
            "dbms": request.task.dbms,
            "source_sql_version_id": sql_version.version_id,
            "source_sql": sql_version.sql,
            "blueprint": _blueprint_summary(blueprint),
            "allowed_schema_scope": allowed_scope,
            "physical_schema_context": physical_context,
            "available_indexes": available_indexes,
            "bottleneck_report": _bottleneck_summary(bottleneck_report),
            "unoptimized_fragments": _unoptimized_fragments(bottleneck_report),
            "rag_confidence_threshold": self.rag_confidence_threshold,
            "deterministic_operator_opportunities": [
                _strategy_summary(strategy) for strategy in operator_strategies
            ],
            "expert_rewrite_priors": [
                _strategy_summary(strategy) for strategy in expert_strategies
            ],
            "historical_similar_rewrites": [
                _strategy_summary(strategy) for strategy in hist_strategies
            ],
            "low_confidence_rag_strategies": [
                _strategy_summary(strategy) for strategy in hist_strategies
            ],
            "failed_free_exploration_directions": failed_directions,
            "current_exploration_direction": "Explore an optimization not covered by high-confidence RAG rules.",
            "prompt": _build_free_exploration_prompt(
                profile=prompt_profile,
                question=request.task.question,
                evidence=evidence,
                source_sql=sql_version.sql,
                available_indexes_text=available_indexes_text,
                analyser_notes_text=analyser_notes_text,
                bottleneck_summary_text=bottleneck_summary_text,
                operator_strategy_text=operator_strategy_text,
                expert_strategy_text=expert_strategy_text,
                hist_strategy_text=hist_strategy_text,
                failed_direction_text=failed_direction_text,
                allowed_tables=allowed_tables,
                allowed_columns=allowed_columns,
                physical_context=physical_context,
            ),
        }

    def _deterministic_rewrite(
        self,
        sql: str,
        rewrite_plan: RewritePlan,
        blueprint: VerifiedContextBlueprint,
    ) -> str | None:
        if rewrite_plan.get("hint_strategy") == "reduce_select_columns":
            return self._rewrite_select_star(sql, rewrite_plan, blueprint)
        if rewrite_plan.get("rule_id") == "builtin_date_extraction_to_range":
            return self._rewrite_date_extraction_to_range(sql)
        if rewrite_plan.get("rule_id") == "builtin_like_prefix_to_range":
            return self._rewrite_like_prefix_to_range(sql)
        if rewrite_plan.get("rule_id") == "builtin_redundant_distinct_elimination":
            return self._rewrite_redundant_distinct_elimination(
                sql,
                rewrite_plan.get("required_fragments") or {},
            )
        if rewrite_plan.get("rule_id") == "builtin_redundant_count_distinct_elimination":
            return self._rewrite_redundant_count_distinct_elimination(
                sql,
                rewrite_plan.get("required_fragments") or {},
            )
        if rewrite_plan.get("rule_id") == "builtin_filter_dimension_before_top1":
            return self._rewrite_filter_dimension_before_top1(
                sql,
                _operator_match_from_plan(rewrite_plan),
            )
        if rewrite_plan.get("rule_id") == "builtin_distinct_join_to_semijoin":
            return self._rewrite_distinct_join_to_semijoin(
                sql,
                _operator_match_from_plan(rewrite_plan),
            )
        if rewrite_plan.get("rule_id") == "builtin_distinct_extrema_to_grouped_having":
            return self._rewrite_distinct_extrema_to_grouped_having(
                sql,
                _operator_match_from_plan(rewrite_plan),
            )
        if rewrite_plan.get("rule_id") == "builtin_distinct_top1_to_grouped_extrema":
            return self._rewrite_distinct_top1_to_grouped_extrema(
                sql,
                _operator_match_from_plan(rewrite_plan),
            )
        if rewrite_plan.get("rule_id") == "builtin_scalar_extrema_anchor_then_lookup_tail":
            return self._rewrite_scalar_extrema_anchor_then_lookup_tail(
                sql,
                _operator_match_from_plan(rewrite_plan),
            )
        if rewrite_plan.get("rule_id") == "builtin_argmax_aggregate_to_topk":
            return self._rewrite_argmax_aggregate_to_topk(
                sql,
                _operator_match_from_plan(rewrite_plan),
            )
        if rewrite_plan.get("rule_id") == "builtin_repeated_rescan_to_conditional_agg":
            return self._rewrite_repeated_rescan_to_conditional_agg(
                sql,
                _operator_match_from_plan(rewrite_plan),
            )
        if rewrite_plan.get("rule_id") == "builtin_redundant_bridge_join_elimination":
            return self._rewrite_redundant_bridge_join_elimination(sql, blueprint)
        if rewrite_plan.get("rule_id") == "builtin_same_key_bridge_join_elimination":
            return self._rewrite_same_key_bridge_join_elimination(
                sql,
                _operator_match_from_plan(rewrite_plan),
            )
        if rewrite_plan.get("rule_id") == "builtin_unused_fk_join_elimination":
            return self._rewrite_unused_fk_join_elimination(
                sql,
                _operator_match_from_plan(rewrite_plan),
            )
        if rewrite_plan.get("rule_id") == "builtin_unused_fk_join_chain_elimination":
            return self._rewrite_unused_fk_join_chain_elimination(
                sql,
                _operator_match_from_plan(rewrite_plan),
            )
        if rewrite_plan.get("rule_id") == "builtin_dimension_key_first_then_fact_probe":
            return self._rewrite_dimension_key_first_then_fact_probe(
                sql,
                _operator_match_from_plan(rewrite_plan),
            )
        if rewrite_plan.get("rule_id") == "builtin_reanchor_join_driver":
            return self._rewrite_reanchor_join_driver(
                sql,
                _operator_match_from_plan(rewrite_plan),
            )
        if rewrite_plan.get("rule_id") == "builtin_top1_anchor_then_lookup_tail":
            return self._rewrite_top1_anchor_then_lookup_tail(
                sql,
                _operator_match_from_plan(rewrite_plan),
            )
        if rewrite_plan.get("rule_id") == "builtin_prefer_summary_table_when_grain_matches":
            return self._rewrite_prefer_summary_table_when_grain_matches(
                sql,
                _operator_match_from_plan(rewrite_plan),
            )
        if rewrite_plan.get("rule_id") == "builtin_symmetric_union_arm_pruning":
            return self._rewrite_symmetric_union_arm_pruning(
                sql,
                _operator_match_from_plan(rewrite_plan),
            )
        if rewrite_plan.get("rule_id") == "builtin_grouped_max_top1_before_join":
            return self._rewrite_grouped_max_top1_before_join(
                sql,
                _operator_match_from_plan(rewrite_plan),
            )
        if rewrite_plan.get("rule_id") == "builtin_topk_before_join":
            return self._rewrite_topk_before_join(
                sql,
                _operator_match_from_plan(rewrite_plan),
            )
        if rewrite_plan.get("rule_id") == "builtin_scalar_maxmin_to_order_limit":
            return self._rewrite_scalar_maxmin_to_topk(
                sql,
                _operator_match_from_plan(rewrite_plan),
            )
        return None

    def _llm_rewrite(
        self,
        sql_version: SQLVersion,
        rewrite_plan: RewritePlan,
        blueprint: VerifiedContextBlueprint,
        reflection_context: dict | None = None,
    ) -> str | None:
        if self.llm_rewriter is None:
            return None
        signature = inspect.signature(self.llm_rewriter)
        if "reflection_context" in signature.parameters:
            return self.llm_rewriter(
                sql_version,
                rewrite_plan,
                blueprint,
                reflection_context=reflection_context,
            )
        return self.llm_rewriter(sql_version, rewrite_plan, blueprint)

    def _rewrite_select_star(
        self,
        sql: str,
        rewrite_plan: RewritePlan,
        blueprint: VerifiedContextBlueprint,
    ) -> str | None:
        try:
            ast = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception:
            return None
        select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
        if select is None:
            return None
        projection_columns = rewrite_plan.get("required_fragments", {}).get("projection_columns")
        if not projection_columns:
            projection_columns = self._projection_columns(sql, blueprint)
        tables, alias_for_table = _extract_table_refs(sql)
        qualify = len(tables) > 1
        projections: list[exp.Expression] = []
        seen: set[tuple[str, str]] = set()
        for col in projection_columns:
            if not isinstance(col, ColumnRef):
                continue
            key = (col.table_name, col.column_name)
            if key in seen:
                continue
            seen.add(key)
            table = alias_for_table.get(col.table_name, col.table_name) if qualify else None
            projections.append(exp.column(col.column_name, table=table))
        if not projections:
            return None
        changed = False
        expressions: list[exp.Expression] = []
        for expression in select.expressions:
            if _is_star_expression(expression):
                expressions.extend(projections)
                changed = True
            else:
                expressions.append(expression)
        if not changed:
            return None
        select.set("expressions", expressions)
        return ast.sql(dialect="sqlite")

    def _rewrite_topk_before_join(self, sql: str, match: Any | None = None) -> str | None:
        try:
            ast = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception:
            return None
        if not isinstance(ast, exp.Select):
            ast = ast.find(exp.Select)
        if ast is None:
            return None
        shape = match if match is not None else topk_before_join_shape(sql)
        if shape is None:
            return None
        from_expr = ast.args.get("from_")
        if from_expr is None or from_expr.this is None or not isinstance(from_expr.this, exp.Table):
            return None

        base_table = from_expr.this
        base_alias = base_table.alias_or_name
        subquery_alias = "topk_base"

        projected_names = set(shape.base_columns)
        order_col = shape.order_by_column
        where_clause = shape.where_clause
        if order_col:
            projected_names.add(order_col)
        if not projected_names:
            return None

        inner_projections = [
            exp.alias_(exp.column(column_name, table=base_alias), column_name)
            for column_name in sorted(projected_names)
        ]
        inner_select = exp.select(*inner_projections).from_(base_table.copy())
        if where_clause is not None:
            inner_select.set("where", where_clause.copy())
        if ast.args.get("order") is not None:
            inner_select.set("order", ast.args["order"].copy())
        if ast.args.get("limit") is not None:
            inner_select.set("limit", ast.args["limit"].copy())

        outer_ast = ast.copy()
        outer_ast.set("order", None)
        outer_ast.set("limit", None)
        outer_ast.set("where", None)
        for column in outer_ast.find_all(exp.Column):
            if column.table in {base_alias, base_table.name}:
                column.set("table", exp.to_identifier(subquery_alias))
        outer_ast.set(
            "from_",
            exp.From(
                this=exp.Subquery(
                    this=inner_select,
                    alias=exp.TableAlias(this=exp.to_identifier(subquery_alias)),
                )
            ),
        )
        return outer_ast.sql(dialect="sqlite")

    def _rewrite_grouped_max_top1_before_join(self, sql: str, match: Any | None = None) -> str | None:
        shape = match if match is not None else grouped_max_top1_before_join_shape(sql)
        if shape is None:
            return None

        base_alias = shape.base_alias
        base_table = shape.base_table
        join_alias = shape.join_alias
        join_table = shape.join_table
        base_join_key = shape.base_join_key
        join_key = shape.join_key
        where_clause = shape.where_clause
        limit_expr = shape.limit_expr
        ordered_expr = shape.ordered_expr
        aggregate_expr = shape.aggregate_expression
        aggregate_function = shape.aggregate_function.upper()

        def _copy_table(name: str, alias: str) -> exp.Table:
            table = exp.to_table(name)
            if alias != name:
                table.set("alias", exp.TableAlias(this=exp.to_identifier(alias)))
            return table

        def _remap_expression(expression: exp.Expression, mapping: dict[str, str]) -> exp.Expression:
            rewritten = expression.copy()
            for column in rewritten.find_all(exp.Column):
                table_name = str(column.table) if column.table is not None else None
                if table_name in mapping:
                    column.set("table", exp.to_identifier(mapping[table_name]))
            return rewritten

        def _combine_predicates(predicates: list[exp.Expression]) -> exp.Expression | None:
            if not predicates:
                return None
            combined = predicates[0].copy()
            for predicate in predicates[1:]:
                combined = exp.and_(combined, predicate.copy())
            return combined

        def _strip_alias(expression: exp.Expression) -> exp.Expression:
            return expression.this.copy() if isinstance(expression, exp.Alias) else expression.copy()

        def _aggregate_tables(expression: exp.Expression) -> set[str]:
            return {
                str(column.table)
                for column in expression.find_all(exp.Column)
                if column.table is not None
            }

        def _split_predicates_by_scope(
            scopes: list[tuple[str, set[str]]],
        ) -> dict[str, list[exp.Expression]] | None:
            assignments = {name: [] for name, _ in scopes}
            if where_clause is None:
                return assignments
            for predicate in _flatten_and_conditions(where_clause.this):
                referenced = {
                    str(column.table)
                    for column in predicate.find_all(exp.Column)
                    if column.table is not None
                }
                if not referenced:
                    assignments[scopes[0][0]].append(predicate.copy())
                    continue
                matching_names = [
                    name
                    for name, allowed_tables in scopes
                    if referenced <= allowed_tables
                ]
                if len(matching_names) != 1:
                    return None
                assignments[matching_names[0]].append(predicate.copy())
            return assignments

        def _combine_aggregate(alias_name: str) -> exp.Expression | None:
            column = exp.column(alias_name, table="fact_agg")
            if aggregate_function in {"COUNT", "SUM"}:
                return exp.Sum(this=column)
            if aggregate_function == "MAX":
                return exp.Max(this=column)
            if aggregate_function == "MIN":
                return exp.Min(this=column)
            return None

        projection_expr = shape.projection_expression.copy()
        group_expressions = tuple(expression.copy() for expression in shape.group_expressions)
        base_tables = {base_alias, base_table}

        if shape.join_count == 0:
            predicate_split = _split_predicates_by_scope([("base", base_tables)])
            if predicate_split is None:
                return None
            base_predicates = predicate_split["base"]
            inner_select = exp.select(
                exp.alias_(projection_expr.copy(), "group_value", quoted=False),
                exp.alias_(aggregate_expr.copy(), "order_value", quoted=False),
            ).from_(_copy_table(base_table, base_alias))
            where_expr = _combine_predicates(base_predicates)
            if where_expr is not None:
                inner_select.set("where", exp.Where(this=where_expr))
            inner_select.set("group", exp.Group(expressions=[expr.copy() for expr in group_expressions]))
            rewritten_order = ordered_expr.copy()
            rewritten_order.set("this", exp.column("order_value"))
            inner_select.set("order", exp.Order(expressions=[rewritten_order]))
            inner_select.set("limit", exp.Limit(expression=limit_expr.copy()))
            outer_select = exp.select(exp.column("group_value", table="grouped_top1"))
            outer_select.set(
                "from_",
                exp.From(
                    this=exp.Subquery(
                        this=inner_select,
                        alias=exp.TableAlias(this=exp.to_identifier("grouped_top1")),
                    )
                ),
            )
            return outer_select.sql(dialect="sqlite")

        if not join_alias or not join_table or not base_join_key or not join_key:
            return None

        join_tables = {join_alias, join_table}
        aggregate_tables = _aggregate_tables(aggregate_expr)
        group_tables = {
            str(column.table)
            for expression in group_expressions
            for column in expression.find_all(exp.Column)
            if column.table is not None
        }

        if shape.join_count == 2:
            if len(shape.join_hops) != 2:
                return None
            first_hop, second_hop = shape.join_hops
            middle_tables = {first_hop.right_alias, first_hop.right_table}
            final_tables = {second_hop.right_alias, second_hop.right_table}
            predicate_split = _split_predicates_by_scope(
                [("base", base_tables), ("middle", middle_tables), ("final", final_tables)]
            )
            if predicate_split is None:
                return None
            base_predicates = predicate_split["base"]
            middle_predicates = predicate_split["middle"]
            final_predicates = predicate_split["final"]
            if aggregate_function not in {"COUNT", "SUM", "MAX", "MIN"}:
                return None
            if group_tables and not (group_tables <= final_tables):
                return None
            projection_tables = {
                str(column.table)
                for column in projection_expr.find_all(exp.Column)
                if column.table is not None
            }
            if projection_tables and not (projection_tables <= final_tables):
                return None
            if aggregate_tables and not (aggregate_tables <= (base_tables | middle_tables)):
                return None

            transport_key_alias = "transport_key"
            order_value_alias = "order_value"
            lookup_alias = "lookup_dim"
            lookup_mapping = {
                second_hop.right_alias: lookup_alias,
                second_hop.right_table: lookup_alias,
            }

            needed_final_columns = {second_hop.right_column}
            for expression in list(group_expressions) + [projection_expr] + final_predicates:
                for column in expression.find_all(exp.Column):
                    if column.table in final_tables:
                        needed_final_columns.add(column.name)

            lookup_source: exp.Expression
            if final_predicates:
                lookup_select = exp.select(
                    *[
                        exp.alias_(
                            exp.column(column_name, table=second_hop.right_alias),
                            column_name,
                            quoted=False,
                        )
                        for column_name in sorted(needed_final_columns)
                    ]
                ).from_(_copy_table(second_hop.right_table, second_hop.right_alias))
                final_where = _combine_predicates(final_predicates)
                if final_where is not None:
                    lookup_select.set("where", exp.Where(this=final_where))
                lookup_source = exp.Subquery(
                    this=lookup_select.distinct(),
                    alias=exp.TableAlias(this=exp.to_identifier(lookup_alias)),
                )
            else:
                lookup_source = _copy_table(second_hop.right_table, lookup_alias)

            inner_fact = exp.select(
                exp.alias_(
                    exp.column(second_hop.left_column, table=first_hop.right_alias),
                    transport_key_alias,
                    quoted=False,
                ),
                exp.alias_(aggregate_expr.copy(), order_value_alias, quoted=False),
            ).from_(_copy_table(base_table, base_alias))
            inner_fact.join(
                _copy_table(first_hop.right_table, first_hop.right_alias),
                on=exp.EQ(
                    this=exp.column(first_hop.left_column, table=base_alias),
                    expression=exp.column(first_hop.right_column, table=first_hop.right_alias),
                ),
                join_type="INNER",
                copy=False,
            )
            if final_predicates:
                inner_fact.join(
                    lookup_source,
                    on=exp.EQ(
                        this=exp.column(second_hop.left_column, table=first_hop.right_alias),
                        expression=exp.column(second_hop.right_column, table=lookup_alias),
                    ),
                    join_type="INNER",
                    copy=False,
                )
            inner_where = _combine_predicates(base_predicates + middle_predicates)
            if inner_where is not None:
                inner_fact.set("where", exp.Where(this=inner_where))
            inner_fact.set(
                "group",
                exp.Group(expressions=[exp.column(second_hop.left_column, table=first_hop.right_alias)]),
            )

            combine_expr = _combine_aggregate(order_value_alias)
            if combine_expr is None:
                return None
            outer_select = exp.select(_remap_expression(projection_expr, lookup_mapping))
            outer_select.set(
                "from_",
                exp.From(
                    this=exp.Subquery(
                        this=inner_fact,
                        alias=exp.TableAlias(this=exp.to_identifier("fact_agg")),
                    )
                ),
            )
            if not final_predicates:
                lookup_source = _copy_table(second_hop.right_table, lookup_alias)
            outer_select.join(
                lookup_source,
                on=exp.EQ(
                    this=exp.column(transport_key_alias, table="fact_agg"),
                    expression=exp.column(second_hop.right_column, table=lookup_alias),
                ),
                join_type="INNER",
                copy=False,
            )
            outer_select.set(
                "group",
                exp.Group(
                    expressions=[
                        _remap_expression(expression, lookup_mapping)
                        for expression in group_expressions
                    ]
                ),
            )
            outer_order = ordered_expr.copy()
            outer_order.set("this", combine_expr)
            outer_select.set("order", exp.Order(expressions=[outer_order]))
            outer_select.set("limit", exp.Limit(expression=limit_expr.copy()))
            return outer_select.sql(dialect="sqlite")

        predicate_split = _split_predicates_by_scope(
            [("base", base_tables), ("join", join_tables)]
        )
        if predicate_split is None:
            return None
        base_predicates = predicate_split["base"]
        join_predicates = predicate_split["join"]

        # Legacy grouped MAX-on-joined-table path, generalized to regroup only on base expressions.
        if aggregate_tables and aggregate_tables <= join_tables and group_tables <= base_tables:
            if len(group_expressions) != 1:
                return None
            group_expression = group_expressions[0]
            if not isinstance(group_expression, exp.Column):
                return None
            if group_expression.table not in {base_alias, base_table} or group_expression.name != base_join_key:
                return None
            filtered_base = exp.select(
                exp.alias_(
                    exp.column(base_join_key, table=base_alias),
                    base_join_key,
                    quoted=False,
                )
            ).distinct().from_(_copy_table(base_table, base_alias))
            base_where = _combine_predicates(base_predicates)
            if base_where is not None:
                filtered_base.set("where", exp.Where(this=base_where))

            aggregated_join = exp.select(
                exp.alias_(
                    exp.column(join_key, table=join_alias),
                    join_key,
                    quoted=False,
                ),
                exp.alias_(aggregate_expr.copy(), "max_order_value", quoted=False),
            ).from_(_copy_table(join_table, join_alias))
            join_where = _combine_predicates(join_predicates)
            if join_where is not None:
                aggregated_join.set("where", exp.Where(this=join_where))
            aggregated_join.set("group", exp.Group(expressions=[exp.column(join_key, table=join_alias)]))

            outer_select = exp.select(_remap_expression(projection_expr, {base_alias: "filtered_base", base_table: "filtered_base"}))
            outer_select.set(
                "from_",
                exp.From(
                    this=exp.Subquery(
                        this=filtered_base,
                        alias=exp.TableAlias(this=exp.to_identifier("filtered_base")),
                    )
                ),
            )
            outer_select.join(
                exp.Subquery(
                    this=aggregated_join,
                    alias=exp.TableAlias(this=exp.to_identifier("agg_join")),
                ),
                on=exp.EQ(
                    this=exp.column(base_join_key, table="filtered_base"),
                    expression=exp.column(join_key, table="agg_join"),
                ),
                join_type="INNER",
                copy=False,
            )
            rewritten_order = ordered_expr.copy()
            rewritten_order.set("this", exp.column("max_order_value", table="agg_join"))
            outer_select.set("order", exp.Order(expressions=[rewritten_order]))
            outer_select.set("limit", exp.Limit(expression=limit_expr.copy()))
            return outer_select.sql(dialect="sqlite")

        if aggregate_function not in {"COUNT", "SUM", "MAX", "MIN"}:
            return None
        if aggregate_tables and not (aggregate_tables <= base_tables):
            return None

        lookup_source: exp.Expression
        lookup_alias = "lookup_dim"
        lookup_group_mapping = {join_alias: lookup_alias, join_table: lookup_alias}
        if join_predicates:
            lookup_select = exp.select(
                exp.alias_(exp.column(join_key, table=join_alias), join_key, quoted=False)
            ).from_(_copy_table(join_table, join_alias))
            for expression in group_expressions:
                if {
                    str(column.table)
                    for column in expression.find_all(exp.Column)
                    if column.table is not None
                } <= join_tables:
                    lookup_select.select(_strip_alias(expression))
            lookup_where = _combine_predicates(join_predicates)
            if lookup_where is not None:
                lookup_select.set("where", exp.Where(this=lookup_where))
            lookup_source = exp.Subquery(
                this=lookup_select.distinct(),
                alias=exp.TableAlias(this=exp.to_identifier(lookup_alias)),
            )
        else:
            lookup_source = _copy_table(join_table, lookup_alias)

        if group_tables and group_tables <= base_tables:
            inner_select = exp.select(
                exp.alias_(projection_expr.copy(), "group_value", quoted=False),
                exp.alias_(aggregate_expr.copy(), "order_value", quoted=False),
            ).from_(_copy_table(base_table, base_alias))
            inner_select.join(
                lookup_source,
                on=exp.EQ(
                    this=exp.column(base_join_key, table=base_alias),
                    expression=exp.column(join_key, table=lookup_alias),
                ),
                join_type="INNER",
                copy=False,
            )
            base_where = _combine_predicates(base_predicates)
            if base_where is not None:
                inner_select.set("where", exp.Where(this=base_where))
            inner_select.set("group", exp.Group(expressions=[expr.copy() for expr in group_expressions]))
            rewritten_order = ordered_expr.copy()
            rewritten_order.set("this", exp.column("order_value"))
            inner_select.set("order", exp.Order(expressions=[rewritten_order]))
            inner_select.set("limit", exp.Limit(expression=limit_expr.copy()))
            outer_select = exp.select(exp.column("group_value", table="grouped_top1"))
            outer_select.set(
                "from_",
                exp.From(
                    this=exp.Subquery(
                        this=inner_select,
                        alias=exp.TableAlias(this=exp.to_identifier("grouped_top1")),
                    )
                ),
            )
            return outer_select.sql(dialect="sqlite")

        if group_tables and not (group_tables <= join_tables):
            return None

        inner_fact = exp.select(
            exp.alias_(
                exp.column(base_join_key, table=base_alias),
                base_join_key,
                quoted=False,
            ),
            exp.alias_(aggregate_expr.copy(), "order_value", quoted=False),
        ).from_(_copy_table(base_table, base_alias))
        if join_predicates:
            inner_fact.join(
                lookup_source,
                on=exp.EQ(
                    this=exp.column(base_join_key, table=base_alias),
                    expression=exp.column(join_key, table=lookup_alias),
                ),
                join_type="INNER",
                copy=False,
            )
        base_where = _combine_predicates(base_predicates)
        if base_where is not None:
            inner_fact.set("where", exp.Where(this=base_where))
        inner_fact.set("group", exp.Group(expressions=[exp.column(base_join_key, table=base_alias)]))

        combine_expr = _combine_aggregate("order_value")
        if combine_expr is None:
            return None
        outer_select = exp.select(_remap_expression(projection_expr, lookup_group_mapping))
        outer_select.set(
            "from_",
            exp.From(
                this=exp.Subquery(
                    this=inner_fact,
                    alias=exp.TableAlias(this=exp.to_identifier("fact_agg")),
                )
            ),
        )
        if not join_predicates:
            lookup_source = _copy_table(join_table, lookup_alias)
        outer_select.join(
            lookup_source,
            on=exp.EQ(
                this=exp.column(base_join_key, table="fact_agg"),
                expression=exp.column(join_key, table=lookup_alias),
            ),
            join_type="INNER",
            copy=False,
        )
        outer_select.set(
            "group",
            exp.Group(expressions=[_remap_expression(expression, lookup_group_mapping) for expression in group_expressions]),
        )
        outer_order = ordered_expr.copy()
        outer_order.set("this", combine_expr)
        outer_select.set("order", exp.Order(expressions=[outer_order]))
        outer_select.set("limit", exp.Limit(expression=limit_expr.copy()))
        return outer_select.sql(dialect="sqlite")

    def _rewrite_filter_dimension_before_top1(self, sql: str, match: Any | None = None) -> str | None:
        shape = match if match is not None else filter_dimension_before_top1_shape(sql)
        if shape is None:
            return None
        try:
            ast = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception:
            return None
        if not isinstance(ast, exp.Select):
            ast = ast.find(exp.Select)
        if ast is None:
            return None

        fact_alias = shape.fact_alias
        dim_alias = shape.dim_alias
        dim_table = shape.dim_table
        dim_columns = shape.dim_columns
        where_clause = shape.where_clause
        new_alias = "filtered_dim"

        inner_projections = [
            exp.alias_(exp.column(column_name, table=dim_alias), column_name, quoted=False)
            for column_name in sorted(dim_columns)
        ]
        inner_select = exp.select(*inner_projections).from_(exp.to_table(dim_table))
        if dim_alias != dim_table:
            inner_select.args["from_"].this.set(
                "alias",
                exp.TableAlias(this=exp.to_identifier(dim_alias)),
            )
        if where_clause is not None:
            inner_select.set("where", where_clause.copy())

        outer_ast = ast.copy()
        outer_ast.set("where", None)
        for column in outer_ast.find_all(exp.Column):
            if column.table in {dim_alias, dim_table}:
                column.set("table", exp.to_identifier(new_alias))

        joins = outer_ast.args.get("joins", [])
        if not joins:
            return None
        outer_join = joins[0]
        outer_join.set(
            "this",
            exp.Subquery(
                this=inner_select,
                alias=exp.TableAlias(this=exp.to_identifier(new_alias)),
            ),
        )
        return outer_ast.sql(dialect="sqlite")

    def _rewrite_date_extraction_to_range(self, sql: str) -> str | None:
        try:
            ast = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception:
            return None
        inferred_table = _single_from_table_name(ast)
        where_clause = ast.find(exp.Where)
        scope_expr = where_clause.this if where_clause is not None else None
        replaced = False

        def _transform(node: exp.Expression) -> exp.Expression:
            nonlocal replaced
            shape = _date_range_replacement_shape(
                node,
                inferred_table=inferred_table,
                scope_expr=scope_expr,
            )
            if shape is None:
                return node
            replaced = True
            return shape["replacement"].copy()

        rewritten = ast.transform(_transform)
        if not replaced:
            return None
        return rewritten.sql(dialect="sqlite")

    def _rewrite_like_prefix_to_range(self, sql: str) -> str | None:
        shape = _like_prefix_to_range_shape(sql)
        if not shape:
            return None
        try:
            ast = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception:
            return None
        target_sql = _normalize_sql(shape["predicate_sql"])
        replacement = _range_predicate_for_bounds(
            column=shape["column"],
            start=shape["range_start"],
            end=shape["range_end"],
        )
        rewritten = ast.transform(
            lambda node: (
                replacement.copy()
                if _normalize_sql(node.sql(dialect="sqlite")) == target_sql
                else node
            )
        )
        return rewritten.sql(dialect="sqlite")

    def _rewrite_redundant_distinct_elimination(
        self,
        sql: str,
        fragments: dict[str, Any] | None = None,
    ) -> str | None:
        shape = _redundant_distinct_elimination_shape(
            sql,
            require_unique_index=False,
            physical_context=None,
            required_fragments=fragments or None,
        )
        if not shape:
            return None
        try:
            ast = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception:
            return None
        select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
        if select is None or select.args.get("distinct") is None:
            return None
        select.set("distinct", None)
        return ast.sql(dialect="sqlite")

    def _rewrite_redundant_count_distinct_elimination(
        self,
        sql: str,
        fragments: dict[str, Any] | None = None,
    ) -> str | None:
        shape = dict(fragments or {})
        counted_column_sql = str(shape.get("counted_column_sql") or "")
        if not counted_column_sql:
            return None
        try:
            ast = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception:
            return None
        select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
        if select is None or len(select.expressions) != 1:
            return None
        outer_expression = select.expressions[0]
        alias_name = outer_expression.alias if isinstance(outer_expression, exp.Alias) else None
        count_expr = outer_expression.this if isinstance(outer_expression, exp.Alias) else outer_expression
        if not isinstance(count_expr, exp.Count) or not isinstance(count_expr.this, exp.Distinct):
            return None
        distinct_expr = count_expr.this
        if len(distinct_expr.expressions) != 1:
            return None
        counted_expr = distinct_expr.expressions[0]
        if not isinstance(counted_expr, exp.Column):
            return None
        if counted_expr.sql(dialect="sqlite") != counted_column_sql:
            return None
        replacement_arg: exp.Expression = (
            exp.Star() if shape.get("can_use_count_star") else counted_expr.copy()
        )
        replacement_count = exp.Count(this=replacement_arg)
        if alias_name:
            select.set("expressions", [exp.alias_(replacement_count, alias_name, quoted=False)])
        else:
            select.set("expressions", [replacement_count])
        return ast.sql(dialect="sqlite")

    def _rewrite_distinct_join_to_semijoin(self, sql: str, match: Any | None = None) -> str | None:
        shape = match if match is not None else distinct_join_to_semijoin_shape(sql)
        if shape is None:
            return None
        try:
            ast = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception:
            return None
        if not isinstance(ast, exp.Select):
            ast = ast.find(exp.Select)
        if ast is None:
            return None

        base_alias = shape.base_alias
        base_predicates = [predicate.copy() for predicate in shape.base_predicates]
        inner_predicates = [predicate.copy() for predicate in shape.inner_predicates]
        correlated_predicates = [predicate.copy() for predicate in shape.correlated_predicates]

        inner_tables = list(shape.inner_tables)
        inner_aliases = list(shape.inner_aliases)
        if not inner_tables or not inner_aliases or len(inner_tables) != len(inner_aliases):
            return None

        subquery = exp.select(exp.Literal.number(1)).from_(exp.to_table(inner_tables[0]))
        if inner_aliases[0] != inner_tables[0]:
            subquery.args["from_"].this.set(
                "alias",
                exp.TableAlias(this=exp.to_identifier(inner_aliases[0])),
            )
        subquery_joins: list[exp.Join] = []
        for idx in range(1, len(inner_tables)):
            join_on = shape.inner_join_ons[idx - 1].copy()
            join_expr = exp.Join(
                this=exp.to_table(inner_tables[idx]),
                kind="INNER",
                on=join_on,
            )
            if inner_aliases[idx] != inner_tables[idx]:
                join_expr.this.set(
                    "alias",
                    exp.TableAlias(this=exp.to_identifier(inner_aliases[idx])),
                )
            subquery_joins.append(join_expr)
        if subquery_joins:
            subquery.set("joins", subquery_joins)

        exists_conditions = inner_predicates + correlated_predicates
        exists_where = _combine_conjuncts(exists_conditions)
        if exists_where is not None:
            subquery.set("where", exp.Where(this=exists_where))

        outer_ast = ast.copy()
        outer_ast.set("from_", exp.From(this=exp.to_table(shape.base_table)))
        if base_alias != shape.base_table:
            outer_ast.args["from_"].this.set(
                "alias",
                exp.TableAlias(this=exp.to_identifier(base_alias)),
            )
        outer_ast.set("joins", [])
        outer_conditions = list(base_predicates)
        outer_conditions.append(exp.Exists(this=exp.Subquery(this=subquery)))
        combined_outer_where = _combine_conjuncts(outer_conditions)
        if combined_outer_where is not None:
            outer_ast.set("where", exp.Where(this=combined_outer_where))
        else:
            outer_ast.set("where", None)
        return outer_ast.sql(dialect="sqlite")

    def _rewrite_distinct_extrema_to_grouped_having(self, sql: str, match: Any | None = None) -> str | None:
        shape = match if match is not None else distinct_extrema_to_grouped_having_shape(sql)
        if shape is None:
            return None
        try:
            ast = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception:
            return None
        if not isinstance(ast, exp.Select):
            ast = ast.find(exp.Select)
        if ast is None:
            return None

        rewritten = ast.copy()
        rewritten.set("distinct", None)
        where_expr = _combine_conjuncts([predicate.copy() for predicate in shape.outer_predicates])
        rewritten.set("where", exp.Where(this=where_expr) if where_expr is not None else None)
        rewritten.set("group", exp.Group(expressions=[projection.copy() for projection in shape.projections]))

        aggregate_expr = (
            exp.Max(this=shape.outer_metric.copy())
            if shape.direction == "DESC"
            else exp.Min(this=shape.outer_metric.copy())
        )
        inner_agg = shape.inner_select.copy()
        inner_agg.set("order", None)
        inner_agg.set("limit", None)
        inner_agg.set(
            "expressions",
            [
                exp.Max(this=shape.inner_metric.copy())
                if shape.direction == "DESC"
                else exp.Min(this=shape.inner_metric.copy())
            ],
        )
        rewritten.set(
            "having",
            exp.Having(
                this=exp.EQ(
                    this=aggregate_expr,
                    expression=exp.Subquery(this=inner_agg),
                )
            ),
        )
        return rewritten.sql(dialect="sqlite")

    def _rewrite_distinct_top1_to_grouped_extrema(self, sql: str, match: Any | None = None) -> str | None:
        shape = match if match is not None else distinct_top1_to_grouped_extrema_shape(sql)
        if shape is None:
            return None
        try:
            ast = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception:
            return None
        if not isinstance(ast, exp.Select):
            ast = ast.find(exp.Select)
        if ast is None:
            return None

        rewritten = ast.copy()
        rewritten.set("distinct", None)
        rewritten.set("group", exp.Group(expressions=[shape.projection.copy()]))
        where_expr = _combine_conjuncts([predicate.copy() for predicate in shape.where_predicates])
        if where_expr is not None:
            rewritten.set("where", exp.Where(this=where_expr))
        else:
            rewritten.set("where", None)
        extrema_expr: exp.Expression
        if shape.direction == "DESC":
            extrema_expr = exp.Max(this=shape.metric.copy())
        else:
            extrema_expr = exp.Min(this=shape.metric.copy())
        rewritten.set(
            "order",
            exp.Order(
                expressions=[
                    exp.Ordered(
                        this=extrema_expr,
                        desc=shape.direction == "DESC",
                    )
                ]
            ),
        )
        rewritten.set("limit", exp.Limit(expression=shape.limit_expr.copy()))
        return rewritten.sql(dialect="sqlite")

    def _rewrite_argmax_aggregate_to_topk(self, sql: str, match: Any | None = None) -> str | None:
        shape = match if match is not None else argmax_aggregate_to_topk_shape(sql)
        if shape is None:
            return None
        try:
            ast = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception:
            return None
        if not isinstance(ast, exp.Select):
            ast = ast.find(exp.Select)
        if ast is None:
            return None

        outer_ast = ast.copy()
        outer_ast.set("having", None)
        aggregate_expression = shape.aggregate_expression.copy()
        direction = "DESC" if shape.best_direction == "DESC" else "ASC"
        outer_ast.set(
            "order",
            exp.Order(
                expressions=[
                    exp.Ordered(
                        this=aggregate_expression,
                        desc=direction == "DESC",
                    )
                ]
            ),
        )
        outer_ast.set("limit", exp.Limit(expression=exp.Literal.number(1)))
        return outer_ast.sql(dialect="sqlite")

    def _rewrite_repeated_rescan_to_conditional_agg(self, sql: str, match: Any | None = None) -> str | None:
        shape = match if match is not None else repeated_rescan_to_conditional_agg_shape(sql)
        if shape is None:
            return None

        fact_table = shape.fact_table
        grouping_key = shape.group_key
        group_alias = shape.group_alias
        scans = shape.scan_specs
        outer_select = shape.outer_select
        combined_alias = "combined_scan"

        combined_projections: list[exp.Expression] = [
            exp.alias_(exp.column(grouping_key), group_alias, quoted=False)
        ]
        having_conditions: list[exp.Expression] = []
        where_predicates: list[exp.Expression] = []

        for index, scan in enumerate(scans, start=1):
            predicate = scan.where
            aggregate_alias = scan.aggregate_alias
            aggregate_expr = scan.aggregate_expr
            condition_counter_name = f"__cond_{index}"
            if predicate is not None:
                where_predicates.append(predicate.copy())
            rewritten_aggregate = _conditionalized_aggregate_expression(
                aggregate_expr,
                predicate.copy() if predicate is not None else None,
            )
            if rewritten_aggregate is None:
                return None
            combined_projections.append(
                exp.alias_(rewritten_aggregate, aggregate_alias, quoted=False)
            )
            if predicate is not None:
                combined_projections.append(
                    exp.alias_(
                        exp.Sum(
                            this=exp.Case(
                                ifs=[exp.If(this=predicate.copy(), true=exp.Literal.number(1))],
                                default=exp.Literal.number(0),
                            )
                        ),
                        condition_counter_name,
                        quoted=False,
                    )
                )
                having_conditions.append(
                    exp.GT(
                        this=exp.column(condition_counter_name, table=combined_alias),
                        expression=exp.Literal.number(0),
                    )
                )

        inner_select = exp.select(*combined_projections).from_(exp.to_table(fact_table))
        combined_where = _combine_conjuncts_or(where_predicates)
        if combined_where is not None:
            inner_select.set("where", exp.Where(this=combined_where))
        inner_select.set("group", exp.Group(expressions=[exp.column(grouping_key)]))

        rewritten_outer = outer_select.copy()
        rewritten_outer.set("with_", None)
        rewritten_outer.set(
            "from_",
            exp.From(
                this=exp.Subquery(
                    this=inner_select,
                    alias=exp.TableAlias(this=exp.to_identifier(combined_alias)),
                )
            ),
        )
        rewritten_outer.set("joins", [])
        rewritten_outer.set(
            "where",
            exp.Where(this=_combine_conjuncts(having_conditions))
            if having_conditions
            else None,
        )

        alias_map = shape.cte_alias_to_output_alias
        for column in rewritten_outer.find_all(exp.Column):
            if column.table in alias_map:
                column.set("table", exp.to_identifier(combined_alias))
        return rewritten_outer.sql(dialect="sqlite")

    def _rewrite_redundant_bridge_join_elimination(
        self,
        sql: str,
        blueprint: VerifiedContextBlueprint,
    ) -> str | None:
        shape = _redundant_bridge_join_elimination_shape(sql, blueprint)
        if not shape:
            return None
        try:
            ast = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception:
            return None
        select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
        if select is None:
            return None
        direct_edge = shape["direct_edge"]
        rewritten = select.copy()
        joins = rewritten.args.get("joins") or []
        if len(joins) < 2:
            return None
        joins.pop(0)
        joins[0].set(
            "on",
            exp.EQ(
                this=exp.column(direct_edge.source_column, table=shape["left_alias"]),
                expression=exp.column(direct_edge.target_column, table=shape["right_alias"]),
            )
            if direct_edge.source_table == shape["left_table"]
            else exp.EQ(
                this=exp.column(direct_edge.target_column, table=shape["left_alias"]),
                expression=exp.column(direct_edge.source_column, table=shape["right_alias"]),
            ),
        )
        rewritten.set("joins", joins)
        bridge_replacements = shape.get("bridge_predicate_replacements") or {}
        if bridge_replacements:
            where_clause = rewritten.args.get("where")
            if where_clause is not None and where_clause.this is not None:
                rewritten_where = where_clause.this.transform(
                    lambda node: (
                        bridge_replacements[_normalize_sql(node.sql(dialect="sqlite"))].copy()
                        if _normalize_sql(node.sql(dialect="sqlite")) in bridge_replacements
                        else node
                    )
                )
                rewritten.set("where", exp.Where(this=rewritten_where))
        return rewritten.sql(dialect="sqlite")

    def _rewrite_same_key_bridge_join_elimination(
        self,
        sql: str,
        match: Any | None = None,
    ) -> str | None:
        shape = match if match is not None else _same_key_bridge_join_elimination_shape(sql)
        if not shape:
            return None
        try:
            ast = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception:
            return None
        select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
        if select is None:
            return None
        rewritten = select.copy()
        joins = rewritten.args.get("joins") or []
        if len(joins) != 2:
            return None
        joins.pop(0)
        joins[0].set(
            "on",
            exp.EQ(
                this=exp.column(shape["left_column"], table=shape["left_alias"]),
                expression=exp.column(shape["right_column"], table=shape["right_alias"]),
            ),
        )
        rewritten.set("joins", joins)
        return rewritten.sql(dialect="sqlite")

    def _rewrite_unused_fk_join_elimination(
        self,
        sql: str,
        match: Any | None = None,
    ) -> str | None:
        shape = match if match is not None else _unused_fk_join_elimination_shape(sql, {})
        if not shape:
            return None
        try:
            ast = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception:
            return None
        select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
        if select is None:
            return None
        rewritten = select.copy()
        joins = rewritten.args.get("joins") or []
        if len(joins) != 1:
            return None
        rewritten.set("joins", [])
        return rewritten.sql(dialect="sqlite")

    def _rewrite_unused_fk_join_chain_elimination(
        self,
        sql: str,
        match: Any | None = None,
    ) -> str | None:
        shape = match if match is not None else _unused_fk_join_chain_elimination_shape(sql, {})
        if not shape:
            return None
        try:
            ast = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception:
            return None
        select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
        if select is None:
            return None
        rewritten = select.copy()
        rewritten.set("joins", [])
        return rewritten.sql(dialect="sqlite")

    def _rewrite_dimension_key_first_then_fact_probe(
        self,
        sql: str,
        match: Any | None = None,
    ) -> str | None:
        shape = match or dimension_key_first_then_fact_probe_shape(sql)
        if not shape:
            return None

        key_subquery = exp.select(
            exp.column(shape.fact_key, table=shape.fact_alias)
        ).from_(exp.to_table(shape.dimension_table))
        if shape.dimension_alias != shape.dimension_table:
            key_subquery.args["from_"].this.set(
                "alias",
                exp.TableAlias(this=exp.to_identifier(shape.dimension_alias)),
            )
        key_subquery.join(
            exp.to_table(shape.fact_table),
            on=exp.EQ(
                this=exp.column(shape.dimension_key, table=shape.dimension_alias),
                expression=exp.column(shape.fact_key, table=shape.fact_alias),
            ),
            join_type="INNER",
            copy=False,
        )
        if shape.fact_alias != shape.fact_table:
            key_subquery.args["joins"][0].this.set(
                "alias",
                exp.TableAlias(this=exp.to_identifier(shape.fact_alias)),
            )
        key_subquery.set("order", shape.order.copy())
        key_subquery.set("limit", shape.limit.copy())

        outer_select = exp.select(
            *[
                _rewrite_table_references(
                    expression.copy(),
                    {shape.fact_alias, shape.fact_table},
                    shape.fact_alias,
                )
                for expression in shape.projections
            ]
        ).from_(exp.to_table(shape.fact_table))
        if shape.fact_alias != shape.fact_table:
            outer_select.args["from_"].this.set(
                "alias",
                exp.TableAlias(this=exp.to_identifier(shape.fact_alias)),
            )
        outer_select.set(
            "where",
            exp.Where(
                this=exp.EQ(
                    this=exp.column(shape.fact_key, table=shape.fact_alias),
                    expression=exp.Subquery(this=key_subquery),
                )
            ),
        )
        return outer_select.sql(dialect="sqlite")

    def _rewrite_reanchor_join_driver(
        self,
        sql: str,
        match: Any | None = None,
    ) -> str | None:
        shape = match or reanchor_join_driver_shape(sql)
        if not shape:
            return None

        driver_subquery = exp.select(
            exp.column(shape.fact_key, table=shape.fact_alias)
        ).from_(exp.to_table(shape.driver_table))
        if shape.driver_alias != shape.driver_table:
            driver_subquery.args["from_"].this.set(
                "alias",
                exp.TableAlias(this=exp.to_identifier(shape.driver_alias)),
            )
        driver_predicates = list(shape.driver_predicates)
        if driver_predicates:
            driver_subquery.set(
                "where",
                exp.Where(this=_combine_conjuncts([predicate.copy() for predicate in driver_predicates])),
            )
        driver_subquery.join(
            exp.to_table(shape.bridge_table),
            on=exp.EQ(
                this=exp.column(shape.driver_key, table=shape.driver_alias),
                expression=exp.column(shape.bridge_key, table=shape.bridge_alias),
            ),
            join_type="INNER",
            copy=False,
        )
        if shape.bridge_alias != shape.bridge_table:
            driver_subquery.args["joins"][0].this.set(
                "alias",
                exp.TableAlias(this=exp.to_identifier(shape.bridge_alias)),
            )
        driver_subquery.join(
            exp.to_table(shape.fact_table),
            on=exp.EQ(
                this=exp.column(shape.bridge_key, table=shape.bridge_alias),
                expression=exp.column(shape.fact_key, table=shape.fact_alias),
            ),
            join_type="INNER",
            copy=False,
        )
        if shape.fact_alias != shape.fact_table:
            driver_subquery.args["joins"][1].this.set(
                "alias",
                exp.TableAlias(this=exp.to_identifier(shape.fact_alias)),
            )
        driver_subquery.set("order", shape.order.copy())
        driver_subquery.set("limit", shape.limit.copy())

        outer_select = exp.select(
            *[
                _rewrite_table_references(
                    expression.copy(),
                    {shape.fact_alias, shape.fact_table},
                    shape.fact_alias,
                )
                for expression in shape.projections
            ]
        ).from_(exp.to_table(shape.fact_table))
        if shape.fact_alias != shape.fact_table:
            outer_select.args["from_"].this.set(
                "alias",
                exp.TableAlias(this=exp.to_identifier(shape.fact_alias)),
            )
        outer_select.set(
            "where",
            exp.Where(
                this=exp.EQ(
                    this=exp.column(shape.fact_key, table=shape.fact_alias),
                    expression=exp.Subquery(this=driver_subquery),
                )
            ),
        )
        return outer_select.sql(dialect="sqlite")

    def _rewrite_top1_anchor_then_lookup_tail(
        self,
        sql: str,
        match: Any | None = None,
    ) -> str | None:
        try:
            ast = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception:
            return None
        select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
        if select is None:
            return None
        shape = match or top1_anchor_then_lookup_tail_shape(sql)
        if shape is None:
            return None
        from_expr = select.args.get("from_")
        joins = list(select.args.get("joins") or [])
        if from_expr is None or not isinstance(from_expr.this, exp.Table) or len(joins) < 2:
            return None

        inner_select = select.copy()
        inner_select.set(
            "expressions",
            [exp.column(shape.predecessor_tail_key, table=shape.predecessor_alias)],
        )
        inner_select.set("joins", [join.copy() for join in joins[:-1]])

        outer_select = exp.select(*[expression.copy() for expression in shape.projections]).from_(
            _copy_table_with_alias(shape.tail_table, shape.tail_alias)
        )
        outer_select.set(
            "where",
            exp.Where(
                this=exp.EQ(
                    this=exp.column(shape.tail_key, table=shape.tail_alias),
                    expression=exp.Subquery(this=inner_select),
                )
            ),
        )
        return outer_select.sql(dialect="sqlite")

    def _rewrite_prefer_summary_table_when_grain_matches(
        self,
        sql: str,
        match: Any | None = None,
    ) -> str | None:
        shape = match or prefer_summary_table_when_grain_matches_shape(sql)
        if not shape:
            return None
        try:
            ast = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception:
            return None
        select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
        if select is None:
            return None

        rewritten = select.copy()
        from_expr = rewritten.args.get("from_")
        if from_expr is None or not isinstance(from_expr.this, exp.Table):
            return None
        summary_table = exp.to_table(shape.summary_table)
        if shape.detail_alias != shape.summary_table:
            summary_table.set(
                "alias",
                exp.TableAlias(this=exp.to_identifier(shape.detail_alias)),
            )
        from_expr.set("this", summary_table)

        rewritten = _rewrite_summary_table_references(
            rewritten,
            source_tables={shape.detail_alias, shape.detail_table},
            target_table=shape.detail_alias,
            column_mapping=shape.column_mapping,
            target_allowed_columns=shape.target_allowed_columns,
        )
        if rewritten is None:
            return None

        where_conditions: list[exp.Expression] = []
        where_clause = rewritten.args.get("where")
        if where_clause is not None and where_clause.this is not None:
            where_conditions.extend(_flatten_and_conditions(where_clause.this))
        where_conditions.append(
            exp.Not(
                this=exp.Is(
                    this=exp.column(shape.summary_metric_column, table=shape.detail_alias),
                    expression=exp.Null(),
                )
            )
        )
        rewritten.set("where", exp.Where(this=_combine_conjuncts(where_conditions)))
        return rewritten.sql(dialect="sqlite")

    def _rewrite_scalar_maxmin_to_topk(
        self,
        sql: str,
        match: Any | None = None,
    ) -> str | None:
        shape = match or scalar_extrema_ladder_shape(sql)
        if not shape:
            return None
        try:
            ast = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception:
            return None
        select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
        if select is None:
            return None

        rewritten = select.copy()
        base_predicates = [predicate.copy() for predicate in shape.base_predicates]
        if base_predicates:
            rewritten.set("where", exp.Where(this=_combine_conjuncts(base_predicates)))
        else:
            rewritten.set("where", None)

        order_expressions = []
        for spec in shape.order_specs:
            ordered = exp.Ordered(
                this=spec.column.copy(),
                desc=spec.direction.upper() == "DESC",
            )
            order_expressions.append(ordered)
        if not order_expressions:
            return None
        rewritten.set("order", exp.Order(expressions=order_expressions))
        rewritten.set("limit", exp.Limit(expression=exp.Literal.number(1)))
        return rewritten.sql(dialect="sqlite")

    def _rewrite_symmetric_union_arm_pruning(
        self,
        sql: str,
        match: Any | None = None,
    ) -> str | None:
        shape = match or symmetric_union_arm_pruning_shape(sql)
        if not shape:
            return None
        if shape.shape_type == "or":
            try:
                ast = sqlglot.parse_one(sql, dialect="sqlite")
            except Exception:
                return None
            if shape.canonical_predicate is None:
                return None
            target_sql = _normalize_sql(shape.target_sql)
            replacement = shape.canonical_predicate.copy()
            rewritten = ast.transform(
                lambda node: (
                    replacement.copy()
                    if _normalize_sql(node.sql(dialect="sqlite")) == target_sql
                    else node
                )
        )
        return rewritten.sql(dialect="sqlite")

    def _rewrite_scalar_extrema_anchor_then_lookup_tail(
        self,
        sql: str,
        match: Any | None = None,
    ) -> str | None:
        try:
            ast = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception:
            return None
        select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
        if select is None:
            return None
        shape = match if match is not None else scalar_extrema_anchor_then_lookup_tail_shape(sql)
        if shape is None:
            return None
        from_expr = select.args.get("from_")
        joins = list(select.args.get("joins") or [])
        if from_expr is None or not isinstance(from_expr.this, exp.Table) or len(joins) < 2:
            return None

        inner_select = exp.select(
            exp.column(shape.predecessor_tail_key, table=shape.predecessor_alias)
        ).from_(from_expr.this.copy())
        inner_select.set("joins", [join.copy() for join in joins[:-1]])
        where_expr = _combine_conjuncts([predicate.copy() for predicate in shape.outer_predicates])
        if where_expr is not None:
            inner_select.set("where", exp.Where(this=where_expr))
        inner_select.set(
            "order",
            exp.Order(
                expressions=[
                    exp.Ordered(
                        this=exp.column(shape.metric_column, table=shape.predecessor_alias),
                        desc=shape.direction == "DESC",
                    )
                ]
            ),
        )
        inner_select.set("limit", exp.Limit(expression=exp.Literal.number(1)))

        outer_select = exp.select(*[expression.copy() for expression in shape.projections]).from_(
            _copy_table_with_alias(shape.tail_table, shape.tail_alias)
        )
        outer_select.set(
            "where",
            exp.Where(
                this=exp.EQ(
                    this=exp.column(shape.tail_key, table=shape.tail_alias),
                    expression=exp.Subquery(this=inner_select),
                )
            ),
        )
        return outer_select.sql(dialect="sqlite")
        if shape.shape_type == "union":
            if shape.canonical_select is None:
                return None
            return shape.canonical_select.sql(dialect="sqlite")
        return None

    def guardrail_check(
        self,
        source_sql: SQLVersion,
        candidate_sql: SQLVersion,
        blueprint: VerifiedContextBlueprint,
    ) -> list[str]:
        errors: list[str] = []
        if not candidate_sql.sql.strip():
            errors.append("Candidate SQL is empty.")
        if _normalize_sql(candidate_sql.sql) == _normalize_sql(source_sql.sql):
            errors.append("Candidate SQL is identical to source SQL.")
        if not candidate_sql.rewrite_rule_ids:
            errors.append("Candidate SQL has no rewrite_rule_ids.")
        errors.extend(_blueprint_violations(candidate_sql.sql, blueprint))
        return _unique(errors)

    def explain_rewrite(
        self,
        source_sql: SQLVersion,
        candidate_sql: SQLVersion,
        strategies: list[RetrievedStrategy],
    ) -> str:
        del strategies
        return (
            f"Rewrote SQL version {source_sql.version_id} using "
            f"{', '.join(candidate_sql.rewrite_rule_ids)}."
        )

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run_rewrite(self, request: AgentRequest) -> dict:
        sql_version = self.select_sql_version(request)
        blueprint = self.select_blueprint(request)
        report = self.select_bottleneck_report(request)
        force_free_exploration = bool(
            (request.input_artifacts or {}).get("force_free_exploration")
        )
        physical_context = _physical_schema_context(
            db_id=request.task.db_id,
            dbms=request.task.dbms,
            blueprint=blueprint,
            cost_snapshot=report.cost_snapshot,
        )
        detected_operator_opportunities = detect_operator_opportunities(
            sql=sql_version.sql,
            report=report,
            blueprint=blueprint,
            physical_context=physical_context,
        )
        detected_operator_opportunities = _filter_operator_opportunities_for_report(
            report,
            detected_operator_opportunities,
        )
        planning_report = _report_with_detected_operator_hints(
            report,
            detected_operator_opportunities,
        )
        reflection_context = request.input_artifacts.get("reflection_context")
        explicit_strategies = self.select_retrieved_strategies(request)
        operator_strategies = build_operator_strategies_from_opportunities(
            detected_operator_opportunities
        )
        expert_strategies: list[RetrievedStrategy] = []
        hist_strategies: list[RetrievedStrategy] = []
        low_confidence_rag_strategies: list[RetrievedStrategy] = []
        free_exploration_context: dict | None = None
        used_free_exploration = False
        rag_confidence_miss = False
        operator_plans = self.plan_operator_candidates(
            sql_version=sql_version,
            strategies=operator_strategies,
            bottleneck_report=planning_report,
            blueprint=blueprint,
        )
        strategies = list(operator_strategies)
        plans = operator_plans
        if not operator_plans:
            rag_candidates = explicit_strategies
            if not rag_candidates and self.rag_engine is not None:
                rag_candidates = self.retrieve_rag_strategies(
                    request.task.question, sql_version, blueprint, report
                )
            _, raw_expert_strategies, raw_hist_strategies = self.partition_strategy_sources(
                rag_candidates
            )
            expert_strategies = self.rank_source_strategies(
                sql_version.sql,
                raw_expert_strategies,
                top_k=self.expert_top_k,
            )
            hist_strategies = self.rank_source_strategies(
                sql_version.sql,
                raw_hist_strategies,
                top_k=self.hist_top_k,
            )
            high_confidence_experts = self.high_confidence_strategies(expert_strategies)
            high_confidence_hist = self.high_confidence_strategies(hist_strategies)
            candidate_strategies = _merge_strategies(
                high_confidence_experts or expert_strategies,
                high_confidence_hist or hist_strategies,
            )
            strategies = candidate_strategies
            generic_plans = self.plan_generic_retrieval_candidates(
                sql_version=sql_version,
                strategies=candidate_strategies,
                bottleneck_report=planning_report,
                blueprint=blueprint,
            )
            plans = self.merge_planned_candidates([], generic_plans)
            if not plans and (expert_strategies or hist_strategies):
                low_confidence_rag_strategies = hist_strategies
                rag_confidence_miss = not high_confidence_experts and not high_confidence_hist
        plans = [
            plan.with_physical_schema_context(physical_context)
            for plan in plans
        ]
        plan: RewritePlan | NoOpRewritePlan = (
            plans[0] if plans else NoOpRewritePlan(reason="no safe rewrite candidate")
        )
        base_artifacts = {
            "source_sql_version": sql_version,
            "rewrite_plan": plan,
            "rewrite_plan_candidates": plans,
            "retrieved_strategies": strategies,
            "detected_operator_opportunities": detected_operator_opportunities,
            "deterministic_operator_strategies": operator_strategies,
            "expert_retrieved_strategies": expert_strategies,
            "hist_retrieved_strategies": hist_strategies,
            "low_confidence_rag_strategies": low_confidence_rag_strategies,
            "used_hint": _hint_for_plan(report, plan) if plan else None,
        }
        def _run_free_exploration_fallback(reason: str) -> dict[str, Any]:
            free_context = self.build_free_exploration_context(
                request=request,
                sql_version=sql_version,
                blueprint=blueprint,
                bottleneck_report=report,
                operator_strategies=operator_strategies,
                expert_strategies=expert_strategies,
                hist_strategies=hist_strategies,
                failed_directions=self.select_free_exploration_history(request),
            )
            candidate, free_skip_reason = self.free_explore_rewrite(
                sql_version,
                free_context,
            )
            if candidate is None:
                return {
                    **base_artifacts,
                    "candidate_sql_version": None,
                    "skip_reason": free_skip_reason,
                    "guardrail_errors": [],
                    "free_exploration_context": free_context,
                    "free_exploration": True,
                    "tool_calls": [
                        {"tool_name": "retrieve_hybrid_strategies", "summary": "retrieval or fallback"},
                        {"tool_name": "free_exploration_fallback", "summary": reason},
                        {
                            "tool_name": "llm_free_exploration",
                            "summary": (
                                "reported no safe optimization space"
                                if free_skip_reason
                                == "LLM free exploration reported no safe optimization space."
                                else "returned empty response"
                            ),
                        },
                    ],
                }
            guardrail_errors = self.guardrail_check(sql_version, candidate, blueprint)
            return {
                **base_artifacts,
                "candidate_sql_version": None if guardrail_errors else candidate,
                "skip_reason": "Candidate failed Rewriter guardrails." if guardrail_errors else None,
                "guardrail_errors": guardrail_errors,
                "free_exploration_context": free_context,
                "free_exploration": True,
                "reflection_allowed_failure_types": [
                    "semantic_inconsistency",
                    "syntax_error",
                ],
                "tool_calls": [
                    {"tool_name": "retrieve_hybrid_strategies", "summary": "retrieval or fallback"},
                    {"tool_name": "free_exploration_fallback", "summary": reason},
                    {"tool_name": "llm_free_exploration", "summary": "candidate generation without a retrieved rule"},
                    {
                        "tool_name": "guardrail_check",
                        "summary": "candidate rejected" if guardrail_errors else "candidate passed",
                    },
                ],
            }
        if force_free_exploration:
            if self.llm_free_explorer is None:
                return {
                    **base_artifacts,
                    "candidate_sql_version": None,
                    "skip_reason": "Forced free exploration requested but no llm_free_explorer is configured.",
                    "guardrail_errors": [],
                    "free_exploration": False,
                    "tool_calls": [
                        {"tool_name": "force_free_exploration", "summary": "requested by controller"},
                        {"tool_name": "llm_free_exploration", "summary": "not configured"},
                    ],
                }
            return _run_free_exploration_fallback(
                "forced by controller after retrieved rewrite was insufficient"
            )
        if not plan and rag_confidence_miss:
            if self.llm_free_explorer is None:
                return {
                    **base_artifacts,
                    "candidate_sql_version": None,
                    "skip_reason": (
                        "RAG recalled only low-confidence strategies and no "
                        "llm_free_explorer is configured."
                    ),
                    "guardrail_errors": [],
                    "free_exploration": False,
                    "tool_calls": [
                        {
                            "tool_name": "retrieve_hybrid_strategies",
                            "summary": "all recalled strategies below confidence threshold",
                        },
                        {
                            "tool_name": "llm_free_exploration",
                            "summary": "not configured",
                        },
                    ],
                }
            return _run_free_exploration_fallback(
                "all recalled strategies below confidence threshold"
            )
        if not plans:
            if self.llm_free_explorer is not None:
                return _run_free_exploration_fallback(
                    "no applicable safe retrieved rewrite plan"
                )
            return {
                **base_artifacts,
                "candidate_sql_version": None,
                "skip_reason": "No safe rewrite plan found.",
                "guardrail_errors": [],
                "tool_calls": [
                    {"tool_name": "retrieve_hybrid_strategies", "summary": "retrieval or fallback"},
                    {"tool_name": "plan_rewrite", "summary": "no applicable safe plan"},
                ],
            }
        attempted_plans: list[dict[str, Any]] = []
        last_attempted_plan: RewritePlan | None = None
        last_skip_reason: str | None = None
        last_guardrail_errors: list[str] = []
        for current_plan in plans:
            last_attempted_plan = current_plan
            candidate, rewrite_skip_reason = self.rewrite(
                sql_version,
                current_plan,
                blueprint,
                reflection_context=reflection_context if isinstance(reflection_context, dict) else None,
            )
            if candidate is None:
                attempted_plans.append(
                    {
                        "rule_id": current_plan["rule_id"],
                        "outcome": "rewrite_skipped",
                        "reason": rewrite_skip_reason,
                    }
                )
                last_skip_reason = rewrite_skip_reason
                continue
            guardrail_errors = self.guardrail_check(sql_version, candidate, blueprint)
            if guardrail_errors:
                attempted_plans.append(
                    {
                        "rule_id": current_plan["rule_id"],
                        "outcome": "guardrail_rejected",
                        "reason": "; ".join(guardrail_errors),
                    }
                )
                last_guardrail_errors = guardrail_errors
                last_skip_reason = "Candidate failed Rewriter guardrails."
                continue
            return {
                **base_artifacts,
                "rewrite_plan": current_plan,
                "used_hint": _hint_for_plan(report, current_plan),
                "candidate_sql_version": candidate,
                "skip_reason": None,
                "guardrail_errors": [],
                "attempted_plans": attempted_plans,
                "free_exploration": used_free_exploration,
                "tool_calls": [
                    {"tool_name": "retrieve_hybrid_strategies", "summary": "retrieval or fallback"},
                    {"tool_name": "plan_rewrite", "summary": str(current_plan["rule_id"])},
                    {"tool_name": "rewrite", "summary": "candidate generation"},
                    {"tool_name": "guardrail_check", "summary": "candidate passed"},
                ],
            }
        rejected_plan = (
            RejectedRewritePlan(
                rule_id=last_attempted_plan.rule_id,
                hint_strategy=last_attempted_plan.hint_strategy,
                rejection_reason=last_skip_reason or "; ".join(last_guardrail_errors) or "all candidate plans were exhausted",
                source_type=last_attempted_plan.source_type,
            )
            if last_attempted_plan is not None
            else plan
        )
        if self.llm_free_explorer is not None and not used_free_exploration:
            return _run_free_exploration_fallback(
                last_skip_reason
                or "; ".join(last_guardrail_errors)
                or "all candidate plans were exhausted"
            )
        return {
            **base_artifacts,
            "rewrite_plan": rejected_plan,
            "candidate_sql_version": None,
            "skip_reason": last_skip_reason or "No safe rewrite plan found.",
            "guardrail_errors": last_guardrail_errors,
            "attempted_plans": attempted_plans,
            "free_exploration": used_free_exploration,
            "tool_calls": [
                {"tool_name": "retrieve_hybrid_strategies", "summary": "retrieval or fallback"},
                {"tool_name": "plan_rewrite", "summary": "explored sibling rewrite plans"},
                {
                    "tool_name": "rewrite",
                    "summary": (
                        "all sibling plans reported no safe optimization space"
                        if last_skip_reason == "LLM rewriter reported no safe optimization space."
                        else "all sibling plans were exhausted"
                    ),
                },
            ],
        }

    def run(self, request: AgentRequest) -> AgentResponse:
        try:
            artifacts = self.run_rewrite(request)
        except Exception as exc:
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status="error",
                output_artifacts={},
                reasoning_summary="SQL rewrite failed before producing a candidate.",
                tool_calls=[],
                errors=[str(exc)],
            )
        candidate = artifacts["candidate_sql_version"]
        status = "success" if candidate is not None else "skipped"
        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            status=status,
            output_artifacts=artifacts,
            reasoning_summary=(
                f"Generated candidate SQL version {candidate.version_id} "
                f"with rule {candidate.rewrite_rule_ids[0]}."
                if candidate is not None
                else artifacts["skip_reason"]
            ),
            tool_calls=artifacts["tool_calls"],
            errors=list(artifacts.get("guardrail_errors") or []),
        )


_MVP_CHECKER_HINTS = {
    "reduce_select_columns",
    "eliminate_redundant_distinct",
    "eliminate_redundant_count_distinct",
    "eliminate_same_key_bridge_join",
    "rewrite_scalar_maxmin_subquery",
    "align_order_by_with_index",
    "avoid_function_on_column",
    "simplify_join_graph",
}


def _has_mvp_checker(hint_strategy: str) -> bool:
    return hint_strategy in _MVP_CHECKER_HINTS


_HINT_TO_RULE_FAMILIES = {
    "reduce_select_columns": {"projection_pruning", "remove_select_star", "reduce_row_width"},
    "eliminate_redundant_distinct": {
        "redundant_distinct_elimination",
        "distinct_elimination",
    },
    "eliminate_redundant_count_distinct": {
        "redundant_count_distinct_elimination",
        "count_grain_canonicalization",
    },
    "eliminate_same_key_bridge_join": {
        "same_key_bridge_join_elimination",
        "join_graph_simplification",
    },
    "rewrite_scalar_maxmin_subquery": {
        "scalar_maxmin_to_order_limit",
        "maxmin_subquery_rewrite",
        "top1_order_limit",
    },
    "align_order_by_with_index": {
        "topk_before_join",
        "top1_before_join",
        "top1_anchor_then_lookup_tail",
        "scalar_extrema_anchor_then_lookup_tail",
        "distinct_extrema_to_grouped_having",
        "distinct_top1_to_grouped_extrema",
        "grouped_max_top1_before_join",
        "filter_dimension_before_top1",
        "argmax_aggregate_to_topk",
        "reanchor_join_driver",
    },
    "replace_correlated_subquery": {"correlated_subquery_to_join", "correlated_subquery_to_cte"},
    "push_down_filter": {
        "predicate_pushdown",
        "filter_before_join",
        "grouped_max_top1_before_join",
        "filter_dimension_before_top1",
        "dimension_key_first_then_fact_probe",
        "reanchor_join_driver",
        "top1_anchor_then_lookup_tail",
        "prefer_summary_table_when_grain_matches",
    },
    "eliminate_redundant_self_join": {"self_join_lookup_simplification"},
    "pre_aggregate_before_join": {
        "pre_aggregation",
        "aggregate_then_join",
        "distinct_top1_to_grouped_extrema",
        "grouped_max_top1_before_join",
        "repeated_rescan_to_conditional_agg",
    },
    "rewrite_or_to_union": {"or_to_union_all", "disjunction_split"},
    "simplify_join_graph": {
        "join_graph_simplification",
        "distinct_join_to_semijoin",
        "redundant_bridge_join_elimination",
        "reanchor_join_driver",
        "symmetric_union_arm_pruning",
    },
    "add_null_guard_for_sort_key": {"null_guard_topk"},
    "avoid_function_on_column": {
        "sargable_predicate",
        "date_extraction_to_range",
        "like_prefix_to_range",
    },
}


def _strategy_families(strategy: RetrievedStrategy) -> set[str]:
    explicit_families = {
        str(family).strip()
        for family in (strategy.families or [])
        if str(family).strip()
    }
    if explicit_families:
        return explicit_families
    text = " ".join(
        [
            strategy.rule_id,
            strategy.rule_name,
            strategy.rewrite_template,
            " ".join(strategy.applicable_when),
        ]
    ).lower()
    families: set[str] = set()
    if "select *" in text or "projection" in text or "row width" in text:
        families.update({"projection_pruning", "remove_select_star", "reduce_row_width"})
    if "redundant distinct" in text and "unique" in text:
        families.add("redundant_distinct_elimination")
    if "same bridge key" in text and "bridge table" in text:
        families.add("same_key_bridge_join_elimination")
    if "max" in text and "order by" in text and "limit" in text:
        families.add("scalar_maxmin_to_order_limit")
    if (
        "strftime('%y'" in text
        or "strftime('%y-%m'" in text
        or "date extraction on column to range predicate" in text
    ):
        families.add("date_extraction_to_range")
    if "like prefix on ordered text/date column to range predicate" in text:
        families.add("like_prefix_to_range")
    if (
        "top-k before downstream joins" in text
        or "order by ... limit 1 uses a base-table sort key" in text
        or (
            "select column from (select column from table order by numeric_column desc limit literal)" in text
            and "order by" in text
            and "limit" in text
        )
    ):
        families.add("topk_before_join")
    if (
        "resolve top-1 anchor key before final lookup tail" in text
        or "the final lookup table can be probed by one uniqueness-backed key after the top-1 anchor is resolved upstream" in text
    ):
        families.add("top1_anchor_then_lookup_tail")
    if (
        "normalize scalar extrema filter into top-1 anchor then final lookup tail" in text
        or "a predecessor metric is filtered by a scalar extrema subquery" in text
    ):
        families.add("scalar_extrema_anchor_then_lookup_tail")
    if (
        "canonicalize distinct extrema filter into grouped having extrema" in text
        or "select distinct projects plain columns while where filters one metric by a scalar extrema subquery" in text
    ):
        families.add("distinct_extrema_to_grouped_having")
    if (
        "canonicalize distinct top-1 into grouped extrema" in text
        or "select distinct projects one column while order by ranks joined rows by one metric with limit 1" in text
    ):
        families.add("distinct_top1_to_grouped_extrema")
    if (
        "pre-aggregate join table before grouped max top-1" in text
        or "group by base key with order by max(join_table.column) limit 1" in text
        or "max(order_column) as max_order_value" in text
    ):
        families.add("grouped_max_top1_before_join")
    if (
        "repeated aggregate argmax to order by aggregate limit" in text
        or "having count(*) = (select max(cnt)" in text
    ):
        families.add("argmax_aggregate_to_topk")
    if "collapse repeated rescans into one grouped conditional aggregation pass" in text:
        families.add("repeated_rescan_to_conditional_agg")
    if "resolve dimension key first, then probe fact table" in text:
        families.add("dimension_key_first_then_fact_probe")
    if "re-anchor join on the selective driver table" in text:
        families.add("reanchor_join_driver")
    if "summary-compatible" in text and "raw detail fact" in text:
        families.add("prefer_summary_table_when_grain_matches")
    if (
        "filter joined dimension before fact top-1" in text
        or "select dim_column from fact_table join (select dim_key, dim_column from dim_table where dim_filter)" in text
        or "dimension-side filter can be pushed into a filtered subquery before order by ... limit 1" in text
    ):
        families.add("filter_dimension_before_top1")
    if "join plus distinct selected key to semi-join" in text:
        families.add("distinct_join_to_semijoin")
    if "eliminate redundant bridge join when a direct narrower join exists" in text:
        families.add("redundant_bridge_join_elimination")
    if "prune symmetric union/or edge duplication under canonical edge storage" in text:
        families.add("symmetric_union_arm_pruning")
    if "correlated" in text and ("join" in text or "cte" in text):
        families.add("correlated_subquery_to_join")
    if "push" in text and "filter" in text:
        families.add("predicate_pushdown")
    if (
        "where text_column = literal" in text
        and "select fk_column from table inner join table on fk_column = fk_column" in text
        and "->" in text
    ):
        families.add("self_join_lookup_simplification")
    if (
        "same-table lookup join to scalar key subquery" in text
        or "where column = (select column from table where text_column = literal)" in text
    ):
        families.add("self_join_lookup_simplification")
    if "pre" in text and "aggregat" in text:
        families.add("pre_aggregation")
    if "or" in text and "union" in text:
        families.add("or_to_union_all")
    if "join graph" in text or "cartesian" in text:
        families.add("join_graph_simplification")
    if "null" in text and ("sort" in text or "top" in text):
        families.add("null_guard_topk")
    if "sargable" in text or "function on column" in text:
        families.add("sargable_predicate")
    return families


def _hint_for_plan(report: BottleneckReport, plan: RewritePlan | NoOpRewritePlan) -> RewriteHint | None:
    strategy = plan.get("hint_strategy")
    for hint in report.rewrite_hints:
        if hint.strategy == strategy:
            return hint
    return None


def _report_with_detected_operator_hints(
    report: BottleneckReport,
    opportunities: list[Any],
) -> BottleneckReport:
    existing = {hint.strategy for hint in report.rewrite_hints}
    merged_hints = list(report.rewrite_hints)
    for opportunity in opportunities:
        strategy = str(getattr(opportunity, "hint_strategy", "") or "")
        if not strategy or strategy in existing:
            continue
        merged_hints.append(
            RewriteHint(
                strategy=strategy,
                target_fragment=getattr(opportunity, "target_fragment", None),
                expected_effect=str(getattr(opportunity, "expected_effect", "") or ""),
                risk="medium",
                requires_validation=bool(getattr(opportunity, "requires_validation", True)),
                dbms_notes=getattr(opportunity, "dbms_notes", None),
            )
        )
        existing.add(strategy)
    if merged_hints == list(report.rewrite_hints):
        return report
    return BottleneckReport(
        sql_version_id=report.sql_version_id,
        bottlenecks=list(report.bottlenecks),
        cost_snapshot=dict(report.cost_snapshot),
        risk_tags=list(report.risk_tags),
        rewrite_hints=merged_hints,
        explanation=report.explanation,
    )


def _filter_operator_opportunities_for_report(
    report: BottleneckReport,
    opportunities: list[Any],
) -> list[Any]:
    if not report.rewrite_hints:
        return opportunities
    allowed_strategies = {hint.strategy for hint in report.rewrite_hints}
    auto_allowed_strategies = {
        "eliminate_redundant_distinct",
        "eliminate_redundant_count_distinct",
        "eliminate_same_key_bridge_join",
    }
    auto_allowed_operators = {
        "top1_anchor_then_lookup_tail",
    }
    filtered = [
        opportunity
        for opportunity in opportunities
        if str(getattr(opportunity, "hint_strategy", "") or "") in allowed_strategies
        or str(getattr(opportunity, "hint_strategy", "") or "") in auto_allowed_strategies
        or str(getattr(opportunity, "operator_name", "") or "") in auto_allowed_operators
    ]
    return filtered


def _merge_strategies(
    primary: list[RetrievedStrategy],
    fallback: list[RetrievedStrategy],
) -> list[RetrievedStrategy]:
    merged: list[RetrievedStrategy] = []
    seen: set[str] = set()
    for strategy in [*primary, *fallback]:
        key = strategy.rule_id or strategy.rule_name
        if key in seen:
            continue
        seen.add(key)
        merged.append(strategy)
    return merged


def _operator_match_from_plan(rewrite_plan: RewritePlan | NoOpRewritePlan | dict[str, Any]) -> Any | None:
    if isinstance(rewrite_plan, OperatorDeterministicRewritePlan):
        return rewrite_plan.operator_match
    fragments = rewrite_plan.get("required_fragments") or {}
    if not isinstance(fragments, dict):
        return None
    return fragments.get("operator_match")


def _has_top_level_select_star(sql: str) -> bool:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return False
    select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
    return bool(select and any(_is_star_expression(e) for e in select.expressions))


def _is_star_expression(expression: exp.Expression) -> bool:
    return isinstance(expression, exp.Star) or (
        isinstance(expression, exp.Column) and isinstance(expression.this, exp.Star)
    )


def _has_scalar_maxmin_subquery(sql: str) -> bool:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return False
    for subquery in ast.find_all(exp.Subquery):
        for func in subquery.find_all(exp.AggFunc):
            if func.key.upper() in {"MAX", "MIN"}:
                return True
    return False


def _sql_join_count(sql: str) -> int:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return 0
    return sum(1 for _ in ast.find_all(exp.Join))


def _date_extraction_to_range_shape(sql: str) -> dict[str, Any] | None:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return None
    inferred_table = _single_from_table_name(ast)
    where_clause = ast.find(exp.Where)
    scope_expr = where_clause.this if where_clause is not None else None
    for node in ast.walk():
        shape = _date_range_replacement_shape(
            node,
            inferred_table=inferred_table,
            scope_expr=scope_expr,
        )
        if shape is not None:
            return shape
    return None


def _redundant_distinct_elimination_shape(
    sql: str,
    *,
    require_unique_index: bool,
    physical_context: dict | None = None,
    required_fragments: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return None
    select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
    if select is None or select.args.get("distinct") is None:
        return None
    if any(select.args.get(key) is not None for key in ("group", "having", "with_", "qualify", "order", "limit")):
        return None
    if any(isinstance(node, exp.SetOperation) for node in ast.walk()):
        return None
    if any(True for _ in select.find_all(exp.AggFunc)):
        return None
    from_expr = select.args.get("from_")
    if from_expr is None or not isinstance(from_expr.this, exp.Table):
        return None
    base_table = from_expr.this
    base_alias = base_table.alias_or_name
    joins = select.args.get("joins") or []
    if len(joins) > 1:
        return None
    projected_columns: list[str] = []
    projected_columns_by_table: dict[str, set[str]] = {}
    allowed_tables = {base_alias, base_table.name}
    join_table: exp.Table | None = None
    join_alias: str | None = None
    if joins:
        join = joins[0]
        if join.side and str(join.side).upper() != "INNER":
            return None
        if not isinstance(join.this, exp.Table):
            return None
        join_table = join.this
        join_alias = join_table.alias_or_name
        allowed_tables.update({join_alias, join_table.name})
    for expression in select.expressions:
        if isinstance(expression, exp.Alias):
            expression = expression.this
        if not isinstance(expression, exp.Column):
            return None
        if expression.table and expression.table not in allowed_tables:
            return None
        projected_columns.append(expression.name)
        resolved_table = (
            base_table.name
            if expression.table in {"", None, base_alias, base_table.name}
            else join_table.name if join_table is not None else None
        )
        if resolved_table is None:
            return None
        projected_columns_by_table.setdefault(resolved_table, set()).add(expression.name)
    if not projected_columns:
        return None

    def _serialise_projection_sets() -> dict[str, tuple[str, ...]]:
        return {
            table_name: tuple(sorted(columns))
            for table_name, columns in projected_columns_by_table.items()
        }

    def _covered_unique_index(table_name: str, columns: set[str]) -> tuple[str, tuple[str, ...]] | None:
        if physical_context is None:
            return None
        indexes = physical_context.get("indexes", {}).get(table_name) or []
        for index in indexes:
            if not index.get("unique"):
                continue
            index_columns = tuple(str(column) for column in (index.get("columns") or []) if str(column))
            if index_columns and set(index_columns) <= columns:
                return str(index.get("name") or ""), index_columns
        return None

    def _fragments_match(shape: dict[str, Any]) -> bool:
        if not required_fragments:
            return True
        for key in ("scope", "table", "join_table", "preserved_table"):
            expected = required_fragments.get(key)
            if expected is not None and shape.get(key) != expected:
                return False
        expected_projection = required_fragments.get("projected_columns_by_table")
        if expected_projection:
            normalized_expected = {
                table_name: tuple(sorted(str(column) for column in columns))
                for table_name, columns in expected_projection.items()
            }
            if shape.get("projected_columns_by_table") != normalized_expected:
                return False
        return True

    single_table_shape = {
        "scope": "single_table",
        "table": base_table.name,
        "projected_columns": tuple(projected_columns),
        "projected_columns_by_table": _serialise_projection_sets(),
        "target_fragment": "SELECT DISTINCT",
    }
    if not joins:
        if require_unique_index:
            projected_index = _covered_unique_index(base_table.name, projected_columns_by_table.get(base_table.name, set()))
            if projected_index is None:
                return None
            single_table_shape["unique_index_name"] = projected_index[0]
            single_table_shape["unique_index_columns"] = projected_index[1]
        return single_table_shape if _fragments_match(single_table_shape) else None

    assert join_table is not None
    assert join_alias is not None
    on_clause = joins[0].args.get("on")
    if on_clause is None:
        return None

    def _join_shape_for(
        preserved_table_name: str,
        preserved_aliases: set[str],
        other_table_name: str,
        other_aliases: set[str],
    ) -> dict[str, Any] | None:
        shape = {
            "scope": "single_join",
            "table": base_table.name,
            "join_table": join_table.name,
            "preserved_table": preserved_table_name,
            "projected_columns": tuple(projected_columns),
            "projected_columns_by_table": _serialise_projection_sets(),
            "target_fragment": "SELECT DISTINCT",
        }
        if require_unique_index:
            preserved_projection = projected_columns_by_table.get(preserved_table_name, set())
            preserved_unique_index = _covered_unique_index(preserved_table_name, preserved_projection)
            if preserved_unique_index is None:
                return None
            other_join_columns: set[str] = set()
            for predicate in _flatten_and_conditions(on_clause):
                if not isinstance(predicate, exp.EQ):
                    return None
                left = predicate.left
                right = predicate.right
                if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                    return None
                if left.table in preserved_aliases and right.table in other_aliases:
                    other_join_columns.add(right.name)
                    continue
                if right.table in preserved_aliases and left.table in other_aliases:
                    other_join_columns.add(left.name)
                    continue
                return None
            other_unique_index = _covered_unique_index(other_table_name, other_join_columns)
            if other_unique_index is None:
                return None
            shape["preserved_unique_index_name"] = preserved_unique_index[0]
            shape["preserved_unique_index_columns"] = preserved_unique_index[1]
            shape["joined_unique_index_name"] = other_unique_index[0]
            shape["joined_unique_index_columns"] = other_unique_index[1]
        return shape if _fragments_match(shape) else None

    return _join_shape_for(
        base_table.name,
        {base_alias, base_table.name},
        join_table.name,
        {join_alias, join_table.name},
    ) or _join_shape_for(
        join_table.name,
        {join_alias, join_table.name},
        base_table.name,
        {base_alias, base_table.name},
    )


def _redundant_count_distinct_elimination_shape(
    sql: str,
    physical_context: dict,
) -> dict[str, Any] | None:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return None
    select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
    if select is None:
        return None
    if any(select.args.get(key) is not None for key in ("distinct", "group", "having", "with_", "qualify")):
        return None
    if any(isinstance(node, exp.SetOperation) for node in ast.walk()):
        return None
    if any(True for _ in select.find_all(exp.Join)):
        return None
    if len(select.expressions) != 1:
        return None

    expression = select.expressions[0]
    if isinstance(expression, exp.Alias):
        expression = expression.this
    if not isinstance(expression, exp.Count) or not isinstance(expression.this, exp.Distinct):
        return None
    distinct_expr = expression.this
    if len(distinct_expr.expressions) != 1:
        return None
    counted_expr = distinct_expr.expressions[0]
    if not isinstance(counted_expr, exp.Column):
        return None

    from_expr = select.args.get("from_")
    if from_expr is None or not isinstance(from_expr.this, exp.Table):
        return None
    base_table = from_expr.this
    base_alias = base_table.alias_or_name
    if counted_expr.table and counted_expr.table not in {base_alias, base_table.name}:
        return None

    indexes = physical_context.get("indexes", {}).get(base_table.name) or []
    column_meta = ((physical_context.get("columns") or {}).get(base_table.name) or {}).get(counted_expr.name) or {}
    for index in indexes:
        if not index.get("unique"):
            continue
        index_columns = [str(column) for column in (index.get("columns") or []) if str(column)]
        if index_columns != [counted_expr.name]:
            continue
        return {
            "table": base_table.name,
            "counted_column": counted_expr.name,
            "counted_column_sql": counted_expr.sql(dialect="sqlite"),
            "unique_index_name": str(index.get("name") or ""),
            "can_use_count_star": bool(column_meta.get("not_null")),
            "target_fragment": expression.sql(dialect="sqlite"),
        }
    return None


def _like_prefix_to_range_shape(sql: str) -> dict[str, Any] | None:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return None
    inferred_table = _single_from_table_name(ast)
    where_clause = ast.find(exp.Where)
    if where_clause is None or where_clause.this is None:
        return None
    for node in where_clause.this.walk():
        if not isinstance(node, exp.Like):
            continue
        if node.args.get("escape") is not None:
            continue
        column = node.this
        pattern = node.expression
        if not isinstance(column, exp.Column) or not isinstance(pattern, exp.Literal):
            continue
        raw_pattern = str(pattern.this)
        if not _is_safe_like_prefix_pattern(raw_pattern):
            continue
        prefix = raw_pattern[:-1]
        next_prefix = _next_ascii_prefix(prefix)
        if next_prefix is None:
            continue
        return {
            "predicate": node,
            "column": column.copy(),
            "table": column.table or inferred_table,
            "range_start": prefix,
            "range_end": next_prefix,
            "predicate_sql": node.sql(dialect="sqlite"),
        }
    return None


def _redundant_bridge_join_elimination_shape(
    sql: str,
    blueprint: VerifiedContextBlueprint,
) -> dict[str, Any] | None:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return None
    select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
    if select is None:
        return None
    from_expr = select.args.get("from_")
    joins = select.args.get("joins") or []
    if (
        from_expr is None
        or not isinstance(from_expr.this, exp.Table)
        or len(joins) != 2
        or not isinstance(joins[0].this, exp.Table)
        or not isinstance(joins[1].this, exp.Table)
    ):
        return None
    left = from_expr.this
    bridge = joins[0].this
    right = joins[1].this
    if select.args.get("having") is not None:
        return None
    direct_edge = _direct_blueprint_join_edge(blueprint, left.name, right.name)
    if direct_edge is None:
        return None
    bridge_aliases = {bridge.alias_or_name, bridge.name}
    for expression in list(select.expressions) + [select.args.get("order"), select.args.get("group")]:
        if expression is None:
            continue
        for column in expression.find_all(exp.Column):
            if column.table in bridge_aliases:
                return None
    bridge_predicate_replacements: dict[str, exp.Expression] = {}
    where_clause = select.args.get("where")
    if where_clause is not None:
        for predicate in _flatten_and_conditions(where_clause.this):
            referenced = _referenced_tables(predicate)
            if not (referenced & bridge_aliases):
                continue
            replacement = _bridge_predicate_replacement(
                predicate=predicate,
                left=left,
                bridge=bridge,
                right=right,
                joins=joins,
                direct_edge=direct_edge,
            )
            if replacement is None:
                return None
            bridge_predicate_replacements[_normalize_sql(predicate.sql(dialect="sqlite"))] = replacement
    return {
        "left_table": left.name,
        "left_alias": left.alias_or_name,
        "bridge_table": bridge.name,
        "right_table": right.name,
        "right_alias": right.alias_or_name,
        "direct_edge": direct_edge,
        "bridge_predicate_replacements": bridge_predicate_replacements,
    }


def _same_key_bridge_join_elimination_shape(sql: str) -> dict[str, Any] | None:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return None
    select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
    if select is None:
        return None
    if any(select.args.get(key) is not None for key in ("group", "having", "with_", "qualify")):
        return None
    from_expr = select.args.get("from_")
    joins = select.args.get("joins") or []
    if (
        from_expr is None
        or not isinstance(from_expr.this, exp.Table)
        or len(joins) != 2
        or any(not isinstance(join.this, exp.Table) for join in joins)
    ):
        return None
    if any(join.side and str(join.side).upper() != "INNER" for join in joins):
        return None

    left = from_expr.this
    bridge = joins[0].this
    right = joins[1].this
    join_one = joins[0].args.get("on")
    join_two = joins[1].args.get("on")
    left_bridge = _column_pair_with_table(join_one, left.alias_or_name, bridge.alias_or_name)
    bridge_right = _column_pair_with_table(join_two, bridge.alias_or_name, right.alias_or_name)
    if left_bridge is None or bridge_right is None:
        return None
    left_col, bridge_col_one = left_bridge
    bridge_col_two, right_col = bridge_right
    if bridge_col_one.name != bridge_col_two.name:
        return None

    bridge_alias = bridge.alias_or_name
    for expression in select.expressions:
        if any(column.table == bridge_alias for column in expression.find_all(exp.Column)):
            return None
    for clause_name in ("where", "order", "limit"):
        clause = select.args.get(clause_name)
        if clause is None:
            continue
        for column in clause.find_all(exp.Column):
            if column.table == bridge_alias:
                return None

    return {
        "left_table": left.name,
        "left_alias": left.alias_or_name,
        "left_column": left_col.name,
        "bridge_table": bridge.name,
        "bridge_alias": bridge.alias_or_name,
        "bridge_column": bridge_col_one.name,
        "right_table": right.name,
        "right_alias": right.alias_or_name,
        "right_column": right_col.name,
        "target_fragment": bridge.name,
    }


def _unused_fk_join_elimination_shape(
    sql: str,
    physical_context: dict,
) -> dict[str, Any] | None:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return None
    select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
    if select is None:
        return None
    if any(select.args.get(key) is not None for key in ("group", "having", "with_", "qualify")):
        return None
    if any(isinstance(node, exp.SetOperation) for node in ast.walk()):
        return None
    from_expr = select.args.get("from_")
    joins = select.args.get("joins") or []
    if (
        from_expr is None
        or not isinstance(from_expr.this, exp.Table)
        or len(joins) != 1
        or not isinstance(joins[0].this, exp.Table)
    ):
        return None
    join = joins[0]
    if join.side and str(join.side).upper() != "INNER":
        return None

    base = from_expr.this
    joined = join.this
    base_aliases = {base.alias_or_name, base.name}
    joined_aliases = {joined.alias_or_name, joined.name}

    for expression in select.expressions:
        if any(column.table in joined_aliases for column in expression.find_all(exp.Column)):
            return None
    for clause_name in ("where", "order", "limit", "distinct"):
        clause = select.args.get(clause_name)
        if clause is None:
            continue
        for column in clause.find_all(exp.Column):
            if column.table in joined_aliases:
                return None

    on_clause = join.args.get("on")
    if on_clause is None:
        return None
    base_join_columns: set[str] = set()
    joined_join_columns: set[str] = set()
    for predicate in _flatten_and_conditions(on_clause):
        if not isinstance(predicate, exp.EQ):
            return None
        left = predicate.left
        right = predicate.right
        if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
            return None
        if left.table in base_aliases and right.table in joined_aliases:
            base_join_columns.add(left.name)
            joined_join_columns.add(right.name)
            continue
        if right.table in base_aliases and left.table in joined_aliases:
            base_join_columns.add(right.name)
            joined_join_columns.add(left.name)
            continue
        return None
    if len(base_join_columns) != 1 or len(joined_join_columns) != 1:
        return None

    foreign_keys = physical_context.get("foreign_keys") or []
    matching_fk = next(
        (
            fk for fk in foreign_keys
            if str(fk.get("from_table")) == base.name
            and str(fk.get("to_table")) == joined.name
            and str(fk.get("from_column")) in base_join_columns
            and str(fk.get("to_column")) in joined_join_columns
        ),
        None,
    )
    if matching_fk is None:
        return None

    joined_indexes = physical_context.get("indexes", {}).get(joined.name) or []
    joined_unique_ok = any(
        index.get("unique")
        and [str(column) for column in (index.get("columns") or []) if str(column)] == [str(matching_fk.get("to_column"))]
        for index in joined_indexes
    )
    if not joined_unique_ok:
        return None

    return {
        "base_table": base.name,
        "base_alias": base.alias_or_name,
        "joined_table": joined.name,
        "joined_alias": joined.alias_or_name,
        "base_join_column": str(matching_fk.get("from_column")),
        "joined_join_column": str(matching_fk.get("to_column")),
        "target_fragment": joined.name,
    }


def _unused_fk_join_chain_elimination_shape(
    sql: str,
    physical_context: dict,
) -> dict[str, Any] | None:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return None
    select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
    if select is None:
        return None
    if any(select.args.get(key) is not None for key in ("group", "having", "with_", "qualify")):
        return None
    if any(isinstance(node, exp.SetOperation) for node in ast.walk()):
        return None
    from_expr = select.args.get("from_")
    joins = select.args.get("joins") or []
    if (
        from_expr is None
        or not isinstance(from_expr.this, exp.Table)
        or len(joins) < 2
        or any(not isinstance(join.this, exp.Table) for join in joins)
    ):
        return None
    if any(join.side and str(join.side).upper() != "INNER" for join in joins):
        return None

    base = from_expr.this
    if any(_is_star_expression(expression) for expression in select.expressions):
        return None

    joined_tables = [join.this for join in joins]
    joined_aliases = {table.alias_or_name for table in joined_tables} | {table.name for table in joined_tables}
    base_aliases = {base.alias_or_name, base.name}

    for expression in select.expressions:
        if not _expression_uses_only_tables(expression, base_aliases):
            return None
    for clause_name in ("where", "order", "limit", "distinct"):
        clause = select.args.get(clause_name)
        if clause is None:
            continue
        for column in clause.find_all(exp.Column):
            if column.table in joined_aliases:
                return None

    foreign_keys = physical_context.get("foreign_keys") or []
    indexes_by_table = physical_context.get("indexes", {}) or {}
    current_table = base
    current_aliases = {base.alias_or_name, base.name}
    joined_names: list[str] = []
    join_steps: list[dict[str, str]] = []
    for join in joins:
        next_table = join.this
        next_aliases = {next_table.alias_or_name, next_table.name}
        on_clause = join.args.get("on")
        if on_clause is None:
            return None
        current_join_columns: set[str] = set()
        next_join_columns: set[str] = set()
        for predicate in _flatten_and_conditions(on_clause):
            if not isinstance(predicate, exp.EQ):
                return None
            left = predicate.left
            right = predicate.right
            if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                return None
            if left.table in current_aliases and right.table in next_aliases:
                current_join_columns.add(left.name)
                next_join_columns.add(right.name)
                continue
            if right.table in current_aliases and left.table in next_aliases:
                current_join_columns.add(right.name)
                next_join_columns.add(left.name)
                continue
            return None
        if len(current_join_columns) != 1 or len(next_join_columns) != 1:
            return None
        current_col = next(iter(current_join_columns))
        next_col = next(iter(next_join_columns))
        matching_fk = next(
            (
                fk for fk in foreign_keys
                if str(fk.get("from_table")) == current_table.name
                and str(fk.get("to_table")) == next_table.name
                and str(fk.get("from_column")) == current_col
                and str(fk.get("to_column")) == next_col
            ),
            None,
        )
        if matching_fk is None:
            return None
        if not any(
            index.get("unique")
            and [str(column) for column in (index.get("columns") or []) if str(column)] == [next_col]
            for index in (indexes_by_table.get(next_table.name) or [])
        ):
            return None
        joined_names.append(next_table.name)
        join_steps.append(
            {
                "from_table": current_table.name,
                "from_column": current_col,
                "to_table": next_table.name,
                "to_column": next_col,
            }
        )
        current_table = next_table
        current_aliases = next_aliases

    return {
        "base_table": base.name,
        "base_alias": base.alias_or_name,
        "joined_tables": tuple(joined_names),
        "join_steps": tuple(join_steps),
        "target_fragment": " -> ".join([base.name, *joined_names]),
    }


def _flatten_and_conditions(expression: exp.Expression) -> list[exp.Expression]:
    if isinstance(expression, exp.And):
        return _flatten_and_conditions(expression.left) + _flatten_and_conditions(expression.right)
    return [expression]


def _column_pair_with_table(
    expression: exp.Expression | None,
    left_table: str,
    right_table: str,
) -> tuple[exp.Column, exp.Column] | None:
    if not isinstance(expression, exp.EQ):
        return None
    left = expression.left
    right = expression.right
    if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
        return None
    if left.table == left_table and right.table == right_table:
        return (left, right)
    if left.table == right_table and right.table == left_table:
        return (right, left)
    return None


def _table_has_single_column_uniqueness(
    physical_context: dict,
    table_name: str,
    column_name: str,
) -> bool:
    table_columns = (physical_context.get("columns", {}) or {}).get(table_name) or {}
    column_meta = table_columns.get(column_name) or {}
    if int(column_meta.get("primary_key_position") or 0) > 0:
        return True
    for index in (physical_context.get("indexes", {}) or {}).get(table_name) or []:
        columns = [str(column) for column in (index.get("columns") or []) if str(column)]
        if index.get("unique") and columns == [column_name]:
            return True
    return False


def _single_join_joined_side_key(sql: str) -> tuple[str | None, str | None]:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return (None, None)
    select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
    if select is None:
        return (None, None)
    joins = select.args.get("joins") or []
    from_expr = select.args.get("from_")
    if (
        from_expr is None
        or not isinstance(from_expr.this, exp.Table)
        or len(joins) != 1
        or not isinstance(joins[0].this, exp.Table)
    ):
        return (None, None)
    base = from_expr.this
    joined = joins[0].this
    base_aliases = {base.alias_or_name, base.name}
    joined_aliases = {joined.alias_or_name, joined.name}
    on_clause = joins[0].args.get("on")
    if on_clause is None:
        return (None, None)
    joined_columns: set[str] = set()
    for predicate in _flatten_and_conditions(on_clause):
        if not isinstance(predicate, exp.EQ):
            return (None, None)
        left = predicate.left
        right = predicate.right
        if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
            return (None, None)
        if left.table in base_aliases and right.table in joined_aliases:
            joined_columns.add(right.name)
            continue
        if right.table in base_aliases and left.table in joined_aliases:
            joined_columns.add(left.name)
            continue
        return (None, None)
    if len(joined_columns) != 1:
        return (None, None)
    return (joined.name, next(iter(joined_columns)))


def _combine_conjuncts(conditions: list[exp.Expression]) -> exp.Expression | None:
    if not conditions:
        return None
    combined = conditions[0]
    for condition in conditions[1:]:
        combined = exp.and_(combined, condition)
    return combined


def _combine_conjuncts_or(conditions: list[exp.Expression]) -> exp.Expression | None:
    if not conditions:
        return None
    combined = conditions[0]
    for condition in conditions[1:]:
        combined = exp.or_(combined, condition)
    return combined


def _referenced_tables(expression: exp.Expression) -> set[str]:
    return {
        str(column.table)
        for column in expression.find_all(exp.Column)
        if column.table is not None
    }


def _single_from_table_name(ast: exp.Expression) -> str | None:
    select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
    if select is None:
        return None
    from_expr = select.args.get("from_")
    if from_expr is None or from_expr.this is None or not isinstance(from_expr.this, exp.Table):
        return None
    return from_expr.this.name


def _direct_blueprint_join_edge(
    blueprint: VerifiedContextBlueprint,
    left_table: str,
    right_table: str,
) -> JoinEdge | None:
    for edge in blueprint.join_topology.edges:
        if {edge.source_table, edge.target_table} == {left_table, right_table}:
            return edge
    return None


def _expression_uses_only_tables(expression: exp.Expression, tables: set[str]) -> bool:
    referenced = _referenced_tables(expression)
    return referenced <= tables


def _rewrite_table_references(
    expression: exp.Expression,
    source_tables: set[str],
    target_table: str,
) -> exp.Expression:
    return expression.transform(
        lambda node: (
            exp.column(node.name, table=target_table)
            if isinstance(node, exp.Column) and node.table in source_tables
            else node
        )
    )


def _rewrite_summary_table_references(
    expression: exp.Expression,
    *,
    source_tables: set[str],
    target_table: str,
    column_mapping: dict[str, str],
    target_allowed_columns: set[str] | frozenset[str],
) -> exp.Expression | None:
    def _replace(node: exp.Expression) -> exp.Expression:
        if (
            isinstance(node, exp.Column)
            and node.table in source_tables
            and node.name in column_mapping
        ):
            return exp.column(column_mapping[node.name], table=target_table)
        return node

    rewritten = expression.transform(_replace)
    for column in rewritten.find_all(exp.Column):
        if column.table == target_table and column.name not in target_allowed_columns:
            return None
    return rewritten


def _copy_table_with_alias(table_name: str, alias: str) -> exp.Table:
    table = exp.to_table(table_name)
    if alias != table_name:
        table.set("alias", exp.TableAlias(this=exp.to_identifier(alias)))
    return table


def _range_predicate_for_bounds(
    *,
    column: exp.Column,
    start: str,
    end: str,
) -> exp.Expression:
    return exp.and_(
        exp.GTE(this=column.copy(), expression=exp.Literal.string(start)),
        exp.LT(this=column.copy(), expression=exp.Literal.string(end)),
    )


def _single_bound_predicate(
    *,
    column: exp.Column,
    lower: str | exp.Expression | None = None,
    upper: str | exp.Expression | None = None,
) -> exp.Expression:
    lower_expr = lower if isinstance(lower, exp.Expression) or lower is None else exp.Literal.string(lower)
    upper_expr = upper if isinstance(upper, exp.Expression) or upper is None else exp.Literal.string(upper)
    if lower is not None and upper is not None:
        return exp.and_(
            exp.GTE(this=column.copy(), expression=lower_expr.copy() if isinstance(lower_expr, exp.Expression) else lower_expr),
            exp.LT(this=column.copy(), expression=upper_expr.copy() if isinstance(upper_expr, exp.Expression) else upper_expr),
        )
    if lower is not None:
        return exp.GTE(this=column.copy(), expression=lower_expr.copy() if isinstance(lower_expr, exp.Expression) else lower_expr)
    if upper is not None:
        return exp.LT(this=column.copy(), expression=upper_expr.copy() if isinstance(upper_expr, exp.Expression) else upper_expr)
    raise ValueError("At least one bound is required.")


def _date_range_replacement_shape(
    node: exp.Expression,
    *,
    inferred_table: str | None,
    scope_expr: exp.Expression | None = None,
) -> dict[str, Any] | None:
    month_shape = _month_bucket_replacement_shape(
        node,
        inferred_table=inferred_table,
        scope_expr=scope_expr,
    )
    if month_shape is not None:
        return month_shape
    if isinstance(node, exp.Between):
        return _date_between_replacement_shape(node, inferred_table=inferred_table)
    if isinstance(node, (exp.EQ, exp.LT, exp.LTE, exp.GT, exp.GTE)):
        return _date_comparison_replacement_shape(node, inferred_table=inferred_table)
    return None


def _date_comparison_replacement_shape(
    node: exp.Expression,
    *,
    inferred_table: str | None,
) -> dict[str, Any] | None:
    left = getattr(node, "left", None)
    right = getattr(node, "right", None)
    if left is None or not isinstance(right, exp.Literal):
        return None
    operator = _comparison_operator_text(node)
    if operator is None:
        return None
    extracted = _extract_date_bucket_function(left, literal_text=str(right.this))
    if extracted is not None:
        column = extracted["column"]
        replacement = _comparison_replacement_for_bucket(
            column=column,
            operator=operator,
            range_start=extracted["range_start"],
            range_end=extracted["range_end"],
        )
        if replacement is None:
            return None
        return {
            "predicate": node,
            "column": column.copy(),
            "table": column.table or inferred_table,
            "predicate_sql": node.sql(dialect="sqlite"),
            "replacement": replacement,
        }
    arithmetic = _year_difference_predicate_shape(
        expr=left,
        operator=operator,
        literal_text=str(right.this),
    )
    if arithmetic is None:
        return None
    column = arithmetic["column"]
    return {
        "predicate": node,
        "column": column.copy(),
        "table": column.table or inferred_table,
        "predicate_sql": node.sql(dialect="sqlite"),
        "replacement": arithmetic["replacement"],
    }


def _month_bucket_replacement_shape(
    node: exp.Expression,
    *,
    inferred_table: str | None,
    scope_expr: exp.Expression | None,
) -> dict[str, Any] | None:
    if scope_expr is None:
        return None
    column: exp.Column | None = None
    months: list[int] | None = None
    if isinstance(node, exp.EQ) and isinstance(node.right, exp.Literal):
        column = _extract_month_bucket_column(node.left)
        months = _parse_month_literals([str(node.right.this)])
    elif isinstance(node, exp.In):
        column = _extract_month_bucket_column(node.this)
        literals = [str(item.this) for item in node.expressions if isinstance(item, exp.Literal)]
        if len(literals) != len(node.expressions):
            return None
        months = _parse_month_literals(literals)
    if column is None or not months:
        return None
    year = _infer_fixed_year_for_column(scope_expr, column)
    if year is None:
        return None
    return {
        "predicate": node,
        "column": column.copy(),
        "table": column.table or inferred_table,
        "predicate_sql": node.sql(dialect="sqlite"),
        "replacement": _or_month_ranges(column=column, year=year, months=months),
    }


def _date_between_replacement_shape(
    node: exp.Between,
    *,
    inferred_table: str | None,
) -> dict[str, Any] | None:
    low = node.args.get("low")
    high = node.args.get("high")
    if not isinstance(low, exp.Literal) or not isinstance(high, exp.Literal):
        return None
    low_shape = _extract_date_bucket_function(node.this, literal_text=str(low.this))
    high_shape = _extract_date_bucket_function(node.this, literal_text=str(high.this))
    if low_shape is None or high_shape is None:
        return None
    low_column = low_shape["column"]
    high_column = high_shape["column"]
    if _normalize_sql(low_column.sql(dialect="sqlite")) != _normalize_sql(
        high_column.sql(dialect="sqlite")
    ):
        return None
    replacement = _single_bound_predicate(
        column=low_column,
        lower=low_shape["range_start"],
        upper=high_shape["range_end"],
    )
    return {
        "predicate": node,
        "column": low_column.copy(),
        "table": low_column.table or inferred_table,
        "predicate_sql": node.sql(dialect="sqlite"),
        "replacement": replacement,
    }


def _extract_date_bucket_function(
    expr: exp.Expression,
    *,
    literal_text: str,
) -> dict[str, Any] | None:
    fmt: exp.Literal | None = None
    column: exp.Column | None = None
    bounds: tuple[str, str] | None = None
    if isinstance(expr, exp.Cast):
        inner = expr.this
        if isinstance(inner, exp.TimeToStr):
            format_arg = inner.args.get("format")
            if isinstance(format_arg, exp.Literal):
                fmt = format_arg
            source = inner.this
            source_inner = source.this if hasattr(source, "this") else source
            if isinstance(source_inner, exp.Column):
                column = source_inner
    elif isinstance(expr, exp.Anonymous) and expr.name.upper() == "STRFTIME":
        args = list(expr.expressions)
        if len(args) == 2 and isinstance(args[0], exp.Literal) and isinstance(args[1], exp.Column):
            fmt = args[0]
            column = args[1]
    elif isinstance(expr, exp.TimeToStr):
        format_arg = expr.args.get("format")
        if isinstance(format_arg, exp.Literal):
            fmt = format_arg
        source = expr.this
        if isinstance(source, exp.Cast) and isinstance(source.this, exp.Column):
            column = source.this
        elif hasattr(source, "this") and isinstance(source.this, exp.DPipe):
            synthesized = _extract_synthesized_yearmonth_source(source.this, format_text=str(format_arg.this) if isinstance(format_arg, exp.Literal) else None, literal_text=literal_text)
            if synthesized is not None:
                return synthesized
        elif hasattr(source, "this") and isinstance(source.this, exp.Column):
            column = source.this
        elif isinstance(source, exp.Column):
            column = source
    elif isinstance(expr, exp.Substring):
        start = expr.args.get("start")
        length = expr.args.get("length")
        if (
            isinstance(expr.this, exp.Column)
            and isinstance(start, exp.Literal)
            and isinstance(length, exp.Literal)
            and str(start.this) == "1"
        ):
            bounds = _substring_range_bounds(int(str(length.this)), literal_text)
            column = expr.this
    elif isinstance(expr, exp.Date) and isinstance(expr.this, exp.Column):
        bounds = _date_wrapper_range_bounds(literal_text)
        column = expr.this
    if fmt is not None and column is not None:
        bounds = _strftime_range_bounds(str(fmt.this), literal_text)
    if column is None or bounds is None:
        return None
    return {
        "column": column.copy(),
        "range_start": bounds[0],
        "range_end": bounds[1],
    }


def _extract_month_bucket_column(expr: exp.Expression) -> exp.Column | None:
    if isinstance(expr, exp.TimeToStr):
        format_arg = expr.args.get("format")
        if not isinstance(format_arg, exp.Literal) or str(format_arg.this) != "%m":
            return None
        source = expr.this
        source_inner = source.this if hasattr(source, "this") else source
        if isinstance(source_inner, exp.Column):
            return source_inner.copy()
        if isinstance(source, exp.Column):
            return source.copy()
    return None


def _parse_month_literals(values: list[str]) -> list[int] | None:
    months: list[int] = []
    for value in values:
        if len(value) != 2 or not value.isdigit():
            return None
        month = int(value)
        if not 1 <= month <= 12:
            return None
        months.append(month)
    return sorted(set(months))


def _infer_fixed_year_for_column(scope_expr: exp.Expression, column: exp.Column) -> int | None:
    extracted_year = _infer_fixed_year_from_extracted_predicates(scope_expr, column)
    if extracted_year is not None:
        return extracted_year
    return _infer_fixed_year_from_raw_range(scope_expr, column)


def _infer_fixed_year_from_extracted_predicates(
    scope_expr: exp.Expression,
    column: exp.Column,
) -> int | None:
    target = _normalize_sql(column.sql(dialect="sqlite"))
    for node in scope_expr.walk():
        if isinstance(node, exp.EQ) and isinstance(node.right, exp.Literal):
            extracted = _extract_year_bucket_column(node.left)
            if extracted and _normalize_sql(extracted.sql(dialect="sqlite")) == target:
                year_text = str(node.right.this)
                if len(year_text) == 4 and year_text.isdigit():
                    return int(year_text)
        elif isinstance(node, exp.Between):
            low = node.args.get("low")
            high = node.args.get("high")
            extracted = _extract_year_bucket_column(node.this)
            if (
                extracted
                and _normalize_sql(extracted.sql(dialect="sqlite")) == target
                and isinstance(low, exp.Literal)
                and isinstance(high, exp.Literal)
            ):
                low_text = str(low.this)
                high_text = str(high.this)
                if (
                    len(low_text) == 4
                    and len(high_text) == 4
                    and low_text.isdigit()
                    and high_text.isdigit()
                    and low_text == high_text
                ):
                    return int(low_text)
    return None


def _extract_year_bucket_column(expr: exp.Expression) -> exp.Column | None:
    if isinstance(expr, exp.Cast):
        expr = expr.this
    if isinstance(expr, exp.TimeToStr):
        format_arg = expr.args.get("format")
        if not isinstance(format_arg, exp.Literal) or str(format_arg.this) != "%Y":
            return None
        source = expr.this
        source_inner = source.this if hasattr(source, "this") else source
        if isinstance(source_inner, exp.Column):
            return source_inner.copy()
        if isinstance(source, exp.Column):
            return source.copy()
    return None


def _infer_fixed_year_from_raw_range(scope_expr: exp.Expression, column: exp.Column) -> int | None:
    target = _normalize_sql(column.sql(dialect="sqlite"))
    lower_year: int | None = None
    upper_year: int | None = None
    between_year: int | None = None
    for node in scope_expr.walk():
        if isinstance(node, (exp.GTE, exp.GT, exp.LT, exp.LTE)):
            left = getattr(node, "left", None)
            right = getattr(node, "right", None)
            if not isinstance(left, exp.Column) or not isinstance(right, exp.Literal):
                continue
            if _normalize_sql(left.sql(dialect="sqlite")) != target:
                continue
            literal_text = str(right.this)
            if isinstance(node, (exp.GTE, exp.GT)) and literal_text.endswith("-01-01"):
                year = _parse_year_start_text(literal_text)
                if year is not None:
                    lower_year = year
            elif isinstance(node, (exp.LT, exp.LTE)) and literal_text.endswith("-01-01"):
                year = _parse_year_start_text(literal_text)
                if year is not None:
                    upper_year = year - 1
        elif isinstance(node, exp.Between):
            low = node.args.get("low")
            high = node.args.get("high")
            if (
                isinstance(node.this, exp.Column)
                and _normalize_sql(node.this.sql(dialect="sqlite")) == target
                and isinstance(low, exp.Literal)
                and isinstance(high, exp.Literal)
            ):
                low_year = _parse_year_start_text(str(low.this))
                high_year = _parse_year_end_text(str(high.this))
                if low_year is not None and high_year is not None and low_year == high_year:
                    between_year = low_year
    if between_year is not None:
        return between_year
    if lower_year is not None and upper_year is not None and lower_year == upper_year:
        return lower_year
    return None


def _parse_year_start_text(text: str) -> int | None:
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        return None
    if parsed.month == 1 and parsed.day == 1:
        return parsed.year
    return None


def _parse_year_end_text(text: str) -> int | None:
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        return None
    if parsed.month == 12 and parsed.day == 31:
        return parsed.year
    return None


def _or_month_ranges(
    *,
    column: exp.Column,
    year: int,
    months: list[int],
) -> exp.Expression:
    expressions = [
        _range_predicate_for_bounds(
            column=column,
            start=f"{year:04d}-{month:02d}-01",
            end=(
                f"{year + 1:04d}-01-01"
                if month == 12
                else f"{year:04d}-{month + 1:02d}-01"
            ),
        )
        for month in months
    ]
    result = expressions[0]
    for expr in expressions[1:]:
        result = exp.or_(result, expr)
    return result


def _extract_synthesized_yearmonth_source(
    expr: exp.Expression,
    *,
    format_text: str | None,
    literal_text: str,
) -> dict[str, Any] | None:
    if format_text not in {"%Y"}:
        return None
    shape = _synthesized_yearmonth_concat_shape(expr)
    if shape is None:
        return None
    if len(literal_text) != 4 or not literal_text.isdigit():
        return None
    year = int(literal_text)
    return {
        "column": shape["column"].copy(),
        "range_start": f"{year:04d}01",
        "range_end": f"{year + 1:04d}01",
    }


def _synthesized_yearmonth_concat_shape(expr: exp.Expression) -> dict[str, Any] | None:
    parts = _flatten_dpipe(expr)
    if len(parts) != 4:
        return None
    year_part, dash1, month_part, day_suffix = parts
    if not (
        isinstance(dash1, exp.Literal)
        and str(dash1.this) == "-"
        and isinstance(day_suffix, exp.Literal)
        and str(day_suffix.this) == "-01"
    ):
        return None
    year_sub = _prefix_substring_shape(year_part, start="1", length="4")
    month_sub = _prefix_substring_shape(month_part, start="5", length="2")
    if year_sub is None or month_sub is None:
        return None
    if _normalize_sql(year_sub.sql(dialect="sqlite")) != _normalize_sql(
        month_sub.sql(dialect="sqlite")
    ):
        return None
    return {"column": year_sub.copy()}


def _flatten_dpipe(expr: exp.Expression) -> list[exp.Expression]:
    if isinstance(expr, exp.DPipe):
        return _flatten_dpipe(expr.left) + _flatten_dpipe(expr.right)
    return [expr]


def _prefix_substring_shape(
    expr: exp.Expression,
    *,
    start: str,
    length: str,
) -> exp.Column | None:
    if not isinstance(expr, exp.Substring):
        return None
    start_expr = expr.args.get("start")
    length_expr = expr.args.get("length")
    if not (
        isinstance(expr.this, exp.Column)
        and isinstance(start_expr, exp.Literal)
        and isinstance(length_expr, exp.Literal)
        and str(start_expr.this) == start
        and str(length_expr.this) == length
    ):
        return None
    return expr.this.copy()


def _comparison_operator_text(node: exp.Expression) -> str | None:
    if isinstance(node, exp.EQ):
        return "="
    if isinstance(node, exp.LT):
        return "<"
    if isinstance(node, exp.LTE):
        return "<="
    if isinstance(node, exp.GT):
        return ">"
    if isinstance(node, exp.GTE):
        return ">="
    return None


def _comparison_replacement_for_bucket(
    *,
    column: exp.Column,
    operator: str,
    range_start: str,
    range_end: str,
) -> exp.Expression | None:
    if operator == "=":
        return _range_predicate_for_bounds(column=column, start=range_start, end=range_end)
    if operator == "<":
        return _single_bound_predicate(column=column, upper=range_start)
    if operator == "<=":
        return _single_bound_predicate(column=column, upper=range_end)
    if operator == ">":
        return _single_bound_predicate(column=column, lower=range_end)
    if operator == ">=":
        return _single_bound_predicate(column=column, lower=range_start)
    return None


def _year_difference_predicate_shape(
    *,
    expr: exp.Expression,
    operator: str,
    literal_text: str,
) -> dict[str, Any] | None:
    if not literal_text.lstrip("-").isdigit() or not isinstance(expr, exp.Sub):
        return None
    current_year = _current_year_now_expr(expr.left)
    column = _year_cast_column_expr(expr.right)
    if current_year is None or column is None:
        return None
    replacement = _year_difference_replacement(
        column=column,
        operator=operator,
        years=int(literal_text),
    )
    if replacement is None:
        return None
    return {"column": column.copy(), "replacement": replacement}


def _current_year_now_expr(expr: exp.Expression) -> str | None:
    if not isinstance(expr, exp.Cast):
        return None
    inner = expr.this
    if not isinstance(inner, exp.TimeToStr):
        return None
    format_arg = inner.args.get("format")
    if not isinstance(format_arg, exp.Literal) or str(format_arg.this) != "%Y":
        return None
    source = inner.this
    source_inner = source.this if hasattr(source, "this") else source
    if not isinstance(source_inner, exp.Literal):
        return None
    if str(source_inner.this).lower() != "now":
        return None
    return "now"


def _year_cast_column_expr(expr: exp.Expression) -> exp.Column | None:
    if not isinstance(expr, exp.Cast):
        return None
    inner = expr.this
    if not isinstance(inner, exp.TimeToStr):
        return None
    format_arg = inner.args.get("format")
    if not isinstance(format_arg, exp.Literal) or str(format_arg.this) != "%Y":
        return None
    source = inner.this
    source_inner = source.this if hasattr(source, "this") else source
    if isinstance(source_inner, exp.Column):
        return source_inner.copy()
    if isinstance(source, exp.Column):
        return source.copy()
    return None


def _year_difference_replacement(
    *,
    column: exp.Column,
    operator: str,
    years: int,
) -> exp.Expression | None:
    if years < 0:
        return None
    if operator == ">=":
        return _single_bound_predicate(column=column, upper=_year_start_expr(-(years - 1)))
    if operator == ">":
        return _single_bound_predicate(column=column, upper=_year_start_expr(-years))
    if operator == "<=":
        return _single_bound_predicate(column=column, lower=_year_start_expr(-years))
    if operator == "<":
        return _single_bound_predicate(column=column, lower=_year_start_expr(-(years - 1)))
    if operator == "=":
        return _single_bound_predicate(
            column=column,
            lower=_year_start_expr(-years),
            upper=_year_start_expr(-(years - 1)),
        )
    return None


def _year_start_expr(year_delta: int) -> exp.Expression:
    modifiers = [exp.Literal.string("start of year")]
    if year_delta != 0:
        modifiers.append(exp.Literal.string(f"{year_delta} years"))
    return exp.Date(this=exp.Literal.string("now"), expressions=modifiers)


def _date_wrapper_range_bounds(literal_text: str) -> tuple[str, str] | None:
    try:
        parsed = date.fromisoformat(literal_text)
    except ValueError:
        return None
    next_day = parsed + timedelta(days=1)
    return (parsed.isoformat(), next_day.isoformat())


def _conditionalized_aggregate_expression(
    aggregate_expr: exp.Expression,
    predicate: exp.Expression | None,
) -> exp.Expression | None:
    if isinstance(aggregate_expr, exp.Count):
        if predicate is None:
            return aggregate_expr.copy()
        if isinstance(aggregate_expr.this, exp.Star):
            return exp.Sum(
                this=exp.Case(
                    ifs=[exp.If(this=predicate, true=exp.Literal.number(1))],
                    default=exp.Literal.number(0),
                )
            )
        if _is_safe_conditional_aggregate_argument(aggregate_expr.this):
            return exp.Count(
                this=exp.Case(
                    ifs=[exp.If(this=predicate, true=aggregate_expr.this.copy())],
                    default=exp.Null(),
                )
            )
        return None
    if isinstance(aggregate_expr, exp.Sum):
        if not _is_safe_conditional_aggregate_argument(aggregate_expr.this):
            return None
        if predicate is None:
            return aggregate_expr.copy()
        return exp.Sum(
            this=exp.Case(
                ifs=[exp.If(this=predicate, true=aggregate_expr.this.copy())],
                default=exp.Null(),
            )
        )
    if isinstance(aggregate_expr, (exp.Avg, exp.Min, exp.Max)):
        if not _is_safe_conditional_aggregate_argument(aggregate_expr.this):
            return None
        if predicate is None:
            return aggregate_expr.copy()
        rewritten_arg = exp.Case(
            ifs=[exp.If(this=predicate, true=aggregate_expr.this.copy())],
            default=exp.Null(),
        )
        return aggregate_expr.__class__(this=rewritten_arg)
    return None


def _is_safe_conditional_aggregate_argument(expression: exp.Expression | None) -> bool:
    if expression is None:
        return False
    for node in expression.walk():
        if isinstance(node, (exp.AggFunc, exp.Window, exp.Subquery, exp.Exists)):
            return False
    return True


def _in_expression_has_only_literals(predicate: exp.In) -> bool:
    expressions = predicate.args.get("expressions") or []
    return bool(expressions) and all(isinstance(item, exp.Literal) for item in expressions)


def _tuple_has_only_literals(tuple_expr: exp.Tuple) -> bool:
    return bool(tuple_expr.expressions) and all(
        isinstance(item, exp.Literal) for item in tuple_expr.expressions
    )


def _bridge_predicate_replacement(
    *,
    predicate: exp.Expression,
    left: exp.Table,
    bridge: exp.Table,
    right: exp.Table,
    joins: list[exp.Join],
    direct_edge: JoinEdge,
) -> exp.Expression | None:
    bridge_aliases = {bridge.alias_or_name, bridge.name}
    bridge_col: exp.Column | None = None
    literal_side: exp.Expression | None = None
    if isinstance(predicate, (exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
        left_expr = predicate.left
        right_expr = predicate.right
        if isinstance(left_expr, exp.Column) and left_expr.table in bridge_aliases and isinstance(right_expr, exp.Literal):
            bridge_col, literal_side = left_expr, right_expr
        elif isinstance(right_expr, exp.Column) and right_expr.table in bridge_aliases and isinstance(left_expr, exp.Literal):
            bridge_col, literal_side = right_expr, left_expr
    if bridge_col is None or literal_side is None:
        if isinstance(predicate, exp.In):
            if isinstance(predicate.this, exp.Column) and predicate.this.table in bridge_aliases:
                if _in_expression_has_only_literals(predicate):
                    bridge_col = predicate.this
                    literal_side = exp.Tuple(
                        expressions=[expr.copy() for expr in predicate.args.get("expressions") or []]
                    )
        elif isinstance(predicate, (exp.Is, exp.NullSafeEQ)):
            left_expr = predicate.left
            right_expr = predicate.right
            if isinstance(left_expr, exp.Column) and left_expr.table in bridge_aliases and isinstance(right_expr, exp.Null):
                bridge_col = left_expr
                literal_side = right_expr
        elif isinstance(predicate, exp.Not) and isinstance(predicate.this, exp.Is):
            inner = predicate.this
            if isinstance(inner.left, exp.Column) and inner.left.table in bridge_aliases and isinstance(inner.right, exp.Null):
                bridge_col = inner.left
                literal_side = exp.Not(this=exp.Null())
        if bridge_col is None or literal_side is None:
            return None
    endpoint_col = _bridge_filter_replacement_column(
        bridge_column=bridge_col.name,
        left=left,
        bridge=bridge,
        right=right,
        joins=joins,
        direct_edge=direct_edge,
    )
    if endpoint_col is None:
        return None
    if isinstance(predicate, exp.In):
        if not isinstance(literal_side, exp.Tuple):
            return None
        return exp.In(this=endpoint_col, expressions=[expr.copy() for expr in literal_side.expressions])
    if isinstance(predicate, exp.Is):
        return exp.Is(this=endpoint_col, expression=exp.Null())
    if isinstance(predicate, exp.Not) and isinstance(predicate.this, exp.Is):
        return exp.Not(this=exp.Is(this=endpoint_col, expression=exp.Null()))
    return predicate.__class__(this=endpoint_col, expression=literal_side.copy())


def _bridge_filter_replacement_column(
    *,
    bridge_column: str,
    left: exp.Table,
    bridge: exp.Table,
    right: exp.Table,
    joins: list[exp.Join],
    direct_edge: JoinEdge,
) -> exp.Column | None:
    left_aliases = {left.alias_or_name, left.name}
    right_aliases = {right.alias_or_name, right.name}
    bridge_aliases = {bridge.alias_or_name, bridge.name}
    mapped_to_left: str | None = None
    mapped_to_right: str | None = None
    for join in joins:
        on_clause = join.args.get("on")
        if on_clause is None:
            continue
        for predicate in _flatten_and_conditions(on_clause):
            if not isinstance(predicate, exp.EQ):
                continue
            if not isinstance(predicate.left, exp.Column) or not isinstance(predicate.right, exp.Column):
                continue
            pairs = ((predicate.left, predicate.right), (predicate.right, predicate.left))
            for bridge_side, other_side in pairs:
                if bridge_side.table not in bridge_aliases or bridge_side.name != bridge_column:
                    continue
                if other_side.table in left_aliases:
                    mapped_to_left = other_side.name
                if other_side.table in right_aliases:
                    mapped_to_right = other_side.name
    if direct_edge.source_table == left.name and direct_edge.source_column == (mapped_to_left or direct_edge.source_column):
        if direct_edge.target_table == right.name:
            return exp.column(direct_edge.target_column, table=right.alias_or_name)
    if direct_edge.target_table == left.name and direct_edge.target_column == (mapped_to_left or direct_edge.target_column):
        if direct_edge.source_table == right.name:
            return exp.column(direct_edge.source_column, table=right.alias_or_name)
    if direct_edge.source_table == right.name and direct_edge.source_column == (mapped_to_right or direct_edge.source_column):
        if direct_edge.target_table == left.name:
            return exp.column(direct_edge.target_column, table=left.alias_or_name)
    if direct_edge.target_table == right.name and direct_edge.target_column == (mapped_to_right or direct_edge.target_column):
        if direct_edge.source_table == left.name:
            return exp.column(direct_edge.source_column, table=left.alias_or_name)
    return None


def _strftime_range_bounds(format_text: str, literal_text: str) -> tuple[str, str] | None:
    if format_text == "%Y" and len(literal_text) == 4 and literal_text.isdigit():
        year = int(literal_text)
        return (
            f"{year:04d}-01-01",
            f"{year + 1:04d}-01-01",
        )
    if format_text == "%Y-%m" and _is_iso_month_text(literal_text):
        year_text, month_text = literal_text.split("-", 1)
        year = int(year_text)
        month = int(month_text)
        next_year = year + 1 if month == 12 else year
        next_month = 1 if month == 12 else month + 1
        return (
            f"{year:04d}-{month:02d}",
            f"{next_year:04d}-{next_month:02d}",
        )
    if format_text == "%Y-%m-%d":
        return _date_wrapper_range_bounds(literal_text)
    return None


def _substring_range_bounds(length: int, literal_text: str) -> tuple[str, str] | None:
    if length == 4 and len(literal_text) == 4 and literal_text.isdigit():
        year = int(literal_text)
        return (f"{year:04d}", f"{year + 1:04d}")
    if length == 7 and _is_iso_month_text(literal_text):
        year_text, month_text = literal_text.split("-", 1)
        year = int(year_text)
        month = int(month_text)
        next_year = year + 1 if month == 12 else year
        next_month = 1 if month == 12 else month + 1
        return (
            f"{year:04d}-{month:02d}",
            f"{next_year:04d}-{next_month:02d}",
        )
    if length == 6 and len(literal_text) == 6 and literal_text.isdigit():
        year = int(literal_text[:4])
        month = int(literal_text[4:])
        if not 1 <= month <= 12:
            return None
        next_year = year + 1 if month == 12 else year
        next_month = 1 if month == 12 else month + 1
        return (
            f"{year:04d}{month:02d}",
            f"{next_year:04d}{next_month:02d}",
        )
    if length == 10:
        return _date_wrapper_range_bounds(literal_text)
    if length == 8 and len(literal_text) == 8 and literal_text.isdigit():
        iso_text = f"{literal_text[:4]}-{literal_text[4:6]}-{literal_text[6:8]}"
        bounds = _date_wrapper_range_bounds(iso_text)
        if bounds is None:
            return None
        return (literal_text, bounds[1].replace("-", ""))
    return None


def _is_iso_month_text(value: str) -> bool:
    if len(value) != 7 or value[4] != "-":
        return False
    year_text, month_text = value.split("-", 1)
    return year_text.isdigit() and month_text.isdigit() and 1 <= int(month_text) <= 12


def _is_safe_like_prefix_pattern(pattern: str) -> bool:
    if not pattern.endswith("%") or pattern.count("%") != 1:
        return False
    if "_" in pattern:
        return False
    prefix = pattern[:-1]
    if not prefix:
        return False
    return all(ord(ch) < 128 and (ch.isalnum() or ch in "-_./:") for ch in prefix)


def _next_ascii_prefix(prefix: str) -> str | None:
    chars = list(prefix)
    for index in range(len(chars) - 1, -1, -1):
        codepoint = ord(chars[index])
        if codepoint >= 126:
            continue
        chars[index] = chr(codepoint + 1)
        return "".join(chars[: index + 1])
    return None


def _rewrite_template_join_counts(template: str) -> tuple[int | None, int | None]:
    if "->" not in template:
        return None, None
    source, target = template.split("->", 1)
    return source.lower().count(" join "), target.lower().count(" join ")


def _is_hist_template_strategy(strategy: RetrievedStrategy) -> bool:
    text = " ".join(
        [
            strategy.rule_name,
            " ".join(strategy.example_cases),
            " ".join(strategy.applicable_when),
        ]
    ).lower()
    return "hist template" in text or "rewrite the source sql template into the target sql template" in text


def _strategy_source_type(strategy: RetrievedStrategy) -> str:
    explicit = str(getattr(strategy, "source_type", "unknown") or "unknown").strip().lower()
    if explicit in {"operator", "expert", "hist"}:
        return explicit
    if str(strategy.rule_id).startswith("builtin_"):
        return "operator"
    rule_text = " ".join([strategy.rule_id, strategy.rule_name]).lower()
    if "expert" in rule_text or "专家" in rule_text:
        return "expert"
    if _is_hist_template_strategy(strategy):
        return "hist"
    return "expert"


def _source_strategy_priority_score(sql: str, strategy: RetrievedStrategy) -> float:
    base = float(strategy.confidence)
    text = " ".join(strategy.applicable_when).lower()
    if _strategy_source_type(strategy) == "hist":
        base += 0.15 * _hist_template_shape_score(
            _template_source_shape(strategy.rewrite_template),
            _sql_shape(sql),
        )
    if text and sql:
        sql_lower = sql.lower()
        matched_terms = sum(1 for token in text.split(",") if token.strip() and token.strip() in sql_lower)
        base += min(0.1, matched_terms * 0.02)
    return base


def _template_source_shape(template: str) -> dict[str, Any]:
    source = template.split("->", 1)[0].lower() if "->" in template else template.lower()
    select_expr_count = 0
    if "select " in source and " from " in source:
        projection = source.split("select ", 1)[1].split(" from ", 1)[0]
        select_expr_count = projection.count(",") + 1 if projection.strip() else 0
    return {
        "join_count": source.count(" join "),
        "has_where": " where " in source,
        "has_order_by": " order by " in source,
        "has_limit": " limit " in source,
        "has_group_by": " group by " in source,
        "has_distinct": "select distinct" in source,
        "has_literal_filter": "literal" in source and " where " in source,
        "has_cast": "cast(" in source,
        "has_division": " / " in source,
        "has_in_subquery": " in (select" in source,
        "has_scalar_subquery": "= (select" in source,
        "select_expression_count": select_expr_count,
    }


def _sql_shape(sql: str) -> dict[str, Any]:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return {
            "join_count": 0,
            "has_where": False,
            "has_order_by": False,
            "has_limit": False,
            "has_group_by": False,
            "has_distinct": False,
            "has_literal_filter": False,
            "has_cast": False,
            "has_division": False,
            "has_in_subquery": False,
            "has_scalar_subquery": False,
            "select_expression_count": 0,
        }
    select = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
    has_literal_filter = False
    where_clause = ast.find(exp.Where)
    if where_clause is not None and where_clause.this is not None:
        for node in where_clause.this.walk():
            if isinstance(node, exp.EQ):
                left = node.left
                right = node.right
                if (
                    isinstance(left, exp.Column)
                    and isinstance(right, exp.Literal)
                    or isinstance(right, exp.Column)
                    and isinstance(left, exp.Literal)
                ):
                    has_literal_filter = True
                    break
    return {
        "join_count": sum(1 for _ in ast.find_all(exp.Join)),
        "has_where": where_clause is not None,
        "has_order_by": ast.find(exp.Order) is not None,
        "has_limit": ast.find(exp.Limit) is not None,
        "has_group_by": ast.find(exp.Group) is not None,
        "has_distinct": bool(select and select.args.get("distinct")),
        "has_literal_filter": has_literal_filter,
        "has_cast": any(isinstance(node, exp.Cast) for node in ast.walk()),
        "has_division": any(isinstance(node, exp.Div) for node in ast.walk()),
        "has_in_subquery": any(
            isinstance(node, exp.In) and isinstance(node.args.get("query"), exp.Subquery)
            for node in ast.walk()
        ),
        "has_scalar_subquery": any(
            isinstance(node, exp.EQ)
            and (
                isinstance(node.left, exp.Subquery)
                or isinstance(node.right, exp.Subquery)
            )
            for node in ast.walk()
        ),
        "select_expression_count": len(select.expressions) if select is not None else 0,
    }


def _hist_template_shape_score(template_shape: dict[str, Any], sql_shape: dict[str, Any]) -> float:
    score = 0.0
    checks = 0
    for key in (
        "has_where",
        "has_order_by",
        "has_limit",
        "has_group_by",
        "has_distinct",
        "has_literal_filter",
        "has_cast",
        "has_division",
        "has_in_subquery",
        "has_scalar_subquery",
    ):
        if template_shape[key]:
            checks += 1
            if sql_shape[key]:
                score += 1.0
    checks += 1
    join_gap = abs(int(template_shape["join_count"]) - int(sql_shape["join_count"]))
    score += max(0.0, 1.0 - 0.5 * join_gap)
    checks += 1
    expr_gap = max(0, int(template_shape["select_expression_count"]) - int(sql_shape["select_expression_count"]))
    score += 1.0 if expr_gap == 0 else 0.0
    return score / max(1, checks)


def _generic_retrieval_rerank_score(
    sql: str,
    hint: RewriteHint,
    strategy: RetrievedStrategy,
    applicability: ApplicabilityResult,
) -> float:
    if _strategy_source_type(strategy) == "operator":
        return 0.0
    score = 0.0
    sql_shape = _sql_shape(sql)
    hint_text = " ".join(
        [hint.strategy, hint.target_fragment or "", hint.expected_effect or ""]
    ).lower()
    strategy_text = " ".join(
        [
            strategy.rule_id,
            strategy.rule_name,
            strategy.rewrite_template,
            " ".join(strategy.applicable_when),
            " ".join(strategy.example_cases),
        ]
    ).lower()

    if "order by" in hint_text and sql_shape["has_order_by"]:
        score += 0.15
    if "limit" in hint_text and sql_shape["has_limit"]:
        score += 0.1
    if "selective predicates" in hint_text and sql_shape["has_where"]:
        score += 0.15
    if "same-table" in strategy_text and _has_same_table_literal_lookup_join(sql):
        score += 0.2
    if "scalar max" in strategy_text and _has_scalar_maxmin_subquery(sql):
        score += 0.2
    if "distinct top-1" in strategy_text and distinct_top1_to_grouped_extrema_shape(sql):
        score += 0.42
    if "grouped max top-1" in strategy_text and grouped_max_top1_before_join_shape(sql):
        score += 0.32
    if "filter joined dimension before fact top-1" in strategy_text and filter_dimension_before_top1_shape(sql):
        score += 0.55
    if "re-anchor join on the selective driver table" in strategy_text and reanchor_join_driver_shape(sql):
        score += 0.62
    if "normalize scalar extrema filter into top-1 anchor then final lookup tail" in strategy_text and scalar_extrema_anchor_then_lookup_tail_shape(sql):
        score += 0.6
    if "canonicalize distinct extrema filter into grouped having extrema" in strategy_text and distinct_extrema_to_grouped_having_shape(sql):
        score += 0.56
    if "resolve top-1 anchor key before final lookup tail" in strategy_text and top1_anchor_then_lookup_tail_shape(sql):
        score += 0.58
    if "top-k before downstream joins" in strategy_text and topk_before_join_shape(sql):
        score += 0.28
    if _is_hist_template_strategy(strategy):
        score += float(applicability.required_fragments.get("template_shape_score") or 0.0) * 0.5
    if "push" in strategy_text and "filter" in strategy_text and sql_shape["has_where"]:
        score += 0.08
    if "top-k" in strategy_text or ("order by" in strategy_text and "limit" in strategy_text):
        if sql_shape["has_order_by"] and sql_shape["has_limit"]:
            score += 0.08
    return score


def _has_same_table_literal_lookup_join(sql: str) -> bool:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return False

    alias_to_table: dict[str, str] = {}
    aliases_by_table: dict[str, set[str]] = {}
    for table in ast.find_all(exp.Table):
        table_name = table.name
        alias = table.alias_or_name
        if not table_name or not alias:
            continue
        alias_to_table[alias] = table_name
        aliases_by_table.setdefault(table_name, set()).add(alias)

    if not any(len(aliases) >= 2 for aliases in aliases_by_table.values()):
        return False

    where_clause = ast.find(exp.Where)
    if where_clause is None or where_clause.this is None:
        return False

    literal_filtered_aliases: set[str] = set()
    for node in where_clause.this.walk():
        if not isinstance(node, exp.EQ):
            continue
        left = node.left
        right = node.right
        for candidate, other in ((left, right), (right, left)):
            if isinstance(candidate, exp.Column) and isinstance(other, exp.Literal):
                if candidate.table:
                    literal_filtered_aliases.add(candidate.table)

    if not literal_filtered_aliases:
        return False

    for join in ast.find_all(exp.Join):
        condition = join.args.get("on")
        if condition is None:
            continue
        for node in condition.walk():
            if not isinstance(node, exp.EQ):
                continue
            left = node.left
            right = node.right
            if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                continue
            if not left.table or not right.table or not left.name or not right.name:
                continue
            left_table = alias_to_table.get(left.table)
            right_table = alias_to_table.get(right.table)
            if left_table is None or right_table is None or left_table != right_table:
                continue
            if left.name != right.name:
                continue
            if left.table in literal_filtered_aliases or right.table in literal_filtered_aliases:
                return True
    return False


def _extract_table_refs(sql: str) -> tuple[set[str], dict[str, str]]:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return set(), {}
    tables: set[str] = set()
    alias_for_table: dict[str, str] = {}
    for table in ast.find_all(exp.Table):
        name = table.name
        if not name:
            continue
        tables.add(name)
        alias_for_table[name] = table.alias or name
    return tables, alias_for_table


def _blueprint_violations(sql: str, blueprint: VerifiedContextBlueprint) -> list[str]:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception as exc:
        return [f"Candidate SQL failed to parse: {exc}"]
    allowed_tables = set(blueprint.selected_tables)
    allowed_columns = {(c.table_name, c.column_name) for c in blueprint.selected_columns}
    allowed_names = {c.column_name for c in blueprint.selected_columns}
    allowed_tables_by_column: dict[str, set[str]] = {}
    for column in blueprint.selected_columns:
        allowed_tables_by_column.setdefault(column.column_name, set()).add(column.table_name)
    # Collect SELECT aliases so they don't trigger false-positive violations
    select_aliases: set[str] = set()
    subquery_output_aliases: dict[str, set[str]] = {}
    for select in ast.find_all(exp.Select):
        for expression in select.expressions:
            if isinstance(expression, exp.Alias):
                alias = expression.alias
                if isinstance(alias, str):
                    select_aliases.add(alias)
    tables: set[str] = set()
    alias_to_table: dict[str, str] = {}
    for table in ast.find_all(exp.Table):
        name = table.name
        if not name:
            continue
        tables.add(name)
        alias_to_table[name] = name
        if table.alias:
            alias_to_table[table.alias] = name
    for subquery in ast.find_all(exp.Subquery):
        alias = subquery.alias
        if not alias:
            continue
        output_aliases: set[str] = set()
        inner_select = subquery.this if isinstance(subquery.this, exp.Select) else subquery.find(exp.Select)
        if inner_select is not None:
            for expression in inner_select.expressions:
                if isinstance(expression, exp.Alias) and isinstance(expression.alias, str):
                    output_aliases.add(expression.alias)
        if output_aliases:
            subquery_output_aliases[alias] = output_aliases
        source_tables: set[str] = set()
        for column in subquery.find_all(exp.Column):
            if not column.table:
                continue
            resolved = alias_to_table.get(column.table, column.table)
            if resolved in allowed_tables:
                source_tables.add(resolved)
        if len(source_tables) == 1:
            alias_to_table[alias] = next(iter(source_tables))
    violations: list[str] = []
    for table in sorted(tables):
        if table not in allowed_tables:
            violations.append(f"Table '{table}' is outside the Blueprint.")
    for column in ast.find_all(exp.Column):
        if isinstance(column.this, exp.Star):
            violations.append("Candidate SQL still contains SELECT *.")
            continue
        name = column.name
        table = column.table
        if not name:
            continue
        if table:
            resolved = alias_to_table.get(table, table)
            if (resolved, name) in allowed_columns:
                continue
            if name in subquery_output_aliases.get(table, set()):
                continue
            # Be permissive when the qualifier cannot be resolved but the
            # column name maps to exactly one allowed Blueprint table.
            candidate_tables = allowed_tables_by_column.get(name, set())
            if table not in alias_to_table and len(candidate_tables) == 1:
                continue
            if (resolved, name) not in allowed_columns:
                violations.append(f"Column '{table}.{name}' is outside the Blueprint.")
        elif name not in allowed_names and name not in select_aliases:
            violations.append(f"Column '{name}' is outside the Blueprint.")
    return _unique(violations)


def _normalize_sql(sql: str) -> str:
    try:
        return sqlglot.parse_one(sql, dialect="sqlite").sql(dialect="sqlite").lower()
    except Exception:
        return " ".join(sql.lower().split())


def _coerce_sql_version(value: Any) -> SQLVersion | None:
    if isinstance(value, SQLVersion):
        return value
    if not isinstance(value, dict):
        return None
    required = {
        "version_id",
        "parent_id",
        "sql",
        "source_agent",
        "rewrite_rule_ids",
        "explanation",
        "created_at",
    }
    if not required.issubset(value):
        return None
    return SQLVersion(
        version_id=str(value["version_id"]),
        parent_id=value["parent_id"],
        sql=str(value["sql"]),
        source_agent=str(value["source_agent"]),
        rewrite_rule_ids=list(value["rewrite_rule_ids"]),
        explanation=str(value["explanation"]),
        created_at=str(value["created_at"]),
    )


def _coerce_blueprint(value: Any) -> VerifiedContextBlueprint | None:
    if isinstance(value, VerifiedContextBlueprint):
        return value
    if not isinstance(value, dict):
        return None
    columns = [
        col
        if isinstance(col, ColumnRef)
        else ColumnRef(
            table_name=str(col["table_name"]),
            column_name=str(col["column_name"]),
            data_type=col.get("data_type"),
            comment=col.get("comment"),
        )
        for col in value.get("selected_columns", [])
    ]
    topology = value.get("join_topology") or {}
    if isinstance(topology, JoinGraph):
        join_topology = topology
    else:
        edges = [
            edge
            if isinstance(edge, JoinEdge)
            else JoinEdge(
                source_table=str(edge["source_table"]),
                source_column=str(edge["source_column"]),
                target_table=str(edge["target_table"]),
                target_column=str(edge["target_column"]),
                join_type=str(edge.get("join_type", "inner")),
            )
            for edge in topology.get("edges", [])
        ]
        join_topology = JoinGraph(
            tables=list(topology.get("tables", value.get("selected_tables", []))),
            edges=edges,
        )
    return VerifiedContextBlueprint(
        db_id=str(value["db_id"]),
        selected_tables=list(value.get("selected_tables", [])),
        selected_columns=columns,
        value_mappings=list(value.get("value_mappings", [])),
        join_topology=join_topology,
        predicate_hints=list(value.get("predicate_hints", [])),
        evidence_trace=list(value.get("evidence_trace", [])),
        confidence=float(value.get("confidence", 0.0)),
    )


def _coerce_bottleneck_report(value: Any) -> BottleneckReport | None:
    if isinstance(value, BottleneckReport):
        return value
    if not isinstance(value, dict):
        return None
    hints = [
        hint
        if isinstance(hint, RewriteHint)
        else RewriteHint(
            strategy=str(hint["strategy"]),
            target_fragment=hint.get("target_fragment"),
            expected_effect=str(hint.get("expected_effect", "")),
            risk=str(hint.get("risk", "medium")),
            requires_validation=bool(hint.get("requires_validation", True)),
            dbms_notes=hint.get("dbms_notes"),
        )
        for hint in value.get("rewrite_hints", [])
    ]
    return BottleneckReport(
        sql_version_id=str(value["sql_version_id"]),
        bottlenecks=list(value.get("bottlenecks", [])),
        cost_snapshot=dict(value.get("cost_snapshot", {})),
        risk_tags=list(value.get("risk_tags", [])),
        rewrite_hints=hints,
        explanation=str(value.get("explanation", "")),
    )


def _coerce_retrieved_strategy(value: Any) -> RetrievedStrategy:
    if isinstance(value, RetrievedStrategy):
        return value
    if not isinstance(value, dict):
        raise ValueError("Retrieved strategy must be a RetrievedStrategy or dict.")
    return RetrievedStrategy(
        rule_id=str(value["rule_id"]),
        rule_name=str(value["rule_name"]),
        applicable_when=list(value.get("applicable_when", [])),
        rewrite_template=str(value.get("rewrite_template", "")),
        risk_notes=list(value.get("risk_notes", [])),
        example_cases=list(value.get("example_cases", [])),
        confidence=float(value.get("confidence", 0.0)),
        source_type=str(value.get("source_type", "unknown")),
        families=list(value.get("families", [])) or None,
        hint_strategies=list(value.get("hint_strategies", [])) or None,
        operator_name=(
            str(value.get("operator_name"))
            if value.get("operator_name") is not None
            else None
        ),
        suppressed_by=list(value.get("suppressed_by", [])) or None,
        preflight_policy=(
            str(value.get("preflight_policy"))
            if value.get("preflight_policy") is not None
            else None
        ),
        preflight_failure_message=(
            str(value.get("preflight_failure_message"))
            if value.get("preflight_failure_message") is not None
            else None
        ),
    )


def _strategy_summary(strategy: RetrievedStrategy) -> dict:
    return {
        "rule_id": strategy.rule_id,
        "rule_name": strategy.rule_name,
        "confidence": strategy.confidence,
        "source_type": _strategy_source_type(strategy),
        "operator_name": strategy.operator_name,
        "families": list(strategy.families or []),
        "applicable_when": list(strategy.applicable_when),
        "rewrite_template": strategy.rewrite_template,
        "risk_notes": list(strategy.risk_notes),
    }


def _normalize_free_exploration_prompt_profile(profile: str | None) -> str:
    normalized = str(profile or "strong_llm").strip().lower()
    if normalized not in FREE_EXPLORATION_PROMPT_PROFILES:
        raise ValueError(
            "free_exploration_prompt_profile must be one of "
            f"{sorted(FREE_EXPLORATION_PROMPT_PROFILES)}."
        )
    return normalized


def infer_free_exploration_prompt_profile(model: str | None) -> str:
    normalized = str(model or "").strip().lower()
    if "coder" in normalized or "codestral" in normalized:
        return "coder"
    return "strong_llm"


def _trim_lines(text: str, *, limit: int) -> str:
    lines = [line.rstrip() for line in str(text or "").splitlines() if line.strip()]
    if len(lines) <= limit:
        return "\n".join(lines) if lines else "- <none>"
    return "\n".join([*lines[:limit], "- ..."])


def _top_free_exploration_priorities(physical_context: dict) -> list[str]:
    cost_snapshot = physical_context.get("cost_snapshot") or {}
    full_scan_tables = cost_snapshot.get("full_scan_tables") or []
    priorities: list[str] = []
    if full_scan_tables:
        priorities.append(f"shrink scan-driving tables first: {full_scan_tables}")
    priorities.extend(
        [
            "remove DISTINCT/GROUP BY fanout work only when duplicate semantics stay exact",
            "reduce rows entering ORDER BY/LIMIT or extreme-value work",
            "rewrite function-on-column or OR predicates only when equivalence is clear",
        ]
    )
    return priorities[:4]


def _free_exploration_focus_blocks(
    *,
    question: str,
    source_sql: str,
    analyser_notes_text: str,
    bottleneck_summary_text: str,
    failed_direction_text: str,
    physical_context: dict,
) -> list[str]:
    lower_sql = source_sql.lower()
    lower_question = question.lower()
    lower_notes = analyser_notes_text.lower()
    lower_bottlenecks = bottleneck_summary_text.lower()
    lower_failed = failed_direction_text.lower()
    combined = "\n".join(
        [lower_sql, lower_question, lower_notes, lower_bottlenecks, lower_failed]
    )
    blocks: list[str] = []
    if any(token in combined for token in ("order by", "limit", "max(", "min(", "topk", "highest", "lowest")):
        blocks.append(
            "Risk block: top-k / extreme-value semantics.\n"
            "- Primary opportunity: reduce rows before ORDER BY/LIMIT or extreme-value work.\n"
            "- Primary risk: preserve tie handling, ordering contract, and LIMIT semantics exactly.\n"
            "- Do not replace a tie-preserving shape with scalar MAX/MIN logic unless equivalence is obvious."
        )
    if any(token in combined for token in ("distinct", "group by", "fanout", "join fanout")):
        blocks.append(
            "Risk block: join fanout / duplicate semantics.\n"
            "- Primary opportunity: eliminate unnecessary fanout work before DISTINCT/GROUP BY.\n"
            "- Primary risk: preserve duplicate behavior and aggregation grain exactly.\n"
            "- If DISTINCT only repairs fanout, prefer a shape that avoids the fanout instead of widening DISTINCT work."
        )
    if any(token in combined for token in (" in (", "= (select", "scalar subquery", "same-table lookup", "cardinality", "multiple matching", "changed clause: where predicate")):
        blocks.append(
            "Risk block: scalar-vs-set lookup semantics.\n"
            "- Primary opportunity: keep lookup filtering selective without collapsing multi-row matches.\n"
            "- Primary risk: do not replace a set-producing lookup with scalar equality when multiple matches may exist.\n"
            "- Prefer IN/semi-join or another set-preserving form when uniqueness is not guaranteed."
        )
    if any(token in combined for token in (" or ", "function-on-column", "like(", "substr(", "coalesce(", "upper(", "lower(")):
        blocks.append(
            "Risk block: sargability rewrites.\n"
            "- Primary opportunity: convert function-on-column or OR-heavy predicates into more selective access paths.\n"
            "- Primary risk: preserve NULL behavior and predicate truth conditions exactly.\n"
            "- If the truth table is not clearly identical, do not apply the rewrite."
        )
    return blocks[:3]


def _build_free_exploration_prompt(
    *,
    profile: str,
    question: str,
    evidence: str,
    source_sql: str,
    available_indexes_text: str,
    analyser_notes_text: str,
    bottleneck_summary_text: str,
    operator_strategy_text: str,
    expert_strategy_text: str,
    hist_strategy_text: str,
    failed_direction_text: str,
    allowed_tables: str,
    allowed_columns: str,
    physical_context: dict,
) -> str:
    core_header = (
        "Role: You are a senior SQLite query optimizer.\n"
        "Task: Rewrite the given source SQL into an equivalent SQLite SELECT query that is "
        "most likely to reduce Scan Rows first, then other costs. Preserve the exact result "
        "semantics, including row set, duplicates, aggregation semantics, NULL behavior, "
        "ordering, LIMIT behavior, and top-k tie behavior.\n"
        f"Original question: {question}\n"
        f"Original evidence: {evidence}\n"
        f"Source SQL:\n{source_sql}\n"
        "Allowed schema scope: use only the listed allowed tables and allowed columns. "
        f"Allowed tables: {allowed_tables}. Allowed columns: {allowed_columns}.\n"
        "Hard schema constraints: do not introduce any table, column, CTE source, joined relation, "
        "or schema object outside the Blueprint. Stay in the same db_id/schema as the source SQL.\n"
    )
    priority_lines = "\n".join(
        f"- {item}" for item in _top_free_exploration_priorities(physical_context)
    )
    focus_blocks = "\n\n".join(
        _free_exploration_focus_blocks(
            question=question,
            source_sql=source_sql,
            analyser_notes_text=analyser_notes_text,
            bottleneck_summary_text=bottleneck_summary_text,
            failed_direction_text=failed_direction_text,
            physical_context=physical_context,
        )
    )
    supporting_context = (
        "Primary optimization priorities:\n"
        f"{priority_lines}\n"
        "Table row counts on Blueprint tables:\n"
        f"{_trim_lines(_available_table_row_summary(physical_context), limit=6)}\n"
        "Available indexes on Blueprint tables:\n"
        f"{_trim_lines(available_indexes_text, limit=6)}\n"
        "Deterministic operator opportunities:\n"
        f"{_trim_lines(operator_strategy_text, limit=4)}\n"
        "Expert rewrite priors:\n"
        f"{_trim_lines(expert_strategy_text, limit=4)}\n"
        "Historical similar rewrites (weak hints only):\n"
        f"{_trim_lines(hist_strategy_text, limit=4)}\n"
        "Analyser improvement notes (concise actionable hints derived from rewrite_hints):\n"
        f"{_trim_lines(analyser_notes_text, limit=4)}\n"
        "Bottleneck summary (broader diagnostic context from the analyser report):\n"
        f"{_trim_lines(bottleneck_summary_text, limit=4)}\n"
        "Previously failed free-exploration directions to avoid repeating:\n"
        f"{_trim_lines(failed_direction_text, limit=3)}\n"
        f"Index and FK context: {physical_context}.\n"
    )
    output_contract = (
        "Output contract: return exactly one fenced SQL block in the form "
        "```sql\\nSELECT ...\\n``` and nothing else. If no safe non-identical equivalent rewrite "
        "exists under these constraints, return exactly NO_OPTIMIZATION_SPACE."
    )
    if profile == "coder":
        return (
            core_header
            + ("Focused risk blocks:\n" + focus_blocks + "\n" if focus_blocks else "")
            + supporting_context
            + "Rules:\n"
            "- Preserve exact result semantics.\n"
            "- Treat analyser notes as the primary rewrite guidance, with Scan Rows reduction as the top optimization objective.\n"
            "- Prefer rewrites that reduce full scans, early row explosion, or post-filter work before any rewrite that only improves shape or formatting.\n"
            "- Use the focused risk blocks to decide which semantics must remain unchanged before attempting any rewrite.\n"
            "- Treat bottleneck summary as supporting context, not a license to make a broader rewrite.\n"
            "- Avoid every previously failed direction listed above; do not repeat the same SQL shape.\n"
            "- Do not change duplicate behavior, NULL behavior, aggregation grain, JOIN type, JOIN keys, "
            "ORDER BY, LIMIT, or tie handling unless equivalence is syntactically obvious.\n"
            "- If DISTINCT only repairs join fanout, prefer a shape that avoids the fanout instead of keeping a wide DISTINCT.\n"
            "- If ORDER BY/LIMIT or MAX/MIN semantics are present, prefer top-k on the smallest equivalent input.\n"
            "- Do not introduce cosmetic rewrites. If unsure, return NO_OPTIMIZATION_SPACE.\n"
            "Internal checklist: derive the exact result contract; identify one concrete bottleneck-backed "
            "rewrite; reject risky rewrites; verify every schema reference is allowed. Do not output this checklist.\n"
            + output_contract
        )
    return (
        core_header
        + ("Focused risk blocks:\n" + focus_blocks + "\n" if focus_blocks else "")
        + supporting_context
        + "Private decision process: first derive the exact result contract from the question, evidence, "
        "and source SQL; then inspect the primary optimization priorities and any focused risk blocks with "
        "Scan Rows reduction as the primary goal; then inspect analyser notes and bottleneck summary; then consider "
        "only safe equivalent rewrites such as predicate pushdown, sargable predicate rewrites, pre-aggregation, "
        "semi-join or join simplification, and top-k simplification that are likely to lower scanned rows; reject candidates that may change "
        "duplicates, NULL behavior, aggregation grain, join cardinality, ORDER BY, LIMIT, or tie "
        "semantics; avoid every previously failed direction listed above; finally verify the candidate "
        "uses only allowed schema elements. Do not output this reasoning.\n"
        + output_contract
    )


def _compact_analyser_improvement_notes(report: BottleneckReport) -> list[str]:
    notes: list[str] = []
    for hint in report.rewrite_hints:
        if hint.strategy == "no_rewrite":
            continue
        parts = [f"{hint.strategy}: {hint.expected_effect}"]
        if hint.target_fragment:
            parts.append(f"target={hint.target_fragment}")
        if hint.risk:
            parts.append(f"risk={hint.risk}")
        notes.append("; ".join(parts))
    if not notes:
        notes.extend(report.bottlenecks)
    if report.explanation and report.explanation not in notes:
        notes.append(f"analysis={report.explanation}")
    return notes[:5]


def _compact_bottleneck_summary_lines(report: BottleneckReport) -> list[str]:
    lines: list[str] = []
    for bottleneck in report.bottlenecks[:3]:
        lines.append(f"bottleneck={bottleneck}")
    if report.risk_tags:
        lines.append(f"risk_tags={', '.join(report.risk_tags[:6])}")
    if report.explanation:
        lines.append(f"analysis={report.explanation}")
    return lines[:5]


def _compact_strategy_lines(
    strategies: list[RetrievedStrategy],
) -> list[str]:
    lines: list[str] = []
    for strategy in strategies[:4]:
        parts = [
            strategy.rule_name or strategy.rule_id,
            f"confidence={strategy.confidence:.2f}",
        ]
        if strategy.rewrite_template:
            parts.append(f"rewrite={strategy.rewrite_template}")
        if strategy.risk_notes:
            parts.append(f"risks={'; '.join(strategy.risk_notes[:2])}")
        lines.append("; ".join(parts))
    return lines


def _compact_failed_direction_lines(failed_directions: list[dict]) -> list[str]:
    lines: list[str] = []
    for item in failed_directions[:4]:
        direction = str(item.get("direction") or "unknown direction")
        failure_reason = str(item.get("failure_reason") or "unknown failure")
        sql = str(item.get("sql") or "").strip()
        sql_summary = ""
        if sql:
            sql_summary = f"; failed_sql={_truncate_text(' '.join(sql.split()), 160)}"
        lines.append(f"direction={direction}; failure_reason={failure_reason}{sql_summary}")
    return lines


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _available_index_summary(physical_context: dict) -> list[str]:
    indexes_by_table = physical_context.get("indexes") or {}
    summaries: list[str] = []
    if not isinstance(indexes_by_table, dict):
        return summaries
    for table, indexes in indexes_by_table.items():
        if not isinstance(indexes, list):
            continue
        for index in indexes:
            if not isinstance(index, dict):
                continue
            columns = ", ".join(str(column) for column in index.get("columns") or [])
            if not columns:
                continue
            unique = " UNIQUE" if index.get("unique") else ""
            summaries.append(f"{table}.{index.get('name')}{unique}({columns})")
    return summaries


def _available_table_row_summary(physical_context: dict) -> str:
    row_counts = physical_context.get("table_row_counts") or {}
    if not isinstance(row_counts, dict) or not row_counts:
        return "- <none>"
    lines: list[str] = []
    for table in sorted(row_counts):
        row_count = row_counts.get(table)
        if isinstance(row_count, int):
            lines.append(f"- {table}: {row_count} rows")
    return "\n".join(lines) if lines else "- <none>"


def _is_no_optimization_response(response: str) -> bool:
    normalized = response.strip().strip("`").strip().upper()
    return normalized in {
        "NO_OPTIMIZATION_SPACE",
        "NO OPTIMIZATION SPACE",
        "NO_OPTIMIZATION",
        "NO OPTIMIZATION",
    }


def extract_sql_fenced_response(response: str) -> str:
    text = response.strip()
    if not text or _is_no_optimization_response(text):
        return text
    if text.startswith("```sql"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower().startswith("```sql"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def _blueprint_summary(blueprint: VerifiedContextBlueprint) -> dict:
    return {
        "db_id": blueprint.db_id,
        "selected_tables": list(blueprint.selected_tables),
        "selected_columns": [
            {
                "table_name": column.table_name,
                "column_name": column.column_name,
                "data_type": column.data_type,
                "comment": column.comment,
            }
            for column in blueprint.selected_columns
        ],
        "join_tables": list(blueprint.join_topology.tables),
        "join_edges": [
            {
                "source_table": edge.source_table,
                "source_column": edge.source_column,
                "target_table": edge.target_table,
                "target_column": edge.target_column,
                "join_type": edge.join_type,
            }
            for edge in blueprint.join_topology.edges
        ],
        "predicate_hints": [
            {
                "predicate_type": hint.predicate_type,
                "expression": hint.expression,
                "source_text": hint.source_text,
                "confidence": hint.confidence,
            }
            for hint in blueprint.predicate_hints
        ],
    }


def _allowed_schema_scope(blueprint: VerifiedContextBlueprint) -> dict:
    columns = [
        f"{column.table_name}.{column.column_name}"
        for column in blueprint.selected_columns
    ]
    return {
        "db_id": blueprint.db_id,
        "tables": list(blueprint.selected_tables),
        "columns": columns,
    }


def _physical_schema_context(
    *,
    db_id: str,
    dbms: str,
    blueprint: VerifiedContextBlueprint,
    cost_snapshot: dict,
) -> dict:
    context: dict[str, Any] = {
        "db_id": db_id,
        "dbms": dbms,
        "cost_snapshot": {
            key: cost_snapshot.get(key)
            for key in (
                "full_scan_tables",
                "uses_temp_sort",
                "uses_temp_group",
                "uses_covering_index",
                "uses_index_condition",
                "join_strategy",
            )
            if key in cost_snapshot
        },
        "indexes": {},
        "columns": {},
        "table_row_counts": {},
        "foreign_keys": [],
        "warnings": [],
    }
    cached_row_counts = get_cached_bird_db_table_row_counts(db_id)
    if cached_row_counts:
        context["table_row_counts"].update(
            {
                table: row_count
                for table, row_count in cached_row_counts.items()
                if table in blueprint.selected_tables
            }
        )
    if str(dbms).lower() != "sqlite":
        context["warnings"].append("Index/FK introspection is only implemented for sqlite.")
        return context
    try:
        conn = connect_bird_database(db_id)
    except Exception as exc:
        context["warnings"].append(f"Failed to open sqlite database: {exc}")
        return context
    try:
        for table in blueprint.selected_tables:
            quoted_table = _quote_sqlite_identifier(table)
            if table not in context["table_row_counts"]:
                context["warnings"].append(
                    f"Missing precomputed row-count cache for {db_id}.{table}."
                )
            indexes: list[dict] = []
            try:
                index_rows = conn.execute(f"PRAGMA index_list({quoted_table})").fetchall()
                for row in index_rows:
                    index_name = str(row[1])
                    quoted_index = _quote_sqlite_identifier(index_name)
                    columns = [
                        str(column_row[2])
                        for column_row in conn.execute(f"PRAGMA index_info({quoted_index})").fetchall()
                        if column_row[2] is not None
                    ]
                    indexes.append(
                        {
                            "name": index_name,
                            "unique": bool(row[2]),
                            "columns": columns,
                        }
                    )
            except Exception as exc:
                context["warnings"].append(f"Failed to read indexes for {table}: {exc}")
            context["indexes"][table] = indexes
            columns: dict[str, dict[str, Any]] = {}
            try:
                for column_row in conn.execute(f"PRAGMA table_info({quoted_table})").fetchall():
                    column_name = str(column_row[1])
                    columns[column_name] = {
                        "type": str(column_row[2] or ""),
                        "not_null": bool(column_row[3]) or int(column_row[5] or 0) > 0,
                        "default": column_row[4],
                        "primary_key_position": int(column_row[5] or 0),
                    }
            except Exception as exc:
                context["warnings"].append(f"Failed to read columns for {table}: {exc}")
            context["columns"][table] = columns
            try:
                for fk_row in conn.execute(f"PRAGMA foreign_key_list({quoted_table})").fetchall():
                    context["foreign_keys"].append(
                        {
                            "from_table": table,
                            "from_column": str(fk_row[3]),
                            "to_table": str(fk_row[2]),
                            "to_column": str(fk_row[4]),
                        }
                    )
            except Exception as exc:
                context["warnings"].append(f"Failed to read foreign keys for {table}: {exc}")
    finally:
        conn.close()
    return context


def _quote_sqlite_identifier(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _bottleneck_summary(report: BottleneckReport) -> dict:
    return {
        "sql_version_id": report.sql_version_id,
        "bottlenecks": list(report.bottlenecks),
        "risk_tags": list(report.risk_tags),
        "rewrite_hints": [
            {
                "strategy": hint.strategy,
                "target_fragment": hint.target_fragment,
                "expected_effect": hint.expected_effect,
                "risk": hint.risk,
                "requires_validation": hint.requires_validation,
                "dbms_notes": hint.dbms_notes,
            }
            for hint in report.rewrite_hints
        ],
        "explanation": report.explanation,
    }


def _unoptimized_fragments(report: BottleneckReport) -> list[dict]:
    fragments: list[dict] = []
    for hint in report.rewrite_hints:
        if not hint.target_fragment:
            continue
        fragments.append(
            {
                "target_fragment": hint.target_fragment,
                "strategy": hint.strategy,
                "expected_effect": hint.expected_effect,
                "risk": hint.risk,
            }
        )
    return fragments


def _unique(values: list[Any]) -> list[Any]:
    result = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
