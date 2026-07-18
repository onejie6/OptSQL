"""Meta-cognitive controller implementation."""

from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Any, Callable

from agent.base import BaseAgent
from agent.explain_analyzer import ExplainAnalyzerAgent
from agent.sql_rewriter import SQLRewriterAgent
from agent.validator import ValidatorAgent
from myTypes import (
    AgentRequest,
    AgentResponse,
    AgentTask,
    AttemptRecord,
    ExecutionMetrics,
    FinalAnswer,
    OptimizationCase,
    ReflectionMemory,
    RuntimeState,
    SQLVersion,
    SchemaGapHint,
    ValidationReport,
    VerifiedContextBlueprint,
)
from utils.db import connect_bird_database
from utils.openai_client import request_chat_text
from utils.openai_client import llm_request_context


PlanClassifierFn = Callable[[dict], dict]
SemanticDiagnosisFn = Callable[[dict], str]


class MetaCognitiveController(BaseAgent):
    """Top-level orchestrator for planning, optimization, and reflection."""

    name = "meta_cognitive_controller"

    def __init__(
        self,
        *,
        explain_analyzer: Any | None = None,
        sql_rewriter: Any | None = None,
        validator: Any | None = None,
        plan_classifier: PlanClassifierFn | None = None,
        knowledge_base: Any | None = None,
        history_improvement_threshold_percent: float = 20.0,
        max_retry: int = 5,
        max_failed_rewrites: int = 3,
        semantic_reflection_max_retry: int = 3,
        semantic_diagnoser: SemanticDiagnosisFn | None = None,
    ) -> None:
        self.explain_analyzer = explain_analyzer or ExplainAnalyzerAgent()
        self.sql_rewriter = sql_rewriter or SQLRewriterAgent()
        self.validator = validator or ValidatorAgent()
        self.plan_classifier = plan_classifier
        self.knowledge_base = knowledge_base
        self.history_improvement_threshold_ratio = history_improvement_threshold_percent / 100.0
        self.max_retry = max_retry
        self.max_failed_rewrites = max_failed_rewrites
        self.semantic_reflection_max_retry = semantic_reflection_max_retry
        self.semantic_diagnoser = semantic_diagnoser
        self._last_case_store_results: list[dict] = []

    def initialize_state(self, task: AgentTask) -> RuntimeState:
        return RuntimeState(
            task=task,
            complexity_score=self.assess_complexity(task),
            strategy="uninitialized",
            blueprint=None,
            sql_versions=[],
            best_sql_version_id=None,
            execution_metrics={},
            validation_reports={},
            reflection_memory=ReflectionMemory(
                attempts=[],
                failed_assumptions=[],
                confirmed_facts=[],
                rejected_rewrite_rules=[],
                next_strategy_hint=None,
            ),
            iteration=0,
            status="initialized",
        )

    def assess_complexity(self, task: AgentTask) -> int:
        text = f"{task.question} {task.evidence or ''}".lower()
        score = 1
        for marker in (" join ", " group ", " order ", " most ", " least ", " average ", " sum "):
            if marker in f" {text} ":
                score += 1
        return min(score, 5)

    def select_strategy(self, state: RuntimeState) -> str:
        return (
            "planning_plus_optimization"
            if state.complexity_score >= 3
            else "planning_only"
        )

    def run_planning_phase(self, state: RuntimeState) -> RuntimeState:
        """Planning is implemented by SchemaFilter/SQLBuilder outside this MVP."""
        return replace(state, status="planning_complete")

    def collect_db_scale_profile(
        self,
        task: AgentTask,
        blueprint: VerifiedContextBlueprint | None = None,
    ) -> dict:
        tables = list(blueprint.selected_tables) if blueprint else _list_sqlite_tables(task.db_id)
        table_rows: dict[str, int] = {}
        total_rows = 0
        try:
            conn = connect_bird_database(task.db_id)
            try:
                if not tables:
                    tables = _list_sqlite_tables_from_connection(conn)
                for table in tables:
                    rows = _table_row_count(conn, table)
                    table_rows[table] = rows
                    total_rows += rows
            finally:
                conn.close()
        except Exception as exc:
            return {
                "total_rows": None,
                "table_count": len(tables),
                "selected_table_rows": table_rows,
                "scale_bucket": "unknown",
                "error": str(exc),
            }
        return {
            "total_rows": total_rows,
            "table_count": len(tables),
            "selected_table_rows": table_rows,
            "scale_bucket": _scale_bucket(total_rows),
        }

    def assess_task_complexity(
        self,
        task: AgentTask,
        sql_version: SQLVersion | None = None,
    ) -> dict:
        text = f"{task.question} {task.evidence or ''}".lower()
        sql = (sql_version.sql if sql_version else "").lower()
        features = {
            "has_join": " join " in f" {text} " or " join " in f" {sql} ",
            "has_aggregation": any(token in text or token in sql for token in ("count(", "sum(", "avg(", "average", "total")),
            "has_group_by": "group by" in sql or " group " in f" {text} ",
            "has_order_by": "order by" in sql or " order " in f" {text} ",
            "has_limit": "limit" in sql or "top " in text or " most " in f" {text} " or " least " in f" {text} ",
            "has_nested_query": "select" in sql[sql.find("select") + 6 :] if "select" in sql else False,
            "question_length": len(task.question or ""),
            "evidence_length": len(task.evidence or ""),
        }
        score = 1 + sum(
            1
            for key in (
                "has_join",
                "has_aggregation",
                "has_group_by",
                "has_order_by",
                "has_limit",
                "has_nested_query",
            )
            if features[key]
        )
        if features["question_length"] > 160 or features["evidence_length"] > 240:
            score += 1
        features["complexity_score"] = min(score, 5)
        features["complexity_bucket"] = (
            "simple"
            if score <= 2
            else "moderate"
            if score <= 3
            else "complex"
        )
        return features

    def classify_plan(
        self,
        *,
        task: AgentTask,
        db_scale: dict,
        task_complexity: dict,
    ) -> dict:
        payload = {
            "question": task.question,
            "evidence": task.evidence,
            "db_scale": db_scale,
            "task_complexity": task_complexity,
            "allowed_plans": ["planning_only", "planning_plus_optimization"],
        }
        if self.plan_classifier is not None:
            decision = dict(self.plan_classifier(payload))
        else:
            decision = self._fallback_plan_classifier(payload)
        plan = decision.get("plan")
        if plan not in {"planning_only", "planning_plus_optimization"}:
            plan = "planning_plus_optimization"
        return {
            "plan": plan,
            "confidence": float(decision.get("confidence", 0.0)),
            "reason": str(decision.get("reason", "")),
            "risk_factors": list(decision.get("risk_factors", [])),
            "classifier": decision.get("classifier", "fallback"),
        }

    def _fallback_plan_classifier(self, payload: dict) -> dict:
        db_scale = payload["db_scale"]
        complexity = payload["task_complexity"]
        risk_factors: list[str] = []
        if db_scale.get("scale_bucket") in {"large", "unknown"}:
            risk_factors.append(f"db_scale={db_scale.get('scale_bucket')}")
        if complexity.get("complexity_bucket") in {"complex"}:
            risk_factors.append("task_complexity=complex")
        if any(
            complexity.get(key)
            for key in ("has_join", "has_group_by", "has_order_by", "has_limit", "has_nested_query")
        ):
            risk_factors.append("query_shape_may_benefit_from_optimization")
        if risk_factors:
            return {
                "plan": "planning_plus_optimization",
                "confidence": 0.65,
                "reason": "DB scale or task complexity suggests optimization may help.",
                "risk_factors": risk_factors,
                "classifier": "fallback",
            }
        return {
            "plan": "planning_only",
            "confidence": 0.62,
            "reason": "Small/simple task; optimization loop is unlikely to help.",
            "risk_factors": [],
            "classifier": "fallback",
        }

    def run_optimization_loop(self, state: RuntimeState) -> RuntimeState:
        """Run Plan A serial optimization: Explain -> Rewrite -> Validate."""
        if state.blueprint is None:
            raise ValueError("Optimization loop requires RuntimeState.blueprint.")
        if not state.sql_versions:
            raise ValueError("Optimization loop requires at least one SQLVersion.")

        sql_versions = list(state.sql_versions)
        base_sql = sql_versions[0]
        execution_metrics = dict(state.execution_metrics)
        validation_reports = dict(state.validation_reports)
        best_sql_version_id = state.best_sql_version_id or sql_versions[-1].version_id
        current_sql = _find_sql_version(sql_versions, best_sql_version_id) or sql_versions[-1]
        latest_explain_plan_cache: dict | None = None
        previous_risk_tags: list[str] = []
        failed_rewrites = 0
        accepted_count = 0
        status = "optimization_started"
        iteration = state.iteration

        for _ in range(self.max_retry):
            explain_response = self._run_explain(
                task=state.task,
                sql_version=current_sql,
                previous_risk_tags=previous_risk_tags,
                explain_plan_cache=latest_explain_plan_cache,
                optimization_round=iteration + 1,
            )
            if explain_response.status != "success":
                status = "optimization_explain_error"
                break

            explain_artifacts = explain_response.output_artifacts
            bottleneck_report = explain_artifacts["bottleneck_report"]
            previous_risk_tags = list(bottleneck_report.risk_tags)
            metrics = explain_artifacts.get("execution_metrics")
            if isinstance(metrics, ExecutionMetrics):
                execution_metrics[current_sql.version_id] = metrics
            optimization_decision = explain_artifacts.get("optimization_decision")
            if _should_stop_before_rewrite(optimization_decision):
                status = (
                    "optimization_converged"
                    if accepted_count
                    else "optimization_skipped_by_explain"
                )
                break

            rewrite_response = self._run_rewriter(
                task=state.task,
                sql_version=current_sql,
                blueprint=state.blueprint,
                bottleneck_report=bottleneck_report,
                optimization_round=iteration + 1,
            )
            candidate_sql = rewrite_response.output_artifacts.get("candidate_sql_version")
            if rewrite_response.status != "success" or candidate_sql is None:
                status = "optimization_converged" if accepted_count else "optimization_skipped"
                break

            sql_versions.append(candidate_sql)
            validation_response = self._run_validator(
                task=state.task,
                source_sql=current_sql,
                candidate_sql=candidate_sql,
                blueprint=state.blueprint,
                optimization_round=iteration + 1,
            )
            if validation_response.status != "success":
                status = "optimization_validation_error"
                break

            validation_report = validation_response.output_artifacts["validation_report"]
            validation_reports[candidate_sql.version_id] = validation_report
            execution_metrics[current_sql.version_id] = validation_report.old_metrics
            execution_metrics[candidate_sql.version_id] = validation_report.new_metrics
            iteration += 1

            if validation_report.accepted:
                accepted_count += 1
                failed_rewrites = 0
                current_sql = candidate_sql
                best_sql_version_id = candidate_sql.version_id
                latest_explain_plan_cache = validation_response.output_artifacts.get(
                    "explain_plan_cache"
                )
                status = "optimization_improved"
                if self.monitor_convergence(
                    replace(
                        state,
                        sql_versions=sql_versions,
                        best_sql_version_id=best_sql_version_id,
                        execution_metrics=execution_metrics,
                        validation_reports=validation_reports,
                        iteration=iteration,
                        status=status,
                    ),
                    validation_report,
                ) == "stop":
                    status = "optimization_converged"
                    break
                continue

            if not _is_free_exploration_candidate(candidate_sql):
                forced_rewrite_response = self._run_rewriter(
                    task=state.task,
                    sql_version=current_sql,
                    blueprint=state.blueprint,
                    bottleneck_report=bottleneck_report,
                    optimization_round=iteration + 1,
                    force_free_exploration=True,
                )
                forced_candidate_sql = forced_rewrite_response.output_artifacts.get(
                    "candidate_sql_version"
                )
                if (
                    forced_rewrite_response.status == "success"
                    and forced_candidate_sql is not None
                ):
                    sql_versions.append(forced_candidate_sql)
                    forced_validation_response = self._run_validator(
                        task=state.task,
                        source_sql=current_sql,
                        candidate_sql=forced_candidate_sql,
                        blueprint=state.blueprint,
                        optimization_round=iteration + 1,
                    )
                    if forced_validation_response.status == "success":
                        forced_validation_report = forced_validation_response.output_artifacts[
                            "validation_report"
                        ]
                        validation_reports[forced_candidate_sql.version_id] = (
                            forced_validation_report
                        )
                        execution_metrics[current_sql.version_id] = (
                            forced_validation_report.old_metrics
                        )
                        execution_metrics[forced_candidate_sql.version_id] = (
                            forced_validation_report.new_metrics
                        )
                        iteration += 1
                        if forced_validation_report.accepted:
                            accepted_count += 1
                            failed_rewrites = 0
                            current_sql = forced_candidate_sql
                            best_sql_version_id = forced_candidate_sql.version_id
                            latest_explain_plan_cache = forced_validation_response.output_artifacts.get(
                                "explain_plan_cache"
                            )
                            status = "optimization_improved"
                            if self.monitor_convergence(
                                replace(
                                    state,
                                    sql_versions=sql_versions,
                                    best_sql_version_id=best_sql_version_id,
                                    execution_metrics=execution_metrics,
                                    validation_reports=validation_reports,
                                    iteration=iteration,
                                    status=status,
                                ),
                                forced_validation_report,
                            ) == "stop":
                                status = "optimization_converged"
                                break
                            continue

            if _needs_validation_reflection(validation_report, candidate_sql):
                reflection_result = self._retry_validation_reflection(
                    state=state,
                    current_sql=current_sql,
                    failed_candidate=candidate_sql,
                    original_rewrite_response=rewrite_response,
                    original_validation_response=validation_response,
                    bottleneck_report=bottleneck_report,
                    sql_versions=sql_versions,
                    validation_reports=validation_reports,
                    execution_metrics=execution_metrics,
                )
                if reflection_result["accepted"]:
                    accepted_count += 1
                    failed_rewrites = 0
                    repaired_sql = reflection_result["candidate_sql"]
                    current_sql = repaired_sql
                    best_sql_version_id = repaired_sql.version_id
                    latest_explain_plan_cache = reflection_result["validation_response"].output_artifacts.get(
                        "explain_plan_cache"
                    )
                    iteration += int(reflection_result["attempts"])
                    state = replace(
                        state,
                        reflection_memory=reflection_result["reflection_memory"],
                    )
                    status = "optimization_improved_after_reflection"
                    if self.monitor_convergence(
                        replace(
                            state,
                            sql_versions=sql_versions,
                            best_sql_version_id=best_sql_version_id,
                            execution_metrics=execution_metrics,
                            validation_reports=validation_reports,
                            iteration=iteration,
                            status=status,
                        ),
                        reflection_result["validation_report"],
                    ) == "stop":
                        status = "optimization_converged"
                        break
                    continue
                iteration += int(reflection_result["attempts"])
                state = replace(
                    state,
                    reflection_memory=reflection_result["reflection_memory"],
                )
                failed_rewrites += 1
                status = "optimization_validation_reflection_failed"
                if failed_rewrites >= self.max_failed_rewrites:
                    status = "optimization_failed_rewrites"
                break

            failed_rewrites += 1
            status = "optimization_rejected"
            if failed_rewrites >= self.max_failed_rewrites:
                status = "optimization_failed_rewrites"
            break
        else:
            status = "optimization_max_retry"

        status = _normalize_terminal_optimization_status(
            status=status,
            base_sql_version_id=base_sql.version_id,
            best_sql_version_id=best_sql_version_id,
            validation_reports=validation_reports,
        )

        self._last_case_store_results = self._persist_final_optimization_case_if_qualified(
            state=state,
            base_sql=base_sql,
            best_sql_version_id=best_sql_version_id,
            sql_versions=sql_versions,
            execution_metrics=execution_metrics,
            validation_reports=validation_reports,
            bottleneck_tags=previous_risk_tags,
        )

        return replace(
            state,
            strategy="planning_plus_optimization",
            sql_versions=sql_versions,
            best_sql_version_id=best_sql_version_id,
            execution_metrics=execution_metrics,
            validation_reports=validation_reports,
            reflection_memory=state.reflection_memory,
            iteration=iteration,
            status=status,
        )

    def _retry_validation_reflection(
        self,
        *,
        state: RuntimeState,
        current_sql: SQLVersion,
        failed_candidate: SQLVersion,
        original_rewrite_response: AgentResponse,
        original_validation_response: AgentResponse,
        bottleneck_report: Any,
        sql_versions: list[SQLVersion],
        validation_reports: dict[str, ValidationReport],
        execution_metrics: dict[str, ExecutionMetrics],
    ) -> dict:
        """Ask the rewriter to repair semantic mismatch or syntax failure."""
        reflection_memory = state.reflection_memory
        last_candidate = failed_candidate
        last_validation_response = original_validation_response
        attempts_used = 0
        reflection_context = self._build_validation_reflection_context(
            task=state.task,
            source_sql=current_sql,
            failed_candidate=failed_candidate,
            rewrite_response=original_rewrite_response,
            validation_response=original_validation_response,
            bottleneck_report=bottleneck_report,
            attempt_index=0,
        )
        reflection_context = self._augment_semantic_reflection_context(reflection_context)
        reflection_memory = _record_validation_reflection_attempt(
            reflection_memory,
            candidate_sql=failed_candidate,
            validation_response=original_validation_response,
            reflection_context=reflection_context,
            status=f"{reflection_context['failure_type']}_detected",
        )

        for retry_index in range(1, self.semantic_reflection_max_retry + 1):
            attempts_used += 1
            repair_response = self._run_validation_reflection_repair(
                task=state.task,
                source_sql=current_sql,
                failed_candidate=last_candidate,
                blueprint=state.blueprint,
                bottleneck_report=bottleneck_report,
                reflection_context={
                    **reflection_context,
                    "attempt_index": retry_index,
                    "max_retry": self.semantic_reflection_max_retry,
                },
            )
            repaired_candidate = repair_response.output_artifacts.get("candidate_sql_version")
            if repair_response.status != "success" or repaired_candidate is None:
                reflection_memory = _record_validation_reflection_attempt(
                    reflection_memory,
                    candidate_sql=last_candidate,
                    validation_response=last_validation_response,
                    reflection_context=reflection_context,
                    status="repair_generation_failed",
                    notes="; ".join(repair_response.errors) or repair_response.reasoning_summary,
                )
                continue

            sql_versions.append(repaired_candidate)
            repaired_validation_response = self._run_validator(
                task=state.task,
                source_sql=current_sql,
                candidate_sql=repaired_candidate,
                blueprint=state.blueprint,
                optimization_round=state.iteration + retry_index + 1,
            )
            if repaired_validation_response.status != "success":
                reflection_memory = _record_validation_reflection_attempt(
                    reflection_memory,
                    candidate_sql=repaired_candidate,
                    validation_response=repaired_validation_response,
                    reflection_context=reflection_context,
                    status="repair_validation_error",
                    notes="; ".join(repaired_validation_response.errors),
                )
                last_candidate = repaired_candidate
                last_validation_response = repaired_validation_response
                continue

            repaired_report = repaired_validation_response.output_artifacts["validation_report"]
            validation_reports[repaired_candidate.version_id] = repaired_report
            execution_metrics[current_sql.version_id] = repaired_report.old_metrics
            execution_metrics[repaired_candidate.version_id] = repaired_report.new_metrics
            reflection_memory = _record_validation_reflection_attempt(
                reflection_memory,
                candidate_sql=repaired_candidate,
                validation_response=repaired_validation_response,
                reflection_context=reflection_context,
                status="repair_accepted" if repaired_report.accepted else "repair_rejected",
            )
            if repaired_report.accepted:
                return {
                    "accepted": True,
                    "attempts": attempts_used,
                    "candidate_sql": repaired_candidate,
                    "validation_report": repaired_report,
                    "validation_response": repaired_validation_response,
                    "reflection_memory": reflection_memory,
                }
            if not _needs_validation_reflection(repaired_report, repaired_candidate):
                return {
                    "accepted": False,
                    "attempts": attempts_used,
                    "candidate_sql": repaired_candidate,
                    "validation_report": repaired_report,
                    "validation_response": repaired_validation_response,
                    "reflection_memory": reflection_memory,
                }
            last_candidate = repaired_candidate
            last_validation_response = repaired_validation_response
            reflection_context = self._build_validation_reflection_context(
                task=state.task,
                source_sql=current_sql,
                failed_candidate=repaired_candidate,
                rewrite_response=repair_response,
                validation_response=repaired_validation_response,
                bottleneck_report=bottleneck_report,
                attempt_index=retry_index,
            )
            reflection_context = self._augment_semantic_reflection_context(reflection_context)

        return {
            "accepted": False,
            "attempts": attempts_used,
            "candidate_sql": last_candidate,
            "validation_report": last_validation_response.output_artifacts.get("validation_report"),
            "validation_response": last_validation_response,
            "reflection_memory": reflection_memory,
        }

    def _build_validation_reflection_context(
        self,
        *,
        task: AgentTask,
        source_sql: SQLVersion,
        failed_candidate: SQLVersion,
        rewrite_response: AgentResponse,
        validation_response: AgentResponse,
        bottleneck_report: Any,
        attempt_index: int,
    ) -> dict:
        validation_artifacts = validation_response.output_artifacts
        report = validation_artifacts.get("validation_report")
        comparison = validation_artifacts.get("equivalence_comparison")
        failure_type = _validation_reflection_failure_type(report)
        optimization_basis = _optimization_basis(
            rewrite_response=rewrite_response,
            bottleneck_report=bottleneck_report,
            candidate_sql=failed_candidate,
        )
        failed_free_exploration_directions = _free_exploration_failed_directions(
            rewrite_response=rewrite_response,
            validation_response=validation_response,
            failed_candidate=failed_candidate,
        )
        syntax_error = None
        new_metrics = getattr(report, "new_metrics", None)
        if failure_type == "syntax_error":
            syntax_error = (
                getattr(new_metrics, "error_message", None)
                or getattr(report, "failure_reason", None)
                or "; ".join(validation_response.errors)
            )
        return {
            "reflection_type": "optimization_validation_failure",
            "failure_type": failure_type,
            "attempt_index": attempt_index,
            "max_retry": self.semantic_reflection_max_retry,
            "question": task.question,
            "evidence": task.evidence,
            "db_id": task.db_id,
            "dbms": task.dbms,
            "sql_before_optimization": source_sql.sql,
            "sql_after_optimization": failed_candidate.sql,
            "source_sql_version_id": source_sql.version_id,
            "failed_sql_version_id": failed_candidate.version_id,
            "optimization_basis": optimization_basis,
            "failed_free_exploration_directions": failed_free_exploration_directions,
            "validator_failure": {
                "executable": getattr(report, "executable", None),
                "equivalent": getattr(report, "equivalent", None),
                "performance_better": getattr(report, "performance_better", None),
                "failure_reason": getattr(report, "failure_reason", None),
                "syntax_error": syntax_error,
                "equivalence_diff_summary": getattr(comparison, "diff_summary", None),
                "source_row_count": getattr(getattr(report, "old_metrics", None), "row_count", None),
                "candidate_row_count": getattr(getattr(report, "new_metrics", None), "row_count", None),
            },
            "semantic_diagnosis_summary": None,
            "semantic_diagnosis_error": None,
            "repair_instruction": (
                "Use chain-of-thought reasoning internally. If failure_type is "
                "semantic_inconsistency, compare the SQL before and after optimization, locate "
                "the exact rewritten clause/expression/join/predicate/order that changed result "
                "semantics, use the provided semantic diagnosis summary to verify the likely root "
                "cause, and repair it. If failure_type is syntax_error, use the concrete syntax/"
                "execution error to locate the invalid token, column, alias, clause, or dialect "
                "construct and repair it. Return one SQL that preserves the optimization basis "
                "where safe and passes Validator. Do not expose private reasoning; include a "
                "concise diagnosis and the repaired SQL."
            ),
        }

    def _augment_semantic_reflection_context(self, reflection_context: dict) -> dict:
        if reflection_context.get("failure_type") != "semantic_inconsistency":
            return reflection_context
        try:
            summary = self._summarize_semantic_inconsistency(reflection_context)
        except Exception as exc:
            return {
                **reflection_context,
                "semantic_diagnosis_summary": None,
                "semantic_diagnosis_error": str(exc),
            }
        if not summary:
            fallback = self._fallback_semantic_diagnosis(reflection_context)
            return {
                **reflection_context,
                "semantic_diagnosis_summary": fallback or None,
                "semantic_diagnosis_error": (
                    "empty diagnosis response; used deterministic fallback"
                    if fallback
                    else "empty diagnosis response"
                ),
            }
        return {
            **reflection_context,
            "semantic_diagnosis_summary": summary or None,
            "semantic_diagnosis_error": None if summary else "empty diagnosis response",
        }

    def _summarize_semantic_inconsistency(self, reflection_context: dict) -> str:
        if self.semantic_diagnoser is not None:
            return str(self.semantic_diagnoser(reflection_context) or "").strip()
        validator_failure = reflection_context.get("validator_failure") or {}
        optimization_basis = reflection_context.get("optimization_basis") or {}
        prompt = (
            "You are diagnosing why an optimized SQLite query changed semantics.\n"
            "Return 3 short lines using exactly these prefixes:\n"
            "Changed clause:\n"
            "Mismatch hypothesis:\n"
            "Repair hint:\n\n"
            f"Question: {reflection_context.get('question')}\n"
            f"Evidence: {reflection_context.get('evidence') or '<none>'}\n"
            f"Source SQL:\n{reflection_context.get('sql_before_optimization')}\n\n"
            f"Failed optimized SQL:\n{reflection_context.get('sql_after_optimization')}\n\n"
            f"Validator failure reason: {validator_failure.get('failure_reason')}\n"
            f"Equivalence diff summary: {validator_failure.get('equivalence_diff_summary')}\n"
            f"Source row count: {validator_failure.get('source_row_count')}\n"
            f"Candidate row count: {validator_failure.get('candidate_row_count')}\n"
            f"Rewrite rule ids: {optimization_basis.get('rewrite_rule_ids')}\n"
            f"Rewrite plan: {optimization_basis.get('rewrite_plan')}\n"
            "Focus on the exact clause or operator that likely changed set semantics, duplicate "
            "behavior, NULL behavior, aggregation grain, or top-k behavior."
        )
        with llm_request_context(
            branch="semantic_diagnoser",
            prompt_profile="diagnosis",
            feature="controller.semantic_diagnoser",
        ):
            return request_chat_text(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You analyze SQL semantic mismatches. Be concise and concrete. "
                            "Do not produce SQL."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=220,
            ).strip()

    def _fallback_semantic_diagnosis(self, reflection_context: dict) -> str:
        optimization_basis = reflection_context.get("optimization_basis") or {}
        rewrite_rule_ids = [
            str(rule_id).lower()
            for rule_id in (optimization_basis.get("rewrite_rule_ids") or [])
        ]
        source_sql = str(reflection_context.get("sql_before_optimization") or "")
        failed_sql = str(reflection_context.get("sql_after_optimization") or "")
        combined = "\n".join(
            [
                source_sql.lower(),
                failed_sql.lower(),
                str(optimization_basis.get("rewrite_plan") or "").lower(),
                str((reflection_context.get("validator_failure") or {}).get("failure_reason") or "").lower(),
                str((reflection_context.get("validator_failure") or {}).get("equivalence_diff_summary") or "").lower(),
            ]
        )
        if (
            "builtin_same_table_lookup_to_scalar_subquery" in rewrite_rule_ids
            or "same-table lookup" in combined
            or "= (select" in combined
        ):
            return (
                "Changed clause: lookup predicate rewritten from same-table join semantics to scalar subquery equality\n"
                "Mismatch hypothesis: scalar equality is not equivalent when the lookup predicate can match multiple rows or multiple keys\n"
                "Repair hint: preserve set semantics with IN, EXISTS, or another semi-join form instead of = (SELECT ...)"
            )
        if "order by" in combined or "limit" in combined or "top-k" in combined:
            return (
                "Changed clause: ordering or top-k reduction path\n"
                "Mismatch hypothesis: the rewrite changed row selection under ORDER BY/LIMIT or tie handling\n"
                "Repair hint: preserve the original ORDER BY/LIMIT contract and only reduce rows in a provably equivalent way"
            )
        if "distinct" in combined or "group by" in combined or "fanout" in combined:
            return (
                "Changed clause: duplicate-control or grouping shape\n"
                "Mismatch hypothesis: the rewrite changed duplicate behavior or aggregation grain\n"
                "Repair hint: restore the original duplicate semantics before attempting further optimization"
            )
        return (
            "Changed clause: rewritten predicate or join boundary\n"
            "Mismatch hypothesis: the optimization changed result-set semantics relative to the source SQL\n"
            "Repair hint: compare the source and failed SQL clause-by-clause and restore source semantics before preserving optimization intent"
        )

    def _run_validation_reflection_repair(
        self,
        *,
        task: AgentTask,
        source_sql: SQLVersion,
        failed_candidate: SQLVersion,
        blueprint: VerifiedContextBlueprint,
        bottleneck_report: Any,
        reflection_context: dict,
    ) -> AgentResponse:
        optimization_basis = reflection_context.get("optimization_basis") or {}
        with llm_request_context(
            optimization_round=reflection_context.get("attempt_index", 0) + 1,
            optimization_phase="validation_reflection_repair",
            reflection_attempt_index=reflection_context.get("attempt_index"),
            reflection_failure_type=reflection_context.get("failure_type"),
            task_question_id=task.question_id,
            task_db_id=task.db_id,
            source_sql_version_id=source_sql.version_id,
            candidate_sql_version_id=failed_candidate.version_id,
            rewrite_rule_ids=list(optimization_basis.get("rewrite_rule_ids") or []),
        ):
            return self.sql_rewriter.run(
                AgentRequest(
                    request_id=f"reflect-rewrite-{uuid.uuid4().hex[:8]}",
                    task=task,
                    runtime_state={},
                    input_artifacts={
                        "sql_version": source_sql,
                        "source_sql_version": source_sql,
                        "failed_candidate_sql_version": failed_candidate,
                        "blueprint": blueprint,
                        "bottleneck_report": bottleneck_report,
                        "reflection_context": reflection_context,
                    },
                    constraints={
                        "mode": "validation_reflection_repair",
                        "max_retry": self.semantic_reflection_max_retry,
                    },
                )
            )

    def _persist_final_optimization_case_if_qualified(
        self,
        *,
        state: RuntimeState,
        base_sql: SQLVersion,
        best_sql_version_id: str | None,
        sql_versions: list[SQLVersion],
        execution_metrics: dict[str, ExecutionMetrics],
        validation_reports: dict[str, ValidationReport],
        bottleneck_tags: list[str],
    ) -> list[dict]:
        """Persist only accepted optimization cases with measurable improvement."""
        if self.knowledge_base is None or not hasattr(self.knowledge_base, "upsert_if_novel_case"):
            return []
        best_sql = _find_sql_version(sql_versions, best_sql_version_id or "")
        if best_sql is None or best_sql.version_id == base_sql.version_id:
            return []
        report = validation_reports.get(best_sql.version_id)
        if report is None or not report.accepted:
            return []
        before_metrics = execution_metrics.get(base_sql.version_id) or report.old_metrics
        after_metrics = execution_metrics.get(best_sql.version_id) or report.new_metrics
        case = OptimizationCase(
            case_id=uuid.uuid4().hex[:12],
            dbms=state.task.dbms,
            nlq=state.task.question,
            evidence=state.task.evidence,
            schema_signature=_schema_signature(state.blueprint),
            src_sql=base_sql.sql,
            dst_sql=best_sql.sql,
            rule_ids=list(best_sql.rewrite_rule_ids),
            bottleneck_tags=list(bottleneck_tags),
            metrics_before=before_metrics,
            metrics_after=after_metrics,
            explanation=best_sql.explanation,
            novelty_score=1.0,
        )
        try:
            inserted = bool(self.knowledge_base.upsert_if_novel_case(case))
        except Exception as exc:
            return [
                {
                    "case_id": case.case_id,
                    "inserted": False,
                    "error": str(exc),
                }
            ]
        return [
            {
                "case_id": case.case_id,
                "inserted": inserted,
                "rule_ids": case.rule_ids,
            }
        ]

    def handle_schema_gap(
        self,
        gap_hints: list[SchemaGapHint],
        state: RuntimeState,
    ) -> RuntimeState:
        attempts = list(state.reflection_memory.attempts)
        for hint in gap_hints:
            attempts.append(
                AttemptRecord(
                    attempt_id=uuid.uuid4().hex[:12],
                    sql_version_id=state.best_sql_version_id,
                    stage="schema_gap",
                    status="pending_recovery",
                    failure_reason=hint.suggestion,
                    notes=f"{hint.gap_type}: {hint.element}",
                )
            )
        return replace(
            state,
            reflection_memory=replace(state.reflection_memory, attempts=attempts),
            status="schema_gap_detected",
        )

    def monitor_convergence(
        self,
        state: RuntimeState,
        validation_report: ValidationReport,
    ) -> str:
        if not validation_report.accepted:
            return "stop"
        if state.iteration >= self.max_retry:
            return "stop"
        return "continue"

    def reflect(self, state: RuntimeState) -> ReflectionMemory:
        return state.reflection_memory

    def finalize(self, state: RuntimeState) -> FinalAnswer:
        best_sql = _find_sql_version(state.sql_versions, state.best_sql_version_id or "")
        validation_summary = "not_validated"
        performance_summary = "not_available"
        if best_sql and best_sql.version_id in state.validation_reports:
            report = state.validation_reports[best_sql.version_id]
            validation_summary = (
                f"accepted={report.accepted}, executable={report.executable}, "
                f"equivalent={report.equivalent}"
            )
            performance_summary = f"performance_better={report.performance_better}"
        return FinalAnswer(
            sql=best_sql.sql if best_sql else "",
            selected_schema=[],
            value_bindings=[],
            join_path=[],
            optimization_steps=[
                version.explanation
                for version in state.sql_versions
                if version.source_agent == "sql_rewriter"
            ],
            validation_summary=validation_summary,
            performance_summary=performance_summary,
            caveats=[] if best_sql else ["No SQL version available."],
        )

    def run(self, request: AgentRequest) -> AgentResponse:
        try:
            state = _state_from_request(request) or _state_from_artifacts(self, request)
            current_sql = _find_sql_version(state.sql_versions, state.best_sql_version_id or "")
            db_scale = self.collect_db_scale_profile(request.task, state.blueprint)
            task_complexity = self.assess_task_complexity(request.task, current_sql)
            plan_decision = self.classify_plan(
                task=request.task,
                db_scale=db_scale,
                task_complexity=task_complexity,
            )
            planned_state = replace(
                state,
                strategy=plan_decision["plan"],
                complexity_score=int(task_complexity["complexity_score"]),
            )
            planned_state = self.run_planning_phase(planned_state)
            if plan_decision["plan"] == "planning_plus_optimization":
                final_state = self.run_optimization_loop(planned_state)
            else:
                final_state = replace(planned_state, status="planning_only_complete")
            final_answer = self.finalize(final_state)
        except Exception as exc:
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status="error",
                output_artifacts={},
                reasoning_summary="Optimization loop failed.",
                tool_calls=[],
                errors=[str(exc)],
            )
        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            status="success",
            output_artifacts={
                "runtime_state": final_state,
                "final_answer": final_answer,
                "plan_decision": plan_decision,
                "db_scale": db_scale,
                "task_complexity": task_complexity,
            },
            reasoning_summary=(
                f"Controller selected {plan_decision['plan']} and finished "
                f"with status {final_state.status}; iterations={final_state.iteration}."
            ),
            tool_calls=[],
            errors=[],
        )

    def _run_explain(
        self,
        *,
        task: AgentTask,
        sql_version: SQLVersion,
        previous_risk_tags: list[str],
        explain_plan_cache: dict | None,
        optimization_round: int,
    ) -> AgentResponse:
        input_artifacts: dict[str, Any] = {
            "sql_version": sql_version,
            "db_id": task.db_id,
            "dbms": task.dbms,
            "previous_risk_tags": previous_risk_tags,
        }
        if explain_plan_cache:
            input_artifacts["explain_plan_cache"] = explain_plan_cache
        with llm_request_context(
            optimization_round=optimization_round,
            optimization_phase="explain",
            task_question_id=task.question_id,
            task_db_id=task.db_id,
            source_sql_version_id=sql_version.version_id,
        ):
            return self.explain_analyzer.run(
                AgentRequest(
                    request_id=f"explain-{uuid.uuid4().hex[:8]}",
                    task=task,
                    runtime_state={},
                    input_artifacts=input_artifacts,
                    constraints={},
                )
            )

    def _run_rewriter(
        self,
        *,
        task: AgentTask,
        sql_version: SQLVersion,
        blueprint: VerifiedContextBlueprint,
        bottleneck_report: Any,
        optimization_round: int,
        force_free_exploration: bool = False,
    ) -> AgentResponse:
        with llm_request_context(
            optimization_round=optimization_round,
            optimization_phase="rewrite",
            task_question_id=task.question_id,
            task_db_id=task.db_id,
            source_sql_version_id=sql_version.version_id,
        ):
            return self.sql_rewriter.run(
                AgentRequest(
                    request_id=f"rewrite-{uuid.uuid4().hex[:8]}",
                    task=task,
                    runtime_state={},
                    input_artifacts={
                        "sql_version": sql_version,
                        "blueprint": blueprint,
                        "bottleneck_report": bottleneck_report,
                        "force_free_exploration": force_free_exploration,
                    },
                    constraints={},
                )
            )

    def _run_validator(
        self,
        *,
        task: AgentTask,
        source_sql: SQLVersion,
        candidate_sql: SQLVersion,
        blueprint: VerifiedContextBlueprint,
        optimization_round: int,
    ) -> AgentResponse:
        with llm_request_context(
            optimization_round=optimization_round,
            optimization_phase="validate",
            task_question_id=task.question_id,
            task_db_id=task.db_id,
            source_sql_version_id=source_sql.version_id,
            candidate_sql_version_id=candidate_sql.version_id,
            rewrite_rule_ids=list(candidate_sql.rewrite_rule_ids),
        ):
            return self.validator.run(
                AgentRequest(
                    request_id=f"validate-{uuid.uuid4().hex[:8]}",
                    task=task,
                    runtime_state={},
                    input_artifacts={
                        "source_sql_version": source_sql,
                        "candidate_sql_version": candidate_sql,
                        "blueprint": blueprint,
                        "db_id": task.db_id,
                        "dbms": task.dbms,
                    },
                    constraints={},
                )
            )


def _find_sql_version(sql_versions: list[SQLVersion], version_id: str) -> SQLVersion | None:
    for version in sql_versions:
        if version.version_id == version_id:
            return version
    return None


def _has_accepted_best_candidate(
    *,
    base_sql_version_id: str,
    best_sql_version_id: str | None,
    validation_reports: dict[str, ValidationReport],
) -> bool:
    if not best_sql_version_id or best_sql_version_id == base_sql_version_id:
        return False
    report = validation_reports.get(best_sql_version_id)
    return bool(report is not None and report.accepted)


def _normalize_terminal_optimization_status(
    *,
    status: str,
    base_sql_version_id: str,
    best_sql_version_id: str | None,
    validation_reports: dict[str, ValidationReport],
) -> str:
    """Preserve success semantics when a better accepted SQL already exists.

    The loop can accept an improved candidate, continue exploring, and later
    encounter a rejected candidate. In that case the final state should still
    report optimization success because the accepted best SQL remains the
    controller's selected output.
    """
    if _has_accepted_best_candidate(
        base_sql_version_id=base_sql_version_id,
        best_sql_version_id=best_sql_version_id,
        validation_reports=validation_reports,
    ):
        if status in {
            "optimization_rejected",
            "optimization_failed_rewrites",
            "optimization_max_retry",
        }:
            return "optimization_improved"
    return status


def _state_from_request(request: AgentRequest) -> RuntimeState | None:
    for container in (request.input_artifacts, request.runtime_state):
        state = container.get("runtime_state") if isinstance(container, dict) else None
        if isinstance(state, RuntimeState):
            return state
    return None


def _state_from_artifacts(self: MetaCognitiveController, request: AgentRequest) -> RuntimeState:
    blueprint = request.input_artifacts.get("blueprint")
    sql_version = request.input_artifacts.get("sql_version")
    initial_metrics = request.input_artifacts.get("execution_metrics")
    if not isinstance(blueprint, VerifiedContextBlueprint) or not isinstance(sql_version, SQLVersion):
        raise ValueError(
            "Controller.run requires a RuntimeState or input_artifacts with "
            "'blueprint' and 'sql_version'."
        )
    execution_metrics = (
        {sql_version.version_id: initial_metrics}
        if isinstance(initial_metrics, ExecutionMetrics)
        else {}
    )
    state = self.initialize_state(request.task)
    return replace(
        state,
        strategy="planning_plus_optimization",
        blueprint=blueprint,
        sql_versions=[sql_version],
        best_sql_version_id=sql_version.version_id,
        execution_metrics=execution_metrics,
        status="planning_complete",
    )


def _list_sqlite_tables(db_id: str) -> list[str]:
    try:
        conn = connect_bird_database(db_id)
        try:
            return _list_sqlite_tables_from_connection(conn)
        finally:
            conn.close()
    except Exception:
        return []


def _list_sqlite_tables_from_connection(conn: Any) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return [str(row[0]) for row in rows]


def _table_row_count(conn: Any, table_name: str) -> int:
    escaped = table_name.replace('"', '""')
    try:
        row = conn.execute(f'SELECT COUNT(*) FROM "{escaped}"').fetchone()
    except Exception:
        return 0
    return int(row[0]) if row and row[0] is not None else 0


def _scale_bucket(total_rows: int) -> str:
    if total_rows < 100_000:
        return "small"
    if total_rows < 1_000_000:
        return "medium"
    return "large"


def _schema_signature(blueprint: VerifiedContextBlueprint | None) -> str:
    if blueprint is None:
        return ""
    columns = sorted(f"{col.table_name}.{col.column_name}" for col in blueprint.selected_columns)
    tables = sorted(blueprint.selected_tables)
    return "|".join([*tables, *columns])


def _should_stop_before_rewrite(optimization_decision: Any) -> bool:
    if optimization_decision is None:
        return False
    if isinstance(optimization_decision, dict):
        should_rewrite = optimization_decision.get("should_rewrite")
        next_action = optimization_decision.get("next_action")
        decision = optimization_decision.get("decision")
    else:
        should_rewrite = getattr(optimization_decision, "should_rewrite", None)
        next_action = getattr(optimization_decision, "next_action", None)
        decision = getattr(optimization_decision, "decision", None)
    if should_rewrite is True or next_action == "call_rewriter" or decision == "rewrite":
        return False
    return next_action in {"return_current_sql", "validate_current_sql"} or should_rewrite is False


def _needs_validation_reflection(
    report: ValidationReport,
    candidate_sql: SQLVersion | None = None,
) -> bool:
    failure_type = _validation_reflection_failure_type(report)
    return failure_type in {"semantic_inconsistency", "syntax_error"}


def _validation_reflection_failure_type(report: Any) -> str:
    if not isinstance(report, ValidationReport):
        return "unknown"
    if report.executable and not report.equivalent:
        return "semantic_inconsistency"
    if not report.executable:
        reason = " ".join(
            str(value or "")
            for value in (
                report.failure_reason,
                report.new_metrics.error_message if report.new_metrics else None,
            )
        ).lower()
        if "outside the blueprint" in reason or "semantic guardrails" in reason:
            return "guardrail_failure"
        return "syntax_error"
    return "unknown"


def _is_free_exploration_candidate(candidate_sql: SQLVersion | None) -> bool:
    if candidate_sql is None:
        return False
    return "llm_free_exploration" in set(candidate_sql.rewrite_rule_ids)


def _optimization_basis(
    *,
    rewrite_response: AgentResponse,
    bottleneck_report: Any,
    candidate_sql: SQLVersion,
) -> dict:
    artifacts = rewrite_response.output_artifacts or {}
    rewrite_plan = artifacts.get("rewrite_plan")
    used_hint = artifacts.get("used_hint")
    return {
        "rewrite_rule_ids": list(candidate_sql.rewrite_rule_ids),
        "candidate_explanation": candidate_sql.explanation,
        "rewrite_plan": rewrite_plan,
        "used_hint": used_hint,
        "bottlenecks": list(getattr(bottleneck_report, "bottlenecks", []) or []),
        "risk_tags": list(getattr(bottleneck_report, "risk_tags", []) or []),
        "rewrite_hints": [
            {
                "strategy": getattr(hint, "strategy", None),
                "target_fragment": getattr(hint, "target_fragment", None),
                "expected_effect": getattr(hint, "expected_effect", None),
                "risk": getattr(hint, "risk", None),
                "requires_validation": getattr(hint, "requires_validation", None),
                "dbms_notes": getattr(hint, "dbms_notes", None),
            }
            for hint in (getattr(bottleneck_report, "rewrite_hints", []) or [])
        ],
        "bottleneck_explanation": getattr(bottleneck_report, "explanation", None),
    }


def _free_exploration_failed_directions(
    *,
    rewrite_response: AgentResponse,
    validation_response: AgentResponse,
    failed_candidate: SQLVersion,
) -> list[dict]:
    if not _is_free_exploration_candidate(failed_candidate):
        return []
    artifacts = rewrite_response.output_artifacts or {}
    exploration_context = artifacts.get("free_exploration_context") or {}
    previous = exploration_context.get("failed_free_exploration_directions") or []
    result: list[dict] = []
    for item in previous:
        if isinstance(item, dict):
            result.append(dict(item))
        else:
            result.append({"direction": str(item)})
    report = validation_response.output_artifacts.get("validation_report")
    failure_reason = getattr(report, "failure_reason", None) or "; ".join(
        validation_response.errors
    )
    result.append(
        {
            "direction": exploration_context.get(
                "current_exploration_direction",
                "LLM free exploration",
            ),
            "sql": failed_candidate.sql,
            "failure_reason": failure_reason,
            "rewrite_rule_ids": list(failed_candidate.rewrite_rule_ids),
        }
    )
    return result


def _record_validation_reflection_attempt(
    memory: ReflectionMemory,
    *,
    candidate_sql: SQLVersion,
    validation_response: AgentResponse,
    reflection_context: dict,
    status: str,
    notes: str | None = None,
) -> ReflectionMemory:
    report = validation_response.output_artifacts.get("validation_report")
    failure = getattr(report, "failure_reason", None) or "; ".join(validation_response.errors)
    failure_type = str(reflection_context.get("failure_type") or "unknown")
    attempt = AttemptRecord(
        attempt_id=uuid.uuid4().hex[:12],
        sql_version_id=candidate_sql.version_id,
        stage=f"optimization_{failure_type}_reflection",
        status=status,
        failure_reason=failure,
        notes=notes
        or (
            f"before={reflection_context.get('source_sql_version_id')} "
            f"after={candidate_sql.version_id}; maxRetry={reflection_context.get('max_retry')}"
        ),
    )
    rejected_rules = list(memory.rejected_rewrite_rules)
    for rule_id in candidate_sql.rewrite_rule_ids:
        if rule_id not in rejected_rules and status not in {"repair_accepted"}:
            rejected_rules.append(rule_id)
    failed_assumptions = list(memory.failed_assumptions)
    if failure and status not in {"repair_accepted"}:
        assumption = f"{failure_type}: {failure}"
        if assumption not in failed_assumptions:
            failed_assumptions.append(assumption)
    confirmed_facts = list(memory.confirmed_facts)
    if status == "repair_accepted":
        fact = f"reflection repaired {failure_type} for {candidate_sql.version_id}"
        if fact not in confirmed_facts:
            confirmed_facts.append(fact)
    return replace(
        memory,
        attempts=[*memory.attempts, attempt],
        failed_assumptions=failed_assumptions,
        confirmed_facts=confirmed_facts,
        rejected_rewrite_rules=rejected_rules,
        next_strategy_hint=(
            f"Use validation_reflection_repair for {failure_type}; "
            "preserve source SQL semantics before optimizing."
        ),
    )
