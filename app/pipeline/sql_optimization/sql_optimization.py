from __future__ import annotations

import math
import re
import sqlite3
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from sqlglot import exp, parse

from app.dataset import BaseDataset, DataItem, load_dataset, save_dataset
from app.llm import LLM
from app.logger import logger
from app.meta_controller import ControllerAction, MetaControllerRuntime
from app.pipeline.validation import validate_pipeline_step
from app.progress import log_progress, should_checkpoint
from app.services import (
    ArtifactStore,
    STAGE_ARTIFACT_FIELDS,
    configure_execution_service,
    configure_schema_service,
    get_execution_service,
    get_schema_service,
    load_stage_dataset,
    reset_execution_service,
    reset_schema_service,
)


_RESULT_PATTERN = re.compile(r"<result>(.*?)</result>", re.DOTALL | re.IGNORECASE)


def _zero_usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _safe_read_query(sql: str) -> bool:
    try:
        statements = parse(sql, read="sqlite")
    except Exception:
        return False
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        return False
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter)
    return not any(statements[0].find(node_type) is not None for node_type in forbidden)


def _order_sensitive(sql: str, question: str) -> bool:
    try:
        statement = parse(sql, read="sqlite")[0]
        if statement.find(exp.Order) is not None:
            return True
    except Exception:
        return True
    lowered = question.lower()
    return any(term in lowered for term in ("order", "sorted", "ascending", "descending", "first", "last", "top "))


def _equivalent_results(base_result: Any, candidate_result: Any, *, order_sensitive: bool) -> bool:
    if base_result.result_rows is None or candidate_result.result_rows is None:
        return False
    if list(base_result.result_cols or []) != list(candidate_result.result_cols or []):
        return False
    base_rows = [tuple(row) for row in base_result.result_rows]
    candidate_rows = [tuple(row) for row in candidate_result.result_rows]
    if order_sensitive:
        return base_rows == candidate_rows
    return Counter(map(repr, base_rows)) == Counter(map(repr, candidate_rows))


def _sqlite_plan(data_item: DataItem, sql: str) -> tuple[list[str], str | None]:
    if getattr(data_item, "db_type", None) not in (None, "sqlite"):
        return [], "execution-plan inspection is available only for SQLite"
    try:
        db_path = str(Path(data_item.database_path).resolve())
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
            rows = connection.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()
        return [str(row[-1]) for row in rows], None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def _plan_risk_score(sql: str, plan: list[str]) -> tuple[int, list[str]]:
    reasons: list[str] = []
    normalized_plan = [detail.upper() for detail in plan]
    scan_count = sum("SCAN " in detail and "USING INDEX" not in detail for detail in normalized_plan)
    if scan_count:
        reasons.append(f"{scan_count} full scan(s)")
    if any("TEMP B-TREE" in detail for detail in normalized_plan):
        reasons.append("temporary B-tree")
    if any("CORRELATED" in detail for detail in normalized_plan):
        reasons.append("correlated subquery")
    try:
        statement = parse(sql, read="sqlite")[0]
        subquery_count = sum(1 for _ in statement.find_all(exp.Subquery))
    except Exception:
        subquery_count = 0
    if subquery_count >= 2:
        reasons.append(f"{subquery_count} nested subqueries")
    score = scan_count
    score += 2 if "temporary B-tree" in reasons else 0
    score += 2 if "correlated subquery" in reasons else 0
    score += 1 if subquery_count >= 2 else 0
    return score, reasons


def _rewrite_prompt(data_item: DataItem, base_sql: str, plan: list[str], reasons: list[str]) -> str:
    schema_profile = get_schema_service().build_schema_profile(
        data_item.database_schema_after_schema_linking or data_item.database_schema,
        include_value_statistics=False,
        include_value_examples=False,
    )
    return f"""You are a conservative SQLite query optimizer.

Rewrite the SQL only when you can preserve exactly the same output columns, rows,
duplicates, NULL behavior, aggregation, ordering, and LIMIT semantics. Do not repair
the intended answer and do not add or remove requested columns. Use only the schema
shown below. Return either <result>NO_CHANGE</result> or one SQL query inside
<result>...</result>. Never return multiple statements or data-changing SQL.

Question:
{data_item.question}

Evidence:
{data_item.evidence}

Schema:
{schema_profile}

Base SQL:
{base_sql}

EXPLAIN QUERY PLAN:
{chr(10).join(plan) if plan else '(unavailable)'}

Detected risks:
{', '.join(reasons) if reasons else 'none'}
"""


def _parse_rewrite(content: str | None) -> str | None:
    if not content:
        return None
    match = _RESULT_PATTERN.search(content)
    if not match:
        return None
    sql = match.group(1).strip()
    if sql.upper() == "NO_CHANGE":
        return None
    if sql.startswith("```sql") and sql.endswith("```"):
        sql = sql[6:-3].strip()
    return sql if _safe_read_query(sql) else None


class SQLOptimizationRunner:
    def __init__(
        self,
        stage_config: Any,
        dataset_config: Any,
        input_save_path: str,
        parallelism: int,
        progress_log_interval: int,
        checkpoint_interval: int,
        meta_controller_config: Any = None,
    ):
        self._stage_config = stage_config
        self._dataset_config = dataset_config
        self._parallelism = max(1, parallelism)
        self._progress_log_interval = max(1, progress_log_interval)
        self._checkpoint_interval = max(1, checkpoint_interval)
        self._artifact_store = ArtifactStore(
            self._stage_config.save_path,
            "sql_optimization",
            STAGE_ARTIFACT_FIELDS["sql_optimization"],
        )
        self._dataset, checkpoint_source = load_stage_dataset(
            load_dataset_fn=load_dataset,
            current_save_path=self._stage_config.save_path,
            fallback_load_path=input_save_path,
            artifact_store=self._artifact_store,
            stage_name="sql_optimization",
        )
        logger.info(f"Initialized SQL optimization dataset from {checkpoint_source}")
        configure_schema_service(max_value_example_length=self._dataset_config.max_value_example_length)
        configure_execution_service(default_timeout=self._dataset_config.sql_execution_timeout)
        self._execution_service = get_execution_service()
        self._llm = LLM(self._stage_config.llm) if self._stage_config.llm is not None else None
        self._meta_controller = (
            MetaControllerRuntime(meta_controller_config)
            if meta_controller_config is not None and meta_controller_config.enabled
            else None
        )
        self._thread_pool = ThreadPoolExecutor(max_workers=self._parallelism)

    @classmethod
    def from_config(cls, app_config=None) -> "SQLOptimizationRunner":
        if app_config is None:
            from app.config import get_config

            app_config = get_config()
        return cls(
            stage_config=app_config.sql_optimization_config,
            dataset_config=app_config.dataset_config,
            input_save_path=app_config.sql_selection_config.save_path,
            parallelism=app_config.run_config.parallelism,
            progress_log_interval=app_config.run_config.progress_log_interval,
            checkpoint_interval=app_config.run_config.checkpoint_interval,
            meta_controller_config=app_config.meta_controller_config,
        )

    def _finish(self, data_item: DataItem, start_time: float, usage: dict[str, int]) -> None:
        data_item.sql_optimization_time = time.time() - start_time
        data_item.sql_optimization_llm_cost = usage
        data_item.total_time += data_item.sql_optimization_time
        for key in usage:
            data_item.total_llm_cost[key] += usage[key]

    def _optimize_one(self, data_item: DataItem) -> None:
        start_time = time.time()
        usage = _zero_usage()
        base_sql = (data_item.final_selected_sql or "").strip()
        data_item.final_optimized_sql = base_sql or "Error"
        trace: dict[str, Any] = {"base_sql": base_sql, "gold_accessed": False}
        if not base_sql or not _safe_read_query(base_sql):
            data_item.optimization_status = "fallback_invalid_base"
            data_item.optimization_trace = trace
            self._finish(data_item, start_time, usage)
            return

        plan, plan_error = _sqlite_plan(data_item, base_sql)
        risk_score, risk_reasons = _plan_risk_score(base_sql, plan)
        trace.update({"plan": plan, "plan_error": plan_error, "risk_score": risk_score, "risk_reasons": risk_reasons})
        decision = (
            self._meta_controller.assess_optimization(
                data_item,
                risk_score=risk_score,
                min_risk_score=self._stage_config.min_risk_score,
                max_rewrites=self._stage_config.max_rewrites,
                trigger_mode=self._stage_config.trigger_mode,
            )
            if self._meta_controller is not None
            else None
        )
        should_optimize = (
            self._stage_config.max_rewrites > 0
            and (self._stage_config.trigger_mode == "all" or risk_score >= self._stage_config.min_risk_score)
        )
        if not should_optimize:
            data_item.optimization_status = "not_triggered"
            data_item.optimization_trace = trace
            self._finish(data_item, start_time, usage)
            if decision is not None:
                self._meta_controller.record_stage_outcome(
                    decision, status=data_item.optimization_status, selected_sql=base_sql, token_usage=usage
                )
            return

        base_result = self._execution_service.execute(data_item, base_sql)
        prompt = _rewrite_prompt(data_item, base_sql, plan, risk_reasons)
        status = "no_change"
        for attempt in range(1, self._stage_config.max_rewrites + 1):
            choices, attempt_usage = self._llm.ask([{"role": "user", "content": prompt}], n=1)
            for key in usage:
                usage[key] += attempt_usage[key]
            candidate_sql = _parse_rewrite(choices[0].content)
            trace.setdefault("attempts", []).append({"attempt": attempt, "candidate_sql": candidate_sql})
            if not candidate_sql or candidate_sql.strip() == base_sql:
                status = "no_change"
                break
            candidate_result = self._execution_service.execute(data_item, candidate_sql)
            equivalent = _equivalent_results(
                base_result,
                candidate_result,
                order_sensitive=_order_sensitive(base_sql, data_item.question),
            )
            trace["attempts"][-1]["equivalent"] = equivalent
            if not equivalent:
                status = "rejected_not_equivalent"
                prompt += "\nThe previous rewrite changed the result. Return NO_CHANGE or a strictly equivalent alternative."
                continue
            base_time = self._execution_service.measure_time(
                data_item, base_sql, repeat=self._stage_config.timing_repeat, use_cache=False
            )
            candidate_time = self._execution_service.measure_time(
                data_item, candidate_sql, repeat=self._stage_config.timing_repeat, use_cache=False
            )
            speedup = base_time / candidate_time if candidate_time > 0 and math.isfinite(candidate_time) else 0.0
            trace["attempts"][-1].update(
                {"base_seconds": base_time, "candidate_seconds": candidate_time, "speedup": speedup}
            )
            if base_time * 1000 < self._stage_config.min_runtime_ms:
                status = "rejected_runtime_too_small"
                break
            if speedup < self._stage_config.min_speedup_ratio:
                status = "rejected_not_faster"
                prompt += "\nThe previous rewrite was equivalent but not measurably faster. Return NO_CHANGE or a better alternative."
                continue
            data_item.final_optimized_sql = candidate_sql
            status = "accepted"
            break

        data_item.optimization_status = status
        data_item.optimization_trace = trace
        self._finish(data_item, start_time, usage)
        if decision is not None:
            self._meta_controller.record_stage_outcome(
                decision,
                status=status,
                selected_sql=data_item.final_optimized_sql,
                token_usage=usage,
            )

    def _optimize_one_safe(self, data_item: DataItem) -> None:
        try:
            self._optimize_one(data_item)
        except Exception as exc:
            logger.exception(f"SQL optimization failed for item {data_item.question_id}: {exc}")
            data_item.final_optimized_sql = data_item.final_selected_sql or "Error"
            data_item.optimization_status = "error_fallback"
            data_item.optimization_trace = {
                "base_sql": data_item.final_selected_sql,
                "gold_accessed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            if data_item.sql_optimization_time is None:
                data_item.sql_optimization_time = 0.0
            if data_item.sql_optimization_llm_cost is None:
                data_item.sql_optimization_llm_cost = _zero_usage()

    def run(self) -> None:
        future_to_item = {}
        for data_item in self._dataset:
            if data_item.is_stage_complete("sql_optimization"):
                continue
            future_to_item[self._thread_pool.submit(self._optimize_one_safe, data_item)] = data_item
        for index, future in enumerate(as_completed(future_to_item), 1):
            future.result()
            self._artifact_store.record_item(future_to_item[future])
            log_progress("Optimizing SQL", index, len(future_to_item), self._progress_log_interval, previous_completed=index - 1)
            if should_checkpoint(index, self._checkpoint_interval):
                self.save_result()
        self._artifact_store.flush()
        validate_pipeline_step(self._dataset, "sql_optimization")
        self.save_result(materialize_snapshot=True)
        self._clean_up()

    def save_result(self, materialize_snapshot: bool = False) -> None:
        self._artifact_store.flush()
        if materialize_snapshot:
            save_dataset(self._dataset, self._stage_config.save_path)
            self._artifact_store.cleanup()

    def _clean_up(self) -> None:
        self._thread_pool.shutdown(wait=True)
        self._artifact_store.close()
        reset_execution_service()
        reset_schema_service()
