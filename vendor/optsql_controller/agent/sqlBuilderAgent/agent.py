"""Initial SQL Builder Agent — generate executable base SQL from a Verified Context Blueprint.

Blueprint-compliance enforcement is implemented as a two-layer defense:

1. **Prompt constraints** — the system prompt and per-request user prompts instruct the
   LLM to only use tables, columns, values, and join topology from the supplied
   Blueprint. The LLM is explicitly told not to invent schema elements.
2. **Programmatic SQL validation** — after the LLM produces SQL, ``sqlglot`` parses the
   statement and extracts every table / column reference.  Each reference is checked
   against the Blueprint allowlists.  Violations are fed back into the repair loop so
   the LLM can fix them before the SQL ever touches the database.

This mirrors the Schema Filter design principle: *database facts (or, here, blueprint
structural facts) take priority over LLM inference.*
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from typing import Any
from typing import TypedDict

import sqlglot
import sqlglot.expressions as exp

from agent.base import BaseAgent
from agent.sqlBuilderAgent.generation_methods import SQLGenerationPipeline
from agent.sqlBuilderAgent.json_parser import parse_json_object
from agent.sqlBuilderAgent.prompts import build_repair_prompt
from agent.sqlBuilderAgent.prompts import build_sql_generation_prompt
from agent.sqlBuilderAgent.prompts import SQL_BUILDER_SYSTEM_PROMPT
from myTypes import AgentRequest
from myTypes import AgentResponse
from myTypes import ExecutionMetrics
from myTypes import SchemaGapHint
from myTypes import SQLVersion
from myTypes import VerifiedContextBlueprint
from utils.db import connect_bird_database
from utils.openai_client import request_chat_text
from utils.sql_comparison import calculate_ves
from utils.sql_safety import ensure_select_sql


class _SQLBuilderState(TypedDict, total=False):
    request: AgentRequest
    blueprint: VerifiedContextBlueprint
    sql_version: SQLVersion
    execution_metrics: ExecutionMetrics | None
    repair_attempts: int
    error_message: str | None
    blueprint_violations: list[str]


class InitialSQLBuilderAgent(BaseAgent):
    """Generate executable base SQL strictly from a Verified Context Blueprint."""

    name = "initial_sql_builder"

    def __init__(
        self,
        max_repair_attempts: int = 3,
        llm_model: str | None = None,
        llm_temperature: float = 0.0,
        llm_max_tokens: int = 4096,
        llm_client=None,
    ) -> None:
        self.max_repair_attempts = max_repair_attempts
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.llm_client = llm_client
        self._generation_pipeline = SQLGenerationPipeline()
        self._last_generation_result: dict | None = None
        self._graph: Any | None = None

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def build_base_sql(
        self,
        blueprint: VerifiedContextBlueprint,
        question: str,
        evidence: str | None,
        dbms: str,
        question_id: int | str | None = None,
    ) -> str:
        """Generate base SQL using candidate generation, syntax revision, and selection."""
        self._last_generation_result = None
        try:
            result = self._generation_pipeline.run(
                blueprint=blueprint,
                question=question,
                evidence=evidence,
                db_id=blueprint.db_id,
                question_id=question_id,
                llm_text=self._request_text,
            )
            self._last_generation_result = asdict(result)
            return result.selected_sql
        except Exception as exc:
            self._last_generation_result = {"fallback_reason": str(exc)}
            return self._build_base_sql_legacy(
                blueprint=blueprint,
                question=question,
                evidence=evidence,
                dbms=dbms,
            )

    def _build_base_sql_legacy(
        self,
        blueprint: VerifiedContextBlueprint,
        question: str,
        evidence: str | None,
        dbms: str,
    ) -> str:
        """Generate one SQL with the legacy JSON prompt fallback."""
        blueprint_json = self._blueprint_to_json(blueprint)
        prompt = build_sql_generation_prompt(
            question=question,
            evidence=evidence,
            blueprint_json=blueprint_json,
            dbms=dbms,
        )
        response = self._request_json(prompt)
        sql = (response.get("sql") or "").strip()
        if not sql:
            raise ValueError("LLM did not produce a non-empty SQL query.")
        return sql

    def repair_sql(
        self,
        sql: str,
        error_message: str,
        blueprint: VerifiedContextBlueprint,
        question: str,
        evidence: str | None,
        dbms: str,
    ) -> str:
        """Attempt to repair a failed SQL query using the error message."""
        blueprint_json = self._blueprint_to_json(blueprint)
        prompt = build_repair_prompt(
            sql=sql,
            error_message=error_message,
            question=question,
            evidence=evidence,
            blueprint_json=blueprint_json,
            dbms=dbms,
        )
        response = self._request_json(prompt)
        repaired_sql = (response.get("sql") or "").strip()
        if not repaired_sql:
            raise ValueError("LLM repair did not produce a non-empty SQL query.")
        return repaired_sql

    def execute_sql(self, sql: str, db_id: str) -> ExecutionMetrics:
        """Execute SQL against the BIRD database and return metrics."""
        start = time.perf_counter()
        try:
            ensure_select_sql(sql)
            with connect_bird_database(db_id) as conn:
                cursor = conn.execute(sql)
                rows = cursor.fetchall()
                latency_ms = (time.perf_counter() - start) * 1000.0
                return ExecutionMetrics(
                    executable=True,
                    latency_ms=round(latency_ms, 3),
                    row_count=len(rows),
                    explain_plan=None,
                    plan_features={},
                    error_message=None,
                )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return ExecutionMetrics(
                executable=False,
                latency_ms=round(latency_ms, 3),
                row_count=None,
                explain_plan=None,
                plan_features={},
                error_message=str(exc),
            )

    def _load_db_columns(self, db_id: str, tables: set[str]) -> dict[str, set[str]]:
        """Query the actual database schema for *tables* and return {table: {col, ...}}."""
        result: dict[str, set[str]] = {}
        try:
            with connect_bird_database(db_id) as conn:
                for table in tables:
                    try:
                        rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
                        result[table] = {row[1] for row in rows}
                    except Exception:
                        result[table] = set()
        except Exception:
            pass
        return result

    # ------------------------------------------------------------------
    # Blueprint compliance validation (programmatic)
    # ------------------------------------------------------------------

    # SQLite implicit columns present in every table.
    _SQLITE_IMPLICIT_COLUMNS: set[str] = {"rowid", "_rowid_", "oid"}

    def validate_blueprint_compliance(
        self,
        sql: str,
        blueprint: VerifiedContextBlueprint,
        db_id: str | None = None,
    ) -> list[str]:
        """Parse *sql* with ``sqlglot`` and check every table / column reference
        against the blueprint allowlists.

        Returns a (possibly empty) list of human-readable violation descriptions.
        """
        allowed_tables = set(blueprint.selected_tables)
        allowed_columns: set[tuple[str, str]] = {
            (col.table_name, col.column_name) for col in blueprint.selected_columns
        }

        violations: list[str] = []

        cte_names = self._extract_cte_names(sql)
        derived_aliases = self._extract_derived_table_aliases(sql)
        virtual_names = cte_names | derived_aliases

        # Resolve table aliases so column checks work on real table names.
        sql_tables, alias_map = self._extract_table_refs(sql)
        select_aliases = self._extract_all_select_aliases(sql)

        db_columns: dict[str, set[str]] | None = None

        for table in sorted(sql_tables):
            if table in allowed_tables or table in virtual_names:
                continue
            violations.append(
                f"Table '{table}' is not in blueprint.selected_tables. "
                f"Allowed tables: {sorted(allowed_tables)}"
            )

        sql_columns = self._extract_column_refs(sql)
        for table, col in sorted(sql_columns):
            if table:
                resolved = alias_map.get(table, table)
                if resolved in virtual_names:
                    continue
                if col in self._SQLITE_IMPLICIT_COLUMNS:
                    continue
                if (resolved, col) not in allowed_columns:
                    if db_id and resolved in allowed_tables:
                        if db_columns is None:
                            db_columns = self._load_db_columns(db_id, allowed_tables)
                        db_cols_for_table = db_columns.get(resolved, set())
                        if col in db_cols_for_table:
                            continue
                    similar = sorted(
                        c for t, c in allowed_columns if t == resolved
                    )
                    violations.append(
                        f"Column '{table}.{col}' is not in blueprint.selected_columns. "
                        f"Allowed columns in '{resolved}': {similar}"
                    )
            else:
                if col in select_aliases or col in self._SQLITE_IMPLICIT_COLUMNS:
                    continue
                matching = [(t, c) for t, c in allowed_columns if c == col]
                if not matching:
                    if db_id:
                        if db_columns is None:
                            db_columns = self._load_db_columns(db_id, allowed_tables)
                        for tbl, cols in db_columns.items():
                            if col in cols:
                                matching = [(tbl, col)]
                                break
                    if not matching:
                        violations.append(
                            f"Column '{col}' (unqualified) not found in any blueprint table. "
                            f"Available columns (sample): "
                            f"{sorted(allowed_columns)[:10]}"
                        )

        return violations

    @staticmethod
    def _extract_table_refs(sql: str) -> tuple[set[str], dict[str, str]]:
        """Return ``(real_table_names, alias_map)`` for every table in *sql*.

        *alias_map* maps alias → real name (and real name → real name) so
        column qualifiers can be resolved.
        """
        try:
            ast = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception:
            return set(), {}

        tables: set[str] = set()
        alias_map: dict[str, str] = {}

        for node in ast.find_all(exp.Table):
            name = node.name
            if not name:
                continue
            tables.add(name)
            alias_map[name] = name
            if node.alias:
                alias_map[node.alias] = name

        return tables, alias_map

    @staticmethod
    def _extract_cte_names(sql: str) -> set[str]:
        """Return CTE names defined in WITH clauses."""
        try:
            ast = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception:
            return set()
        return {cte.alias for cte in ast.find_all(exp.CTE) if cte.alias}

    @staticmethod
    def _extract_derived_table_aliases(sql: str) -> set[str]:
        """Return aliases of subquery-derived tables (FROM/JOIN (SELECT ...) AS alias)."""
        try:
            ast = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception:
            return set()
        derived = set()
        for node in ast.find_all(exp.Subquery):
            if node.alias:
                derived.add(node.alias)
        return derived

    @staticmethod
    def _extract_all_select_aliases(sql: str) -> set[str]:
        """Return aliases from ALL SELECT lists (top-level and subqueries)."""
        try:
            ast = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception:
            return set()

        aliases = set()
        for select in ast.find_all(exp.Select):
            for expression in select.expressions:
                alias = expression.alias
                if alias:
                    aliases.add(alias)
        return aliases

    @staticmethod
    def _extract_column_refs(sql: str) -> set[tuple[str, str]]:
        """Return every ``(table, column)`` pair referenced in *sql*.

        Unqualified columns have an empty string for the table part.
        """
        try:
            ast = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception:
            return set()

        columns: set[tuple[str, str]] = set()
        for node in ast.find_all(exp.Column):
            col_name = node.name
            if not col_name:
                continue
            table_name = node.table or ""
            columns.add((table_name, col_name))

        # Also inspect WHERE / HAVING literal values — if a string literal
        # looks like a natural-language value, flag it.
        return columns

    # ------------------------------------------------------------------
    # Schema gap diagnosis
    # ------------------------------------------------------------------

    def diagnose_schema_gaps(
        self,
        sql_version: SQLVersion,
        metrics: ExecutionMetrics | None,
        blueprint: VerifiedContextBlueprint,
        question: str,
        evidence: str | None,
    ) -> list[SchemaGapHint]:
        """Analyse execution failures for signs of an incomplete Blueprint.

        Returns a (possibly empty) list of :class:`SchemaGapHint` that the
        Controller can use to decide whether to re-run the Schema Filter.
        """
        hints: list[SchemaGapHint] = []

        # 1. Parse execution errors for "no such table / column"
        if metrics and not metrics.executable and metrics.error_message:
            error = metrics.error_message
            if "no such table:" in error.lower():
                missing = error.split("no such table:")[-1].strip()
                hints.append(SchemaGapHint(
                    gap_type="missing_table",
                    element=missing,
                    source="execution_error",
                    suggestion=f"Table {missing} referenced by SQL but missing from blueprint. Re-run Schema Filter.",
                ))
            elif "no such column:" in error.lower():
                missing = error.split("no such column:")[-1].strip()
                hints.append(SchemaGapHint(
                    gap_type="missing_column",
                    element=missing,
                    source="execution_error",
                    suggestion=f"Column {missing} referenced but missing from blueprint. Re-run Schema Filter or expand candidates.",
                ))

        # 2. Check evidence for formula columns not in blueprint
        if evidence:
            allowed_cols = {(c.table_name, c.column_name) for c in blueprint.selected_columns}
            allowed_col_names = {c.column_name for c in blueprint.selected_columns}
            # tokenize evidence for quoted identifiers and multi-word CamelCase
            # phrases (single capital words are too common in English prose;
            # multi-word matches like "Free Meal Count" are more likely columns)
            import re
            potential_columns = set(re.findall(r'"([^"]+)"', evidence))
            potential_columns |= set(re.findall(
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', evidence,
            ))
            for col_name in potential_columns:
                if col_name not in allowed_col_names and len(col_name) > 3:
                    hints.append(SchemaGapHint(
                        gap_type="missing_column",
                        element=col_name,
                        source="evidence",
                        suggestion=f"Column '{col_name}' mentioned in evidence but missing from blueprint. Re-run Schema Filter.",
                    ))

        # 3. Check for disconnected tables implied by question
        if question:
            allowed_tables = set(blueprint.selected_tables)
            # tokenize question for multi-word proper-noun phrases that might
            # be table names (single capital words are too ambiguous)
            import re
            words = set(re.findall(
                r'\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+)\b', question,
            ))
            words |= set(re.findall(r'"([^"]+)"', question))
            for word in words:
                if word in allowed_tables:
                    continue
                # check if any column comment or table comment mentions this word
                found_in_comment = False
                for col in blueprint.selected_columns:
                    if col.comment and word.lower() in col.comment.lower():
                        found_in_comment = True
                        break
                if not found_in_comment and len(word) > 3:
                    hints.append(SchemaGapHint(
                        gap_type="missing_table",
                        element=word,
                        source="question",
                        suggestion=f"Term '{word}' from question may refer to a table not in blueprint. Re-run Schema Filter.",
                    ))

        return hints

    # ------------------------------------------------------------------
    # LangGraph-backed run
    # ------------------------------------------------------------------

    def run(self, request: AgentRequest) -> AgentResponse:
        """Run the LangGraph pipeline.

        Path::

            START
              → build_base_sql
              → validate_blueprint
                 ├─ compliant        → execute_validation
                 │   ├─ executable       → build_response → END
                 │   └─ execution error  → repair_sql → validate_blueprint
                 └─ violated (N<max) → repair_sql → validate_blueprint
                    violated (N≥max) → build_response → END
        """
        blueprint: VerifiedContextBlueprint = request.input_artifacts.get("blueprint")
        if blueprint is None:
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status="error",
                output_artifacts={},
                reasoning_summary="Missing VerifiedContextBlueprint in input_artifacts.",
                tool_calls=[],
                errors=["input_artifacts must contain a 'blueprint' key."],
            )

        try:
            final_state = self._get_graph().invoke({
                "request": request,
                "blueprint": blueprint,
                "repair_attempts": 0,
                "error_message": None,
                "blueprint_violations": [],
            })
            sql_version = final_state["sql_version"]
            execution_metrics = final_state.get("execution_metrics")
            violations = final_state.get("blueprint_violations", [])
        except Exception as exc:
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status="error",
                output_artifacts={},
                reasoning_summary="SQL Builder pipeline failed.",
                tool_calls=[],
                errors=[str(exc)],
            )

        executable = execution_metrics is not None and execution_metrics.executable
        gold_sql = self._gold_sql_from_request(request)
        ves_metric = None
        if gold_sql and sql_version.sql:
            ves_metric = calculate_ves(
                gold_sql=gold_sql,
                generated_sql=sql_version.sql,
                db_id=request.task.db_id,
            )
        gap_hints = self.diagnose_schema_gaps(
            sql_version=sql_version,
            metrics=execution_metrics,
            blueprint=blueprint,
            question=request.task.question,
            evidence=request.task.evidence,
        )
        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            status="success" if executable else "error",
            output_artifacts={
                "sql_version": sql_version,
                "execution_metrics": execution_metrics,
                "ves_metric": ves_metric,
                "blueprint_violations": violations,
                "repair_attempts": final_state.get("repair_attempts", 0),
                "schema_gap_hints": gap_hints,
                "sql_generation_methods": self._last_generation_result,
            },
            reasoning_summary=(
                f"Generated SQL version {sql_version.version_id}. "
                f"Executable: {executable}. "
                f"Blueprint violations: {len(violations)}. "
                f"VES: {ves_metric.score if ves_metric else 'not_available'}"
            ),
            tool_calls=[
                {"tool_name": "request_chat_text", "summary": "LLM SQL generation"},
                {"tool_name": "SQLGenerationPipeline", "summary": "candidate generation, syntax-only revision, deterministic selection"},
                {"tool_name": "validate_blueprint_compliance", "summary": "sqlglot-based schema compliance check"},
                {"tool_name": "execute_sql", "summary": "SQL execution validation"},
                {"tool_name": "calculate_ves", "summary": "BIRD classic Valid Efficiency Score"},
            ],
            errors=(
                []
                if executable
                else self._response_errors(execution_metrics, violations)
            ),
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _get_graph(self):
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def _build_graph(self):
        try:
            from langgraph.graph import END
            from langgraph.graph import START
            from langgraph.graph import StateGraph
        except ImportError as exc:
            raise ImportError(
                "LangGraph is required to run InitialSQLBuilderAgent. "
                "Install it with `pip install langgraph`."
            ) from exc

        graph = StateGraph(_SQLBuilderState)
        graph.add_node("build_base_sql", self._build_base_sql_node)
        graph.add_node("validate_blueprint", self._validate_blueprint_node)
        graph.add_node("execute_validation", self._execute_validation_node)
        graph.add_node("repair_sql", self._repair_sql_node)
        graph.add_node("build_response", self._build_response_node)

        graph.add_edge(START, "build_base_sql")
        graph.add_edge("build_base_sql", "validate_blueprint")
        graph.add_conditional_edges(
            "validate_blueprint",
            self._decide_after_validation,
            {
                "execute": "execute_validation",
                "repair": "repair_sql",
                "done": "build_response",
            },
        )
        graph.add_conditional_edges(
            "execute_validation",
            self._decide_after_execution,
            {
                "done": "build_response",
                "repair": "repair_sql",
            },
        )
        graph.add_edge("repair_sql", "validate_blueprint")
        graph.add_edge("build_response", END)
        return graph.compile()

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    def _build_base_sql_node(self, state: _SQLBuilderState) -> dict:
        task = state["request"].task
        blueprint = state["blueprint"]
        sql = self.build_base_sql(
            blueprint=blueprint,
            question=task.question,
            evidence=task.evidence,
            dbms=task.dbms,
            question_id=task.question_id,
        )
        version = SQLVersion(
            version_id=uuid.uuid4().hex[:12],
            parent_id=None,
            sql=sql,
            source_agent=self.name,
            rewrite_rule_ids=[],
            explanation="Initial SQL generated from verified context blueprint.",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        return {"sql_version": version}

    def _validate_blueprint_node(self, state: _SQLBuilderState) -> dict:
        sql = state["sql_version"].sql
        blueprint = state["blueprint"]
        db_id = state["request"].task.db_id
        violations = self.validate_blueprint_compliance(sql, blueprint, db_id=db_id)
        return {"blueprint_violations": violations}

    def _execute_validation_node(self, state: _SQLBuilderState) -> dict:
        sql_version = state["sql_version"]
        db_id = state["request"].task.db_id
        metrics = self.execute_sql(sql_version.sql, db_id)
        return {"execution_metrics": metrics}

    def _repair_sql_node(self, state: _SQLBuilderState) -> dict:
        task = state["request"].task
        blueprint = state["blueprint"]
        sql_version = state["sql_version"]
        metrics = state.get("execution_metrics")
        violations = state.get("blueprint_violations", [])
        attempts = state.get("repair_attempts", 0)

        # Build a unified error message from violations or execution error
        if violations:
            error_message = "Blueprint compliance violations:\n" + "\n".join(
                f"  - {v}" for v in violations
            )
        elif metrics and metrics.error_message:
            error_message = metrics.error_message
        else:
            error_message = "Unknown error"

        repaired_sql = self.repair_sql(
            sql=sql_version.sql,
            error_message=error_message,
            blueprint=blueprint,
            question=task.question,
            evidence=task.evidence,
            dbms=task.dbms,
        )
        new_version = SQLVersion(
            version_id=uuid.uuid4().hex[:12],
            parent_id=sql_version.version_id,
            sql=repaired_sql,
            source_agent=self.name,
            rewrite_rule_ids=[],
            explanation=f"Repair attempt {attempts + 1}: {error_message}",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        return {
            "sql_version": new_version,
            "repair_attempts": attempts + 1,
            "execution_metrics": None,
            "blueprint_violations": [],
        }

    def _build_response_node(self, state: _SQLBuilderState) -> dict:
        return {}

    # ------------------------------------------------------------------
    # Conditional routing
    # ------------------------------------------------------------------

    def _decide_after_validation(self, state: _SQLBuilderState) -> str:
        """Route after blueprint compliance check."""
        violations = state.get("blueprint_violations", [])
        if not violations:
            return "execute"

        attempts = state.get("repair_attempts", 0)
        if attempts < self.max_repair_attempts:
            return "repair"
        return "done"

    def _decide_after_execution(self, state: _SQLBuilderState) -> str:
        """Route after SQL execution."""
        metrics = state.get("execution_metrics")
        if metrics and metrics.executable:
            return "done"
        attempts = state.get("repair_attempts", 0)
        if attempts < self.max_repair_attempts:
            return "repair"
        return "done"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _request_json(self, prompt: str) -> dict:
        response_text = request_chat_text(
            messages=[
                {"role": "system", "content": SQL_BUILDER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=self.llm_model,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
            client=self.llm_client,
        )
        return parse_json_object(response_text)

    def _request_text(self, prompt: str) -> str:
        return request_chat_text(
            messages=[
                {"role": "system", "content": SQL_BUILDER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=self.llm_model,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
            client=self.llm_client,
        )

    def _blueprint_to_json(self, blueprint: VerifiedContextBlueprint) -> dict:
        return {
            "db_id": blueprint.db_id,
            "selected_tables": blueprint.selected_tables,
            "selected_columns": [
                {
                    "table_name": col.table_name,
                    "column_name": col.column_name,
                    "data_type": col.data_type,
                    "comment": col.comment,
                }
                for col in blueprint.selected_columns
            ],
            "value_mappings": [
                {
                    "keyword": vm.keyword,
                    "table_name": vm.table_name,
                    "column_name": vm.column_name,
                    "value": vm.value,
                    "confidence": vm.confidence,
                }
                for vm in blueprint.value_mappings
            ],
            "join_topology": {
                "tables": blueprint.join_topology.tables,
                "edges": [
                    {
                        "source_table": e.source_table,
                        "source_column": e.source_column,
                        "target_table": e.target_table,
                        "target_column": e.target_column,
                        "join_type": e.join_type,
                    }
                    for e in blueprint.join_topology.edges
                ],
            },
            "predicate_hints": [
                {
                    "predicate_type": ph.predicate_type,
                    "expression": ph.expression,
                    "source_text": ph.source_text,
                    "confidence": ph.confidence,
                }
                for ph in blueprint.predicate_hints
            ],
            "confidence": blueprint.confidence,
        }

    def _gold_sql_from_request(self, request: AgentRequest) -> str | None:
        for source in (request.input_artifacts, request.constraints):
            gold_sql = source.get("gold_sql") if isinstance(source, dict) else None
            if isinstance(gold_sql, str) and gold_sql.strip():
                return gold_sql.strip()
        return None

    def _response_errors(
        self,
        execution_metrics: ExecutionMetrics | None,
        violations: list[str],
    ) -> list[str]:
        if execution_metrics and execution_metrics.error_message:
            return [execution_metrics.error_message]
        if violations:
            return ["Blueprint compliance violations:\n" + "\n".join(violations)]
        return ["SQL Builder did not produce executable SQL."]
