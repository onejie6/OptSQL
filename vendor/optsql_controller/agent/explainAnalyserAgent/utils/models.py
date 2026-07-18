"""Shared data models for Explain Analyser tools.

These dataclasses are deliberately lightweight and serializable. Tool
implementations should return structured evidence instead of free-form plan
text so the agent can build stable Bottleneck Reports across SQLite and MySQL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Dbms = Literal["sqlite", "mysql"]
ExplainMode = Literal["estimated", "analyze"]


@dataclass(frozen=True)
class DetectDbCapabilitiesInput:
    db_id: str
    dbms: Dbms


@dataclass(frozen=True)
class DetectDbCapabilitiesOutput:
    dbms: Dbms
    version: str | None
    supports_explain_query_plan: bool
    supports_explain_json: bool
    supports_explain_analyze: bool
    supports_runtime_timeout: bool
    supports_optimizer_trace: bool
    default_explain_mode: ExplainMode
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class JoinInfo:
    join_type: str
    left_table: str | None
    right_table: str | None
    condition: str | None
    columns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PredicateInfo:
    clause: Literal["where", "having", "join", "case", "unknown"]
    expression: str
    columns: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    operators: list[str] = field(default_factory=list)
    risky_patterns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SubqueryInfo:
    location: Literal["select", "from", "where", "having", "unknown"]
    query: str
    correlated: bool
    referenced_outer_columns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ParseSqlStructureInput:
    sql: str
    dbms: Dbms


@dataclass(frozen=True)
class ParseSqlStructureOutput:
    statement_type: Literal["select", "with", "unknown"]
    tables: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    joins: list[JoinInfo] = field(default_factory=list)
    predicates: list[PredicateInfo] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    order_by: list[str] = field(default_factory=list)
    limit: int | None = None
    subqueries: list[SubqueryInfo] = field(default_factory=list)
    has_distinct: bool = False
    has_union: bool = False
    has_window_functions: bool = False
    has_select_star: bool = False
    risky_patterns: list[str] = field(default_factory=list)
    parse_warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GetExplainPlanInput:
    sql: str
    db_id: str
    dbms: Dbms
    mode: ExplainMode = "estimated"
    timeout_ms: int = 5000


@dataclass(frozen=True)
class GetExplainPlanOutput:
    dbms: Dbms
    mode: ExplainMode
    explain_sql: str
    raw_plan: dict[str, Any] | list[Any] | str | None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class PlanEdge:
    source_node_id: str
    target_node_id: str
    edge_type: Literal["parent_child", "join_input", "subquery_input"]


@dataclass(frozen=True)
class PlanNode:
    node_id: str
    parent_id: str | None
    operation: str
    table: str | None = None
    index: str | None = None
    access_type: str | None = None
    predicate: str | None = None
    join_condition: str | None = None
    estimated_rows: float | None = None
    actual_rows: float | None = None
    estimated_cost: float | None = None
    actual_time_ms: float | None = None
    loops: int | None = None
    flags: list[str] = field(default_factory=list)
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanIR:
    dbms: Dbms
    nodes: list[PlanNode] = field(default_factory=list)
    edges: list[PlanEdge] = field(default_factory=list)
    global_flags: list[str] = field(default_factory=list)
    raw_plan: dict[str, Any] | list[Any] | str | None = None
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class NormalizePlanInput:
    dbms: Dbms
    raw_plan: dict[str, Any] | list[Any] | str | None
    sql_structure: ParseSqlStructureOutput


@dataclass(frozen=True)
class NormalizePlanOutput:
    plan_ir: PlanIR


@dataclass(frozen=True)
class CollectSchemaStatsInput:
    db_id: str
    dbms: Dbms
    tables: list[str]
    columns: list[str] = field(default_factory=list)
    include_samples: bool = False
    timeout_ms: int = 5000


@dataclass(frozen=True)
class TableStats:
    table: str
    row_count: int | None
    row_count_kind: Literal["exact", "estimated", "unknown"]
    primary_key: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ColumnStats:
    table: str
    column: str
    data_type: str | None = None
    nullable: bool | None = None
    approx_distinct: int | None = None
    sample_values: list[Any] | None = None


@dataclass(frozen=True)
class IndexStats:
    table: str
    index_name: str
    columns: list[str] = field(default_factory=list)
    unique: bool = False
    origin: str | None = None


@dataclass(frozen=True)
class ForeignKeyStats:
    source_table: str
    source_columns: list[str]
    target_table: str
    target_columns: list[str]


@dataclass(frozen=True)
class CollectSchemaStatsOutput:
    tables: dict[str, TableStats] = field(default_factory=dict)
    columns: dict[str, ColumnStats] = field(default_factory=dict)
    indexes: dict[str, list[IndexStats]] = field(default_factory=dict)
    foreign_keys: list[ForeignKeyStats] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OptimizationDecision:
    should_rewrite: bool
    decision: Literal["rewrite", "skip_optimization", "need_validation_only"]
    confidence: float
    reason: str
    evidence: list[str] = field(default_factory=list)
    risk_tags: list[str] = field(default_factory=list)
    next_action: Literal["call_rewriter", "return_current_sql", "validate_current_sql"] = (
        "validate_current_sql"
    )
