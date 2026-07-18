from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class BirdTask:
    question_id: int
    db_id: str
    question: str
    evidence: str
    sql: str
    difficulty: str

    @classmethod
    def from_dict(cls, raw_task: dict) -> "BirdTask":
        return cls(
            question_id=raw_task["question_id"],
            db_id=raw_task["db_id"],
            question=raw_task["question"],
            evidence=raw_task["evidence"],
            sql=raw_task["SQL"],
            difficulty=raw_task["difficulty"],
        )


@dataclass(frozen=True)
class AgentTask:
    task_id: str
    question_id: int | None
    db_id: str
    question: str
    evidence: str | None
    dbms: str
    user_constraints: dict


@dataclass(frozen=True)
class ColumnRef:
    table_name: str
    column_name: str
    data_type: str | None
    comment: str | None


@dataclass(frozen=True)
class ValueMapping:
    keyword: str
    table_name: str
    column_name: str
    value: object
    confidence: float
    evidence: str


@dataclass(frozen=True)
class JoinEdge:
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    join_type: str


@dataclass(frozen=True)
class JoinGraph:
    tables: list[str]
    edges: list[JoinEdge]


@dataclass(frozen=True)
class PredicateHint:
    predicate_type: str
    expression: str
    source_text: str
    confidence: float


@dataclass(frozen=True)
class EvidenceTrace:
    artifact_type: str
    artifact_id: str
    reason: str
    tool_name: str | None
    fact: object


@dataclass(frozen=True)
class SchemaGapHint:
    """A structured signal that the VerifiedContextBlueprint is incomplete.

    Produced by SQL Builder when execution errors or evidence analysis suggest
    the Schema Filter missed a table, column, or join path.  The Controller
    reads these hints to decide whether to re-run Schema Filter.
    """

    gap_type: str  # "missing_table" | "missing_column" | "missing_join_path"
    element: str  # e.g. "grades" or "frpm.Grade Level" or "frpm↔grades"
    source: str  # "evidence" | "question" | "execution_error"
    suggestion: str  # human-readable recovery suggestion


@dataclass(frozen=True)
class VerifiedContextBlueprint:
    db_id: str
    selected_tables: list[str]
    selected_columns: list[ColumnRef]
    value_mappings: list[ValueMapping]
    join_topology: JoinGraph
    predicate_hints: list[PredicateHint]
    evidence_trace: list[EvidenceTrace]
    confidence: float


@dataclass(frozen=True)
class SQLVersion:
    version_id: str
    parent_id: str | None
    sql: str
    source_agent: str
    rewrite_rule_ids: list[str]
    explanation: str
    created_at: str


@dataclass(frozen=True)
class ExecutionMetrics:
    executable: bool
    latency_ms: float | None
    row_count: int | None
    explain_plan: str | None
    plan_features: dict
    error_message: str | None


@dataclass(frozen=True)
class ResultComparison:
    """Outcome of comparing generated SQL results against Gold SQL results."""

    equivalent: bool
    gold_sql: str
    generated_sql: str
    gold_row_count: int | None
    generated_row_count: int | None
    gold_has_order_by: bool
    gold_has_distinct: bool
    comparison_mode: str  # "list" | "set" | "multiset"
    diff_summary: str | None  # None when equivalent, else describes the mismatch
    gold_error: str | None  # execution error from gold SQL
    generated_error: str | None  # execution error from generated SQL


@dataclass(frozen=True)
class VESMetric:
    """Classic BIRD Valid Efficiency Score for one SQL prediction."""

    valid: bool
    score: float
    gold_latency_ms: float | None
    generated_latency_ms: float | None
    speed_ratio: float | None
    error_message: str | None


@dataclass(frozen=True)
class ColumnMismatchDetail:
    """A single column-level mismatch between generated and gold SQL."""

    generated_column: str  # e.g. "schools.DOCType"
    gold_column: str  # e.g. "schools.DOC"
    llm_reason: str | None  # Schema Filter LLM's reasoning for this selection
    semantic_role: str | None  # detected role (code/description/data)
    schema_evidence: str | None  # evidence trace excerpt from Schema Filter
    clause_source: str | None  # NLQ clause that triggered this selection


@dataclass(frozen=True)
class MismatchDiagnosis:
    """Root cause diagnosis for generated-vs-gold SQL result mismatches."""

    comparison: ResultComparison
    column_mismatches: list[ColumnMismatchDetail]
    root_cause: str  # "semantic_ambiguity" | "schema_gap" | "value_mismatch" | "unknown"
    summary: str  # human-readable explanation


@dataclass(frozen=True)
class AmbiguityCorrection:
    """A single column correction from description→code for semantic ambiguity."""

    original_column: str  # e.g. "schools.DOCType"
    corrected_column: str  # e.g. "schools.DOC"
    reason: str  # why this correction was applied


@dataclass(frozen=True)
class AmbiguityCorrectionResult:
    """Log of semantic ambiguity corrections applied to a Blueprint."""

    corrections: list[AmbiguityCorrection]
    original_blueprint_columns: list[str]  # snapshot before correction
    corrected_blueprint_columns: list[str]  # snapshot after correction


@dataclass(frozen=True)
class EXMetrics:
    """Execution Accuracy metrics across a set of test cases.

    Pass@1 EX — the proportion of cases where the *first* generated SQL
    (before any repair) produces results equivalent to Gold SQL.

    Pass@K EX — the proportion of cases where the *final* generated SQL
    (after up to K repair attempts) matches Gold.
    """

    total: int
    pass_at_1: int
    pass_at_k: int
    k: int  # max_repair_attempts
    pass_at_1_rate: float  # pass_at_1 / total
    pass_at_k_rate: float  # pass_at_k / total
    corrected_count: int = 0  # via semantic ambiguity resolution


@dataclass(frozen=True)
class AttemptRecord:
    attempt_id: str
    sql_version_id: str | None
    stage: str
    status: str
    failure_reason: str | None
    notes: str


@dataclass(frozen=True)
class ReflectionMemory:
    attempts: list[AttemptRecord]
    failed_assumptions: list[str]
    confirmed_facts: list[str]
    rejected_rewrite_rules: list[str]
    next_strategy_hint: str | None


@dataclass(frozen=True)
class RewriteHint:
    strategy: str
    target_fragment: str | None
    expected_effect: str
    risk: str
    requires_validation: bool
    dbms_notes: str | None


@dataclass(frozen=True)
class BottleneckReport:
    sql_version_id: str
    bottlenecks: list[str]
    cost_snapshot: dict
    risk_tags: list[str]
    rewrite_hints: list[RewriteHint]
    explanation: str


@dataclass(frozen=True)
class ValidationReport:
    executable: bool
    equivalent: bool
    performance_better: bool
    old_metrics: ExecutionMetrics
    new_metrics: ExecutionMetrics
    failure_reason: str | None
    accepted: bool


@dataclass(frozen=True)
class OptimizationCase:
    case_id: str
    dbms: str
    nlq: str
    evidence: str | None
    schema_signature: str
    src_sql: str
    dst_sql: str
    rule_ids: list[str]
    bottleneck_tags: list[str]
    metrics_before: ExecutionMetrics
    metrics_after: ExecutionMetrics
    explanation: str
    novelty_score: float


@dataclass(frozen=True)
class RetrievedStrategy:
    rule_id: str
    rule_name: str
    applicable_when: list[str]
    rewrite_template: str
    risk_notes: list[str]
    example_cases: list[str]
    confidence: float
    source_type: str = "unknown"
    families: list[str] | None = None
    hint_strategies: list[str] | None = None
    operator_name: str | None = None
    suppressed_by: list[str] | None = None
    preflight_policy: str | None = None
    preflight_failure_message: str | None = None


@dataclass(frozen=True)
class AgentRequest:
    request_id: str
    task: AgentTask
    runtime_state: dict
    input_artifacts: dict
    constraints: dict


@dataclass(frozen=True)
class AgentResponse:
    request_id: str
    agent_name: str
    status: str
    output_artifacts: dict
    reasoning_summary: str
    tool_calls: list[dict]
    errors: list[str]


@dataclass(frozen=True)
class RuntimeState:
    task: AgentTask
    complexity_score: int
    strategy: str
    blueprint: VerifiedContextBlueprint | None
    sql_versions: list[SQLVersion]
    best_sql_version_id: str | None
    execution_metrics: dict[str, ExecutionMetrics]
    validation_reports: dict[str, ValidationReport]
    reflection_memory: ReflectionMemory
    iteration: int
    status: str


@dataclass(frozen=True)
class RewritePlan(Mapping[str, object]):
    plan_type: str
    source_type: str
    rule_id: str
    rule_name: str
    hint_strategy: str
    source_sql_version_id: str
    target_fragment: str | None
    rewrite_template: str
    risk: str
    expected_effect: str
    requires_validation: bool
    dbms_notes: str | None
    matched_conditions: tuple[str, ...]
    semantic_risks: tuple[str, ...]
    required_fragments: dict[str, object] = field(default_factory=dict)
    strategy_confidence: float = 0.0
    applicability_confidence: float = 0.0
    retrieval_rerank_score: float = 0.0
    hist_template: bool = False
    physical_schema_context: dict[str, object] | None = None

    def __getitem__(self, key: str) -> object:
        data = self.as_dict()
        if key not in data:
            raise KeyError(key)
        return data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.as_dict())

    def __len__(self) -> int:
        return len(self.as_dict())

    def get(self, key: str, default: object = None) -> object:
        return self.as_dict().get(key, default)

    def as_dict(self) -> dict[str, object]:
        return {
            "plan_type": self.plan_type,
            "source_type": self.source_type,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "hint_strategy": self.hint_strategy,
            "source_sql_version_id": self.source_sql_version_id,
            "target_fragment": self.target_fragment,
            "rewrite_template": self.rewrite_template,
            "risk": self.risk,
            "expected_effect": self.expected_effect,
            "requires_validation": self.requires_validation,
            "dbms_notes": self.dbms_notes,
            "matched_conditions": list(self.matched_conditions),
            "semantic_risks": list(self.semantic_risks),
            "required_fragments": dict(self.required_fragments),
            "strategy_confidence": self.strategy_confidence,
            "applicability_confidence": self.applicability_confidence,
            "retrieval_rerank_score": self.retrieval_rerank_score,
            "hist_template": self.hist_template,
            "physical_schema_context": self.physical_schema_context,
        }

    def with_physical_schema_context(self, physical_schema_context: dict[str, object]) -> "RewritePlan":
        return replace(self, physical_schema_context=physical_schema_context)


@dataclass(frozen=True)
class OperatorDeterministicRewritePlan(RewritePlan):
    operator_match: object | None = None

    def as_dict(self) -> dict[str, object]:
        data = super().as_dict()
        fragments = dict(self.required_fragments)
        if self.operator_match is not None:
            fragments["operator_match"] = self.operator_match
        data["required_fragments"] = fragments
        data["operator_match"] = self.operator_match
        return data


@dataclass(frozen=True)
class GenericStrategyRewritePlan(RewritePlan):
    llm_strategy: bool = True

    def as_dict(self) -> dict[str, object]:
        data = super().as_dict()
        data["llm_strategy"] = self.llm_strategy
        return data


@dataclass(frozen=True)
class NoOpRewritePlan(Mapping[str, object]):
    reason: str
    source_type: str = "none"
    plan_type: str = "noop"

    def __bool__(self) -> bool:
        return False

    def __getitem__(self, key: str) -> object:
        data = self.as_dict()
        if key not in data:
            raise KeyError(key)
        return data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.as_dict())

    def __len__(self) -> int:
        return len(self.as_dict())

    def get(self, key: str, default: object = None) -> object:
        return self.as_dict().get(key, default)

    def as_dict(self) -> dict[str, object]:
        return {
            "plan_type": self.plan_type,
            "source_type": self.source_type,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RejectedRewritePlan(Mapping[str, object]):
    rule_id: str
    hint_strategy: str
    rejection_reason: str
    source_type: str = "unknown"
    plan_type: str = "rejected"

    def __bool__(self) -> bool:
        return False

    def __getitem__(self, key: str) -> object:
        data = self.as_dict()
        if key not in data:
            raise KeyError(key)
        return data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.as_dict())

    def __len__(self) -> int:
        return len(self.as_dict())

    def get(self, key: str, default: object = None) -> object:
        return self.as_dict().get(key, default)

    def as_dict(self) -> dict[str, object]:
        return {
            "plan_type": self.plan_type,
            "source_type": self.source_type,
            "rule_id": self.rule_id,
            "hint_strategy": self.hint_strategy,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class FinalAnswer:
    sql: str
    selected_schema: list[str]
    value_bindings: list[str]
    join_path: list[str]
    optimization_steps: list[str]
    validation_summary: str
    performance_summary: str
    caveats: list[str]
