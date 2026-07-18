"""Template-based RAG engine for SQL optimization rules and cases."""

from __future__ import annotations

import difflib
import re
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import sqlglot
import sqlglot.expressions as exp

from myTypes import OptimizationCase
from myTypes import RetrievedStrategy
from myTypes import VerifiedContextBlueprint


DescriptionFn = Callable[[dict], str]
TemplateMinimizerFn = Callable[[dict], dict | None]

WETUNE_RULE_SOURCE = "wetune_rules.py"


RAG_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_template_records (
    record_id TEXT PRIMARY KEY,
    expert INTEGER NOT NULL CHECK (expert IN (0, 1)),
    src_sql_template TEXT NOT NULL,
    dst_sql_template TEXT NOT NULL,
    condition TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    raw_src_sql TEXT NOT NULL DEFAULT '',
    raw_dst_sql TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS rag_template_records_expert_idx
ON rag_template_records (expert);

CREATE INDEX IF NOT EXISTS rag_template_records_created_at_idx
ON rag_template_records (created_at);
"""


@dataclass(frozen=True)
class RAGTemplateRecord:
    """Unified template record for expert rules and historical cases."""

    record_id: str
    expert: bool
    srcSQL: str
    dstSQL: str
    condition: str
    description: str
    raw_srcSQL: str
    raw_dstSQL: str
    created_at: str

    @property
    def type(self) -> str:
        return "专家" if self.expert else "hist"


class RAGEngine:
    """Retrieve and persist optimization knowledge as anonymized SQL templates."""

    def __init__(
        self,
        *,
        records: list[RAGTemplateRecord | dict] | None = None,
        storage_path: str | Path | None = None,
        description_fn: DescriptionFn | None = None,
        template_minimizer_fn: TemplateMinimizerFn | None = None,
        duplicate_similarity_threshold: float = 0.92,
    ) -> None:
        self.storage_path = Path(storage_path) if storage_path else None
        self.description_fn = description_fn
        self.template_minimizer_fn = template_minimizer_fn
        self.duplicate_similarity_threshold = duplicate_similarity_threshold
        loaded_records = self._load_records()
        self.records: list[RAGTemplateRecord] = [
            _coerce_record(record) for record in [*(records or []), *loaded_records]
        ]

    # ------------------------------------------------------------------
    # Public storage API
    # ------------------------------------------------------------------

    def upsert_raw_template(
        self,
        *,
        src_sql: str,
        dst_sql: str,
        record_type: str,
        condition: str = "",
        description: str | None = None,
    ) -> RAGTemplateRecord:
        """Insert an expert or history template from raw SQL."""
        normalized_type = _normalize_record_type(record_type)
        if normalized_type == "hist":
            existing = self.find_similar_pair(src_sql=src_sql, dst_sql=dst_sql, top_k=1)
            if existing and existing[0][1] >= self.duplicate_similarity_threshold:
                return existing[0][0]
            src_template, dst_template = self._minimize_history_templates(
                src_sql=src_sql,
                dst_sql=dst_sql,
                condition=condition,
            )
            existing_template = self.find_similar_template_pair(
                src_template=src_template,
                dst_template=dst_template,
                top_k=1,
            )
            if existing_template and existing_template[0][1] >= self.duplicate_similarity_threshold:
                return existing_template[0][0]
        else:
            src_template = format_sql_template(src_sql)
            dst_template = format_sql_template(dst_sql)

        payload = {
            "type": normalized_type,
            "expert": normalized_type == "专家",
            "srcSQL": src_template,
            "dstSQL": dst_template,
            "condition": condition,
            "raw_srcSQL": src_sql,
            "raw_dstSQL": dst_sql,
        }
        final_description = description or self._describe_transform(payload)
        record = RAGTemplateRecord(
            record_id=uuid.uuid4().hex[:12],
            expert=normalized_type == "专家",
            srcSQL=src_template,
            dstSQL=dst_template,
            condition=condition,
            description=final_description,
            raw_srcSQL=src_sql,
            raw_dstSQL=dst_sql,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self.records.append(record)
        self._save_records()
        return record

    def upsert_expert_rule(
        self,
        *,
        src_sql: str,
        dst_sql: str,
        condition: str,
        description: str | None = None,
    ) -> RAGTemplateRecord:
        return self.upsert_raw_template(
            src_sql=src_sql,
            dst_sql=dst_sql,
            record_type="专家",
            condition=condition,
            description=description,
        )

    def upsert_history_case(
        self,
        *,
        src_sql: str,
        dst_sql: str,
        condition: str = "",
        description: str | None = None,
    ) -> RAGTemplateRecord:
        return self.upsert_raw_template(
            src_sql=src_sql,
            dst_sql=dst_sql,
            record_type="hist",
            condition=condition,
            description=description,
        )

    def upsert_if_novel_case(self, optimization_case: OptimizationCase) -> bool:
        before = len(self.records)
        condition = _condition_from_optimization_case(optimization_case)
        self.upsert_history_case(
            src_sql=optimization_case.src_sql,
            dst_sql=optimization_case.dst_sql,
            condition=condition,
        )
        return len(self.records) > before

    # ------------------------------------------------------------------
    # Retrieval API expected by SQLRewriterAgent
    # ------------------------------------------------------------------

    def retrieve_hybrid_strategies(
        self,
        question: str,
        sql: str,
        blueprint: VerifiedContextBlueprint,
        bottleneck_tags: list[str],
        top_k: int,
    ) -> list[RetrievedStrategy]:
        del question, blueprint, bottleneck_tags
        matches = self.retrieve_by_src_sql(sql, top_k=top_k)
        return [
            _record_to_strategy(record=record, confidence=confidence)
            for record, confidence in matches
        ]

    def retrieve_by_src_sql(
        self,
        sql: str,
        *,
        top_k: int = 5,
    ) -> list[tuple[RAGTemplateRecord, float]]:
        query_template = format_sql_template(sql)
        scored = [
            (record, template_similarity(query_template, record.srcSQL))
            for record in self.records
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def retrieve_negative_cases(
        self,
        rule_ids: list[str],
        blueprint: VerifiedContextBlueprint,
        sql: str,
    ) -> list[OptimizationCase]:
        del rule_ids, blueprint, sql
        return []

    def find_similar_pair(
        self,
        *,
        src_sql: str,
        dst_sql: str,
        top_k: int = 5,
    ) -> list[tuple[RAGTemplateRecord, float]]:
        src_template = format_sql_template(src_sql)
        dst_template = format_sql_template(dst_sql)
        min_src_template, min_dst_template = _fallback_minimized_pair(src_template, dst_template)
        scored = [
            (
                record,
                max(
                    (
                        template_similarity(src_template, record.srcSQL)
                        + template_similarity(dst_template, record.dstSQL)
                    )
                    / 2.0,
                    (
                        template_similarity(min_src_template, record.srcSQL)
                        + template_similarity(min_dst_template, record.dstSQL)
                    )
                    / 2.0,
                ),
            )
            for record in self.records
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def find_similar_template_pair(
        self,
        *,
        src_template: str,
        dst_template: str,
        top_k: int = 5,
    ) -> list[tuple[RAGTemplateRecord, float]]:
        scored = [
            (
                record,
                (
                    template_similarity(src_template, record.srcSQL)
                    + template_similarity(dst_template, record.dstSQL)
                )
                / 2.0,
            )
            for record in self.records
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def list_expert_rules(self) -> list[dict]:
        return [asdict(record) for record in self.records if record.expert]

    def initialize_wetune_expert_rules(self) -> list[RAGTemplateRecord]:
        """Seed the RAG store with expert rewrite templates from wetune_rules.py."""
        from wetune_rules import RULES

        seeded_records: list[RAGTemplateRecord] = []
        existing_by_id = {record.record_id: index for index, record in enumerate(self.records)}
        for level, rules in RULES.items():
            for rule_number, rule in sorted(rules.items()):
                record_id = f"wetune_{str(level).lower()}_{rule_number}"
                src_sql = str(rule.get("src", "")).strip()
                dst_sql = str(rule.get("dst", "")).strip()
                condition = _condition_from_wetune_rule(rule)
                payload = {
                    "type": "专家",
                    "expert": True,
                    "srcSQL": _format_wetune_rule_sql(src_sql),
                    "dstSQL": _format_wetune_rule_sql(dst_sql),
                    "condition": condition,
                    "raw_srcSQL": src_sql,
                    "raw_dstSQL": dst_sql,
                    "source": WETUNE_RULE_SOURCE,
                    "rule_group": str(level),
                    "rule_number": rule_number,
                }
                record = RAGTemplateRecord(
                    record_id=record_id,
                    expert=True,
                    srcSQL=payload["srcSQL"],
                    dstSQL=payload["dstSQL"],
                    condition=condition,
                    description=_describe_wetune_rule_transform(
                        level=str(level),
                        rule_number=rule_number,
                        src_sql=src_sql,
                        dst_sql=dst_sql,
                    ),
                    raw_srcSQL=src_sql,
                    raw_dstSQL=dst_sql,
                    created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )
                if record_id in existing_by_id:
                    self.records[existing_by_id[record_id]] = record
                else:
                    existing_by_id[record_id] = len(self.records)
                    self.records.append(record)
                seeded_records.append(record)
        self._save_records()
        return seeded_records

    def initialize_high_value_expert_rules(self) -> list[RAGTemplateRecord]:
        """Seed the RAG store with curated high-value expert rules."""
        from high_value_rules import RULES

        seeded_records: list[RAGTemplateRecord] = []
        existing_by_id = {record.record_id: index for index, record in enumerate(self.records)}
        for rule in RULES:
            record_id = str(rule["rule_id"])
            record = RAGTemplateRecord(
                record_id=record_id,
                expert=True,
                srcSQL=format_sql_template(str(rule["src"])),
                dstSQL=format_sql_template(str(rule["dst"])),
                condition=str(rule.get("condition") or ""),
                description=str(rule.get("description") or record_id),
                raw_srcSQL=str(rule["src"]),
                raw_dstSQL=str(rule["dst"]),
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            if record_id in existing_by_id:
                self.records[existing_by_id[record_id]] = record
            else:
                existing_by_id[record_id] = len(self.records)
                self.records.append(record)
            seeded_records.append(record)
        self._save_records()
        return seeded_records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _minimize_history_templates(
        self,
        *,
        src_sql: str,
        dst_sql: str,
        condition: str,
    ) -> tuple[str, str]:
        full_src_template = format_sql_template(src_sql)
        full_dst_template = format_sql_template(dst_sql)
        if self.template_minimizer_fn is None:
            return _fallback_minimized_pair(full_src_template, full_dst_template)
        payload = {
            "srcSQL": src_sql,
            "dstSQL": dst_sql,
            "srcSQL_template": full_src_template,
            "dstSQL_template": full_dst_template,
            "condition": condition,
            "instruction": (
                "Minimize the SQL optimization transform template. If the transform is local "
                "and independently reusable, return only the local srcSQL/dstSQL template pair; "
                "otherwise return the full templates. Keep table, column, and literal values "
                "anonymized as keywords."
            ),
        }
        minimized = self.template_minimizer_fn(payload) or {}
        src_template = str(minimized.get("srcSQL") or minimized.get("srcSQL_template") or "")
        dst_template = str(minimized.get("dstSQL") or minimized.get("dstSQL_template") or "")
        return (
            src_template or full_src_template,
            dst_template or full_dst_template,
        )

    def _describe_transform(self, payload: dict) -> str:
        if self.description_fn is not None:
            description = (self.description_fn(payload) or "").strip()
            if description:
                return description
        return _fallback_description(payload["srcSQL"], payload["dstSQL"])

    def _load_records(self) -> list[dict]:
        if self.storage_path is None:
            return []
        self._ensure_storage()
        with sqlite3.connect(self.storage_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    record_id,
                    expert,
                    src_sql_template,
                    dst_sql_template,
                    condition,
                    description,
                    raw_src_sql,
                    raw_dst_sql,
                    created_at
                FROM rag_template_records
                ORDER BY created_at ASC, record_id ASC
                """
            ).fetchall()
        return [_record_from_sqlite_row(row) for row in rows]

    def _save_records(self) -> None:
        if self.storage_path is None:
            return
        self._ensure_storage()
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("DELETE FROM rag_template_records")
            conn.executemany(
                """
                INSERT INTO rag_template_records (
                    record_id,
                    expert,
                    src_sql_template,
                    dst_sql_template,
                    condition,
                    description,
                    raw_src_sql,
                    raw_dst_sql,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                [
                    (
                        record.record_id,
                        1 if record.expert else 0,
                        record.srcSQL,
                        record.dstSQL,
                        record.condition,
                        record.description,
                        record.raw_srcSQL,
                        record.raw_dstSQL,
                        record.created_at,
                    )
                    for record in self.records
                ],
            )

    def _ensure_storage(self) -> None:
        if self.storage_path is None:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.storage_path) as conn:
            _migrate_rag_template_records_schema(conn)
            conn.executescript(RAG_SQLITE_SCHEMA)


def format_sql_template(sql: str) -> str:
    """Return a normalized SQL template with schema/data identifiers anonymized."""
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return _fallback_format_sql_template(sql)

    def anonymize(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Table):
            node.set("this", exp.Identifier(this="TABLE", quoted=False))
            node.set("alias", None)
            return node
        if isinstance(node, exp.Column):
            node.set("this", exp.Identifier(this=_column_template_token(node.name), quoted=False))
            node.set("table", None)
            return node
        if isinstance(node, exp.Literal):
            return exp.Var(this="LITERAL")
        return node

    template_ast = ast.copy().transform(anonymize)
    return _normalize_template_text(template_ast.sql(dialect="sqlite"))


def _format_wetune_rule_sql(sql: str) -> str:
    """Pretty-format WeTune's already-abstract expert SQL without anonymizing placeholders."""
    stripped = sql.strip().rstrip(";")
    if not stripped:
        return ""
    try:
        ast = sqlglot.parse_one(stripped, dialect="sqlite")
    except Exception:
        return _format_unparsed_wetune_sql(stripped)
    return ast.sql(dialect="sqlite", pretty=True)


def _format_unparsed_wetune_sql(sql: str) -> str:
    text = re.sub(r"\s+", " ", sql).strip()
    keyword_pattern = (
        "SELECT|FROM|WHERE|INNER JOIN|LEFT JOIN|JOIN|ON|GROUP BY|ORDER BY|HAVING|WITH|"
        "AND|OR|IN|AS|DISTINCT|COUNT|SUM|ASC|DESC"
    )
    text = re.sub(
        rf"\b({keyword_pattern})\b",
        lambda match: match.group(1).upper(),
        text,
        flags=re.IGNORECASE,
    )
    return text


def template_similarity(left: str, right: str) -> float:
    left_norm = _normalize_template_text(left)
    right_norm = _normalize_template_text(right)
    if not left_norm and not right_norm:
        return 1.0
    if not left_norm or not right_norm:
        return 0.0
    sequence_score = difflib.SequenceMatcher(None, left_norm, right_norm).ratio()
    left_tokens = set(_template_tokens(left_norm))
    right_tokens = set(_template_tokens(right_norm))
    token_score = (
        len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
        if left_tokens or right_tokens
        else 0.0
    )
    return round(sequence_score * 0.65 + token_score * 0.35, 6)


def _fallback_format_sql_template(sql: str) -> str:
    text = re.sub(r"'(?:''|[^'])*'", " LITERAL ", sql)
    text = re.sub(r'"(?:""|[^"])*"', " COLUMN ", text)
    text = re.sub(r"`[^`]*`", " COLUMN ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", " LITERAL ", text)
    return _normalize_template_text(text)


def _column_template_token(column_name: str) -> str:
    name = (column_name or "").strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    parts = [part for part in normalized.split("_") if part]
    part_set = set(parts)
    if not normalized:
        return "COLUMN"
    if normalized in {"id", "identifier"} or normalized.endswith("_id"):
        return "ID_COLUMN" if normalized in {"id", "identifier"} else "FK_COLUMN"
    if part_set & {
        "date",
        "time",
        "year",
        "month",
        "day",
        "timestamp",
        "created",
        "updated",
        "birth",
        "dob",
    }:
        return "DATE_COLUMN"
    if part_set & {
        "count",
        "num",
        "number",
        "amount",
        "total",
        "sum",
        "avg",
        "average",
        "price",
        "cost",
        "score",
        "rate",
        "ratio",
        "percent",
        "percentage",
        "age",
        "rank",
        "size",
        "length",
        "height",
        "weight",
        "enrollment",
        "population",
        "salary",
        "income",
        "revenue",
    }:
        return "NUMERIC_COLUMN"
    if part_set & {
        "name",
        "title",
        "type",
        "category",
        "status",
        "city",
        "county",
        "state",
        "country",
        "address",
        "description",
        "text",
        "code",
        "gender",
        "email",
        "phone",
    }:
        return "TEXT_COLUMN"
    return "COLUMN"


def _normalize_template_text(sql: str) -> str:
    return " ".join(sql.replace("\n", " ").split()).upper()


def _template_tokens(sql: str) -> list[str]:
    return re.findall(r"[A-Z_]+|\d+", sql.upper())


def _normalize_record_type(record_type: str) -> str:
    normalized = str(record_type).strip().lower()
    if normalized in {"expert", "experts", "rule", "专家"}:
        return "专家"
    if normalized in {"hist", "history", "case", "历史"}:
        return "hist"
    raise ValueError("record_type must be '专家'/'expert' or 'hist'/'history'.")


def _coerce_record(record: RAGTemplateRecord | dict) -> RAGTemplateRecord:
    if isinstance(record, RAGTemplateRecord):
        return record
    expert_value = record.get("expert")
    expert = (
        _coerce_expert_flag(expert_value)
        if expert_value is not None
        else _normalize_record_type(str(record["type"])) == "专家"
    )
    return RAGTemplateRecord(
        record_id=str(record.get("record_id") or uuid.uuid4().hex[:12]),
        expert=expert,
        srcSQL=str(record["srcSQL"]),
        dstSQL=str(record["dstSQL"]),
        condition=str(record.get("condition", "")),
        description=str(record.get("description", "")),
        raw_srcSQL=str(record.get("raw_srcSQL", "")),
        raw_dstSQL=str(record.get("raw_dstSQL", "")),
        created_at=str(record.get("created_at", "")),
    )


def _coerce_expert_flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "expert", "专家"}
    return bool(value)


def _record_from_sqlite_row(row: sqlite3.Row) -> dict:
    return {
        "record_id": row["record_id"],
        "expert": bool(row["expert"]),
        "srcSQL": row["src_sql_template"],
        "dstSQL": row["dst_sql_template"],
        "condition": row["condition"],
        "description": row["description"],
        "raw_srcSQL": row["raw_src_sql"],
        "raw_dstSQL": row["raw_dst_sql"],
        "created_at": row["created_at"],
    }


def _migrate_rag_template_records_schema(conn: sqlite3.Connection) -> None:
    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'rag_template_records'"
    ).fetchone()
    if table_exists is None:
        return
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(rag_template_records)").fetchall()
    }
    if "expert" in columns:
        return
    if "record_type" not in columns:
        return
    conn.executescript(
        """
        DROP INDEX IF EXISTS rag_template_records_type_idx;
        DROP INDEX IF EXISTS rag_template_records_expert_idx;
        DROP INDEX IF EXISTS rag_template_records_created_at_idx;
        ALTER TABLE rag_template_records RENAME TO rag_template_records_old;
        """
    )
    conn.executescript(RAG_SQLITE_SCHEMA)
    conn.execute(
        """
        INSERT INTO rag_template_records (
            record_id,
            expert,
            src_sql_template,
            dst_sql_template,
            condition,
            description,
            raw_src_sql,
            raw_dst_sql,
            created_at,
            updated_at
        )
        SELECT
            record_id,
            CASE WHEN record_type = '专家' THEN 1 ELSE 0 END,
            src_sql_template,
            dst_sql_template,
            condition,
            description,
            raw_src_sql,
            raw_dst_sql,
            created_at,
            updated_at
        FROM rag_template_records_old
        """
    )
    conn.execute("DROP TABLE rag_template_records_old")


def _record_to_strategy(record: RAGTemplateRecord, confidence: float) -> RetrievedStrategy:
    return RetrievedStrategy(
        rule_id=record.record_id,
        rule_name=f"{record.type} template: {record.description}",
        applicable_when=[record.condition] if record.condition else [],
        rewrite_template=f"{record.srcSQL} -> {record.dstSQL}",
        risk_notes=[],
        example_cases=[record.description],
        confidence=confidence,
        source_type="expert" if record.expert else "hist",
    )


def _condition_from_optimization_case(case: OptimizationCase) -> str:
    tags = ", ".join(case.bottleneck_tags)
    rules = ", ".join(case.rule_ids)
    parts = []
    if tags:
        parts.append(f"bottleneck tags: {tags}")
    if rules:
        parts.append(f"rewrite rules: {rules}")
    return "; ".join(parts)


def _condition_from_wetune_rule(rule: dict) -> str:
    parts = []
    constraint = _normalize_wetune_condition_text(str(rule.get("constraint") or ""))
    extra = _normalize_wetune_condition_text(str(rule.get("extra") or ""))
    if constraint:
        parts.append(f"Constraints: {constraint}")
    if extra:
        parts.append(f"Additional applicability: {extra}")
    return "; ".join(parts)


def _describe_wetune_rule_transform(
    *,
    level: str,
    rule_number: int,
    src_sql: str,
    dst_sql: str,
) -> str:
    src_upper = _normalize_template_text(src_sql)
    dst_upper = _normalize_template_text(dst_sql)
    actions: list[str] = []

    inner_join_delta = src_upper.count("INNER JOIN") - dst_upper.count("INNER JOIN")
    left_join_delta = src_upper.count("LEFT JOIN") - dst_upper.count("LEFT JOIN")
    if inner_join_delta > 0:
        actions.append(
            _pluralize(
                inner_join_delta,
                "remove a redundant INNER JOIN whose right-side columns are not needed",
                "remove redundant INNER JOINs whose right-side columns are not needed",
            )
        )
    if left_join_delta > 0:
        actions.append(
            _pluralize(
                left_join_delta,
                "eliminate a redundant LEFT JOIN while preserving rows from the left relation",
                "eliminate redundant LEFT JOINs while preserving rows from the left relation",
            )
        )
    if "SELECT DISTINCT *" in src_upper and "SELECT DISTINCT *" not in dst_upper:
        actions.append("drop the DISTINCT subquery wrapper and join directly against the base relation")
    if " IN (SELECT" in src_upper and " IN (SELECT" not in dst_upper:
        actions.append("remove a redundant semi-join/IN subquery filter")
    elif src_upper.count(" IN (SELECT") > dst_upper.count(" IN (SELECT"):
        actions.append("simplify nested semi-join/IN subquery filters")
    if src_upper.startswith("WITH ") and not dst_upper.startswith("WITH "):
        actions.append("inline CTEs into the remaining base-table query")
    if "DISTINCT" in src_upper and "DISTINCT" not in dst_upper:
        actions.append("remove DISTINCT after constraints make duplicate generation impossible")
    if "WHERE" in src_upper and "WHERE" in dst_upper:
        actions.append("push surviving predicates onto the reduced base relation")
    if "GROUP BY" in src_upper and "GROUP BY" in dst_upper:
        actions.append("preserve the aggregation grouping over the reduced relation")
    if "ORDER BY" in src_upper and "ORDER BY" in dst_upper:
        actions.append("preserve the final ordering after the redundant inputs are removed")

    if not actions and src_upper != dst_upper:
        actions.append("rewrite the source rule pattern into the simpler destination pattern")
    if not actions:
        actions.append("keep the source and destination patterns equivalent")

    return f"WeTune {level.upper()} rule {rule_number}: {', '.join(actions)}."


def _pluralize(count: int, singular: str, plural: str) -> str:
    return singular if count == 1 else plural


def _normalize_wetune_condition_text(text: str) -> str:
    normalized = _join_wetune_condition_lines(text)
    if not normalized:
        return ""
    replacements = {
        "外键引用": "REFERENCE",
        "IS UNIQUE": "UNIQUE",
        "LAEGE": "LARGE",
        "and": "AND",
        "or": "OR",
    }
    for source, target in replacements.items():
        normalized = re.sub(re.escape(source), target, normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+(AND|OR)\s+", r" \1 ", normalized, flags=re.IGNORECASE)
    return normalized


def _join_wetune_condition_lines(text: str) -> str:
    lines = [" ".join(line.strip().split()) for line in text.strip().splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    normalized = lines[0]
    for line in lines[1:]:
        previous_has_connector = bool(re.search(r"\b(AND|OR)\s*$", normalized, re.IGNORECASE))
        next_has_connector = bool(re.match(r"^(AND|OR)\b", line, re.IGNORECASE))
        separator = " " if previous_has_connector or next_has_connector else " AND "
        normalized = f"{normalized}{separator}{line}"
    return " ".join(normalized.split())


def _fallback_minimized_pair(src_template: str, dst_template: str) -> tuple[str, str]:
    if "SELECT *" in src_template and "SELECT *" not in dst_template:
        return "SELECT *", "SELECT COLUMN"
    return src_template, dst_template


def _fallback_description(src_template: str, dst_template: str) -> str:
    if "SELECT *" in src_template and "SELECT *" not in dst_template:
        return "Replace broad projection with explicit projected columns."
    if src_template != dst_template:
        return "Rewrite the source SQL template into the target SQL template."
    return "Keep the SQL template unchanged."
