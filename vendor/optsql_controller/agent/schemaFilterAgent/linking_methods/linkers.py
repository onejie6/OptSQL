"""Embedded schema linkers without external runtime imports."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from agent.schemaFilterAgent.linking_methods.prompts import build_direct_linking_prompt
from agent.schemaFilterAgent.linking_methods.prompts import build_reversed_linking_prompt
from agent.schemaFilterAgent.linking_methods.utils import build_schema_profile
from agent.schemaFilterAgent.linking_methods.utils import parse_direct_linking_response
from agent.schemaFilterAgent.linking_methods.utils import parse_sql_linking_response
from utils.db import connect_bird_database
from utils.schema_grounding import quote_identifier


LLMTextFn = Callable[[str], str]


class DirectLinker:
    """Direct linker: ask the LLM to select schema elements."""

    def link(
        self,
        *,
        schema_columns: list[dict[str, Any]],
        question: str,
        evidence: str | None,
        llm_text: LLMTextFn,
    ) -> tuple[dict[str, list[str]], str]:
        prompt = build_direct_linking_prompt(
            schema_profile=build_schema_profile(schema_columns),
            question=question,
            evidence=evidence,
        )
        response = llm_text(prompt)
        return parse_direct_linking_response(response, schema_columns) or {}, response


class ReversedLinker:
    """Reversed linker: generate SQL, then extract referenced schema."""

    def link(
        self,
        *,
        schema_columns: list[dict[str, Any]],
        question: str,
        evidence: str | None,
        llm_text: LLMTextFn,
    ) -> tuple[dict[str, list[str]], str]:
        prompt = build_reversed_linking_prompt(
            schema_profile=build_schema_profile(schema_columns),
            question=question,
            evidence=evidence,
        )
        response = llm_text(prompt)
        return parse_sql_linking_response(response, schema_columns) or {}, response


class ValueLinker:
    """Value linker backed by DISTINCT text-column scans.

    For each text column, scan distinct string values truncated to 100
    characters, lowercase both terms and values, and apply regex substring
    matching to estimate consistency.
    """

    def __init__(
        self,
        *,
        value_distance_threshold: float = 0.35,
        max_terms: int = 12,
        text_types: tuple[str, ...] = ("text", "varchar", "char"),
    ) -> None:
        self.value_distance_threshold = value_distance_threshold
        self.consistency_threshold = 1.0 - value_distance_threshold
        self.max_terms = max_terms
        self.text_types = text_types

    def link(
        self,
        *,
        db_id: str,
        schema_columns: list[dict[str, Any]],
        terms: list[str],
    ) -> tuple[dict[str, list[str]], dict[str, dict[str, list[dict[str, Any]]]]]:
        linked: dict[str, list[str]] = defaultdict(list)
        retrieved_values: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
        cleaned_terms = [term for term in dict.fromkeys(terms) if len(term.strip()) >= 2]
        cleaned_terms = cleaned_terms[: self.max_terms]
        if not cleaned_terms:
            return {}, {}

        normalized_terms = [
            (term, _normalize_for_value_match(term))
            for term in cleaned_terms
        ]
        normalized_terms = [
            (term, normalized)
            for term, normalized in normalized_terms
            if normalized
        ]
        if not normalized_terms:
            return {}, {}

        try:
            conn = connect_bird_database(db_id)
        except Exception:
            return {}, {}

        for column in schema_columns:
            data_type = str(column.get("data_type") or "").lower()
            if data_type and not any(token in data_type for token in self.text_types):
                continue
            table_name = str(column["table_name"])
            column_name = str(column["column_name"])
            matches: list[dict[str, Any]] = []
            try:
                distinct_values = _select_distinct_text_values(conn, table_name, column_name)
            except Exception:
                continue
            for value in distinct_values:
                normalized_value = _normalize_for_value_match(value)
                if not normalized_value:
                    continue
                for term, normalized_term in normalized_terms:
                    score = _regex_substring_consistency(normalized_term, normalized_value)
                    if score >= self.consistency_threshold:
                        matches.append(
                            {
                                "keyword": term,
                                "value": value,
                                "consistency": round(score, 4),
                                "score": round(score, 4),
                            }
                        )
            if matches:
                linked[table_name].append(column_name)
                matches.sort(key=lambda item: item["score"], reverse=True)
                retrieved_values[table_name][column_name] = matches[:5]

        conn.close()

        return dict(linked), {
            table_name: dict(columns)
            for table_name, columns in retrieved_values.items()
        }


def _select_distinct_text_values(conn, table_name: str, column_name: str) -> list[str]:
    table_sql = quote_identifier(table_name)
    column_sql = quote_identifier(column_name)
    sql = (
        f"SELECT DISTINCT substr(CAST({column_sql} AS TEXT), 1, 100) AS value "
        f"FROM {table_sql} "
        f"WHERE {column_sql} IS NOT NULL AND CAST({column_sql} AS TEXT) != ''"
    )
    rows = conn.execute(sql).fetchall()
    return [str(row[0])[:100] for row in rows if row and row[0] is not None]


def _normalize_for_value_match(value: object) -> str:
    return " ".join(str(value or "").lower().split())[:100]


def _regex_substring_consistency(term: str, value: str) -> float:
    if not term or not value:
        return 0.0
    if term == value:
        return 1.0

    term_pattern = _flexible_substring_pattern(term)
    value_pattern = _flexible_substring_pattern(value)
    if term_pattern.search(value):
        return _substring_score(term, value, base=0.86)
    if value_pattern.search(term):
        return _substring_score(value, term, base=0.8)
    return 0.0


def _flexible_substring_pattern(value: str):
    import re

    escaped_tokens = [re.escape(token) for token in value.split() if token]
    pattern = r"\s+".join(escaped_tokens)
    return re.compile(pattern)


def _substring_score(shorter: str, longer: str, *, base: float) -> float:
    shorter_tokens = set(shorter.split())
    longer_tokens = set(longer.split())
    token_score = (
        len(shorter_tokens & longer_tokens) / len(shorter_tokens | longer_tokens)
        if shorter_tokens and longer_tokens
        else 0.0
    )
    coverage = min(1.0, len(shorter) / max(len(longer), 1))
    return min(0.99, base + 0.08 * token_score + 0.05 * coverage)
