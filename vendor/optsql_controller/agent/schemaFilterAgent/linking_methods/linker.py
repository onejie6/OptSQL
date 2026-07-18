"""Coordinator for embedded schema linking methods."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from agent.schemaFilterAgent.linking_methods.linkers import DirectLinker
from agent.schemaFilterAgent.linking_methods.linkers import ReversedLinker
from agent.schemaFilterAgent.linking_methods.linkers import ValueLinker
from agent.schemaFilterAgent.linking_methods.models import LinkerOutput
from agent.schemaFilterAgent.linking_methods.prompts import build_direct_linking_prompt
from agent.schemaFilterAgent.linking_methods.prompts import build_reversed_linking_prompt
from agent.schemaFilterAgent.linking_methods.utils import build_schema_profile
from agent.schemaFilterAgent.linking_methods.utils import merge_schema_linking_results
from utils.openai_client import request_chat_text


class SchemaLinkingPipeline:
    """Run direct, reversed, and value linking and normalize into candidates."""

    def __init__(
        self,
        *,
        llm_model: str | None = None,
        llm_temperature: float = 0.0,
        llm_max_tokens: int = 4096,
        llm_client=None,
        value_distance_threshold: float = 0.35,
    ) -> None:
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.llm_client = llm_client
        self.direct_linker = DirectLinker()
        self.reversed_linker = ReversedLinker()
        self.value_linker = ValueLinker(value_distance_threshold=value_distance_threshold)

    def link(
        self,
        *,
        db_id: str,
        question: str,
        evidence: str | None,
        clauses: list[dict],
        schema_columns: list[dict[str, Any]],
        max_columns: int,
    ) -> LinkerOutput:
        schema_profile = build_schema_profile(schema_columns)
        prompts = {
            "direct": build_direct_linking_prompt(
                schema_profile=schema_profile,
                question=question,
                evidence=evidence,
            ),
            "reversed": build_reversed_linking_prompt(
                schema_profile=schema_profile,
                question=question,
                evidence=evidence,
            ),
        }

        direct_linked, direct_response = self.direct_linker.link(
            schema_columns=schema_columns,
            question=question,
            evidence=evidence,
            llm_text=self._request_text,
        )
        reversed_linked, reversed_response = self.reversed_linker.link(
            schema_columns=schema_columns,
            question=question,
            evidence=evidence,
            llm_text=self._request_text,
        )
        value_terms = _value_terms_from_clauses(clauses, question, evidence)
        value_linked, retrieved_values = self.value_linker.link(
            db_id=db_id,
            schema_columns=schema_columns,
            terms=value_terms,
        )
        merged = merge_schema_linking_results([direct_linked, reversed_linked, value_linked])
        candidates = self._build_candidates(
            schema_columns=schema_columns,
            merged=merged,
            direct_linked=direct_linked,
            reversed_linked=reversed_linked,
            value_linked=value_linked,
            retrieved_values=retrieved_values,
            max_columns=max_columns,
        )
        return LinkerOutput(
            selected_columns=candidates,
            direct_linked=direct_linked,
            reversed_linked=reversed_linked,
            value_linked=value_linked,
            retrieved_values=retrieved_values,
            prompts=prompts,
            raw_responses={
                "direct": direct_response,
                "reversed": reversed_response,
            },
            token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    def _request_text(self, prompt: str) -> str:
        return request_chat_text(
            messages=[{"role": "user", "content": prompt}],
            model=self.llm_model,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
            client=self.llm_client,
        )

    def _build_candidates(
        self,
        *,
        schema_columns: list[dict[str, Any]],
        merged: dict[str, list[str]],
        direct_linked: dict[str, list[str]],
        reversed_linked: dict[str, list[str]],
        value_linked: dict[str, list[str]],
        retrieved_values: dict[str, dict[str, list[dict[str, Any]]]],
        max_columns: int,
    ) -> list[dict[str, Any]]:
        schema_by_key = {
            (column["table_name"], column["column_name"]): column
            for column in schema_columns
        }
        candidates: list[dict[str, Any]] = []
        for table_name, column_names in merged.items():
            for column_name in column_names:
                key = (table_name, column_name)
                if key not in schema_by_key:
                    continue
                sources = []
                if column_name in direct_linked.get(table_name, []):
                    sources.append("direct")
                if column_name in reversed_linked.get(table_name, []):
                    sources.append("reversed")
                if column_name in value_linked.get(table_name, []):
                    sources.append("value")
                value_matches = [
                    {
                        "keyword": item["keyword"],
                        "value": item["value"],
                        "confidence": item["score"],
                        "evidence": "value_linker",
                    }
                    for item in retrieved_values.get(table_name, {}).get(column_name, [])
                ]
                candidates.append(
                    {
                        **schema_by_key[key],
                        "score": _score_for_sources(sources),
                        "matched_clause_ids": [],
                        "value_candidates": [str(item["keyword"]) for item in value_matches],
                        "value_matches": value_matches,
                        "reasons": [f"{source}_linker" for source in sources],
                        "llm_selected": "direct" in sources or "reversed" in sources,
                        "linking_sources": sources,
                    }
                )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:max_columns]


def _score_for_sources(sources: list[str]) -> float:
    score = 0.0
    if "direct" in sources:
        score += 0.42
    if "reversed" in sources:
        score += 0.36
    if "value" in sources:
        score += 0.28
    return min(1.0, max(0.55, round(score, 4)))


def _value_terms_from_clauses(
    clauses: list[dict],
    question: str,
    evidence: str | None,
) -> list[str]:
    terms: list[str] = []
    for clause in clauses:
        terms.append(str(clause.get("text") or ""))
        for entity in clause.get("entities", []) or []:
            terms.append(str(entity))
        for operator in clause.get("operators", []) or []:
            if isinstance(operator, dict):
                terms.append(str(operator.get("expression") or ""))
    terms.extend(_quoted_or_capitalized_terms(question))
    terms.extend(_quoted_or_capitalized_terms(evidence or ""))
    return [
        term.strip()
        for term in dict.fromkeys(terms)
        if term and term.strip()
    ]


def _quoted_or_capitalized_terms(text: str) -> list[str]:
    import re

    terms = re.findall(r'"([^"]+)"|\'([^\']+)\'', text or "")
    flattened = [left or right for left, right in terms]
    flattened.extend(
        re.findall(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*)\b", text or "")
    )
    return flattened
