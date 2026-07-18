"""Candidate SQL generators adapted into local contracts."""

from __future__ import annotations

from typing import Callable

from agent.sqlBuilderAgent.generation_methods.models import SQLCandidate
from agent.sqlBuilderAgent.generation_methods.prompts import build_generation_prompt
from agent.sqlBuilderAgent.generation_methods.utils import parse_xml_result


LLMTextFn = Callable[[str], str]


class CandidateSQLGenerator:
    """Run neutral prompt-based SQL generation methods."""

    def __init__(
        self,
        *,
        enabled_methods: tuple[str, ...] = ("divide_and_conquer", "skeleton"),
    ) -> None:
        self.enabled_methods = enabled_methods

    def generate(
        self,
        *,
        schema_profile: str,
        question: str,
        evidence: str | None,
        few_shot_examples: list[dict] | None = None,
        llm_text: LLMTextFn,
    ) -> tuple[list[SQLCandidate], dict[str, str], dict[str, str]]:
        candidates: list[SQLCandidate] = []
        prompts: dict[str, str] = {}
        raw_responses: dict[str, str] = {}
        examples = few_shot_examples or []
        for method in self.enabled_methods:
            if method == "icl" and not examples:
                continue
            prompt = build_generation_prompt(
                method=method,
                schema_profile=schema_profile,
                question=question,
                evidence=evidence,
                few_shot_examples=examples,
            )
            prompts[method] = prompt
            response = llm_text(prompt)
            raw_responses[method] = response
            sql = parse_xml_result(response)
            if sql:
                candidates.append(SQLCandidate(sql=sql, source=method))
        return candidates, prompts, raw_responses
