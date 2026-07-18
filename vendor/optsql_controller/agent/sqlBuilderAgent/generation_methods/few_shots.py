"""Few-shot example loading for ICL SQL generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import PROJECT_ROOT


DEFAULT_FEW_SHOT_PATH = PROJECT_ROOT / "data" / "bird_few_shots.json"


class BirdFewShotStore:
    """Load and serve BIRD few-shot examples keyed by question id."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_FEW_SHOT_PATH
        self._examples_by_question_id: dict[str, list[dict[str, str]]] | None = None

    def get_examples(self, question_id: int | str | None) -> list[dict[str, str]]:
        if question_id is None:
            return []
        examples = self._load().get(str(question_id), [])
        return list(examples)

    def _load(self) -> dict[str, list[dict[str, str]]]:
        if self._examples_by_question_id is not None:
            return self._examples_by_question_id
        if not self.path.is_file():
            self._examples_by_question_id = {}
            return self._examples_by_question_id
        with self.path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        self._examples_by_question_id = _normalize_examples(raw)
        return self._examples_by_question_id


def _normalize_examples(raw: Any) -> dict[str, list[dict[str, str]]]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, list[dict[str, str]]] = {}
    for key, examples in raw.items():
        if not isinstance(examples, list):
            continue
        valid_examples = []
        for example in examples:
            if not isinstance(example, dict):
                continue
            question = str(example.get("question") or "").strip()
            sql = str(example.get("sql") or "").strip()
            if question and sql:
                valid_examples.append({"question": question, "sql": sql})
        if valid_examples:
            normalized[str(key)] = valid_examples
    return normalized
