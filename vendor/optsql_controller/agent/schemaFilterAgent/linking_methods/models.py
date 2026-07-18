"""Small data contracts for embedded schema linking."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LinkerOutput:
    selected_columns: list[dict[str, Any]]
    direct_linked: dict[str, list[str]]
    reversed_linked: dict[str, list[str]]
    value_linked: dict[str, list[str]]
    retrieved_values: dict[str, dict[str, list[dict[str, Any]]]]
    prompts: dict[str, str]
    raw_responses: dict[str, str | None]
    token_usage: dict[str, int]
