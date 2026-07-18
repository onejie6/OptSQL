"""JSON parsing helpers for LLM responses."""

import json
import re


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_json_object(response_text: str) -> dict:
    """Parse a JSON object from a raw LLM response."""
    if not response_text or not response_text.strip():
        raise ValueError("LLM response is empty.")

    text = response_text.strip()
    fenced_match = _FENCED_JSON_RE.search(text)
    if fenced_match:
        text = fenced_match.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object.")

    return parsed
