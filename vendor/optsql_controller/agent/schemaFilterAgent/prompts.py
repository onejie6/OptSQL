"""Prompt builders for the Evidence-guided Schema Filter Agent."""

import json


SCHEMA_FILTER_SYSTEM_PROMPT = """You are an evidence-guided Text-to-SQL schema grounding agent.
Return only valid JSON.
Use only table and column names provided in the schema inventory.
Prefer exact schema names from table_name and column_name.
Do not invent tables, columns, or database values.
For database values, output natural-language candidates only; they will be verified by tools."""


def build_clause_prompt(question: str, evidence: str | None) -> str:
    """Build the clause decomposition prompt."""
    return f"""Decompose the natural-language question and evidence into semantic clauses.

Return this JSON shape:
{{
  "clauses": [
    {{
      "id": "question_0",
      "source": "question",
      "text": "clause text",
      "entities": ["business entity or literal value candidate"],
      "operators": [
        {{"type": "filter|aggregation|group_by|order_by|limit|calculation|comparison", "expression": "short expression", "confidence": 0.0}}
      ]
    }}
  ]
}}

Rules:
- Extract business entities and literal values, not schema labels.
- Keep formulas from evidence as calculation operators.
- Do not treat grade notation like K-12 as the numeric value 12 unless the user asks for the number 12 itself.
- Confidence must be between 0 and 1.

Question:
{question}

Evidence:
{evidence or ""}
"""


def build_schema_selection_prompt(
    question: str,
    evidence: str | None,
    clauses: list[dict],
    schema_inventory: list[dict],
    max_columns: int,
) -> str:
    """Build the schema selection prompt."""
    return f"""Select the minimal schema context needed to answer the question.

Return this JSON shape:
{{
  "selected_columns": [
    {{
      "table_name": "exact table_name",
      "column_name": "exact column_name",
      "reason": "why this column is needed",
      "confidence": 0.0,
      "value_candidates": ["natural-language value to verify in this column"]
    }}
  ],
  "selected_tables": ["exact table_name"],
  "predicate_hints": [
    {{"type": "filter|aggregation|group_by|order_by|limit|calculation|comparison", "expression": "short expression", "source_text": "NLQ/evidence span", "confidence": 0.0}}
  ]
}}

Rules:
- Use at most {max_columns} selected_columns.
- selected_columns must use exact table_name and column_name from the schema inventory.
- Include join key columns only when they are directly needed for semantics; topology tools will add missing join keys later.
- Put possible database values in value_candidates for the specific column where they should be verified.
- Do not put schema labels or formulas in value_candidates.
- If evidence contains a formula, include all referenced formula columns.
- Each inventory column includes lightweight recall priors:
  - `recall_score`: lexical/semantic similarity between the question/evidence clauses and the column metadata, including small expansions to same-table keys, name-family columns, code/description pairs, and foreign-key neighbors.
  - `recall_hints`: why the column was expanded, such as direct similarity, same-table neighbor, role pair, or FK neighbor.
  - `matched_query_terms`: question/evidence spans that triggered the score.
- Treat high `recall_score` columns as strong candidates, but still verify that the column is semantically needed for the SQL.
- Use expanded neighbor hints to include related columns that the wording did not mention directly, especially keys, formula inputs, category/code-label pairs, and FK bridge columns.
- `semantic_role` is a compatibility hint (`code`, `description`, or `data`). When both a code and description column are plausible, prefer the one that matches the requested output: code/category value for type/category filters, description/name only when full text/name/description is requested.
- Return only JSON.

Question:
{question}

Evidence:
{evidence or ""}

Clauses JSON:
{json.dumps(clauses, ensure_ascii=False)}

Schema inventory JSON:
{json.dumps(schema_inventory, ensure_ascii=False)}
"""
