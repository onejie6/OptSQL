"""Prompt builders for the Initial SQL Builder Agent."""

import json


SQL_BUILDER_SYSTEM_PROMPT = """You are a strict SQL generation agent. You generate executable SQLite SQL.
Return only valid JSON.
You MUST only use the tables, columns, values, and join topology provided in the Verified Context Blueprint.
Do not invent tables, columns, or values.
Do not use natural-language values as exact filter conditions unless they appear in value_mappings.
All JOINs must follow the join topology exactly."""


def build_sql_generation_prompt(
    question: str,
    evidence: str | None,
    blueprint_json: dict,
    dbms: str,
) -> str:
    """Build the prompt for generating base SQL from a verified blueprint."""
    return f"""Generate an executable {dbms} SQL query from the verified context blueprint.

Return this JSON shape:
{{
  "sql": "the generated SQL query",
  "explanation": "step-by-step reasoning for the generated SQL",
  "tables_used": ["list of tables referenced in the SQL"],
  "columns_used": ["list of columns referenced in the SQL"]
}}

Rules:
- Only use tables listed in selected_tables of the blueprint.
- Only use columns listed in selected_columns of the blueprint.
- All JOINs must follow edges in join_topology exactly. Use the source_table, source_column, target_table, target_column, and join_type from each edge.
- Use value_mappings to translate natural-language values to exact column=value conditions.
- Apply predicate_hints for filters, aggregations, GROUP BY, ORDER BY, LIMIT, and calculations.
- If evidence contains a formula, include all referenced columns and the calculation logic explicitly in SELECT, WHERE, HAVING, or ORDER BY.
- Use standard SQLite syntax (double-quoted identifiers if needed).
- Ensure the SQL is complete and executable — no placeholder comments or missing clauses.
- Return only JSON.

Question:
{question}

Evidence:
{evidence or ""}

Verified Context Blueprint:
{json.dumps(blueprint_json, ensure_ascii=False)}
"""


def build_repair_prompt(
    sql: str,
    error_message: str,
    question: str,
    evidence: str | None,
    blueprint_json: dict,
    dbms: str,
) -> str:
    """Build the prompt for repairing a failed SQL query."""
    return f"""The following SQL query failed to execute on {dbms}.

Failed SQL:
{sql}

Error message:
{error_message}

Repair the SQL so it executes successfully. Return this JSON shape:
{{
  "sql": "the repaired SQL query",
  "explanation": "what was wrong and how you fixed it"
}}

Rules:
- Only use tables, columns, values, and joins from the Verified Context Blueprint.
- Do not invent schema elements.
- Return only JSON.

Question:
{question}

Evidence:
{evidence or ""}

Verified Context Blueprint:
{json.dumps(blueprint_json, ensure_ascii=False)}
"""
