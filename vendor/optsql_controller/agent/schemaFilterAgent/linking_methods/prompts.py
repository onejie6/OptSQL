"""Prompt templates for embedded schema linking."""

DIRECT_LINKING_PROMPT = """
# Task:
You are an expert data analyst. Examine the database schema, question, and hint,
then select the specific tables and columns needed to answer the question.

# Instructions:
1. Select any column that is needed by SELECT, WHERE, JOIN, GROUP BY, ORDER BY, LIMIT, or calculations.
2. If a column may contain values related to the question, include it.
3. If multiple tables must be joined, include the join-key columns when visible.
4. It is safer to include a necessary neighboring key column than to omit it.

# Output Format:
Only output XML:
<reasoning>
    Concise reasoning.
</reasoning>
<result>
    <table table_name="table_name">
        <column column_name="column_name" />
    </table>
</result>

# Database Schema:
{DATABASE_SCHEMA}

# Question:
{QUESTION}

# Hint:
{HINT}
"""


REVERSED_LINKING_PROMPT = """
# Task:
Generate one SQLite SQL query that answers the question using only the provided
schema. This SQL is used only to infer which schema elements are needed.

# Rules:
1. Use exact table and column names from the schema.
2. Prefer explicit joins using visible key relationships.
3. Do not invent schema objects.

# Output Format:
Only output XML:
<reasoning>
    Concise SQL construction reasoning.
</reasoning>
<result>
    SELECT ...
</result>

# Database Schema:
{DATABASE_SCHEMA}

# Question:
{QUESTION}

# Hint:
{HINT}
"""


def build_direct_linking_prompt(
    *,
    schema_profile: str,
    question: str,
    evidence: str | None,
) -> str:
    return DIRECT_LINKING_PROMPT.format(
        DATABASE_SCHEMA=schema_profile,
        QUESTION=question,
        HINT=evidence or "",
    )


def build_reversed_linking_prompt(
    *,
    schema_profile: str,
    question: str,
    evidence: str | None,
) -> str:
    return REVERSED_LINKING_PROMPT.format(
        DATABASE_SCHEMA=schema_profile,
        QUESTION=question,
        HINT=evidence or "",
    )
