"""Prompt templates for neutral SQL generation methods."""

from __future__ import annotations


BASE_RULES = """
# Rules:
1. Generate SQLite SELECT SQL only.
2. Use only the provided tables, columns, value mappings, predicate hints, and join edges.
3. Do not invent schema objects or exact values.
4. Quote table or column identifiers when they contain spaces or punctuation.
5. Return only XML with a single <result> SQL block.
"""


DC_SQL_PROMPT = """
# Task:
Generate SQL with a recursive divide-and-conquer strategy.

First decompose the question into SQL sub-problems, then combine them into one executable query.
{BASE_RULES}

# Database Schema:
{DATABASE_SCHEMA}

# Question:
{QUESTION}

# Hint:
{HINT}

# Output:
<reasoning>
Brief decomposition and combination reasoning.
</reasoning>
<result>
SELECT ...
</result>
"""


SKELETON_SQL_PROMPT = """
# Task:
Generate SQL with a plan -> skeleton -> complete strategy.

Plan the required SELECT/FROM/JOIN/WHERE/GROUP BY/HAVING/ORDER BY/LIMIT pieces, then fill the skeleton with exact schema names.
{BASE_RULES}

# Database Schema:
{DATABASE_SCHEMA}

# Question:
{QUESTION}

# Hint:
{HINT}

# Output:
<reasoning>
Brief plan and skeleton reasoning.
</reasoning>
<result>
SELECT ...
</result>
"""


ICL_SQL_PROMPT = """
# Task:
Generate SQL by adapting the few-shot examples to the target schema.
{BASE_RULES}

# Few-Shot Examples:
{FEW_SHOT_EXAMPLES}

# Target Database Schema:
{DATABASE_SCHEMA}

# Target Question:
{QUESTION}

# Hint:
{HINT}

# Output:
<reasoning>
Brief example-pattern adaptation reasoning.
</reasoning>
<result>
SELECT ...
</result>
"""


SYNTAX_REVISION_PROMPT = """
# Task:
The SQL below failed syntax or execution validation. Revise it into executable SQLite SELECT SQL.

# Rules:
1. Preserve the question intent.
2. Use only the provided schema profile and verified values.
3. Fix syntax, identifier quoting, function usage, and other execution blockers.
4. Do not perform semantic rewrites beyond what is needed to make the SQL executable.
5. Return only XML with a single <result> SQL block.

# Database Schema:
{DATABASE_SCHEMA}

# Question:
{QUESTION}

# Hint:
{HINT}

# Failed SQL:
{SQL}

# Error / Result:
{ERROR}

# Output:
<reasoning>
Brief syntax fix explanation.
</reasoning>
<result>
SELECT ...
</result>
"""


def build_generation_prompt(
    *,
    method: str,
    schema_profile: str,
    question: str,
    evidence: str | None,
    few_shot_examples: list[dict] | None = None,
) -> str:
    template = {
        "divide_and_conquer": DC_SQL_PROMPT,
        "skeleton": SKELETON_SQL_PROMPT,
        "icl": ICL_SQL_PROMPT,
    }[method]
    examples = "\n".join(
        f"- Question: {item.get('question', '')}\n  SQL: {item.get('sql', '')}"
        for item in (few_shot_examples or [])
    )
    return template.format(
        BASE_RULES=BASE_RULES,
        DATABASE_SCHEMA=schema_profile,
        QUESTION=question,
        HINT=evidence or "",
        FEW_SHOT_EXAMPLES=examples,
    )


def build_syntax_revision_prompt(
    *,
    schema_profile: str,
    question: str,
    evidence: str | None,
    sql: str,
    error: str,
) -> str:
    return SYNTAX_REVISION_PROMPT.format(
        DATABASE_SCHEMA=schema_profile,
        QUESTION=question,
        HINT=evidence or "",
        SQL=sql,
        ERROR=error,
    )
