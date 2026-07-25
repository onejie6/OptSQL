from .prompt_template import *
from .spider2_prompt_template import *
from typing import List, Dict, Any, Tuple, Optional


def _is_spider2_db_type(db_type: Optional[str]) -> bool:
    """Check if the database type is a Spider2 database."""
    return db_type is not None and db_type in ("bigquery", "snowflake", "sqlite")


def _format_few_shot_example(index: int, example: Dict[str, Any]) -> str:
    lines = [f"- Example {index}:", f"Question: {example['question']}"]
    evidence = str(example.get("evidence", example.get("hint", ""))).strip()
    if evidence:
        lines.append(f"Hint: {evidence}")
    lines.append(f"SQL: {example['sql']}")
    return "\n".join(lines)


class PromptFactory:
    
    @staticmethod
    def format_keywords_extraction_prompt(question: str, hint: str) -> str:
        return KEYWORDS_EXTRACTION_PROMPT.format(QUESTION=question, HINT=hint)
    
    @staticmethod
    def format_direct_linking_prompt(database_schema: str, question: str, hint: str, db_type: Optional[str] = None) -> str:
        if _is_spider2_db_type(db_type):
            return SPIDER2_DIRECT_LINKING_PROMPT.format(DATABASE_SCHEMA=database_schema, QUESTION=question, HINT=hint, DATABASE_ENGINE=db_type.upper())
        return DIRECT_LINKING_PROMPT.format(DATABASE_SCHEMA=database_schema, QUESTION=question, HINT=hint)
    
    @staticmethod
    def format_skeleton_sql_generation_prompt(database_schema: str, question: str, hint: str, db_type: Optional[str] = None) -> str:
        if _is_spider2_db_type(db_type):
            return SPIDER2_SKELETON_SQL_GENERATION_PROMPT.format(DATABASE_SCHEMA=database_schema, QUESTION=question, HINT=hint, DATABASE_ENGINE=db_type.upper())
        return SKELETON_SQL_GENERATION_PROMPT.format(DATABASE_SCHEMA=database_schema, QUESTION=question, HINT=hint)

    @staticmethod
    def format_evidence_blueprint_prompt(
        database_schema: str,
        question: str,
        hint: str,
        relationship_hints: str = "No additional deterministic hints.",
    ) -> str:
        return EVIDENCE_BLUEPRINT_PROMPT.format(
            DATABASE_SCHEMA=database_schema,
            QUESTION=question,
            HINT=hint,
            RELATIONSHIP_HINTS=relationship_hints,
        )

    @staticmethod
    def format_evidence_sql_generation_prompt(
        database_schema: str,
        question: str,
        hint: str,
        blueprint: str = "No separate blueprint was available; derive it from the contract rules.",
        relationship_hints: str = "No additional deterministic hints.",
    ) -> str:
        return EVIDENCE_SQL_GENERATION_PROMPT.format(
            DATABASE_SCHEMA=database_schema,
            QUESTION=question,
            HINT=hint,
            BLUEPRINT=blueprint,
            RELATIONSHIP_HINTS=relationship_hints,
        )
    
    @staticmethod
    def format_dc_sql_generation_prompt(database_schema: str, question: str, hint: str, db_type: Optional[str] = None) -> str:
        if _is_spider2_db_type(db_type):
            return SPIDER2_DC_SQL_GENERATION_PROMPT.format(DATABASE_SCHEMA=database_schema, QUESTION=question, HINT=hint, DATABASE_ENGINE=db_type.upper())
        return DC_SQL_GENERATION_PROMPT.format(DATABASE_SCHEMA=database_schema, QUESTION=question, HINT=hint)
    
    @staticmethod
    def format_icl_sql_generation_prompt(few_shot_examples: List[Dict[str, Any]], database_schema: str, question: str, hint: str, db_type: Optional[str] = None) -> str:
        few_shot_examples_str = "\n".join(
            _format_few_shot_example(i + 1, example)
            for i, example in enumerate(few_shot_examples)
        )
        if _is_spider2_db_type(db_type):
            return SPIDER2_ICL_SQL_GENERATION_PROMPT.format(FEW_SHOT_EXAMPLES=few_shot_examples_str, DATABASE_SCHEMA=database_schema, QUESTION=question, HINT=hint, DATABASE_ENGINE=db_type.upper())
        return ICL_SQL_GENERATION_PROMPT.format(FEW_SHOT_EXAMPLES=few_shot_examples_str, DATABASE_SCHEMA=database_schema, QUESTION=question, HINT=hint)

    @staticmethod
    def format_execution_checker_prompt(database_schema: str, question: str, hint: str, sql: str, execution_result: str, db_type: Optional[str] = None) -> str:
        if _is_spider2_db_type(db_type):
            return SPIDER2_EXECUTION_CHECKER_PROMPT.format(DATABASE_SCHEMA=database_schema, QUESTION=question, HINT=hint, QUERY=sql, RESULT=execution_result, DATABASE_ENGINE=db_type.upper())
        return EXECUTION_CHECKER_PROMPT.format(DATABASE_SCHEMA=database_schema, QUESTION=question, HINT=hint, QUERY=sql, RESULT=execution_result)
    
    @staticmethod
    def format_common_checker_prompt(database_schema: str, question: str, hint: str, sql: str, suggestions: str, db_type: Optional[str] = None) -> str:
        if _is_spider2_db_type(db_type):
            return SPIDER2_COMMON_CHECKER_PROMPT.format(DATABASE_SCHEMA=database_schema, QUESTION=question, HINT=hint, QUERY=sql, SUGGESTIONS=suggestions, DATABASE_ENGINE=db_type.upper())
        return COMMON_CHECKER_PROMPT.format(DATABASE_SCHEMA=database_schema, QUESTION=question, HINT=hint, QUERY=sql, SUGGESTIONS=suggestions)
    
    @staticmethod
    def format_br_pair_selection_prompt(database_schema: str, question: str, hint: str, query_a: str, result_a: str, query_b: str, result_b: str, db_type: Optional[str] = None) -> str:
        if _is_spider2_db_type(db_type):
            return SPIDER2_BR_PAIR_SELECTION_PROMPT.format(DATABASE_SCHEMA=database_schema, QUESTION=question, HINT=hint, QUERY_A=query_a, RESULT_A=result_a, QUERY_B=query_b, RESULT_B=result_b, DATABASE_ENGINE=db_type.upper())
        return BR_PAIR_SELECTION_PROMPT.format(DATABASE_SCHEMA=database_schema, QUESTION=question, HINT=hint, QUERY_A=query_a, RESULT_A=result_a, QUERY_B=query_b, RESULT_B=result_b)

    @staticmethod
    def format_evidence_listwise_selection_prompt(
        database_schema: str,
        question: str,
        hint: str,
        candidates: List[Tuple],
    ) -> str:
        if len(candidates) > 26:
            raise ValueError("Listwise selection supports at most 26 candidates")
        candidate_blocks = []
        for index, candidate in enumerate(candidates):
            query, result = candidate[:2]
            support = candidate[2] if len(candidate) > 2 else None
            audit_findings = candidate[3] if len(candidate) > 3 else []
            label = chr(ord("A") + index)
            support_line = (
                f"Execution-cluster support: {support:.1%}\n" if support is not None else ""
            )
            audit_line = (
                "Deterministic audit warnings (advisory only): "
                + ("; ".join(audit_findings) if audit_findings else "none")
                + "\n"
            )
            candidate_blocks.append(
                f"## Candidate {label}\n{support_line}{audit_line}SQL:\n{query}\n\n"
                f"Execution result:\n{result[:4000]}"
            )
        return EVIDENCE_LISTWISE_SELECTION_PROMPT.format(
            DATABASE_SCHEMA=database_schema,
            QUESTION=question,
            HINT=hint,
            CANDIDATES="\n\n".join(candidate_blocks),
        )
