import re
from typing import Dict, List, Tuple

from app.dataset import DataItem
from app.llm import LLM
from app.logger import logger
from app.prompt import PromptFactory

from .base import BaseSQLGenerator


def _contains_superlative(question: str) -> bool:
    return bool(
        re.search(
            r"\b(?:biggest|largest|highest|lowest|smallest|most|least|maximum|minimum)\b",
            question,
            re.IGNORECASE,
        )
    )


class EvidenceGenerator(BaseSQLGenerator):
    @staticmethod
    def _find_schema_columns(tables: dict, column_name: str) -> List[str]:
        return [
            f"{table_name}.{candidate_name}"
            for table_name, table in tables.items()
            for candidate_name in table.get("columns", {})
            if candidate_name.lower() == column_name.lower()
        ]

    @staticmethod
    def _build_relationship_hints(data_item: DataItem) -> str:
        schema = data_item.database_schema_after_schema_linking
        tables = schema.get("tables", {}) if isinstance(schema, dict) else {}
        intent = f"{data_item.question}\n{data_item.evidence}".lower()
        counted_terms = {
            match.group(1).rstrip("s")
            for match in re.finditer(
                r"count\s*[\[(]\s*(?:distinct\s+)?(?:female\s+|male\s+)?([a-z_]+)",
                intent,
            )
        }
        lines = []
        cohort_match = re.search(
            r"\bpercentage\s+of\s+(?:the\s+)?(.+?)\s+(?:who|that|which|whose|with)\b",
            data_item.question,
            re.IGNORECASE,
        )
        if cohort_match:
            cohort = cohort_match.group(1).lower()
            for column_name, value in re.findall(
                r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*['\"]([^'\"]+)['\"]",
                str(data_item.evidence),
            ):
                if value.lower().rstrip("s") not in cohort.rstrip("s"):
                    continue
                qualified = EvidenceGenerator._find_schema_columns(tables, column_name)
                binding = qualified[0] if len(qualified) == 1 else column_name
                lines.append(
                    f"Denominator cohort binding: {binding} = '{value}' defines the X population; "
                    "apply it before COUNT, while other conditions define the numerator"
                )

        if _contains_superlative(data_item.question):
            for clause in str(data_item.evidence).split(";"):
                metrics = re.finditer(
                    r"\b([A-Za-z_][A-Za-z0-9_]*)\s+(?:contains|refers to)\b",
                    clause,
                    re.IGNORECASE,
                )
                for metric in metrics:
                    qualified = EvidenceGenerator._find_schema_columns(tables, metric.group(1))
                    if qualified:
                        table_name, column_name = qualified[0].split(".", 1)
                        column_type = (
                            tables[table_name].get("columns", {}).get(column_name, {}).get("column_type")
                            or "unknown"
                        )
                        lines.append(
                            f"Superlative metric binding: {qualified[0]} (declared type {column_type}); "
                            "aggregate at the requested metric/group grain before LIMIT, do not preselect "
                            "an arbitrary entity key, and do not add CAST without an explicit conversion"
                        )

        structured_ids = re.findall(
            r"\b([A-Za-z0-9]+(?:_[A-Za-z0-9]+){2,})\b",
            f"{data_item.question} {data_item.evidence}",
        )
        if structured_ids:
            lines.append(
                "Structured identifier detected: "
                + ", ".join(dict.fromkeys(structured_ids))
                + "; verify whether delimiter-separated components encode the requested IDs "
                "before using an auxiliary relationship table"
            )
            if re.search(r"\b(?:bond|connect|relationship|edge|link)\w*\b", data_item.question, re.IGNORECASE):
                lines.append(
                    "Encoded relationship endpoint shape: return one relationship row with two endpoint "
                    "columns derived from the identifier; do not emit two UNION rows or duplicate both "
                    "forward and reverse stored orientations"
                )
        for clause in str(data_item.evidence).split(";"):
            mapping = re.match(
                r"\s*(.+?)\s+refers\s+to\s+([A-Za-z_][A-Za-z0-9_.]*)\s*$",
                clause,
                re.IGNORECASE,
            )
            if mapping:
                lines.append(
                    f'Binding output/schema mapping: "{mapping.group(1).strip()}" -> '
                    f"{mapping.group(2)}; use this exact column rather than a lookup display value"
                )
        if all(operator in intent for operator in ("subtract(", "divide(", "multiply(")):
            lines.append(
                "Binding arithmetic formula detected: preserve all matching historical rows with "
                "SUM(CASE WHEN ...) operands; do not substitute AVG or latest-row selection"
            )
        for table_name, table in tables.items():
            phrase = table_name.replace("_", " ").lower()
            mentioned = re.search(rf"(?<!\w){re.escape(phrase)}(?:s|es)?(?!\w)", intent)
            counted = phrase.rstrip("s") in counted_terms
            if not mentioned:
                continue
            if counted:
                lines.append(f"Likely counting row grain from evidence: {table_name}")
            singular_name = phrase.rstrip("s")
            if re.search(
                rf"(?<!\w)(?:name|names)\s+of\s+(?:all\s+)?{re.escape(phrase)}(?!\w)",
                intent,
            ):
                for column_name in table.get("columns", {}):
                    if column_name.lower().replace("_", " ") in {"name", singular_name}:
                        lines.append(
                            f"Likely entity-name projection: {table_name}.{column_name}; "
                            "prefer the requested entity table over similarly named joined columns"
                        )
            for column_name, column in table.get("columns", {}).items():
                description = column.get("description", "")
                for target_table, target_column in column.get("foreign_keys", []):
                    lines.append(
                        f"Direct FK: {table_name}.{column_name} -> "
                        f"{target_table}.{target_column}; description: {description or '(none)'}"
                    )
        return "\n".join(dict.fromkeys(lines)) or "No additional deterministic hints."

    @staticmethod
    def _route_focus(sample_index: int) -> str:
        if sample_index == 0:
            return "Primary route: implement the most literal schema-grounded evidence contract."
        return (
            "Adversarial route: independently challenge the obvious first-pass interpretation. "
            "Audit denominator cohort, aggregation grain before superlative LIMIT, repeated metric "
            "values, output shape, and whether structured identifiers encode requested components."
        )

    def generate(
        self,
        data_item: DataItem,
        llm: LLM,
        sampling_budget: int = 1,
    ) -> Tuple[List[str], Dict[str, int]]:
        if sampling_budget == 0:
            return [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        relationship_hints = self._build_relationship_hints(data_item)

        candidates = []
        for sample_index in range(sampling_budget):
            route_hints = f"{relationship_hints}\n{self._route_focus(sample_index)}"

            def blueprint_prompt_format_func(schema_profile: str) -> str:
                return PromptFactory.format_evidence_blueprint_prompt(
                    schema_profile,
                    data_item.question,
                    data_item.evidence,
                    route_hints,
                )

            blueprint_prompt, _ = self._generate_with_progressive_stripping(
                data_item,
                llm,
                blueprint_prompt_format_func,
            )
            blueprint = ""
            if blueprint_prompt is not None:
                blueprints, blueprint_usage = self._get_extractor().extract_with_retry(
                    llm=llm,
                    messages=[{"role": "user", "content": blueprint_prompt}],
                    rule_parser=self._parse_llm_response,
                    fix_end_token=llm.llm_config.fix_end_token,
                    end_token="</result>",
                    n=1,
                )
                for key in total_usage:
                    total_usage[key] += blueprint_usage[key]
                if blueprints:
                    blueprint = blueprints[0]

            def prompt_format_func(schema_profile: str) -> str:
                return PromptFactory.format_evidence_sql_generation_prompt(
                    schema_profile,
                    data_item.question,
                    data_item.evidence,
                    blueprint or "No separate blueprint was available; derive it from the contract rules.",
                    route_hints,
                )

            final_prompt, _ = self._generate_with_progressive_stripping(
                data_item,
                llm,
                prompt_format_func,
            )
            if final_prompt is None:
                logger.error(
                    f"Evidence-contract prompt for item {data_item.question_id} exceeds the token limit"
                )
                continue
            generated, generation_usage = self._get_extractor().extract_with_retry(
                llm=llm,
                messages=[{"role": "user", "content": final_prompt}],
                rule_parser=self._parse_llm_response,
                fix_end_token=llm.llm_config.fix_end_token,
                end_token="</result>",
                n=1,
            )
            for key in total_usage:
                total_usage[key] += generation_usage[key]
            candidates.extend(generated)

        return list(dict.fromkeys(candidates)), total_usage
