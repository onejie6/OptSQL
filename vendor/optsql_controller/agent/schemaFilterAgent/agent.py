"""Evidence-guided Schema Filter Agent implementation backed by LangGraph."""

from dataclasses import asdict
from typing import Any
from typing import TypedDict

from agent.base import BaseAgent
from agent.schemaFilterAgent.linking_methods import SchemaLinkingPipeline
from agent.schemaFilterAgent.json_parser import parse_json_object
from agent.schemaFilterAgent.prompts import SCHEMA_FILTER_SYSTEM_PROMPT
from agent.schemaFilterAgent.prompts import build_clause_prompt
from myTypes import AgentRequest
from myTypes import AgentResponse
from myTypes import ColumnRef
from myTypes import EvidenceTrace
from myTypes import JoinEdge
from myTypes import JoinGraph
from myTypes import PredicateHint
from myTypes import ValueMapping
from myTypes import VerifiedContextBlueprint
from utils.openai_client import request_chat_text
from utils.schema_grounding import build_topology_closure
from utils.schema_grounding import inspect_column_info
from utils.schema_grounding import list_foreign_key_edges
from utils.schema_grounding import list_schema_columns
from utils.schema_grounding import normalize_text
from utils.schema_grounding import probe_similar_values
from utils.schema_grounding import tokenize
from utils.schema_grounding import verify_exact_value


class _SchemaFilterState(TypedDict, total=False):
    request: AgentRequest
    clauses: list[dict]
    candidates: list[dict]
    verified_candidates: list[dict]
    value_mappings: list[ValueMapping]
    selected_tables: list[str]
    join_topology: JoinGraph
    predicate_hints: list[PredicateHint]
    blueprint: VerifiedContextBlueprint


class EvidenceGuidedSchemaFilterAgent(BaseAgent):
    """Ground NLQ and evidence to verified schema, values, and join topology."""

    name = "evidence_guided_schema_filter"

    def __init__(
        self,
        split: str = "dev",
        max_candidates: int = 40,
        similar_value_threshold: float = 0.65,
        llm_model: str | None = None,
        llm_temperature: float = 0.0,
        llm_max_tokens: int = 4096,
        llm_client=None,
    ) -> None:
        self.split = split
        self.max_candidates = max_candidates
        self.similar_value_threshold = similar_value_threshold
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.llm_client = llm_client
        self._last_clause_response: dict | None = None
        self._last_recall_profiles: list[dict] | None = None
        self._last_schema_response: dict | None = None
        self._last_schema_linking: dict | None = None
        self._schema_linker = SchemaLinkingPipeline(
            llm_model=llm_model,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            llm_client=llm_client,
            value_distance_threshold=1.0 - similar_value_threshold,
        )
        self._graph: Any | None = None

    def decompose_clauses(self, question: str, evidence: str | None) -> list[dict]:
        """Use an LLM to decompose NLQ/evidence into semantic clauses."""
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string.")

        response = self._request_json(build_clause_prompt(question, evidence))
        raw_clauses = response.get("clauses", [])
        if not isinstance(raw_clauses, list) or not raw_clauses:
            raise ValueError("LLM clause response must contain a non-empty clauses list.")

        clauses = []
        for index, raw_clause in enumerate(raw_clauses):
            if not isinstance(raw_clause, dict):
                continue

            text = str(raw_clause.get("text") or "").strip()
            if not text:
                continue

            source = str(raw_clause.get("source") or "question")
            clause_id = str(raw_clause.get("id") or f"{source}_{index}")
            clauses.append(
                {
                    "id": clause_id,
                    "source": source,
                    "text": text,
                    "entities": self._coerce_string_list(raw_clause.get("entities", [])),
                    "operators": self._normalize_operator_hints(raw_clause.get("operators", []), text),
                }
            )

        if not clauses:
            raise ValueError("LLM clause response did not contain valid clauses.")

        self._last_clause_response = response
        return clauses

    def soft_prune(self, clauses: list[dict], db_id: str) -> list[dict]:
        """Run embedded schema linking to select candidates."""
        schema_columns = list_schema_columns(db_id, split=self.split)
        recall_profiles = self._build_similarity_recall_profiles(
            schema_columns,
            clauses,
            db_id,
        )
        self._last_recall_profiles = self._recall_profiles_for_artifacts(recall_profiles)

        linker_output = self._schema_linker.link(
            db_id=db_id,
            question=self._question_from_clauses(clauses),
            evidence=self._evidence_from_clauses(clauses),
            clauses=clauses,
            schema_columns=schema_columns,
            max_columns=self.max_candidates,
        )

        if not linker_output.selected_columns:
            raise ValueError("Embedded schema linking did not select any valid schema columns.")

        self._last_schema_linking = asdict(linker_output)
        self._last_schema_response = self._schema_response_from_linking_output(
            linker_output.selected_columns
        )
        return linker_output.selected_columns

    def hard_prune(self, candidates: list[dict], db_id: str) -> list[dict]:
        """Verify LLM-proposed value candidates against real database values."""
        verified_candidates = []
        for candidate in candidates:
            verified_candidate = {
                **candidate,
                "value_matches": list(candidate.get("value_matches", [])),
            }
            for value_candidate in candidate.get("value_candidates", []):
                exact_match = verify_exact_value(
                    db_id,
                    candidate["table_name"],
                    candidate["column_name"],
                    value_candidate,
                )
                if exact_match["exists"]:
                    verified_candidate["value_matches"].append(
                        {
                            "keyword": value_candidate,
                            "value": exact_match["examples"][0],
                            "confidence": 1.0,
                            "evidence": "verify_exact_value",
                        }
                    )
                    verified_candidate["reasons"].append(f"exact_value={value_candidate}")
                    continue

                similar_values = probe_similar_values(
                    db_id,
                    candidate["table_name"],
                    candidate["column_name"],
                    value_candidate,
                    top_k=1,
                )
                if similar_values and similar_values[0]["score"] >= self.similar_value_threshold:
                    verified_candidate["value_matches"].append(
                        {
                            "keyword": value_candidate,
                            "value": similar_values[0]["value"],
                            "confidence": float(similar_values[0]["score"]),
                            "evidence": "probe_similar_values",
                        }
                    )
                    verified_candidate["reasons"].append(
                        f"similar_value={value_candidate}->{similar_values[0]['value']}"
                    )

            verified_candidates.append(verified_candidate)

        return verified_candidates

    def correct_values(self, verified_candidates: list[dict], db_id: str) -> list[ValueMapping]:
        """Convert verified database facts into structured value mappings."""
        del db_id
        mappings = []
        seen = set()
        for candidate in verified_candidates:
            for match in candidate.get("value_matches", []):
                key = (
                    match["keyword"],
                    candidate["table_name"],
                    candidate["column_name"],
                    str(match["value"]),
                )
                if key in seen:
                    continue
                seen.add(key)
                mappings.append(
                    ValueMapping(
                        keyword=match["keyword"],
                        table_name=candidate["table_name"],
                        column_name=candidate["column_name"],
                        value=match["value"],
                        confidence=float(match["confidence"]),
                        evidence=match["evidence"],
                    )
                )

        return mappings

    def close_topology(self, selected_tables: list[str], db_id: str) -> JoinGraph:
        """Build a connected foreign-key graph for selected tables when possible."""
        closure = build_topology_closure(db_id, selected_tables, split=self.split)
        return JoinGraph(
            tables=closure["tables"],
            edges=[
                JoinEdge(
                    source_table=edge["source_table"],
                    source_column=edge["source_column"],
                    target_table=edge["target_table"],
                    target_column=edge["target_column"],
                    join_type=edge["join_type"],
                )
                for edge in closure["edges"]
            ],
        )

    def infer_predicate_hints(
        self,
        clauses: list[dict],
        value_mappings: list[ValueMapping],
    ) -> list[PredicateHint]:
        """Use LLM-produced operator hints plus verified value bindings."""
        hints: list[PredicateHint] = []

        schema_response_hints = []
        if self._last_schema_response:
            raw_hints = self._last_schema_response.get("predicate_hints", [])
            if isinstance(raw_hints, list):
                schema_response_hints = raw_hints

        for raw_hint in schema_response_hints:
            if not isinstance(raw_hint, dict):
                continue
            hints.append(
                PredicateHint(
                    predicate_type=str(raw_hint.get("type") or "unknown"),
                    expression=str(raw_hint.get("expression") or ""),
                    source_text=str(raw_hint.get("source_text") or ""),
                    confidence=self._coerce_confidence(raw_hint.get("confidence"), default=0.7),
                )
            )

        if not hints:
            for clause in clauses:
                for operator in clause.get("operators", []):
                    hints.append(
                        PredicateHint(
                            predicate_type=operator["type"],
                            expression=operator["expression"],
                            source_text=clause["text"],
                            confidence=operator["confidence"],
                        )
                    )

        for mapping in value_mappings:
            hints.append(
                PredicateHint(
                    predicate_type="filter",
                    expression=f"{mapping.table_name}.{mapping.column_name} = {mapping.value!r}",
                    source_text=mapping.keyword,
                    confidence=mapping.confidence,
                )
            )

        return hints

    def build_blueprint(
        self,
        db_id: str,
        verified_candidates: list[dict],
        value_mappings: list[ValueMapping],
        join_topology: JoinGraph,
        predicate_hints: list[PredicateHint],
    ) -> VerifiedContextBlueprint:
        """Package verified schema, values, topology, and evidence traces."""
        selected_tables = list(dict.fromkeys(
            [candidate["table_name"] for candidate in verified_candidates]
            + join_topology.tables
        ))
        selected_columns = self._build_column_refs(db_id, verified_candidates, join_topology)
        evidence_trace = self._build_evidence_trace(
            verified_candidates,
            value_mappings,
            join_topology,
        )
        confidence = self._blueprint_confidence(verified_candidates, value_mappings)

        return VerifiedContextBlueprint(
            db_id=db_id,
            selected_tables=selected_tables,
            selected_columns=selected_columns,
            value_mappings=value_mappings,
            join_topology=join_topology,
            predicate_hints=predicate_hints,
            evidence_trace=evidence_trace,
            confidence=confidence,
        )

    def run(self, request: AgentRequest) -> AgentResponse:
        """Run the LangGraph-backed evidence-guided schema filtering pipeline."""
        try:
            self._last_clause_response = None
            self._last_recall_profiles = None
            self._last_schema_response = None
            self._last_schema_linking = None
            final_state = self._get_graph().invoke({"request": request})
            clauses = final_state["clauses"]
            candidates = final_state["candidates"]
            verified_candidates = final_state["verified_candidates"]
            value_mappings = final_state["value_mappings"]
            join_topology = final_state["join_topology"]
            predicate_hints = final_state["predicate_hints"]
            blueprint = final_state["blueprint"]
        except Exception as exc:
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status="error",
                output_artifacts={
                    "llm_clause_response": self._last_clause_response,
                    "schema_recall_profiles": self._last_recall_profiles,
                    "llm_schema_response": self._last_schema_response,
                    "schema_linking": self._last_schema_linking,
                },
                reasoning_summary="LangGraph-backed schema filtering failed before producing a blueprint.",
                tool_calls=[],
                errors=[str(exc)],
            )

        return AgentResponse(
            request_id=request.request_id,
            agent_name=self.name,
            status="success",
            output_artifacts={
                "blueprint": blueprint,
                "clauses": clauses,
                "candidates": candidates,
                "verified_candidates": verified_candidates,
                "value_mappings": value_mappings,
                "join_topology": join_topology,
                "predicate_hints": predicate_hints,
                "llm_clause_response": self._last_clause_response,
                "schema_recall_profiles": self._last_recall_profiles,
                "llm_schema_response": self._last_schema_response,
                "schema_linking": self._last_schema_linking,
            },
            reasoning_summary=(
                f"Selected {len(blueprint.selected_tables)} tables, "
                f"{len(blueprint.selected_columns)} columns, "
                f"{len(blueprint.value_mappings)} value mappings via LangGraph grounding."
            ),
            tool_calls=[
                {"tool_name": "request_chat_text", "summary": "LLM clause decomposition"},
                {"tool_name": "request_chat_text", "summary": "direct schema linking"},
                {"tool_name": "request_chat_text", "summary": "reversed SQL schema linking"},
                {"tool_name": "probe_similar_values", "summary": "embedded value linking"},
                {"tool_name": "verify_exact_value", "summary": "fact-based value verification"},
                {"tool_name": "probe_similar_values", "summary": "fact-based value correction"},
                {"tool_name": "build_topology_closure", "summary": "join topology completion"},
            ],
            errors=[],
        )

    def _get_graph(self):
        """Build and cache the LangGraph state machine used by this agent."""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def _build_graph(self):
        try:
            from langgraph.graph import END
            from langgraph.graph import START
            from langgraph.graph import StateGraph
        except ImportError as exc:
            raise ImportError(
                "LangGraph is required to run EvidenceGuidedSchemaFilterAgent. "
                "Install it with `pip install langgraph`."
            ) from exc

        graph = StateGraph(_SchemaFilterState)
        graph.add_node("decompose_clauses", self._decompose_clauses_node)
        graph.add_node("soft_prune", self._soft_prune_node)
        graph.add_node("hard_prune", self._hard_prune_node)
        graph.add_node("correct_values", self._correct_values_node)
        graph.add_node("close_topology", self._close_topology_node)
        graph.add_node("infer_predicate_hints", self._infer_predicate_hints_node)
        graph.add_node("build_blueprint", self._build_blueprint_node)

        graph.add_edge(START, "decompose_clauses")
        graph.add_edge("decompose_clauses", "soft_prune")
        graph.add_edge("soft_prune", "hard_prune")
        graph.add_edge("hard_prune", "correct_values")
        graph.add_edge("correct_values", "close_topology")
        graph.add_edge("close_topology", "infer_predicate_hints")
        graph.add_edge("infer_predicate_hints", "build_blueprint")
        graph.add_edge("build_blueprint", END)
        return graph.compile()

    def _decompose_clauses_node(self, state: _SchemaFilterState) -> dict:
        task = state["request"].task
        return {"clauses": self.decompose_clauses(task.question, task.evidence)}

    def _soft_prune_node(self, state: _SchemaFilterState) -> dict:
        task = state["request"].task
        return {"candidates": self.soft_prune(state["clauses"], task.db_id)}

    def _hard_prune_node(self, state: _SchemaFilterState) -> dict:
        task = state["request"].task
        return {"verified_candidates": self.hard_prune(state["candidates"], task.db_id)}

    def _correct_values_node(self, state: _SchemaFilterState) -> dict:
        task = state["request"].task
        return {
            "value_mappings": self.correct_values(
                state["verified_candidates"],
                task.db_id,
            )
        }

    def _close_topology_node(self, state: _SchemaFilterState) -> dict:
        task = state["request"].task
        selected_tables = self._select_tables(state["verified_candidates"])
        return {
            "selected_tables": selected_tables,
            "join_topology": self.close_topology(selected_tables, task.db_id),
        }

    def _infer_predicate_hints_node(self, state: _SchemaFilterState) -> dict:
        return {
            "predicate_hints": self.infer_predicate_hints(
                state["clauses"],
                state["value_mappings"],
            )
        }

    def _build_blueprint_node(self, state: _SchemaFilterState) -> dict:
        task = state["request"].task
        return {
            "blueprint": self.build_blueprint(
                db_id=task.db_id,
                verified_candidates=state["verified_candidates"],
                value_mappings=state["value_mappings"],
                join_topology=state["join_topology"],
                predicate_hints=state["predicate_hints"],
            )
        }

    def _request_json(self, prompt: str) -> dict:
        response_text = request_chat_text(
            messages=[
                {"role": "system", "content": SCHEMA_FILTER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=self.llm_model,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
            client=self.llm_client,
        )
        return parse_json_object(response_text)

    def _schema_response_from_linking_output(self, candidates: list[dict]) -> dict:
        """Expose schema linking output through the legacy artifact key."""
        return {
            "selected_columns": [
                {
                    "table_name": candidate["table_name"],
                    "column_name": candidate["column_name"],
                    "confidence": candidate.get("score", 0.0),
                    "value_candidates": candidate.get("value_candidates", []),
                    "reason": "; ".join(candidate.get("reasons", [])),
                    "linking_sources": candidate.get("linking_sources", []),
                }
                for candidate in candidates
            ],
            "predicate_hints": [],
            "source": "embedded_schema_linking",
        }

    def _schema_inventory_for_prompt(
        self,
        schema_columns: list[dict],
        recall_profiles: dict[tuple[str, str], dict] | None = None,
    ) -> list[dict]:
        """Build a prompt-ready schema inventory enriched with recall hints."""
        roles = self._detect_semantic_roles(schema_columns)
        inventory = []
        for column in schema_columns:
            key = (column["table_name"], column["column_name"])
            recall_profile = recall_profiles.get(key, {}) if recall_profiles else {}
            inventory.append(
                {
                    "table_name": column["table_name"],
                    "table_comment": column["table_comment"],
                    "column_name": column["column_name"],
                    "column_comment": column["column_comment"],
                    "data_type": column["data_type"],
                    "is_primary_key": column["is_primary_key"],
                    "is_foreign_key": column["is_foreign_key"],
                    "semantic_role": roles.get(
                        key, "data",
                    ),
                    "recall_score": recall_profile.get("score", 0.0),
                    "recall_hints": recall_profile.get("hints", []),
                    "matched_query_terms": recall_profile.get("matched_query_terms", []),
                }
            )
        return inventory

    def _build_similarity_recall_profiles(
        self,
        schema_columns: list[dict],
        clauses: list[dict],
        db_id: str,
    ) -> dict[tuple[str, str], dict]:
        """Build query-aware lightweight similarity and expansion signals.

        The profiles are broad recall priors for the schema-selection LLM.
        They do not prune the schema inventory; they mark direct matches,
        same-table neighbors, code/description pairs, and FK edge columns.
        """
        if not schema_columns:
            return {}

        queries = self._recall_queries_from_clauses(clauses)
        if not queries:
            return {}

        profiles: dict[tuple[str, str], dict] = {}
        columns_by_key = {
            (column["table_name"], column["column_name"]): column
            for column in schema_columns
        }

        for key, column in columns_by_key.items():
            column_text = self._column_recall_text(column)
            scored_terms = [
                (query, self._recall_similarity(query, column_text))
                for query in queries
            ]
            best_query, best_score = max(scored_terms, key=lambda item: item[1])
            hints = []
            if best_score >= 0.12:
                hints.append(f"direct_similarity:{best_query[:80]}")

            profiles[key] = {
                "score": round(best_score, 4),
                "hints": hints,
                "matched_query_terms": [best_query] if best_score >= 0.12 else [],
                "direct_score": best_score,
            }

        ranked_direct = sorted(
            profiles.items(),
            key=lambda item: item[1]["direct_score"],
            reverse=True,
        )
        seed_limit = min(len(ranked_direct), max(self.max_candidates, 20))
        seed_keys = [
            key
            for key, profile in ranked_direct[:seed_limit]
            if profile["direct_score"] >= 0.18
        ]

        self._expand_recall_with_same_table_neighbors(schema_columns, profiles, seed_keys)
        self._expand_recall_with_foreign_keys(db_id, profiles, seed_keys)

        for profile in profiles.values():
            profile["score"] = round(min(1.0, profile["score"]), 4)
            profile["hints"] = profile["hints"][:4]
            profile["matched_query_terms"] = profile["matched_query_terms"][:3]

        return profiles

    def _expand_recall_with_same_table_neighbors(
        self,
        schema_columns: list[dict],
        profiles: dict[tuple[str, str], dict],
        seed_keys: list[tuple[str, str]],
    ) -> None:
        """Diffuse recall to useful columns that live beside strong matches."""
        by_table: dict[str, list[dict]] = {}
        for column in schema_columns:
            by_table.setdefault(column["table_name"], []).append(column)

        roles = self._detect_semantic_roles(schema_columns)
        for seed_key in seed_keys:
            seed_table, seed_column = seed_key
            seed_profile = profiles[seed_key]
            if seed_profile["direct_score"] < 0.28:
                continue

            seed_base = self._column_base_name(seed_column)
            for neighbor in by_table.get(seed_table, []):
                neighbor_key = (neighbor["table_name"], neighbor["column_name"])
                if neighbor_key == seed_key:
                    continue

                reason = None
                multiplier = 0.0
                if neighbor.get("is_primary_key") or neighbor.get("is_foreign_key"):
                    reason = f"same_table_key:{seed_table}.{seed_column}"
                    multiplier = 0.45
                elif self._column_base_name(neighbor["column_name"]) == seed_base:
                    reason = f"same_table_name_family:{seed_table}.{seed_column}"
                    multiplier = 0.7
                elif self._is_role_pair(roles.get(seed_key), roles.get(neighbor_key)):
                    reason = f"same_table_role_pair:{seed_table}.{seed_column}"
                    multiplier = 0.65

                if reason is None:
                    continue

                expanded_score = seed_profile["direct_score"] * multiplier
                self._merge_recall_hint(
                    profiles[neighbor_key],
                    expanded_score,
                    reason,
                    seed_profile["matched_query_terms"],
                )

    def _expand_recall_with_foreign_keys(
        self,
        db_id: str,
        profiles: dict[tuple[str, str], dict],
        seed_keys: list[tuple[str, str]],
    ) -> None:
        """Diffuse recall from strong table matches onto adjacent FK columns."""
        seed_tables = {
            table_name
            for table_name, column_name in seed_keys
            if profiles[(table_name, column_name)]["direct_score"] >= 0.28
        }
        if not seed_tables:
            return

        try:
            edges = list_foreign_key_edges(db_id, split=self.split)
        except (FileNotFoundError, LookupError, ValueError):
            return

        for edge in edges:
            edge_pairs = [
                (edge["source_table"], edge["source_column"], edge["target_table"]),
                (edge["target_table"], edge["target_column"], edge["source_table"]),
            ]
            for table_name, column_name, other_table in edge_pairs:
                key = (table_name, column_name)
                if key not in profiles:
                    continue
                if table_name not in seed_tables and other_table not in seed_tables:
                    continue

                seed_score = max(
                    (
                        profiles[seed_key]["direct_score"]
                        for seed_key in seed_keys
                        if seed_key[0] in {table_name, other_table}
                    ),
                    default=0.0,
                )
                self._merge_recall_hint(
                    profiles[key],
                    seed_score * 0.4,
                    f"fk_neighbor:{other_table}",
                    [],
                )

    def _recall_profiles_for_artifacts(
        self,
        recall_profiles: dict[tuple[str, str], dict],
    ) -> list[dict]:
        return [
            {
                "table_name": table_name,
                "column_name": column_name,
                "recall_score": profile["score"],
                "recall_hints": profile["hints"],
                "matched_query_terms": profile["matched_query_terms"],
            }
            for (table_name, column_name), profile in sorted(
                recall_profiles.items(),
                key=lambda item: item[1]["score"],
                reverse=True,
            )
            if profile["score"] >= 0.12
        ]

    @staticmethod
    def _detect_semantic_roles(schema_columns: list[dict]) -> dict[tuple[str, str], str]:
        """Tag columns as ``"code"``, ``"description"``, or ``"data"``.

        Uses a combination of column-comment signals and naming patterns
        (X/XType, XCode/XName, X/XName) to help the LLM distinguish
        between a short code column and its verbose description counterpart.
        """
        roles: dict[tuple[str, str], str] = {}

        by_table: dict[str, list[dict]] = {}
        for col in schema_columns:
            by_table.setdefault(col["table_name"], []).append(col)

        for _table_name, columns in by_table.items():
            col_names = {col["column_name"] for col in columns}

            # Pass 1 — comment-based: explicit "describes another column" signals
            for col in columns:
                key = (col["table_name"], col["column_name"])
                comment = (col.get("column_comment") or "").lower()

                # Only tag as "description" when the comment literally says it
                # *is the text description of* a sibling column (e.g. DOCType
                # describes DOC).  "short text description" columns are codes.
                if any(
                    phrase in comment
                    for phrase in (
                        "is the text description of",
                        "is the long text description of",
                    )
                ):
                    roles[key] = "description"

                if any(
                    phrase in comment
                    for phrase in (
                        "numeric code used to identify",
                        "identification number",
                    )
                ):
                    roles[key] = "code"

            # Pass 2 — naming pairs: X/XType, X/XName, XCode/XName.
            # When two columns share a base name and differ only by a
            # code/type/name suffix, they form a code↔description pair.
            _SUFFIX_ROLE = {
                "Type": "description",
                "Name": "description",
            }
            _CODE_SUFFIXES = {"Code", "Type", "Name"}

            for col in columns:
                col_name = col["column_name"]

                for suffix, role in _SUFFIX_ROLE.items():
                    if not col_name.endswith(suffix):
                        continue
                    base = col_name[: -len(suffix)]
                    if not base:
                        continue

                    # XType/XName exists → find the matching code column
                    candidate_code = base  # e.g. DOCType → DOC
                    candidate_code_alt = base + "Code"  # e.g. EILName → EILCode
                    code_col = None
                    if candidate_code in col_names:
                        code_col = candidate_code
                    elif candidate_code_alt in col_names:
                        code_col = candidate_code_alt

                    if code_col is not None:
                        suffix_key = (col["table_name"], col_name)
                        code_key = (col["table_name"], code_col)
                        if suffix_key not in roles:
                            roles[suffix_key] = "description"
                        if code_key not in roles:
                            roles[code_key] = "code"
                        break

            # Pass 3 — Code suffix columns not yet tagged: if a column ends
            # with "Code" and there's a matching "Name" column, it's a code.
            for col in columns:
                col_name = col["column_name"]
                key = (col["table_name"], col_name)
                if key in roles:
                    continue
                if not col_name.endswith("Code"):
                    continue
                base = col_name[: -len("Code")]
                if not base:
                    continue
                # Check if XName or X exists alongside XCode
                if base + "Name" in col_names or base in col_names:
                    roles[key] = "code"

        return roles

    def _recall_queries_from_clauses(self, clauses: list[dict]) -> list[str]:
        queries = []
        for clause in clauses:
            queries.append(str(clause.get("text") or ""))
            queries.extend(self._coerce_string_list(clause.get("entities", [])))
            for operator in clause.get("operators", []):
                if isinstance(operator, dict):
                    queries.append(str(operator.get("expression") or ""))

        return list(dict.fromkeys(query.strip() for query in queries if query.strip()))

    def _column_recall_text(self, column: dict) -> str:
        return " ".join(
            str(value)
            for value in (
                column.get("column_name"),
                column.get("column_comment"),
                column.get("semantic_column_name"),
                column.get("column_description"),
                column.get("data_format"),
                column.get("value_description"),
                column.get("data_type"),
            )
            if value
        )

    def _recall_similarity(self, query: str, column_text: str) -> float:
        raw_query_tokens = tokenize(query)
        query_tokens = self._expanded_tokens(query)
        column_tokens = self._expanded_tokens(column_text)
        if not query_tokens or not column_tokens:
            return 0.0

        overlap = query_tokens & column_tokens
        if not overlap:
            return 0.0

        if len(raw_query_tokens) == 1:
            precision = len(overlap) / len(column_tokens)
            return min(0.85, 0.85 * precision)

        coverage = len(overlap) / len(query_tokens)
        precision = len(overlap) / len(column_tokens)
        score = (0.75 * coverage) + (0.25 * precision)

        query_norm = normalize_text(query)
        column_norm = normalize_text(column_text)
        if query_norm and query_norm in column_norm:
            score = max(score, 0.9)

        return min(1.0, score)

    def _expanded_tokens(self, value: str) -> set[str]:
        tokens = tokenize(value)
        expanded = set(tokens)
        for token in tokens:
            if len(token) > 3 and token.endswith("s"):
                expanded.add(token[:-1])
        return expanded

    def _merge_recall_hint(
        self,
        profile: dict,
        score: float,
        hint: str,
        matched_query_terms: list[str],
    ) -> None:
        if score > profile["score"]:
            profile["score"] = score
        if hint not in profile["hints"]:
            profile["hints"].append(hint)
        for term in matched_query_terms:
            if term not in profile["matched_query_terms"]:
                profile["matched_query_terms"].append(term)

    def _column_base_name(self, column_name: str) -> str:
        normalized = column_name.lower().replace("_", " ").replace("-", " ")
        for suffix in (" code", " type", " name", " id"):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break
        for suffix in ("code", "type", "name", "id"):
            if normalized.endswith(suffix) and len(normalized) > len(suffix):
                normalized = normalized[: -len(suffix)]
                break
        return "".join(ch for ch in normalized if ch.isalnum())

    @staticmethod
    def _is_role_pair(left_role: str | None, right_role: str | None) -> bool:
        return {left_role, right_role} == {"code", "description"}

    def _question_from_clauses(self, clauses: list[dict]) -> str:
        return " ".join(
            clause["text"]
            for clause in clauses
            if clause.get("source") == "question"
        ) or " ".join(clause["text"] for clause in clauses)

    def _evidence_from_clauses(self, clauses: list[dict]) -> str:
        return " ".join(
            clause["text"]
            for clause in clauses
            if clause.get("source") == "evidence"
        )

    def _coerce_string_list(self, value: object) -> list[str]:
        if not isinstance(value, list):
            return []

        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    def _coerce_confidence(self, value: object, default: float) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return default

        return min(1.0, max(0.0, confidence))

    def _normalize_operator_hints(self, raw_operators: object, source_text: str) -> list[dict]:
        if not isinstance(raw_operators, list):
            return []

        operators = []
        for raw_operator in raw_operators:
            if not isinstance(raw_operator, dict):
                continue

            operator_type = str(raw_operator.get("type") or "").strip()
            expression = str(raw_operator.get("expression") or source_text).strip()
            if not operator_type or not expression:
                continue

            operators.append(
                {
                    "type": operator_type,
                    "expression": expression,
                    "confidence": self._coerce_confidence(
                        raw_operator.get("confidence"),
                        default=0.7,
                    ),
                }
            )

        return operators

    def _matched_clause_ids(self, clauses: list[dict], raw_column: dict) -> list[str]:
        raw_clause_ids = raw_column.get("matched_clause_ids")
        if isinstance(raw_clause_ids, list):
            known_ids = {clause["id"] for clause in clauses}
            matched_ids = [
                str(clause_id)
                for clause_id in raw_clause_ids
                if str(clause_id) in known_ids
            ]
            if matched_ids:
                return matched_ids

        return [clause["id"] for clause in clauses]

    def _select_tables(self, verified_candidates: list[dict]) -> list[str]:
        table_scores: dict[str, float] = {}
        for candidate in verified_candidates:
            table_name = candidate["table_name"]
            table_scores[table_name] = max(table_scores.get(table_name, 0.0), candidate["score"])

        ranked_tables = sorted(table_scores.items(), key=lambda item: item[1], reverse=True)
        return [table_name for table_name, _score in ranked_tables]

    def _build_column_refs(
        self,
        db_id: str,
        verified_candidates: list[dict],
        join_topology: JoinGraph,
    ) -> list[ColumnRef]:
        refs: list[ColumnRef] = []
        seen = set()

        for candidate in verified_candidates:
            key = (candidate["table_name"], candidate["column_name"])
            if key in seen:
                continue
            seen.add(key)
            refs.append(
                ColumnRef(
                    table_name=candidate["table_name"],
                    column_name=candidate["column_name"],
                    data_type=candidate.get("data_type"),
                    comment=candidate.get("column_comment"),
                )
            )

        join_columns = [
            (edge.source_table, edge.source_column)
            for edge in join_topology.edges
        ] + [
            (edge.target_table, edge.target_column)
            for edge in join_topology.edges
        ]
        for table_name, column_name in join_columns:
            key = (table_name, column_name)
            if key in seen:
                continue
            seen.add(key)
            try:
                info = inspect_column_info(db_id, table_name, column_name, split=self.split)
                refs.append(
                    ColumnRef(
                        table_name=table_name,
                        column_name=column_name,
                        data_type=info.get("data_type"),
                        comment=info.get("column_comment"),
                    )
                )
            except LookupError:
                refs.append(
                    ColumnRef(
                        table_name=table_name,
                        column_name=column_name,
                        data_type=None,
                        comment=None,
                    )
                )

        return refs

    def _build_evidence_trace(
        self,
        verified_candidates: list[dict],
        value_mappings: list[ValueMapping],
        join_topology: JoinGraph,
    ) -> list[EvidenceTrace]:
        traces: list[EvidenceTrace] = []

        for candidate in verified_candidates:
            traces.append(
                EvidenceTrace(
                    artifact_type="column",
                    artifact_id=f"{candidate['table_name']}.{candidate['column_name']}",
                    reason=", ".join(candidate.get("reasons", [])),
                    tool_name="request_chat_text",
                    fact={
                        "score": candidate["score"],
                        "matched_clause_ids": candidate.get("matched_clause_ids", []),
                    },
                )
            )

        for mapping in value_mappings:
            traces.append(
                EvidenceTrace(
                    artifact_type="value",
                    artifact_id=f"{mapping.table_name}.{mapping.column_name}",
                    reason=f"{mapping.keyword} -> {mapping.value}",
                    tool_name=mapping.evidence,
                    fact={"confidence": mapping.confidence},
                )
            )

        for edge in join_topology.edges:
            traces.append(
                EvidenceTrace(
                    artifact_type="join",
                    artifact_id=(
                        f"{edge.source_table}.{edge.source_column}->"
                        f"{edge.target_table}.{edge.target_column}"
                    ),
                    reason="topology_closure",
                    tool_name="build_topology_closure",
                    fact={"join_type": edge.join_type},
                )
            )

        return traces

    def _blueprint_confidence(
        self,
        verified_candidates: list[dict],
        value_mappings: list[ValueMapping],
    ) -> float:
        if not verified_candidates:
            return 0.0

        score = sum(candidate["score"] for candidate in verified_candidates) / len(verified_candidates)
        if value_mappings:
            score += min(0.2, 0.05 * len(value_mappings))
        return min(1.0, round(score, 4))
