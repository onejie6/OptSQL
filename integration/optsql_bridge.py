"""Bridge a completed OptSQL generation item into the OptSQL meta-controller.

This module intentionally parses OptSQL generation snapshots as plain JSON.  OptSQL generation and
OptSQL both expose a top-level ``app`` package, so keeping the bridge process
free of OptSQL generation imports prevents ambiguous module resolution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


def _load_optsql(optsql_root: Path) -> dict[str, Any]:
    root = str(optsql_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)

    from agent.controller import MetaCognitiveController
    from agent.validator import ValidatorAgent
    from myTypes import (
        AgentRequest,
        AgentTask,
        AttemptRecord,
        ColumnRef,
        EvidenceTrace,
        JoinEdge,
        JoinGraph,
        PredicateHint,
        ReflectionMemory,
        SQLVersion,
        ValueMapping,
        VerifiedContextBlueprint,
    )
    from utils.db import register_database_path
    from utils.schema_grounding import (
        build_sqlite_metadata_from_ddl,
        register_database_metadata,
    )

    return locals()


def iter_snapshot_items(snapshot_path: Path) -> Iterable[dict[str, Any]]:
    """Yield item records from a OptSQL generation .snapshot manifest or JSONL file."""
    snapshot_path = snapshot_path.resolve()
    if snapshot_path.suffix == ".jsonl":
        items_path = snapshot_path
    else:
        manifest = json.loads(snapshot_path.read_text(encoding="utf-8"))
        items_file = manifest.get("items_file", "items.jsonl")
        items_path = snapshot_path.with_name(f"{snapshot_path.name}.data") / items_file

    with items_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield _flatten_snapshot_record(json.loads(line))


def _flatten_snapshot_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize both legacy flat items and structured snapshot records."""
    if "input" not in record and "pipeline_artifacts" not in record:
        return record
    flattened = dict(record.get("input") or {})
    for stage_artifacts in (record.get("pipeline_artifacts") or {}).values():
        if isinstance(stage_artifacts, dict):
            flattened.update(stage_artifacts)
    return flattened


def select_snapshot_item(
    snapshot_path: Path,
    *,
    question_id: int | None = None,
    position: int = 1,
) -> dict[str, Any]:
    if position < 1:
        raise ValueError("position must be >= 1")
    for current_position, item in enumerate(iter_snapshot_items(snapshot_path), 1):
        if question_id is not None and item.get("question_id") == question_id:
            return item
        if question_id is None and current_position == position:
            return item
    selector = f"question_id={question_id}" if question_id is not None else f"position={position}"
    raise LookupError(f"No OptSQL generation snapshot item matched {selector}")


def build_bridge_artifacts(item: dict[str, Any], optsql_root: Path) -> dict[str, Any]:
    """Map one completed OptSQL generation item to OptSQL's typed controller artifacts."""
    types = _load_optsql(optsql_root)
    final_sql = (item.get("final_selected_sql") or "").strip()
    if not final_sql:
        raise ValueError("OptSQL generation item has no final_selected_sql; finish SQL selection first")

    db_id = str(item["database_id"])
    db_path = Path(item["database_path"]).expanduser()
    if not db_path.is_absolute():
        db_path = (Path.cwd() / db_path).resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"OptSQL generation database does not exist: {db_path}")

    linked = item.get("final_linked_tables_and_columns") or {}
    schema = item.get("database_schema_after_schema_linking") or item.get("database_schema") or {}
    schema_tables = schema.get("tables") or {}
    selected_tables = list(linked) if linked else list(schema_tables)
    selected_columns = []
    for table_name in selected_tables:
        table = schema_tables.get(table_name) or {}
        columns = table.get("columns") or {}
        selected_names = linked.get(table_name) if linked else list(columns)
        for column_name in selected_names or []:
            column = columns.get(column_name) or {}
            selected_columns.append(
                types["ColumnRef"](
                    table_name=table_name,
                    column_name=column_name,
                    data_type=column.get("column_type"),
                    comment=column.get("description") or None,
                )
            )

    edges = []
    selected_table_set = set(selected_tables)
    for table_name in selected_tables:
        columns = (schema_tables.get(table_name) or {}).get("columns") or {}
        for column_name, column in columns.items():
            for target in column.get("foreign_keys") or []:
                if len(target) >= 2 and target[0] in selected_table_set:
                    edges.append(
                        types["JoinEdge"](
                            source_table=table_name,
                            source_column=column_name,
                            target_table=str(target[0]),
                            target_column=str(target[1]),
                            join_type="foreign_key",
                        )
                    )

    value_mappings = []
    for table_name, columns in (item.get("retrieved_values") or {}).items():
        for column_name, values in (columns or {}).items():
            for value in values or []:
                if isinstance(value, dict):
                    grounded_value = value.get("value")
                    distance = value.get("distance")
                    confidence = (
                        max(0.0, min(1.0, 1.0 - float(distance)))
                        if distance is not None
                        else 1.0
                    )
                else:
                    grounded_value = value
                    confidence = 1.0
                value_mappings.append(
                    types["ValueMapping"](
                        keyword=str(grounded_value),
                        table_name=table_name,
                        column_name=column_name,
                        value=grounded_value,
                        confidence=confidence,
                        evidence="OptSQL generation value retrieval",
                    )
                )

    evidence = item.get("evidence") or None
    predicate_hints = (
        [
            types["PredicateHint"](
                predicate_type="bird_evidence",
                expression=str(evidence),
                source_text=str(evidence),
                confidence=1.0,
            )
        ]
        if evidence
        else []
    )
    blueprint = types["VerifiedContextBlueprint"](
        db_id=db_id,
        selected_tables=selected_tables,
        selected_columns=selected_columns,
        value_mappings=value_mappings,
        join_topology=types["JoinGraph"](tables=selected_tables, edges=edges),
        predicate_hints=predicate_hints,
        evidence_trace=[
            types["EvidenceTrace"](
                artifact_type="base_snapshot",
                artifact_id=f"qid-{item.get('question_id')}",
                reason="OptSQL generation schema linking and grounding output",
                tool_name="OptSQL",
                fact={
                    "linked_tables": len(selected_tables),
                    "linked_columns": len(selected_columns),
                    "grounded_values": len(value_mappings),
                },
            )
        ],
        confidence=1.0 if linked else 0.5,
    )

    version_id = "optsql_generation-" + hashlib.sha256(final_sql.encode("utf-8")).hexdigest()[:12]
    sql_version = types["SQLVersion"](
        version_id=version_id,
        parent_id=None,
        sql=final_sql,
        source_agent="optsql_base_selection",
        rewrite_rule_ids=[],
        explanation="OptSQL generation final selected SQL; OptSQL optimization baseline",
        created_at=datetime.now(UTC).isoformat(),
    )

    attempts = []
    history_groups = (
        ("optsql_generation", item.get("sql_candidates") or []),
        ("base_revision", item.get("sql_candidates_after_revision") or []),
    )
    for stage, candidates in history_groups:
        for index, candidate in enumerate(candidates, 1):
            digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:12]
            attempts.append(
                types["AttemptRecord"](
                    attempt_id=f"{stage}-{index}-{digest}",
                    sql_version_id=version_id if candidate.strip() == final_sql else None,
                    stage=stage,
                    status="selected" if candidate.strip() == final_sql else "candidate",
                    failure_reason=None,
                    notes=candidate,
                )
            )

    task = types["AgentTask"](
        task_id=f"optsql_generation-qid-{item.get('question_id')}",
        question_id=item.get("question_id"),
        db_id=db_id,
        question=str(item.get("question") or ""),
        evidence=evidence,
        dbms="sqlite",
        user_constraints={"db_path": str(db_path), "source": "base_snapshot"},
    )
    return {
        "task": task,
        "blueprint": blueprint,
        "sql_version": sql_version,
        "reflection_memory": types["ReflectionMemory"](
            attempts=attempts,
            failed_assumptions=[],
            confirmed_facts=[
                f"OptSQL generation linked {len(selected_tables)} tables and {len(selected_columns)} columns",
                f"OptSQL generation grounded {len(value_mappings)} values",
            ],
            rejected_rewrite_rules=[],
            next_strategy_hint="Treat OptSQL generation final SQL as immutable semantic baseline; optimize only after validation.",
        ),
        "db_path": db_path,
        "types": types,
    }


def run_controller(item: dict[str, Any], optsql_root: Path) -> dict[str, Any]:
    artifacts = build_bridge_artifacts(item, optsql_root)
    types = artifacts["types"]
    db_id = artifacts["task"].db_id
    db_path = artifacts["db_path"]
    metadata, description_metadata = types["build_sqlite_metadata_from_ddl"](db_id, db_path)
    types["register_database_path"](db_id, db_path)
    types["register_database_metadata"](db_id, metadata, description_metadata)

    controller = types["MetaCognitiveController"]()
    metrics = types["ValidatorAgent"]().validate_syntax(
        artifacts["sql_version"], db_id, "sqlite"
    )
    state = controller.initialize_state(artifacts["task"])
    state = replace(
        state,
        strategy="planning_plus_optimization",
        blueprint=artifacts["blueprint"],
        sql_versions=[artifacts["sql_version"]],
        best_sql_version_id=artifacts["sql_version"].version_id,
        execution_metrics={artifacts["sql_version"].version_id: metrics},
        reflection_memory=artifacts["reflection_memory"],
        status="planning_complete",
    )
    response = controller.run(
        types["AgentRequest"](
            request_id=f"bridge-{artifacts['task'].task_id}",
            task=artifacts["task"],
            runtime_state={"runtime_state": state},
            input_artifacts={},
            constraints={"semantic_baseline": "optsql_generation_final_selected_sql"},
        )
    )
    if response.status != "success":
        raise RuntimeError("OptSQL controller failed: " + "; ".join(response.errors))
    final_state = response.output_artifacts["runtime_state"]
    final_answer = response.output_artifacts["final_answer"]
    return {
        "question_id": item.get("question_id"),
        "db_id": db_id,
        "base_sql": artifacts["sql_version"].sql,
        "base_execution_metrics": asdict(metrics),
        "plan_decision": response.output_artifacts["plan_decision"],
        "final_status": final_state.status,
        "final_sql": final_answer.sql,
        "final_answer": asdict(final_answer),
        "working_memory_attempts": len(final_state.reflection_memory.attempts),
    }


def describe_mapping(item: dict[str, Any], optsql_root: Path) -> dict[str, Any]:
    artifacts = build_bridge_artifacts(item, optsql_root)
    return {
        "question_id": item.get("question_id"),
        "db_id": artifacts["task"].db_id,
        "mapping": {
            "final_sql_to_sql_version": artifacts["sql_version"].version_id,
            "linked_schema_to_blueprint": {
                "tables": len(artifacts["blueprint"].selected_tables),
                "columns": len(artifacts["blueprint"].selected_columns),
                "join_edges": len(artifacts["blueprint"].join_topology.edges),
            },
            "grounded_values_to_bindings": len(artifacts["blueprint"].value_mappings),
            "candidate_history_to_working_memory": len(artifacts["reflection_memory"].attempts),
            "execution_result_to_metrics": "computed immediately before Controller",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--optsql-root", type=Path, required=True)
    parser.add_argument("--question-id", type=int)
    parser.add_argument("--position", type=int, default=1)
    parser.add_argument("--run-controller", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    item = select_snapshot_item(
        args.snapshot, question_id=args.question_id, position=args.position
    )
    result = (
        run_controller(item, args.optsql_root)
        if args.run_controller
        else describe_mapping(item, args.optsql_root)
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
