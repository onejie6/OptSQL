"""Tool: normalize_plan.

Converts SQLite and MySQL raw explain output into a shared PlanIR. The
normalizer keeps raw details as evidence and only fills rows/cost fields when
the DBMS actually exposes them.
"""

from __future__ import annotations

import re
from typing import Any

from agent.explainAnalyserAgent.utils.common import unique_preserve_order
from agent.explainAnalyserAgent.utils.models import NormalizePlanInput
from agent.explainAnalyserAgent.utils.models import NormalizePlanOutput
from agent.explainAnalyserAgent.utils.models import PlanEdge
from agent.explainAnalyserAgent.utils.models import PlanIR
from agent.explainAnalyserAgent.utils.models import PlanNode


def normalize_plan(input_data: NormalizePlanInput) -> NormalizePlanOutput:
    """Normalize DBMS-specific raw plan into PlanIR."""
    if input_data.dbms == "sqlite":
        plan_ir = _normalize_sqlite_plan(input_data.raw_plan)
    elif input_data.dbms == "mysql":
        plan_ir = _normalize_mysql_plan(input_data.raw_plan)
    else:
        raise ValueError(f"Unsupported dbms: {input_data.dbms}")
    return NormalizePlanOutput(plan_ir=plan_ir)


def _normalize_sqlite_plan(raw_plan: Any) -> PlanIR:
    warnings: list[str] = []
    if raw_plan is None:
        return PlanIR(
            dbms="sqlite",
            raw_plan=raw_plan,
            confidence=0.0,
            warnings=["No SQLite raw plan was provided."],
        )
    if not isinstance(raw_plan, list):
        return PlanIR(
            dbms="sqlite",
            raw_plan=raw_plan,
            confidence=0.0,
            warnings=["SQLite raw plan is not a row list."],
        )

    nodes: list[PlanNode] = []
    edges: list[PlanEdge] = []
    global_flags: list[str] = []

    id_to_node_id: dict[int, str] = {}
    for idx, row in enumerate(raw_plan):
        detail = str(_row_get(row, "detail", 3) or "")
        sqlite_id = _to_int(_row_get(row, "id", 0))
        parent_id = _to_int(_row_get(row, "parent", 1))
        node_id = f"sqlite_{sqlite_id if sqlite_id is not None else idx}"
        id_to_node_id[sqlite_id if sqlite_id is not None else idx] = node_id
        operation, flags, table, index = _classify_sqlite_detail(detail)
        global_flags.extend(flags)
        nodes.append(
            PlanNode(
                node_id=node_id,
                parent_id=None,
                operation=operation,
                table=table,
                index=index,
                access_type="full_scan" if "full_table_scan" in flags else None,
                estimated_rows=None,
                estimated_cost=None,
                flags=flags,
                detail=detail,
                extra={"sqlite_id": sqlite_id, "sqlite_parent": parent_id},
            )
        )

    nodes_by_id = {node.node_id: node for node in nodes}
    for node in nodes:
        parent_id = node.extra.get("sqlite_parent")
        if parent_id is None:
            continue
        parent_node_id = id_to_node_id.get(parent_id)
        if parent_node_id and parent_node_id != node.node_id:
            edges.append(
                PlanEdge(
                    source_node_id=parent_node_id,
                    target_node_id=node.node_id,
                    edge_type="parent_child",
                )
            )
            current = nodes_by_id[node.node_id]
            nodes_by_id[node.node_id] = PlanNode(
                **{**current.__dict__, "parent_id": parent_node_id}
            )

    final_nodes = [nodes_by_id[node.node_id] for node in nodes]
    return PlanIR(
        dbms="sqlite",
        nodes=final_nodes,
        edges=edges,
        global_flags=unique_preserve_order(global_flags),
        raw_plan=raw_plan,
        confidence=0.75 if final_nodes else 0.25,
        warnings=warnings,
    )


def _classify_sqlite_detail(detail: str) -> tuple[str, list[str], str | None, str | None]:
    upper = detail.upper()
    flags: list[str] = []
    table = _extract_sqlite_table(detail)
    index = _extract_sqlite_index(detail)

    if "USE TEMP B-TREE FOR ORDER BY" in upper:
        return "temp_sort", ["temp_sort"], None, None
    if "USE TEMP B-TREE FOR GROUP BY" in upper:
        return "temp_group", ["temp_group_by"], None, None
    if "USE TEMP B-TREE FOR DISTINCT" in upper:
        return "distinct", ["temp_distinct"], None, None
    if "CORRELATED" in upper and "SUBQUERY" in upper:
        return "correlated_subquery", ["correlated_subquery"], None, None
    if upper.startswith("MATERIALIZE"):
        return "materialize", ["materialized_subquery"], table, index
    if upper.startswith("CO-ROUTINE"):
        return "subquery", [], table, index
    if upper.startswith("SCAN"):
        flags.append("full_table_scan")
        return "table_scan", flags, table, index
    if upper.startswith("SEARCH"):
        if "USING INTEGER PRIMARY KEY" in upper or "USING PRIMARY KEY" in upper:
            return "index_lookup", flags, table, index or "PRIMARY"
        if "COVERING INDEX" in upper:
            flags.append("covering_index")
            return "covering_index_lookup", flags, table, index
        return "index_lookup" if index else "table_scan", flags, table, index
    return "unknown", flags, table, index


def _extract_sqlite_table(detail: str) -> str | None:
    tokens = detail.split()
    if len(tokens) >= 2 and tokens[0].upper() in {"SCAN", "SEARCH"}:
        return tokens[1]
    if len(tokens) >= 2 and tokens[0].upper() in {"MATERIALIZE", "CO-ROUTINE"}:
        return tokens[1]
    return None


def _extract_sqlite_index(detail: str) -> str | None:
    upper_tokens = [token.upper() for token in detail.split()]
    tokens = detail.split()
    if "INDEX" not in upper_tokens:
        if "PRIMARY" in upper_tokens and "KEY" in upper_tokens:
            return "PRIMARY"
        return None
    index_pos = upper_tokens.index("INDEX")
    if index_pos + 1 < len(tokens):
        return tokens[index_pos + 1]
    return None


def _normalize_mysql_plan(raw_plan: Any) -> PlanIR:
    if raw_plan is None:
        return PlanIR(
            dbms="mysql",
            raw_plan=raw_plan,
            confidence=0.0,
            warnings=["No MySQL raw plan was provided."],
        )

    if _looks_like_mysql_analyze_plan(raw_plan):
        return _normalize_mysql_analyze_plan(raw_plan)
    if isinstance(raw_plan, list):
        return _normalize_mysql_tabular_plan(raw_plan)
    if isinstance(raw_plan, dict):
        return _normalize_mysql_json_plan(raw_plan)
    if isinstance(raw_plan, str):
        return _normalize_mysql_analyze_plan(raw_plan)
    return PlanIR(
        dbms="mysql",
        raw_plan=raw_plan,
        confidence=0.1,
        warnings=["Unsupported MySQL raw plan format."],
    )


def _normalize_mysql_json_plan(raw_plan: dict[str, Any]) -> PlanIR:
    nodes: list[PlanNode] = []
    edges: list[PlanEdge] = []
    global_flags: list[str] = []

    query_block = raw_plan.get("query_block", raw_plan)
    _walk_mysql_json_node(
        query_block,
        parent_id=None,
        nodes=nodes,
        edges=edges,
        global_flags=global_flags,
        counter={"value": 0},
    )
    return PlanIR(
        dbms="mysql",
        nodes=nodes,
        edges=edges,
        global_flags=unique_preserve_order(global_flags),
        raw_plan=raw_plan,
        confidence=0.85 if nodes else 0.3,
        warnings=[],
    )


def _walk_mysql_json_node(
    payload: Any,
    *,
    parent_id: str | None,
    nodes: list[PlanNode],
    edges: list[PlanEdge],
    global_flags: list[str],
    counter: dict[str, int],
) -> None:
    if isinstance(payload, list):
        previous_id: str | None = None
        for item in payload:
            before_count = len(nodes)
            _walk_mysql_json_node(
                item,
                parent_id=parent_id,
                nodes=nodes,
                edges=edges,
                global_flags=global_flags,
                counter=counter,
            )
            if previous_id is not None and len(nodes) > before_count:
                edges.append(
                    PlanEdge(
                        source_node_id=previous_id,
                        target_node_id=nodes[-1].node_id,
                        edge_type="join_input",
                    )
                )
            if nodes:
                previous_id = nodes[-1].node_id
        return

    if not isinstance(payload, dict):
        return

    if "nested_loop" in payload:
        _walk_mysql_json_node(
            payload["nested_loop"],
            parent_id=parent_id,
            nodes=nodes,
            edges=edges,
            global_flags=global_flags,
            counter=counter,
        )

    table_payload = payload.get("table")
    current_id = parent_id
    if isinstance(table_payload, dict):
        counter["value"] += 1
        node_id = f"mysql_{counter['value']}"
        operation, flags = _classify_mysql_access_type(table_payload.get("access_type"))
        if table_payload.get("using_filesort"):
            flags.extend(["filesort", "temp_sort"])
            global_flags.extend(["filesort", "temp_sort"])
        if table_payload.get("using_temporary_table"):
            flags.extend(["temp_table", "temp_group_by"])
            global_flags.extend(["temp_table", "temp_group_by"])
        if _truthy_mysql_flag(table_payload, "using_index") or _truthy_mysql_flag(
            table_payload, "using_index_for_group_by"
        ):
            flags.append("covering_index")
        if _truthy_mysql_flag(table_payload, "using_index_condition"):
            flags.append("index_condition_pushdown")
        if table_payload.get("attached_condition"):
            flags.append("post_filter")
        global_flags.extend(flags)
        cost_info = table_payload.get("cost_info") or {}
        node = PlanNode(
            node_id=node_id,
            parent_id=parent_id,
            operation=operation,
            table=table_payload.get("table_name"),
            index=table_payload.get("key"),
            access_type=table_payload.get("access_type"),
            predicate=table_payload.get("attached_condition"),
            estimated_rows=_to_float(table_payload.get("rows_examined_per_scan")),
            estimated_cost=_to_float(
                cost_info.get("prefix_cost")
                or cost_info.get("query_cost")
                or cost_info.get("read_cost")
            ),
            flags=unique_preserve_order(flags),
            detail=str(table_payload.get("access_type") or ""),
            extra={
                "possible_keys": table_payload.get("possible_keys"),
                "used_key_parts": table_payload.get("used_key_parts"),
                "rows_produced_per_join": table_payload.get("rows_produced_per_join"),
                "filtered": table_payload.get("filtered"),
                "cost_info": cost_info,
            },
        )
        nodes.append(node)
        if parent_id:
            edges.append(
                PlanEdge(
                    source_node_id=parent_id,
                    target_node_id=node_id,
                    edge_type="parent_child",
                )
            )
        current_id = node_id

    for key in ("query_block", "grouping_operation", "ordering_operation", "duplicates_removal"):
        if key in payload:
            _walk_mysql_json_node(
                payload[key],
                parent_id=current_id,
                nodes=nodes,
                edges=edges,
                global_flags=global_flags,
                counter=counter,
            )


def _normalize_mysql_tabular_plan(raw_plan: list[Any]) -> PlanIR:
    nodes: list[PlanNode] = []
    global_flags: list[str] = []
    for idx, row in enumerate(raw_plan):
        if not isinstance(row, dict):
            continue
        access_type = row.get("type")
        operation, flags = _classify_mysql_access_type(access_type)
        extra = str(row.get("Extra") or "")
        extra_lower = extra.lower()
        if "using index condition" in extra_lower:
            flags.append("index_condition_pushdown")
        if "using index" in extra_lower and "using index condition" not in extra_lower:
            flags.append("covering_index")
        if "using where" in extra_lower:
            flags.append("post_filter")
        if "using filesort" in extra_lower:
            flags.extend(["filesort", "temp_sort"])
        if "using temporary" in extra_lower:
            flags.extend(["temp_table", "temp_group_by"])
        global_flags.extend(flags)
        nodes.append(
            PlanNode(
                node_id=f"mysql_{idx + 1}",
                parent_id=None,
                operation=operation,
                table=row.get("table"),
                index=row.get("key"),
                access_type=access_type,
                estimated_rows=_to_float(row.get("rows")),
                estimated_cost=None,
                flags=unique_preserve_order(flags),
                detail=extra,
                extra={
                    "select_type": row.get("select_type"),
                    "possible_keys": row.get("possible_keys"),
                    "filtered": row.get("filtered"),
                    "raw": row,
                },
            )
        )
    return PlanIR(
        dbms="mysql",
        nodes=nodes,
        edges=[],
        global_flags=unique_preserve_order(global_flags),
        raw_plan=raw_plan,
        confidence=0.65 if nodes else 0.2,
        warnings=["MySQL tabular EXPLAIN has less structure than FORMAT=JSON."],
    )


def _classify_mysql_access_type(access_type: Any) -> tuple[str, list[str]]:
    access = str(access_type or "").upper()
    if access == "ALL":
        return "table_scan", ["full_table_scan"]
    if access in {"INDEX", "RANGE"}:
        return "index_scan", []
    if access in {"REF", "EQ_REF", "CONST", "SYSTEM"}:
        return "index_lookup", []
    return "unknown", []


_MYSQL_ACTUAL_RE = re.compile(
    r"actual\s+time\s*=\s*(?P<start>[0-9]+(?:\.[0-9]+)?)(?:\.\.(?P<end>[0-9]+(?:\.[0-9]+)?))?"
    r"\s+rows\s*=\s*(?P<actual_rows>[0-9]+(?:\.[0-9]+)?)\s+loops\s*=\s*(?P<loops>[0-9]+)",
    re.IGNORECASE,
)
_MYSQL_ESTIMATE_RE = re.compile(
    r"cost\s*=\s*(?P<cost_start>[0-9]+(?:\.[0-9]+)?)(?:\.\.(?P<cost_end>[0-9]+(?:\.[0-9]+)?))?"
    r"\s+rows\s*=\s*(?P<estimated_rows>[0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_MYSQL_ON_TABLE_RE = re.compile(r"\bon\s+`?(?P<table>[A-Za-z_][\w$]*)`?", re.IGNORECASE)
_MYSQL_USING_INDEX_RE = re.compile(
    r"\busing\s+(?:covering\s+)?index\s+`?(?P<index>[A-Za-z_][\w$]*)`?",
    re.IGNORECASE,
)


def _looks_like_mysql_analyze_plan(raw_plan: Any) -> bool:
    text = _mysql_plan_text(raw_plan)
    lowered = text.lower()
    return "actual time=" in lowered and "loops=" in lowered


def _normalize_mysql_analyze_plan(raw_plan: Any) -> PlanIR:
    text = _mysql_plan_text(raw_plan)
    if not text:
        return PlanIR(
            dbms="mysql",
            raw_plan=raw_plan,
            confidence=0.1,
            warnings=["MySQL EXPLAIN ANALYZE output was empty or unsupported."],
        )

    nodes: list[PlanNode] = []
    edges: list[PlanEdge] = []
    global_flags: list[str] = []
    stack: list[tuple[int, str]] = []
    for line in text.splitlines():
        if "actual time=" not in line.lower():
            continue
        depth = _mysql_tree_depth(line)
        detail = line.strip()
        if detail.startswith("->"):
            detail = detail[2:].strip()
        node_id = f"mysql_analyze_{len(nodes) + 1}"
        parent_id = _parent_for_depth(stack, depth)
        operation, flags = _classify_mysql_analyze_operation(detail)
        table = _extract_mysql_analyze_table(detail)
        index = _extract_mysql_analyze_index(detail)
        estimate_match = _MYSQL_ESTIMATE_RE.search(detail)
        actual_match = _MYSQL_ACTUAL_RE.search(detail)
        estimated_rows = (
            _to_float(estimate_match.group("estimated_rows")) if estimate_match else None
        )
        estimated_cost = None
        if estimate_match:
            estimated_cost = _to_float(
                estimate_match.group("cost_end") or estimate_match.group("cost_start")
            )
        actual_rows = _to_float(actual_match.group("actual_rows")) if actual_match else None
        actual_time = None
        loops = None
        if actual_match:
            actual_time = _to_float(actual_match.group("end") or actual_match.group("start"))
            loops = _to_int(actual_match.group("loops"))
        global_flags.extend(flags)
        nodes.append(
            PlanNode(
                node_id=node_id,
                parent_id=parent_id,
                operation=operation,
                table=table,
                index=index,
                estimated_rows=estimated_rows,
                actual_rows=actual_rows,
                estimated_cost=estimated_cost,
                actual_time_ms=actual_time,
                loops=loops,
                flags=unique_preserve_order(flags),
                detail=detail,
            )
        )
        if parent_id:
            edges.append(
                PlanEdge(
                    source_node_id=parent_id,
                    target_node_id=node_id,
                    edge_type="parent_child",
                )
            )
        while stack and stack[-1][0] >= depth:
            stack.pop()
        stack.append((depth, node_id))

    return PlanIR(
        dbms="mysql",
        nodes=nodes,
        edges=edges,
        global_flags=unique_preserve_order(global_flags),
        raw_plan=raw_plan,
        confidence=0.8 if nodes else 0.2,
        warnings=["MySQL EXPLAIN ANALYZE executes the query."],
    )


def _mysql_plan_text(raw_plan: Any) -> str:
    if isinstance(raw_plan, str):
        return raw_plan
    if not isinstance(raw_plan, list):
        return ""
    fragments: list[str] = []
    for row in raw_plan:
        if isinstance(row, dict):
            values = list(row.values())
            if len(values) == 1:
                fragments.append(str(values[0]))
                continue
            for key, value in row.items():
                if "explain" in str(key).lower() or isinstance(value, str):
                    fragments.append(str(value))
                    break
        else:
            fragments.append(str(row))
    return "\n".join(fragments)


def _mysql_tree_depth(line: str) -> int:
    arrow_pos = line.find("->")
    if arrow_pos >= 0:
        return arrow_pos
    return len(line) - len(line.lstrip())


def _parent_for_depth(stack: list[tuple[int, str]], depth: int) -> str | None:
    parent_id: str | None = None
    for candidate_depth, candidate_id in stack:
        if candidate_depth < depth:
            parent_id = candidate_id
    return parent_id


def _classify_mysql_analyze_operation(detail: str) -> tuple[str, list[str]]:
    lowered = detail.lower()
    flags: list[str] = []
    if "covering index" in lowered:
        flags.append("covering_index")
    elif "using index" in lowered and "index condition" not in lowered:
        flags.append("covering_index")
    if "index condition" in lowered:
        flags.append("index_condition_pushdown")
    if "filter:" in lowered or "where" in lowered:
        flags.append("post_filter")
    if "temporary" in lowered or "materialize" in lowered:
        flags.append("temp_table")
    if "filesort" in lowered or "sort:" in lowered:
        flags.extend(["filesort", "temp_sort"])
    if "table scan" in lowered:
        return "table_scan", flags + ["full_table_scan"]
    if "index lookup" in lowered or "single-row index lookup" in lowered:
        return "index_lookup", flags
    if "index range scan" in lowered or "index scan" in lowered:
        return "index_scan", flags
    if "nested loop" in lowered:
        return "nested_loop", flags
    if "sort" in lowered:
        return "temp_sort", flags
    if "filter" in lowered:
        return "filter", flags
    return "unknown", flags


def _extract_mysql_analyze_table(detail: str) -> str | None:
    match = _MYSQL_ON_TABLE_RE.search(detail)
    return match.group("table") if match else None


def _extract_mysql_analyze_index(detail: str) -> str | None:
    match = _MYSQL_USING_INDEX_RE.search(detail)
    return match.group("index") if match else None


def _truthy_mysql_flag(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    if isinstance(value, str):
        return value.lower() in {"true", "yes", "1"}
    return bool(value)


def _row_get(row: Any, key: str, index: int) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[index]
    except Exception:
        return None


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None
