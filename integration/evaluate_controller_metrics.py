"""Evaluate OptSQL generation base and OptSQL controller outputs on BIRD-dev.

This evaluator reports the paper metrics (EX, VES, CR, AR@0.8 and AR@1)
plus controller routing and acceptance statistics. It uses SQLite execution
results for correctness, repeated runtimes for VES, and OptSQL's SQLite plan
cost estimator for CR/AR.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any


def _normalize_rows(rows: list[tuple[Any, ...]]) -> set[tuple[Any, ...]]:
    return set(rows)


def _execute(db_path: str, sql: str, timeout: float) -> tuple[list[tuple[Any, ...]] | None, float | None, str | None]:
    import sqlite3

    started = time.perf_counter()
    connection = sqlite3.connect(db_path, timeout=timeout)
    deadline = started + timeout
    connection.set_progress_handler(
        lambda: 1 if time.perf_counter() >= deadline else 0,
        10_000,
    )
    try:
        rows = connection.execute(sql).fetchall()
        return rows, time.perf_counter() - started, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"
    finally:
        connection.close()


def _table_counts(connection: Any) -> dict[str, int]:
    names = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    counts: dict[str, int] = {}
    for name in names:
        escaped = name.replace('"', '""')
        try:
            counts[name.lower()] = int(
                connection.execute(f'SELECT COUNT(*) FROM "{escaped}"').fetchone()[0]
            )
        except Exception:
            continue
    return counts


def _plan_cost(db_path: str, sql: str, timeout: float) -> tuple[int | None, str | None]:
    import re
    import sqlite3
    import sqlglot
    from sqlglot import exp

    connection = sqlite3.connect(db_path, timeout=timeout)
    deadline = time.perf_counter() + timeout
    connection.set_progress_handler(
        lambda: 1 if time.perf_counter() >= deadline else 0,
        10_000,
    )
    try:
        counts = _table_counts(connection)
        aliases: dict[str, str] = {}
        try:
            ast = sqlglot.parse_one(sql, dialect="sqlite")
            for table in ast.find_all(exp.Table):
                if not table.name:
                    continue
                aliases[table.name.lower()] = table.name.lower()
                if table.alias:
                    aliases[table.alias.lower()] = table.name.lower()
        except Exception:
            pass
        plan = connection.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
        total = 0
        found = False
        for row in plan:
            detail = str(row[-1])
            match = re.search(r"\b(?:SCAN|SEARCH)\s+(?:TABLE\s+)?[`\"\[]?([^`\"\]\s]+)", detail, re.I)
            if not match:
                continue
            table = match.group(1).lower()
            if table in {"subquery", "constant", "temp"}:
                continue
            table = aliases.get(table, table)
            if table in counts:
                total += counts[table]
                found = True
        return (total if found else 0), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        connection.close()


def _evaluate_one(payload: dict[str, Any]) -> dict[str, Any]:
    db_path = payload["db_path"]
    timeout = payload["timeout"]
    repeats = payload["repeats"]
    gold_rows, _, gold_error = _execute(db_path, payload["gold_sql"], timeout)
    record: dict[str, Any] = {
        "question_id": payload["question_id"],
        "db_id": payload["db_id"],
        "difficulty": payload["difficulty"],
        "gold_error": gold_error,
    }
    gold_cost, gold_plan_error = _plan_cost(db_path, payload["gold_sql"], timeout)
    record["gold_cost"] = gold_cost
    record["gold_plan_error"] = gold_plan_error

    for label in ("base", "final"):
        sql = payload[f"{label}_sql"]
        rows, _, error = _execute(db_path, sql, timeout)
        correct = gold_rows is not None and rows is not None and _normalize_rows(rows) == _normalize_rows(gold_rows)
        runtimes: list[float] = []
        gold_runtimes: list[float] = []
        if correct:
            for _ in range(repeats):
                _, pred_time, pred_error = _execute(db_path, sql, timeout)
                _, gold_time, timing_gold_error = _execute(db_path, payload["gold_sql"], timeout)
                if pred_error or timing_gold_error or pred_time is None or gold_time is None:
                    continue
                runtimes.append(pred_time)
                gold_runtimes.append(gold_time)
        cost, plan_error = _plan_cost(db_path, sql, timeout)
        ratio = None
        if correct and runtimes and gold_runtimes:
            pred_median = statistics.median(runtimes)
            gold_median = statistics.median(gold_runtimes)
            if pred_median > 0:
                ratio = gold_median / pred_median
        cost_ratio = None
        if correct and cost is not None and gold_cost is not None:
            if cost == 0:
                cost_ratio = 1.0 if gold_cost == 0 else float("inf")
            else:
                cost_ratio = gold_cost / cost
        record[label] = {
            "correct": correct,
            "error": error,
            "runtime_ratio": ratio,
            "cost": cost,
            "cost_ratio": cost_ratio,
            "plan_error": plan_error,
        }
    return record


def _summary(details: list[dict[str, Any]], label: str) -> dict[str, Any]:
    valid_gold = [row for row in details if row["gold_error"] is None]
    correct = [row for row in valid_gold if row[label]["correct"]]
    ves_components = [
        math.sqrt(row[label]["runtime_ratio"])
        if row[label]["correct"] and row[label]["runtime_ratio"] is not None
        else 0.0
        for row in valid_gold
    ]
    cost_rows = [row for row in correct if row[label]["cost_ratio"] is not None]
    return {
        "denominator": len(valid_gold),
        "gold_errors": len(details) - len(valid_gold),
        "correct": len(correct),
        "ex_percent": 100 * len(correct) / len(valid_gold),
        "ves_percent": 100 * sum(ves_components) / len(valid_gold),
        "cr": (
            sum(math.sqrt(row[label]["cost_ratio"]) for row in cost_rows) / len(cost_rows)
            if cost_rows else None
        ),
        "ar_at_0_8_percent": 100 * sum(row[label]["cost_ratio"] >= 0.8 for row in cost_rows) / len(valid_gold),
        "ar_at_1_percent": 100 * sum(row[label]["cost_ratio"] >= 1.0 for row in cost_rows) / len(valid_gold),
        "sql_errors": sum(row[label]["error"] is not None for row in valid_gold),
        "plan_errors": sum(row[label]["plan_error"] is not None for row in valid_gold),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--controller-results", type=Path, required=True)
    parser.add_argument("--optsql-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.snapshot.parents[3] / "OptSQL"))
    from app.dataset import load_dataset

    dataset = load_dataset(str(args.snapshot))
    controller_rows = {
        int(row["question_id"]): row
        for row in (
            json.loads(line)
            for line in args.controller_results.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    payloads = []
    for item in dataset:
        controller = controller_rows[int(item.question_id)]
        payloads.append({
            "question_id": int(item.question_id),
            "db_id": item.database_id,
            "db_path": item.database_path,
            "difficulty": item.difficulty,
            "gold_sql": item.gold_sql,
            "base_sql": controller["base_sql"],
            "final_sql": controller["final_sql"],
            "timeout": args.timeout,
            "repeats": args.repeats,
        })

    details: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_evaluate_one, payload) for payload in payloads]
        for index, future in enumerate(as_completed(futures), 1):
            details.append(future.result())
            if index % 100 == 0:
                print(f"evaluated {index}/{len(payloads)}", flush=True)
    details.sort(key=lambda row: row["question_id"])

    statuses: dict[str, int] = {}
    plans: dict[str, int] = {}
    changed = 0
    for row in controller_rows.values():
        statuses[row["final_status"]] = statuses.get(row["final_status"], 0) + 1
        plan = row["plan_decision"]["plan"]
        plans[plan] = plans.get(plan, 0) + 1
        changed += row["base_sql"].strip() != row["final_sql"].strip()

    base_correct = {row["question_id"] for row in details if row["base"]["correct"]}
    final_correct = {row["question_id"] for row in details if row["final"]["correct"]}
    output = {
        "base": _summary(details, "base"),
        "final": _summary(details, "final"),
        "controller": {
            "total": len(controller_rows),
            "plan_counts": plans,
            "status_counts": statuses,
            "changed_sql": changed,
            "trigger_rate_percent": 100 * plans.get("planning_plus_optimization", 0) / len(controller_rows),
            "accepted_rewrite_rate_among_triggered_percent": 100 * changed / max(1, plans.get("planning_plus_optimization", 0)),
            "correctness_fixed": len(final_correct - base_correct),
            "correctness_broken": len(base_correct - final_correct),
        },
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "details"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
