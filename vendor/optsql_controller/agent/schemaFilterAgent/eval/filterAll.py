"""Evaluate EvidenceGuidedSchemaFilterAgent on all BIRD tasks.

Iterates over every task in a BIRD split, runs schema filtering, persists
per-task blueprints and metrics to disk cache, and writes an aggregate
summary.

Run from the repository root:
    python -m agent.schemaFilterAgent.eval.filterAll dev
    python agent/schemaFilterAgent/eval/filterAll.py dev
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Ensure the project root is on sys.path so both direct execution and
# module invocation resolve absolute imports correctly.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent.schemaFilterAgent import EvidenceGuidedSchemaFilterAgent
from agent.schemaFilterAgent.eval import (
    average_fpr,
    calculate_fpr,
    calculate_slr,
    extract_ground_truth_schema_columns,
    normalize_schema_columns,
)
from myTypes import AgentRequest
from myTypes import AgentTask
from utils.tasks import load_bird_tasks

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
CACHE_DIR = RESULTS_DIR / "cache"
SUMMARY_PATH = RESULTS_DIR / "summary.txt"


def _check_prerequisites() -> None:
    missing = []
    if not os.getenv("DS_API_KEY"):
        missing.append("  DS_API_KEY environment variable")
    if not os.getenv("DS_BASE_URL"):
        missing.append("  DS_BASE_URL environment variable")

    try:
        import openai  # noqa: F401
    except ImportError:
        missing.append("  openai SDK (pip install openai)")

    try:
        import langgraph  # noqa: F401
    except ImportError:
        missing.append("  langgraph (pip install langgraph)")

    if missing:
        print("Missing prerequisites:", file=sys.stderr)
        for item in missing:
            print(item, file=sys.stderr)
        sys.exit(1)


def _cache_path(split: str, question_id: int) -> Path:
    return CACHE_DIR / split / f"q{question_id}.json"


def _load_cache(split: str, question_id: int) -> dict | None:
    path = _cache_path(split, question_id)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "status" not in data:
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def _save_cache(split: str, question_id: int, data: dict) -> None:
    path = _cache_path(split, question_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _build_request(task, split: str, array_index: int) -> AgentRequest:
    return AgentRequest(
        request_id=f"schema-filter-{split}-qid-{task.question_id}",
        task=AgentTask(
            task_id=f"{split}-index-{array_index}-qid-{task.question_id}",
            question_id=task.question_id,
            db_id=task.db_id,
            question=task.question,
            evidence=task.evidence,
            dbms="sqlite",
            user_constraints={},
        ),
        runtime_state={},
        input_artifacts={},
        constraints={},
    )


def main() -> None:
    _check_prerequisites()

    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    tasks = load_bird_tasks(split)
    total = len(tasks)

    print(f"Split: {split}  Total tasks: {total}", file=sys.stderr)
    print(f"Cache dir: {CACHE_DIR / split}", file=sys.stderr)
    print(f"Summary:   {SUMMARY_PATH}", file=sys.stderr)
    print(file=sys.stderr)

    agent = EvidenceGuidedSchemaFilterAgent(split=split)
    results: list[dict] = []

    for index, task in enumerate(tasks):
        qid = task.question_id
        label = f"[{index + 1:>{len(str(total))}}/{total}] qid={qid:<4} {task.db_id}"
        print(f"{label} ...", file=sys.stderr, end=" ", flush=True)

        cached = _load_cache(split, qid)
        if cached is not None:
            results.append(cached)
            if cached["status"] == "success":
                print(f"cached  fpr={cached.get('fpr', 0):.3f}", file=sys.stderr)
            else:
                print(f"cached  FAIL: {cached.get('error', '?')}", file=sys.stderr)
            continue

        start = time.monotonic()
        request = _build_request(task, split, index)

        try:
            response = agent.run(request)
        except Exception as exc:
            elapsed = time.monotonic() - start
            error_data = _error_cache_entry(task, str(exc), elapsed)
            _save_cache(split, qid, error_data)
            results.append(error_data)
            print(f"FAIL: {exc}", file=sys.stderr)
            continue

        elapsed = time.monotonic() - start
        gt_columns = extract_ground_truth_schema_columns(task, split)

        if response.status == "success":
            blueprint = response.output_artifacts.get("blueprint")
            retrieved = normalize_schema_columns(blueprint) if blueprint else set()
            fpr = calculate_fpr(blueprint, gt_columns) if blueprint else 0.0

            cache_data = {
                "question_id": qid,
                "db_id": task.db_id,
                "status": "success",
                "error": None,
                "ground_truth_columns": sorted([list(c) for c in gt_columns]),
                "retrieved_columns": sorted([list(c) for c in retrieved]),
                "ground_truth_count": len(gt_columns),
                "retrieved_count": len(retrieved),
                "fpr": round(fpr, 4),
                "elapsed_seconds": round(elapsed, 1),
                "blueprint": asdict(blueprint) if blueprint else None,
            }
            _save_cache(split, qid, cache_data)
            results.append(cache_data)
            print(f"fpr={fpr:.3f}  ({elapsed:.1f}s)", file=sys.stderr)
        else:
            error_msg = "; ".join(response.errors) if response.errors else "Unknown error"
            error_data = _error_cache_entry(task, error_msg, elapsed, gt_columns)
            _save_cache(split, qid, error_data)
            results.append(error_data)
            print(f"FAIL: {error_msg}", file=sys.stderr)

    _write_summary(split, results)
    _print_summary_stderr(results)


def _error_cache_entry(
    task,
    error: str,
    elapsed: float,
    gt_columns: set | None = None,
) -> dict:
    gt_list = sorted([list(c) for c in gt_columns]) if gt_columns else []
    return {
        "question_id": task.question_id,
        "db_id": task.db_id,
        "status": "error",
        "error": error,
        "ground_truth_columns": gt_list,
        "retrieved_columns": [],
        "ground_truth_count": len(gt_list),
        "retrieved_count": 0,
        "fpr": None,
        "elapsed_seconds": round(elapsed, 1),
        "blueprint": None,
    }


def _write_summary(split: str, results: list[dict]) -> None:
    success = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] == "error"]

    fpr_pairs = []
    for r in success:
        retrieved_set = {tuple(c) for c in r.get("retrieved_columns", [])}
        gt_set = {tuple(c) for c in r.get("ground_truth_columns", [])}
        fpr_pairs.append((retrieved_set, gt_set))

    avg_fpr = average_fpr(fpr_pairs) if fpr_pairs else 0.0
    slr = calculate_slr(fpr_pairs) if fpr_pairs else 0.0

    db_width = max(25, max((len(r["db_id"]) for r in results), default=25))

    lines = [
        "Schema Filter Evaluation Summary",
        "=" * 78,
        "",
        f"Split:       {split}",
        f"Date:        {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total:       {len(results)}",
        f"Successful:  {len(success)}",
        f"Errors:      {len(errors)}",
        "",
        "Overall Metrics",
        "-" * 78,
        f"  Average FPR:  {avg_fpr:.4f}",
        f"  SLR:          {slr:.4f}",
        "",
        "Per-Task Breakdown",
        "-" * 78,
    ]

    header = (
        f"  {'qid':>5s}  {('db_id').ljust(db_width)}  "
        f"{'FPR':>8s}  {'|Ret|':>6s}  {'|GT|':>6s}  {'Elapsed':>8s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for r in sorted(results, key=lambda r: r["question_id"]):
        if r["status"] == "success":
            fpr_str = f"{r['fpr']:.4f}"
            elapsed_str = f"{r['elapsed_seconds']:.1f}s"
            lines.append(
                f"  {r['question_id']:5d}  {r['db_id'].ljust(db_width)}  "
                f"{fpr_str:>8s}  {r['retrieved_count']:6d}  "
                f"{r['ground_truth_count']:6d}  {elapsed_str:>8s}"
            )
        else:
            elapsed_str = f"{r['elapsed_seconds']:.1f}s"
            lines.append(
                f"  {r['question_id']:5d}  {r['db_id'].ljust(db_width)}  "
                f"{'error':>8s}  {'-':>6s}  {r['ground_truth_count']:6d}  "
                f"{elapsed_str:>8s}"
            )

    if errors:
        lines.append("")
        lines.append("Failed Tasks")
        lines.append("-" * 78)
        lines.append(f"  {'qid':>5s}  {('db_id').ljust(db_width)}  Error")
        lines.append("  " + "-" * (db_width + 25))
        for r in sorted(errors, key=lambda r: r["question_id"]):
            lines.append(
                f"  {r['question_id']:5d}  {r['db_id'].ljust(db_width)}  "
                f"{r.get('error', '?')}"
            )

    lines.append("")
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def _print_summary_stderr(results: list[dict]) -> None:
    success = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] == "error"]

    fpr_pairs = []
    for r in success:
        retrieved_set = {tuple(c) for c in r.get("retrieved_columns", [])}
        gt_set = {tuple(c) for c in r.get("ground_truth_columns", [])}
        fpr_pairs.append((retrieved_set, gt_set))

    avg_fpr = average_fpr(fpr_pairs) if fpr_pairs else 0.0
    slr = calculate_slr(fpr_pairs) if fpr_pairs else 0.0

    print(file=sys.stderr)
    print(f"Done.  {len(success)}/{len(results)} success, {len(errors)} errors", file=sys.stderr)
    print(f"Average FPR: {avg_fpr:.4f}", file=sys.stderr)
    print(f"SLR:         {slr:.4f}", file=sys.stderr)
    print(f"Summary:     {SUMMARY_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
