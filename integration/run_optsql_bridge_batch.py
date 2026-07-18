"""Run the OptSQL generation -> OptSQL bridge over a snapshot with JSONL checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from optsql_bridge import iter_snapshot_items, run_controller


def _completed_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "success" and record.get("question_id") is not None:
            completed.add(int(record["question_id"]))
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--optsql-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "results.jsonl"
    predictions_path = args.output_dir / "predictions.json"
    completed = _completed_ids(checkpoint_path)
    predictions = (
        json.loads(predictions_path.read_text(encoding="utf-8"))
        if predictions_path.exists()
        else {}
    )

    processed = 0
    for item in iter_snapshot_items(args.snapshot):
        question_id = int(item["question_id"])
        if question_id in completed:
            continue
        if args.max_items is not None and processed >= args.max_items:
            break

        started = time.monotonic()
        try:
            result = run_controller(item, args.optsql_root)
            result.update(
                status="success",
                elapsed_seconds=round(time.monotonic() - started, 3),
            )
            predictions[str(question_id)] = result["final_sql"]
        except Exception as exc:
            result = {
                "question_id": question_id,
                "db_id": item.get("database_id"),
                "status": "error",
                "error": str(exc),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }

        with checkpoint_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
        predictions_path.write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        processed += 1
        print(
            json.dumps(
                {
                    "processed_this_run": processed,
                    "question_id": question_id,
                    "status": result["status"],
                    "elapsed_seconds": result["elapsed_seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
