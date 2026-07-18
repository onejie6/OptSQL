"""Validate BIRD prediction completeness without using ground-truth SQL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()

    records = json.loads(args.test_json.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    expected_ids = {
        str(record.get("question_id", position))
        for position, record in enumerate(records)
    }
    actual_ids = set(predictions)
    missing = sorted(expected_ids - actual_ids, key=int)
    empty = sorted(
        (question_id for question_id in expected_ids if not str(predictions.get(question_id, "")).strip()),
        key=int,
    )
    extra = sorted(actual_ids - expected_ids, key=int)
    report = {
        "expected": len(expected_ids),
        "predicted": len(actual_ids),
        "missing": len(missing),
        "empty": len(empty),
        "extra": len(extra),
    }
    print(json.dumps(report, ensure_ascii=False))
    if missing or empty or extra:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
