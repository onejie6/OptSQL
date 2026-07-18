"""Clear selected few-shot records so the resumable runner regenerates them."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.dataset import load_dataset, save_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--question-id", type=int, action="append", required=True)
    args = parser.parse_args()

    requested = set(args.question_id)
    dataset = load_dataset(str(args.snapshot))
    reset = set()
    for item in dataset:
        if int(item.question_id) not in requested:
            continue
        item.few_shot_examples = []
        item.few_shot_preliminary_sql = None
        item.few_shot_preparation_metadata = None
        reset.add(int(item.question_id))

    missing = requested - reset
    if missing:
        raise ValueError(f"Question IDs not found in snapshot: {sorted(missing)}")
    save_dataset(dataset, str(args.snapshot))
    print(f"Reset few-shot items: {sorted(reset)}", flush=True)


if __name__ == "__main__":
    main()
