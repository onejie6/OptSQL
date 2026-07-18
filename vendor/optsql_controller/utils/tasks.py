import json
from pathlib import Path

from config import BIRD_BASE
from myTypes import BirdTask


def get_bird_tasks_path(split: str) -> Path:
    """Return the JSON file path for the given BIRD split."""
    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a non-empty string.")

    normalized_split = split.strip().lower()
    if normalized_split not in {"dev", "train"}:
        raise ValueError("split must be one of: dev, train.")

    split_dir_map = {
        "dev": "dev_20240627",
        "train": "train_20240627",
    }
    tasks_path = Path(BIRD_BASE) / split_dir_map[normalized_split] / f"{normalized_split}.json"

    if not tasks_path.is_file():
        raise FileNotFoundError(
            f"BIRD tasks file for split '{normalized_split}' does not exist: {tasks_path}"
        )

    return tasks_path


def load_bird_tasks(split: str) -> list[BirdTask]:
    """Load all tasks for the given BIRD split."""
    tasks_path = get_bird_tasks_path(split)
    with tasks_path.open("r", encoding="utf-8") as file:
        raw_tasks = json.load(file)

    if not isinstance(raw_tasks, list):
        raise ValueError(f"BIRD tasks file must contain a list: {tasks_path}")

    return [BirdTask.from_dict(raw_task) for raw_task in raw_tasks]


def get_bird_task_by_question_id(split: str, question_id: int) -> BirdTask:
    """Return the task matching the given question_id within a BIRD split."""
    if not isinstance(question_id, int):
        raise ValueError("question_id must be an integer.")

    for task in load_bird_tasks(split):
        if task.question_id == question_id:
            return task

    raise LookupError(
        f"Question ID '{question_id}' does not exist in BIRD split '{split}'."
    )
