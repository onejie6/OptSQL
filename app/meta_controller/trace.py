from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .models import ControllerDecision


class ControllerTraceStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append_decision(self, decision: ControllerDecision, **extra: Any) -> None:
        record = {"event": "decision", **decision.model_dump(mode="json"), **extra}
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
