from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ControllerStage(str, Enum):
    SCHEMA_LINKING = "schema_linking"
    SQL_GENERATION = "sql_generation"
    SQL_SELECTION = "sql_selection"
    OPTIMIZATION = "optimization"


class ControllerAction(str, Enum):
    CONTINUE = "continue"
    ESCALATE = "escalate"
    REFLECT = "reflect"
    OPTIMIZE = "optimize"
    FALLBACK = "fallback"
    STOP = "stop"


class ControllerDecision(BaseModel):
    question_id: int
    database_id: str
    stage: ControllerStage
    action: ControllerAction
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    signals: dict[str, Any] = Field(default_factory=dict)
    budget_before: dict[str, int] = Field(default_factory=dict)
    budget_after: dict[str, int] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
