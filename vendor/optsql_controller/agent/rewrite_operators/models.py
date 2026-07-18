"""Models for structured rewrite-operator detection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorOpportunity:
    operator_name: str
    hint_strategy: str
    matched: bool
    confidence: float
    target_fragment: str | None
    expected_effect: str
    semantic_risks: list[str]
    requires_validation: bool = True
    dbms_notes: str | None = None


@dataclass(frozen=True)
class OperatorStrategyMetadata:
    rule_id: str
    operator_name: str
    families: tuple[str, ...]
    hint_strategies: tuple[str, ...]
    suppressed_by: tuple[str, ...] = ()
    preflight_policy: str | None = None
    preflight_failure_message: str | None = None


@dataclass(frozen=True)
class OperatorDefinition:
    rule_id: str
    rule_name: str
    operator_name: str
    applicable_when: tuple[str, ...]
    rewrite_template: str
    risk_notes: tuple[str, ...]
    confidence: float
    hint_strategies: tuple[str, ...]
    families: tuple[str, ...]
    activation_hint_groups: tuple[tuple[str, ...], ...]
    example_cases: tuple[str, ...] = ()
    suppressed_by: tuple[str, ...] = ()
    preflight_policy: str | None = None
    preflight_failure_message: str | None = None
