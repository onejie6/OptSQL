"""Structured rewrite operators for deterministic SQL optimization paths."""

from agent.rewrite_operators.detector import detect_operator_opportunities
from agent.rewrite_operators.models import OperatorOpportunity
from agent.rewrite_operators.registry import build_operator_strategies
from agent.rewrite_operators.registry import build_operator_strategies_from_opportunities
from agent.rewrite_operators.registry import get_operator_strategy_metadata

__all__ = [
    "OperatorOpportunity",
    "build_operator_strategies",
    "build_operator_strategies_from_opportunities",
    "detect_operator_opportunities",
    "get_operator_strategy_metadata",
]
