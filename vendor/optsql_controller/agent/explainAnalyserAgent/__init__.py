"""Explain Analyser Agent package.

This package contains execution-plan analysis utilities for the Optimization
Loop. The utilities are intentionally DBMS-aware at the tool boundary and
return normalized structures for agent-level reasoning.
"""

__all__ = ["ExplainAnalyzerAgent"]


def __getattr__(name: str):
    """Lazily expose ExplainAnalyzerAgent without creating import cycles."""
    if name == "ExplainAnalyzerAgent":
        from agent.explainAnalyserAgent.agent import ExplainAnalyzerAgent

        return ExplainAnalyzerAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
