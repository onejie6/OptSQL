"""Utility tools for Explain Analyser Agent.

The exported functions implement the minimal tool set documented in
tool_design.md: capability detection, SQL structure parsing, explain-plan
fetching, plan normalization, and schema statistics collection.
"""

from agent.explainAnalyserAgent.utils.db_capabilities import detect_db_capabilities
from agent.explainAnalyserAgent.utils.decision import decide_optimization_action
from agent.explainAnalyserAgent.utils.explain_plan import get_explain_plan
from agent.explainAnalyserAgent.utils.normalize_plan import normalize_plan
from agent.explainAnalyserAgent.utils.schema_stats import collect_schema_stats
from agent.explainAnalyserAgent.utils.sql_structure import parse_sql_structure

__all__ = [
    "collect_schema_stats",
    "decide_optimization_action",
    "detect_db_capabilities",
    "get_explain_plan",
    "normalize_plan",
    "parse_sql_structure",
]
