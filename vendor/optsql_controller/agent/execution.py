"""SQL execution and analysis engine interface."""

from myTypes import ExecutionMetrics


class SQLExecutionAnalysisEngine:
    """Tools for SQL execution, explain-plan analysis, and result comparison."""

    def execute_sql(self, db_id: str, sql: str, timeout_ms: int | None = None) -> ExecutionMetrics:
        raise NotImplementedError

    def explain_analyzer(self, db_id: str, sql: str, dbms: str) -> ExecutionMetrics:
        raise NotImplementedError

    def normalize_result(self, result_set: list[tuple], order_sensitive: bool) -> list[tuple]:
        raise NotImplementedError

    def check_equivalence(
        self,
        db_id: str,
        source_sql: str,
        candidate_sql: str,
        order_sensitive: bool,
    ) -> dict:
        raise NotImplementedError
