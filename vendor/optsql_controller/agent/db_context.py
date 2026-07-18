"""DB context grounding engine interface."""

from myTypes import JoinGraph


class DBContextGroundingEngine:
    """Tools for grounding schema, values, and join topology against a database."""

    def inspect_column_info(self, db_id: str, table_name: str, column_name: str) -> dict:
        raise NotImplementedError

    def verify_exact_value(
        self,
        db_id: str,
        table_name: str,
        column_name: str,
        value: object,
    ) -> dict:
        raise NotImplementedError

    def probe_similar_values(
        self,
        db_id: str,
        table_name: str,
        column_name: str,
        value: object,
        top_k: int,
    ) -> list[dict]:
        raise NotImplementedError

    def get_column_enums(
        self,
        db_id: str,
        table_name: str,
        column_name: str,
        limit: int,
    ) -> list[dict]:
        raise NotImplementedError

    def sample_column_format(
        self,
        db_id: str,
        table_name: str,
        column_name: str,
        sample_size: int,
    ) -> list[object]:
        raise NotImplementedError

    def find_similar_columns(self, db_id: str, value_or_description: str, top_k: int) -> list[dict]:
        raise NotImplementedError

    def route_topology(self, db_id: str, source_table: str, target_table: str) -> JoinGraph:
        raise NotImplementedError
