"""SQL safety checks for user/model generated statements."""

from __future__ import annotations

import sqlglot
import sqlglot.expressions as exp


def ensure_select_sql(sql: str, *, dialect: str = "sqlite") -> None:
    """Reject multi-statement and non-SELECT SQL before execution.

    This guard is intended for model/user generated SQL that will be executed
    against benchmark databases. Internal metadata queries such as PRAGMA or
    sqlite_master introspection should not use this helper.
    """
    parsed = sqlglot.parse(sql, dialect=dialect)
    if len(parsed) != 1:
        raise ValueError("Only a single SQL statement is allowed.")
    expression = parsed[0]
    if isinstance(expression, exp.Select):
        return
    if isinstance(expression, exp.Union):
        return
    raise ValueError("Only SELECT/WITH read-only SQL statements are allowed.")
