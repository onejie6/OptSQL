"""Registry for builtin deterministic rewrite operators."""

from __future__ import annotations

from agent.rewrite_operators.models import OperatorDefinition
from agent.rewrite_operators.models import OperatorOpportunity
from agent.rewrite_operators.models import OperatorStrategyMetadata
from myTypes import RetrievedStrategy


_OPERATOR_DEFINITIONS: tuple[OperatorDefinition, ...] = (
    OperatorDefinition(
        rule_id="builtin_filter_dimension_before_top1",
        rule_name="Filter joined dimension before fact top-1",
        operator_name="filter_dimension_before_top1",
        applicable_when=(
            "Selected column comes from a joined dimension table",
            "Dimension-side filter can be pushed into a filtered subquery before ORDER BY ... LIMIT 1",
        ),
        rewrite_template=(
            "SELECT DIM_COLUMN FROM FACT_TABLE JOIN DIM_TABLE ON FACT_KEY = DIM_KEY "
            "WHERE DIM_FILTER ORDER BY FACT_ORDER_COLUMN ASC LIMIT 1 -> "
            "SELECT DIM_COLUMN FROM FACT_TABLE JOIN "
            "(SELECT DIM_KEY, DIM_COLUMN FROM DIM_TABLE WHERE DIM_FILTER) ON FACT_KEY = DIM_KEY "
            "ORDER BY FACT_ORDER_COLUMN ASC LIMIT 1"
        ),
        risk_notes=(
            "Only safe when the pushed filter depends on the joined dimension table and the ORDER BY key stays on the fact table.",
        ),
        confidence=0.93,
        hint_strategies=("push_down_filter", "align_order_by_with_index"),
        families=("filter_dimension_before_top1", "predicate_pushdown"),
        activation_hint_groups=(
            ("push_down_filter", "align_order_by_with_index"),
        ),
    ),
    OperatorDefinition(
        rule_id="builtin_projection_pruning",
        rule_name="Projection pruning for SELECT star",
        operator_name="projection_pruning",
        applicable_when=(
            "SQL contains SELECT *",
            "Blueprint selected columns provide the required projection",
        ),
        rewrite_template="Replace SELECT * with Blueprint selected columns.",
        risk_notes=("Projection pruning may remove columns if Blueprint is incomplete.",),
        confidence=0.72,
        hint_strategies=("reduce_select_columns",),
        families=("projection_pruning", "remove_select_star", "reduce_row_width"),
        activation_hint_groups=(("reduce_select_columns",),),
    ),
    OperatorDefinition(
        rule_id="builtin_redundant_distinct_elimination",
        rule_name="Eliminate redundant DISTINCT when projected columns are already unique",
        operator_name="redundant_distinct_elimination",
        applicable_when=(
            "Top-level SELECT DISTINCT reads from one base table",
            "Projected plain columns already contain a unique key, so DISTINCT cannot remove rows",
        ),
        rewrite_template=(
            "SELECT DISTINCT UNIQUE_KEY, OTHER_COL FROM TABLE WHERE PRED -> "
            "SELECT UNIQUE_KEY, OTHER_COL FROM TABLE WHERE PRED"
        ),
        risk_notes=(
            "Only safe when uniqueness is established from the projected columns themselves, not from duplicate elimination after joins.",
        ),
        confidence=0.96,
        hint_strategies=("eliminate_redundant_distinct",),
        families=("redundant_distinct_elimination", "distinct_elimination"),
        activation_hint_groups=(("eliminate_redundant_distinct",),),
    ),
    OperatorDefinition(
        rule_id="builtin_redundant_count_distinct_elimination",
        rule_name="Eliminate redundant COUNT DISTINCT when counted column is already unique",
        operator_name="redundant_count_distinct_elimination",
        applicable_when=(
            "Top-level SELECT COUNT(DISTINCT column) reads from one base table",
            "The counted column is covered by a unique key, so DISTINCT does not change the counted non-null set",
        ),
        rewrite_template=(
            "SELECT COUNT(DISTINCT UNIQUE_COL) FROM TABLE WHERE PRED -> "
            "SELECT COUNT(UNIQUE_COL) FROM TABLE WHERE PRED, "
            "or COUNT(*) when the unique column is also NOT NULL"
        ),
        risk_notes=(
            "Only safe when uniqueness is established from schema metadata and NULL-counting semantics are preserved exactly.",
        ),
        confidence=0.97,
        hint_strategies=("eliminate_redundant_count_distinct",),
        families=("redundant_count_distinct_elimination", "count_grain_canonicalization"),
        activation_hint_groups=(("eliminate_redundant_count_distinct",),),
    ),
    OperatorDefinition(
        rule_id="builtin_scalar_maxmin_to_order_limit",
        rule_name="Scalar MAX/MIN subquery to ORDER BY LIMIT",
        operator_name="scalar_maxmin_to_topk",
        applicable_when=(
            "Single-table scalar MAX/MIN query",
            "Tie semantics are not required",
        ),
        rewrite_template="Rewrite scalar MAX/MIN subquery to ORDER BY ... LIMIT 1.",
        risk_notes=("May change semantics when multiple rows tie.",),
        confidence=0.55,
        hint_strategies=("rewrite_scalar_maxmin_subquery",),
        families=("scalar_maxmin_to_order_limit", "maxmin_subquery_rewrite", "top1_order_limit"),
        activation_hint_groups=(("rewrite_scalar_maxmin_subquery",),),
    ),
    OperatorDefinition(
        rule_id="builtin_date_extraction_to_range",
        rule_name="Date extraction on column to range predicate",
        operator_name="date_extraction_to_range",
        applicable_when=(
            "WHERE STRFTIME('%Y', date_column) = 'YYYY' or STRFTIME('%Y-%m', date_column) = 'YYYY-MM'",
            "The extracted predicate can be rewritten as a half-open range on the raw column",
        ),
        rewrite_template=(
            "WHERE STRFTIME('%Y', DATE_COLUMN) = '1996' -> "
            "WHERE DATE_COLUMN >= '1996-01-01' AND DATE_COLUMN < '1997-01-01'"
        ),
        risk_notes=(
            "Only safe for ISO-like date strings or date-typed columns with comparable ordering semantics.",
        ),
        confidence=0.95,
        hint_strategies=("avoid_function_on_column",),
        families=("date_extraction_to_range", "sargable_predicate"),
        activation_hint_groups=(("avoid_function_on_column",),),
    ),
    OperatorDefinition(
        rule_id="builtin_like_prefix_to_range",
        rule_name="LIKE prefix on ordered text/date column to range predicate",
        operator_name="like_prefix_to_range",
        applicable_when=(
            "WHERE column LIKE 'prefix%' with no underscore wildcard or ESCAPE clause",
            "The column uses SQLite binary-style ordering that matches prefix lexicographic range semantics",
        ),
        rewrite_template=(
            "WHERE DATE_COLUMN LIKE '1991-10%' -> "
            "WHERE DATE_COLUMN >= '1991-10' AND DATE_COLUMN < '1991-11'"
        ),
        risk_notes=(
            "Only safe for conservative ASCII prefixes under stable collation semantics.",
        ),
        confidence=0.88,
        hint_strategies=("avoid_function_on_column",),
        families=("like_prefix_to_range", "sargable_predicate"),
        activation_hint_groups=(("avoid_function_on_column",),),
    ),
    OperatorDefinition(
        rule_id="builtin_distinct_join_to_semijoin",
        rule_name="JOIN plus DISTINCT selected key to semi-join",
        operator_name="distinct_join_to_semijoin",
        applicable_when=(
            "SELECT DISTINCT projects only columns from the outer table",
            "The joined table is only used for existence filtering and causes fanout",
        ),
        rewrite_template=(
            "SELECT DISTINCT BASE_COL FROM BASE INNER JOIN DIM ON JOIN_COND WHERE PRED -> "
            "SELECT DISTINCT BASE_COL FROM BASE WHERE BASE_PRED AND EXISTS "
            "(SELECT 1 FROM DIM WHERE DIM_PRED AND CORRELATED_JOIN_COND)"
        ),
        risk_notes=(
            "Only safe when ORDER BY, GROUP BY, and projected expressions do not depend on the fanout side.",
        ),
        confidence=0.91,
        hint_strategies=("simplify_join_graph",),
        families=("distinct_join_to_semijoin", "join_graph_simplification"),
        activation_hint_groups=(("simplify_join_graph",),),
    ),
    OperatorDefinition(
        rule_id="builtin_distinct_top1_to_grouped_extrema",
        rule_name="Canonicalize DISTINCT top-1 into grouped extrema",
        operator_name="distinct_top1_to_grouped_extrema",
        applicable_when=(
            "SELECT DISTINCT projects one column while ORDER BY ranks joined rows by one metric with LIMIT 1",
            "The DISTINCT semantics are equivalent to keeping the best metric per projected value, then taking top-1",
        ),
        rewrite_template=(
            "SELECT DISTINCT PROJ_COL FROM FACT JOIN DIM ON JOIN_COND WHERE PRED "
            "ORDER BY METRIC DESC LIMIT 1 -> "
            "SELECT PROJ_COL FROM FACT JOIN DIM ON JOIN_COND WHERE PRED "
            "GROUP BY PROJ_COL ORDER BY MAX(METRIC) DESC LIMIT 1"
        ),
        risk_notes=(
            "Only safe for single-column projection, single metric ordering, LIMIT 1 without OFFSET, and a single inner join.",
        ),
        confidence=0.9,
        hint_strategies=("align_order_by_with_index", "pre_aggregate_before_join"),
        families=("distinct_top1_to_grouped_extrema", "top1_order_limit"),
        activation_hint_groups=(("align_order_by_with_index",),),
        preflight_policy="must_reduce_scan_rows",
        preflight_failure_message=(
            "Performance guardrail: grouped-extrema candidate does not reduce scan rows in preflight."
        ),
    ),
    OperatorDefinition(
        rule_id="builtin_redundant_bridge_join_elimination",
        rule_name="Eliminate redundant bridge join when a direct narrower join exists",
        operator_name="redundant_bridge_join_elimination",
        applicable_when=(
            "A three-table chain joins through an unused bridge table",
            "Blueprint exposes a direct join edge between the outer tables",
        ),
        rewrite_template=(
            "SELECT DISTINCT OUTER_COL FROM A "
            "INNER JOIN B ON A.K1 = B.K1 "
            "INNER JOIN C ON B.K2 = C.K2 WHERE OUTER_PRED AND INNER_PRED -> "
            "SELECT DISTINCT OUTER_COL FROM A "
            "INNER JOIN C ON A.K3 = C.K3 WHERE OUTER_PRED AND INNER_PRED"
        ),
        risk_notes=(
            "Only safe when the bridge table contributes no projected/filter columns and a direct join preserves the same answer grain.",
        ),
        confidence=0.92,
        hint_strategies=("simplify_join_graph",),
        families=("redundant_bridge_join_elimination", "join_graph_simplification"),
        activation_hint_groups=(("simplify_join_graph",),),
    ),
    OperatorDefinition(
        rule_id="builtin_same_key_bridge_join_elimination",
        rule_name="Eliminate a bridge join chained through the same bridge key",
        operator_name="same_key_bridge_join_elimination",
        applicable_when=(
            "A three-table chain joins through one bridge-table key column on both sides",
            "The bridge table contributes no projected columns or non-join predicates",
        ),
        rewrite_template=(
            "SELECT OUT_COL FROM LEFT "
            "INNER JOIN BRIDGE ON LEFT.K = BRIDGE.BK "
            "INNER JOIN RIGHT ON BRIDGE.BK = RIGHT.K WHERE PRED -> "
            "SELECT OUT_COL FROM LEFT INNER JOIN RIGHT ON LEFT.K = RIGHT.K WHERE PRED"
        ),
        risk_notes=(
            "Only safe when the bridge table is used purely as a key relay through the same bridge column.",
        ),
        confidence=0.95,
        hint_strategies=("eliminate_same_key_bridge_join",),
        families=("same_key_bridge_join_elimination", "join_graph_simplification"),
        activation_hint_groups=(("eliminate_same_key_bridge_join",),),
    ),
    OperatorDefinition(
        rule_id="builtin_unused_fk_join_elimination",
        rule_name="Eliminate an unused foreign-key-protected join",
        operator_name="unused_fk_join_elimination",
        applicable_when=(
            "A single inner join only checks existence of a referenced dimension row",
            "The joined table contributes no projected, filtered, grouped, or ordered columns",
        ),
        rewrite_template=(
            "SELECT LEFT_COLS FROM FACT "
            "INNER JOIN DIM ON FACT.FK = DIM.PK WHERE FACT_PRED -> "
            "SELECT LEFT_COLS FROM FACT WHERE FACT_PRED"
        ),
        risk_notes=(
            "Only safe when the join columns match a foreign key from the preserved table to a unique key on the removed table.",
        ),
        confidence=0.96,
        hint_strategies=("simplify_join_graph",),
        families=("unused_fk_join_elimination", "join_graph_simplification"),
        activation_hint_groups=(("simplify_join_graph",),),
    ),
    OperatorDefinition(
        rule_id="builtin_unused_fk_join_chain_elimination",
        rule_name="Eliminate an unused foreign-key-protected join chain",
        operator_name="unused_fk_join_chain_elimination",
        applicable_when=(
            "A linear inner-join chain only checks existence of referenced rows along foreign-key edges",
            "All joined tables are otherwise unused outside their ON clauses",
        ),
        rewrite_template=(
            "SELECT BASE_COLS FROM BASE "
            "INNER JOIN DIM1 ON BASE.FK1 = DIM1.PK1 "
            "INNER JOIN DIM2 ON DIM1.FK2 = DIM2.PK2 WHERE BASE_PRED -> "
            "SELECT BASE_COLS FROM BASE WHERE BASE_PRED"
        ),
        risk_notes=(
            "Only safe when each join follows a foreign key from the previous table to a unique key on the next table, and no joined table contributes output or filtering semantics.",
        ),
        confidence=0.97,
        hint_strategies=("simplify_join_graph",),
        families=("unused_fk_join_chain_elimination", "join_graph_simplification"),
        activation_hint_groups=(("simplify_join_graph",),),
    ),
    OperatorDefinition(
        rule_id="builtin_symmetric_union_arm_pruning",
        rule_name="Prune symmetric UNION/OR edge duplication under canonical edge storage",
        operator_name="symmetric_union_arm_pruning",
        applicable_when=(
            "Two UNION/OR arms differ only by swapping symmetric edge endpoint columns",
            "The schema/query semantics guarantee one canonical edge orientation is sufficient",
        ),
        rewrite_template=(
            "SELECT ... FROM EDGE WHERE (A = L1 AND B = L2) OR (A = L2 AND B = L1) -> "
            "SELECT ... FROM EDGE WHERE A = LEAST_LITERAL AND B = GREATEST_LITERAL"
        ),
        risk_notes=(
            "Only safe for guarded symmetric-edge schemas with canonical orientation semantics.",
        ),
        confidence=0.83,
        hint_strategies=("simplify_join_graph",),
        families=("symmetric_union_arm_pruning", "join_graph_simplification"),
        activation_hint_groups=(("simplify_join_graph",),),
    ),
    OperatorDefinition(
        rule_id="builtin_grouped_max_top1_before_join",
        rule_name="Pre-aggregate join table before grouped MAX top-1",
        operator_name="grouped_max_top1_before_join",
        applicable_when=(
            "GROUP BY base key with ORDER BY MAX(join_table.column) LIMIT 1",
            "Base-table filters can be applied before joining the aggregated table",
        ),
        rewrite_template=(
            "SELECT BASE_KEY FROM BASE_TABLE JOIN JOIN_TABLE ON BASE_KEY = JOIN_KEY "
            "WHERE BASE_PREDICATE GROUP BY BASE_KEY ORDER BY MAX(ORDER_COLUMN) DESC LIMIT 1 -> "
            "SELECT BASE_KEY FROM (SELECT DISTINCT BASE_KEY FROM BASE_TABLE WHERE BASE_PREDICATE) "
            "JOIN (SELECT JOIN_KEY, MAX(ORDER_COLUMN) AS MAX_ORDER_VALUE FROM JOIN_TABLE GROUP BY JOIN_KEY) "
            "ON BASE_KEY = JOIN_KEY ORDER BY MAX_ORDER_VALUE DESC LIMIT 1"
        ),
        risk_notes=(
            "Only safe when the grouped key is the selected result and base-side filtering does not depend on downstream joins.",
        ),
        confidence=0.9,
        hint_strategies=("align_order_by_with_index", "push_down_filter", "pre_aggregate_before_join"),
        families=("grouped_max_top1_before_join",),
        activation_hint_groups=(("align_order_by_with_index",),),
        preflight_policy="must_improve_any_metric",
        preflight_failure_message=(
            "Performance guardrail: grouped top-1 candidate shows no measured improvement in preflight."
        ),
    ),
    OperatorDefinition(
        rule_id="builtin_topk_before_join",
        rule_name="Top-k before downstream joins",
        operator_name="topk_before_join",
        applicable_when=(
            "ORDER BY ... LIMIT 1 uses a base-table sort key",
            "The query joins downstream tables after the base table",
        ),
        rewrite_template=(
            "SELECT COLUMN FROM TABLE JOIN TABLE ON COLUMN = COLUMN "
            "ORDER BY NUMERIC_COLUMN DESC LIMIT LITERAL -> "
            "SELECT COLUMN FROM (SELECT COLUMN FROM TABLE ORDER BY NUMERIC_COLUMN DESC LIMIT LITERAL) "
            "JOIN TABLE ON COLUMN = COLUMN"
        ),
        risk_notes=("May change semantics when downstream joins duplicate the top-k base row.",),
        confidence=0.86,
        hint_strategies=("align_order_by_with_index",),
        families=("topk_before_join", "top1_before_join"),
        activation_hint_groups=(("align_order_by_with_index",),),
        suppressed_by=("builtin_reanchor_join_driver",),
    ),
    OperatorDefinition(
        rule_id="builtin_top1_anchor_then_lookup_tail",
        rule_name="Resolve top-1 anchor key before final lookup tail",
        operator_name="top1_anchor_then_lookup_tail",
        applicable_when=(
            "A linear inner-join chain orders by a non-tail table and LIMIT 1 while projecting only the final lookup table",
            "The final lookup table can be probed by one uniqueness-backed key after the top-1 anchor is resolved upstream",
        ),
        rewrite_template=(
            "SELECT TAIL_COLS FROM PREFIX1 "
            "INNER JOIN PREFIX2 ON ... "
            "INNER JOIN TAIL ON PREFIX2.TAIL_FK = TAIL.TAIL_PK "
            "WHERE PREFIX_PRED ORDER BY FACT_METRIC ASC LIMIT 1 -> "
            "SELECT TAIL_COLS FROM TAIL WHERE TAIL.TAIL_PK = "
            "(SELECT PREFIX2.TAIL_FK FROM PREFIX1 INNER JOIN PREFIX2 ON ... "
            "WHERE PREFIX_PRED ORDER BY FACT_METRIC ASC LIMIT 1)"
        ),
        risk_notes=(
            "Only safe for linear inner-join chains where all projections come from the final lookup tail, tail-side filters are absent, and the tail key is unique.",
        ),
        confidence=0.92,
        hint_strategies=("align_order_by_with_index", "push_down_filter"),
        families=("top1_anchor_then_lookup_tail", "top1_order_limit", "predicate_pushdown"),
        activation_hint_groups=(("align_order_by_with_index",),),
    ),
    OperatorDefinition(
        rule_id="builtin_scalar_extrema_anchor_then_lookup_tail",
        rule_name="Normalize scalar extrema filter into top-1 anchor then final lookup tail",
        operator_name="scalar_extrema_anchor_then_lookup_tail",
        applicable_when=(
            "A linear inner-join chain projects only the final lookup tail while a predecessor metric is filtered by a scalar extrema subquery",
            "The scalar extrema can be replaced with one upstream top-1 over the same filtered prefix, then a uniqueness-backed tail probe",
        ),
        rewrite_template=(
            "SELECT TAIL_COLS FROM PREFIX INNER JOIN PREDECESSOR ON ... INNER JOIN TAIL ON PREDECESSOR.TAIL_FK = TAIL.TAIL_PK "
            "WHERE PREFIX_PRED AND PREDECESSOR.METRIC = "
            "(SELECT METRIC FROM PREDECESSOR WHERE CORRELATED_PRED ORDER BY METRIC DESC LIMIT 1) -> "
            "SELECT TAIL_COLS FROM TAIL WHERE TAIL.TAIL_PK = "
            "(SELECT PREDECESSOR.TAIL_FK FROM PREFIX INNER JOIN PREDECESSOR ON ... "
            "WHERE PREFIX_PRED ORDER BY PREDECESSOR.METRIC DESC LIMIT 1)"
        ),
        risk_notes=(
            "Only safe for the guarded linear scalar-extrema shape where the inner subquery ranges over the predecessor table alone, predecessor-side predicates match, and the final tail key is unique.",
        ),
        confidence=0.91,
        hint_strategies=("align_order_by_with_index", "rewrite_scalar_maxmin_subquery"),
        families=("scalar_extrema_anchor_then_lookup_tail", "top1_order_limit", "predicate_pushdown"),
        activation_hint_groups=(("align_order_by_with_index",),),
    ),
    OperatorDefinition(
        rule_id="builtin_distinct_extrema_to_grouped_having",
        rule_name="Canonicalize DISTINCT extrema filter into grouped HAVING extrema",
        operator_name="distinct_extrema_to_grouped_having",
        applicable_when=(
            "SELECT DISTINCT projects plain columns while WHERE filters one metric by a scalar extrema subquery",
            "The DISTINCT semantics are equivalent to grouping by the projected values and keeping groups whose best metric matches the same extrema subquery",
        ),
        rewrite_template=(
            "SELECT DISTINCT PROJ_COLS FROM JOIN_GRAPH "
            "WHERE OUTER_PRED AND METRIC = (SELECT METRIC FROM INNER_GRAPH ORDER BY METRIC DESC LIMIT 1) -> "
            "SELECT PROJ_COLS FROM JOIN_GRAPH WHERE OUTER_PRED "
            "GROUP BY PROJ_COLS HAVING MAX(METRIC) = (SELECT MAX(METRIC) FROM INNER_GRAPH)"
        ),
        risk_notes=(
            "Only safe for the guarded scalar-extrema equality shape; duplicate-sensitive semantics stay inside the same join graph and are expressed with GROUP BY plus HAVING.",
        ),
        confidence=0.9,
        hint_strategies=("align_order_by_with_index", "rewrite_scalar_maxmin_subquery"),
        families=("distinct_extrema_to_grouped_having", "top1_order_limit", "pre_aggregation"),
        activation_hint_groups=(("align_order_by_with_index",),),
        preflight_policy="must_reduce_scan_rows",
        preflight_failure_message=(
            "Performance guardrail: grouped-HAVING candidate does not reduce scan rows in preflight."
        ),
    ),
    OperatorDefinition(
        rule_id="builtin_argmax_aggregate_to_topk",
        rule_name="Repeated aggregate argmax to ORDER BY aggregate LIMIT",
        operator_name="argmax_aggregate_to_topk",
        applicable_when=(
            "An outer HAVING compares COUNT/SUM/AVG aggregate against MAX/MIN of the same grouped aggregate subquery",
            "The query can be reduced to one grouped pass with ORDER BY aggregate DESC/ASC LIMIT 1",
        ),
        rewrite_template=(
            "SELECT GROUP_KEY FROM BASE GROUP BY GROUP_KEY "
            "HAVING COUNT(*) = (SELECT MAX(CNT) FROM (SELECT COUNT(*) AS CNT FROM BASE GROUP BY GROUP_KEY)) -> "
            "SELECT GROUP_KEY FROM BASE GROUP BY GROUP_KEY ORDER BY COUNT(*) DESC LIMIT 1"
        ),
        risk_notes=("Tie semantics may differ if the original query returns multiple groups.",),
        confidence=0.9,
        hint_strategies=("align_order_by_with_index",),
        families=("argmax_aggregate_to_topk",),
        activation_hint_groups=(("align_order_by_with_index",),),
        preflight_policy="must_improve_any_metric",
        preflight_failure_message=(
            "Performance guardrail: aggregate argmax candidate shows no measured improvement in preflight."
        ),
    ),
    OperatorDefinition(
        rule_id="builtin_same_table_lookup_to_scalar_subquery",
        rule_name="Same-table lookup join to scalar key subquery",
        operator_name="same_table_lookup_to_scalar_subquery",
        applicable_when=(
            "Same-table alias join filtered by a literal on the lookup alias",
            "The rewrite should replace the lookup alias with a direct key subquery",
        ),
        rewrite_template=(
            "SELECT TEXT_COLUMN FROM TABLE INNER JOIN TABLE ON FK_COLUMN = ID_COLUMN "
            "INNER JOIN TABLE ON COLUMN = COLUMN WHERE TEXT_COLUMN = LITERAL "
            "ORDER BY COLUMN ASC LIMIT LITERAL -> "
            "SELECT TEXT_COLUMN FROM TABLE INNER JOIN TABLE ON FK_COLUMN = ID_COLUMN "
            "WHERE COLUMN = (SELECT COLUMN FROM TABLE WHERE TEXT_COLUMN = LITERAL) "
            "ORDER BY COLUMN ASC LIMIT LITERAL"
        ),
        risk_notes=("Only safe when the lookup alias contributes a key-equivalent filter.",),
        confidence=0.92,
        hint_strategies=("eliminate_redundant_self_join",),
        families=("self_join_lookup_simplification",),
        activation_hint_groups=(("eliminate_redundant_self_join",),),
    ),
    OperatorDefinition(
        rule_id="builtin_repeated_rescan_to_conditional_agg",
        rule_name="Collapse repeated rescans into one grouped conditional aggregation pass",
        operator_name="repeated_rescan_to_conditional_agg",
        applicable_when=(
            "Two sibling grouped subqueries or CTEs scan the same fact table with different predicates",
            "The outer query joins those siblings back on the same grouping key",
        ),
        rewrite_template=(
            "WITH A AS (SELECT KEY, SUM(VAL) AS V1 FROM FACT WHERE P1 GROUP BY KEY), "
            "B AS (SELECT KEY, SUM(VAL) AS V2 FROM FACT WHERE P2 GROUP BY KEY) "
            "SELECT A.KEY, A.V1, B.V2 FROM A INNER JOIN B ON A.KEY = B.KEY -> "
            "SELECT KEY, SUM(CASE WHEN P1 THEN VAL END) AS V1, "
            "SUM(CASE WHEN P2 THEN VAL END) AS V2 FROM FACT WHERE P1 OR P2 GROUP BY KEY "
            "HAVING SUM(CASE WHEN P1 THEN 1 ELSE 0 END) > 0 AND SUM(CASE WHEN P2 THEN 1 ELSE 0 END) > 0"
        ),
        risk_notes=(
            "Only safe for tightly matched sibling grouped scans over the same base table and grouping key.",
        ),
        confidence=0.94,
        hint_strategies=("pre_aggregate_before_join",),
        families=("repeated_rescan_to_conditional_agg", "pre_aggregation", "aggregate_then_join"),
        activation_hint_groups=(("pre_aggregate_before_join",),),
    ),
    OperatorDefinition(
        rule_id="builtin_dimension_key_first_then_fact_probe",
        rule_name="Resolve dimension key first, then probe fact table",
        operator_name="dimension_key_first_then_fact_probe",
        applicable_when=(
            "A selective dimension/top-k subquery determines one key before touching a larger fact table",
            "The outer query only needs fact-table columns for that resolved key",
        ),
        rewrite_template=(
            "SELECT FACT_COL FROM DIM JOIN FACT ON DIM_KEY = FACT_KEY "
            "WHERE DIM_PRED ORDER BY DIM_ORDER DESC LIMIT 1 -> "
            "SELECT FACT_COL FROM FACT WHERE FACT_KEY = "
            "(SELECT DIM_KEY FROM DIM WHERE DIM_PRED ORDER BY DIM_ORDER DESC LIMIT 1)"
        ),
        risk_notes=(
            "Only safe when the resolved key is unique enough for the downstream fact probe semantics.",
        ),
        confidence=0.93,
        hint_strategies=("push_down_filter",),
        families=("dimension_key_first_then_fact_probe", "predicate_pushdown", "filter_before_join"),
        activation_hint_groups=(("push_down_filter",),),
        suppressed_by=("builtin_reanchor_join_driver",),
    ),
    OperatorDefinition(
        rule_id="builtin_reanchor_join_driver",
        rule_name="Re-anchor join on the selective driver table and probe the fact table by resolved key",
        operator_name="reanchor_join_driver",
        applicable_when=(
            "A three-table driver/bridge/fact join orders or filters on the driver table but projects only fact-table columns",
            "The bridge key used to reach the fact table can be resolved from the narrower driver first",
        ),
        rewrite_template=(
            "SELECT FACT_COL FROM DRIVER JOIN BRIDGE ON DRIVER_KEY = BRIDGE_KEY "
            "JOIN FACT ON BRIDGE_KEY = FACT_KEY WHERE DRIVER_PRED ORDER BY DRIVER_ORDER DESC LIMIT 1 -> "
            "SELECT FACT_COL FROM FACT WHERE FACT_KEY = "
            "(SELECT DRIVER_KEY FROM DRIVER WHERE DRIVER_PRED ORDER BY DRIVER_ORDER DESC LIMIT 1)"
        ),
        risk_notes=(
            "Only safe when the bridge contributes no required output/filter columns and the resolved driver key preserves the original top-1 semantics.",
        ),
        confidence=0.95,
        hint_strategies=("push_down_filter", "align_order_by_with_index", "simplify_join_graph"),
        families=("reanchor_join_driver", "predicate_pushdown", "filter_before_join"),
        activation_hint_groups=(
            ("push_down_filter",),
            ("align_order_by_with_index",),
            ("simplify_join_graph",),
        ),
    ),
    OperatorDefinition(
        rule_id="builtin_prefer_summary_table_when_grain_matches",
        rule_name="Use a summary-compatible smaller table instead of raw detail fact",
        operator_name="prefer_summary_table_when_grain_matches",
        applicable_when=(
            "A raw detail fact contributes only a summary-compatible fastest/minimum metric",
            "A smaller summary table exposes the same grain key and metric semantics",
        ),
        rewrite_template=(
            "SELECT MIN(DETAIL_METRIC) FROM DETAIL JOIN DIM ON DETAIL_KEY = DIM_KEY WHERE DIM_PRED -> "
            "SELECT MIN(SUMMARY_METRIC) FROM SUMMARY JOIN DIM ON SUMMARY_KEY = DIM_KEY WHERE DIM_PRED"
        ),
        risk_notes=("Only safe for explicitly known summary/detail metric mappings.",),
        confidence=0.88,
        hint_strategies=("push_down_filter",),
        families=("prefer_summary_table_when_grain_matches", "predicate_pushdown"),
        activation_hint_groups=(("push_down_filter",),),
    ),
)

_DEFINITION_BY_RULE_ID = {
    definition.rule_id: definition
    for definition in _OPERATOR_DEFINITIONS
}
_DEFINITION_BY_OPERATOR_NAME = {
    definition.operator_name: definition
    for definition in _OPERATOR_DEFINITIONS
}


def _definition_is_activated(
    definition: OperatorDefinition,
    hint_names: set[str],
) -> bool:
    return any(set(group) & hint_names for group in definition.activation_hint_groups)


def _strategy_from_definition(definition: OperatorDefinition) -> RetrievedStrategy:
    return RetrievedStrategy(
        rule_id=definition.rule_id,
        rule_name=definition.rule_name,
        applicable_when=list(definition.applicable_when),
        rewrite_template=definition.rewrite_template,
        risk_notes=list(definition.risk_notes),
        example_cases=list(definition.example_cases),
        confidence=definition.confidence,
        source_type="operator",
        families=list(definition.families),
        hint_strategies=list(definition.hint_strategies),
        operator_name=definition.operator_name,
        suppressed_by=list(definition.suppressed_by),
        preflight_policy=definition.preflight_policy,
        preflight_failure_message=definition.preflight_failure_message,
    )


def get_operator_strategy_metadata(rule_id: str) -> OperatorStrategyMetadata | None:
    definition = _DEFINITION_BY_RULE_ID.get(rule_id)
    if definition is None:
        return None
    return OperatorStrategyMetadata(
        rule_id=definition.rule_id,
        operator_name=definition.operator_name,
        families=definition.families,
        hint_strategies=definition.hint_strategies,
        suppressed_by=definition.suppressed_by,
        preflight_policy=definition.preflight_policy,
        preflight_failure_message=definition.preflight_failure_message,
    )


def build_operator_strategies(hint_names: set[str]) -> list[RetrievedStrategy]:
    return [
        _strategy_from_definition(definition)
        for definition in _OPERATOR_DEFINITIONS
        if _definition_is_activated(definition, hint_names)
    ]


def build_operator_strategies_from_opportunities(
    opportunities: list[OperatorOpportunity],
) -> list[RetrievedStrategy]:
    by_rule_id = {
        strategy.rule_id: strategy
        for strategy in build_operator_strategies(
            {opportunity.hint_strategy for opportunity in opportunities}
        )
    }
    opportunity_names = {opportunity.operator_name for opportunity in opportunities}
    suppressed_rule_ids = {
        definition.rule_id
        for definition in _OPERATOR_DEFINITIONS
        if any(suppressor in opportunity_names for suppressor in definition.suppressed_by)
    }

    ordered: list[RetrievedStrategy] = []
    seen: set[str] = set()
    for opportunity in opportunities:
        definition = _DEFINITION_BY_OPERATOR_NAME.get(opportunity.operator_name)
        if definition is None:
            continue
        rule_id = definition.rule_id
        if rule_id in seen or rule_id in suppressed_rule_ids:
            continue
        strategy = by_rule_id.get(rule_id)
        if strategy is None:
            strategy = _strategy_from_definition(definition)
        seen.add(rule_id)
        ordered.append(strategy)
    return ordered
