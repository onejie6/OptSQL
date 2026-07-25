from app.dataset import BaseDataset, load_dataset, save_dataset, DataItem
from app.llm import LLM
from app.logger import logger
from app.prompt import PromptFactory
from app.llm_extractor import LLMExtractor
from app.pipeline.sql_selection.semantic_risk import audit_semantic_risks
from app.meta_controller import MetaControllerRuntime
from app.pipeline.validation import validate_pipeline_step
from app.progress import log_progress, should_checkpoint
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import re
from collections import Counter
from tqdm import tqdm
import time
from app.services import ArtifactStore, STAGE_ARTIFACT_FIELDS, configure_execution_service, configure_schema_service, get_execution_service, get_schema_service, load_stage_dataset, reset_execution_service, reset_schema_service
from app.llm_extractor import LLMExtractor


class SQLSelectionRunner:
    
    _llm: LLM = None
    _dataset: BaseDataset = None
    _thread_pool_executor: ThreadPoolExecutor = None
    _inner_thread_pool_executor: ThreadPoolExecutor = None
    _artifact_store: ArtifactStore = None
    _execution_service = None
    _extractor_max_retry: int = 3
    _stage_config = None
    _input_save_path: str = ""
    _dataset_config = None
    _timing_refine_repeat: int = 2
    _extractor: LLMExtractor = None
    _parallelism: int = 16
    _progress_log_interval: int = 50
    _checkpoint_interval: int = 20
    _meta_controller: MetaControllerRuntime | None = None
    
    def __init__(
        self,
        stage_config,
        dataset_config,
        input_save_path: str,
        extractor_max_retry: int,
        parallelism: int,
        progress_log_interval: int,
        checkpoint_interval: int,
        meta_controller_config=None,
    ):
        self._stage_config = stage_config
        self._dataset_config = dataset_config
        self._input_save_path = input_save_path
        self._extractor_max_retry = extractor_max_retry
        self._parallelism = max(1, parallelism)
        self._progress_log_interval = max(1, progress_log_interval)
        self._checkpoint_interval = max(1, checkpoint_interval)
        if meta_controller_config is not None and meta_controller_config.enabled:
            self._meta_controller = MetaControllerRuntime(meta_controller_config)
        self._artifact_store = ArtifactStore(
            self._stage_config.save_path,
            "sql_selection",
            STAGE_ARTIFACT_FIELDS["sql_selection"],
        )
        self._dataset, checkpoint_source = load_stage_dataset(
            load_dataset_fn=load_dataset,
            current_save_path=self._stage_config.save_path,
            fallback_load_path=self._input_save_path,
            artifact_store=self._artifact_store,
            stage_name="sql_selection",
        )
        logger.info(f"Initialized SQL selection dataset from {checkpoint_source}")
        configure_schema_service(max_value_example_length=self._dataset_config.max_value_example_length)
        configure_execution_service(
            default_timeout=self._dataset_config.sql_execution_timeout,
            bigquery_credential_path=self._dataset_config.bigquery_credential_path,
            snowflake_credential_path=self._dataset_config.snowflake_credential_path,
        )
        self._llm = LLM(self._stage_config.llm)
        logger.info(f"SQL selection parallelism: {self._parallelism}")
        self._thread_pool_executor = ThreadPoolExecutor(max_workers=self._parallelism)
        self._inner_thread_pool_executor = ThreadPoolExecutor(max_workers=self._parallelism)
        self._execution_service = get_execution_service()
        self._extractor = LLMExtractor(max_retry=self._extractor_max_retry)

    @classmethod
    def from_config(cls, app_config=None) -> "SQLSelectionRunner":
        if app_config is None:
            from app.config import get_config

            app_config = get_config()
        return cls(
            stage_config=app_config.sql_selection_config,
            dataset_config=app_config.dataset_config,
            input_save_path=app_config.sql_revision_config.save_path,
            extractor_max_retry=app_config.llm_extractor_config.max_retry,
            parallelism=app_config.run_config.parallelism,
            progress_log_interval=app_config.run_config.progress_log_interval,
            checkpoint_interval=app_config.run_config.checkpoint_interval,
            meta_controller_config=app_config.meta_controller_config,
        )
    
    def _parse_llm_response(self, response: str) -> Optional[List[Dict[str, Any]]]:
        """
        Parse the llm response and return the eval scores.
        """
        
        try:
            answer_match = re.search(r"<result>(.*?)</result>", response, re.DOTALL)
            if not answer_match:
                logger.warning("No <result> tag found in LLM response")
                logger.warning(f"Response content: {response}")
                return None
            answer_content = answer_match.group(1).strip().upper()
            logger.debug(f"Parsed LLM response: {answer_content}")
            # check the answer content is 'A', 'B', or 'TIE'
            if answer_content not in ["A", "B", "TIE"]:
                logger.error("Answer content is not 'A', 'B', or 'TIE'")
                return None
            return answer_content
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.debug(f"Response content: {response}")
            return None

    @staticmethod
    def _parse_listwise_response(response: str, candidate_count: int) -> Optional[int]:
        answer_match = re.search(r"<result>\s*([A-Z])\s*</result>", response, re.IGNORECASE)
        if not answer_match:
            return None
        candidate_index = ord(answer_match.group(1).upper()) - ord("A")
        if candidate_index < 0 or candidate_index >= candidate_count:
            return None
        return candidate_index

    @staticmethod
    def _get_position_balanced_orders(candidate_count: int, vote_count: int) -> List[List[int]]:
        base_order = list(range(candidate_count))
        orders = []
        for vote_index in range(max(1, vote_count)):
            shift = (vote_index * candidate_count) // max(1, vote_count)
            order = base_order[shift:] + base_order[:shift]
            if vote_index % 2 == 1:
                order = list(reversed(order))
            orders.append(order)
        return orders

    def _get_top_k_sql_candidates(self, data_item: DataItem) -> List[Tuple[str, str, float, float]]:
        valid_sql_candidates = []
        fallback_sql_candidates = []
        sql_map_to_execution_result = {}
        sql_map_to_result_hash = {}
        for sql_candidate in data_item.sql_candidates_after_revision:
            execution_result = sql_map_to_execution_result.get(sql_candidate)
            if execution_result is None:
                execution_result = self._execution_service.execute(data_item, sql_candidate)
                sql_map_to_execution_result[sql_candidate] = execution_result
            if execution_result.result_rows is None:
                continue
            result_hash = sql_map_to_result_hash.get(sql_candidate)
            if result_hash is None:
                result_hash = self._execution_service.hash_result(data_item, execution_result.result_rows)
                sql_map_to_result_hash[sql_candidate] = result_hash
            fallback_sql_candidates.append((sql_candidate, result_hash))
            if len(execution_result.result_rows) > 0:
                valid_sql_candidates.append((sql_candidate, result_hash))

        if len(valid_sql_candidates) == 0 and len(fallback_sql_candidates) > 0:
            logger.warning("No successful SQL candidates, backing to SQL candidates with not none result_rows")
            valid_sql_candidates = fallback_sql_candidates

        if len(valid_sql_candidates) == 0:
            return []
        
        counter = Counter(execution_result for _, execution_result in valid_sql_candidates)
        
        deduplicated_valid_sql_candidates = []
        seen_result_set = set()
        for sql_candidate, result_hash in valid_sql_candidates:
            if result_hash not in seen_result_set:
                execution_result = sql_map_to_execution_result[sql_candidate]
                execution_time = execution_result.execution_time if execution_result.execution_time is not None else np.inf
                deduplicated_valid_sql_candidates.append(
                    (sql_candidate, result_hash, counter[result_hash] / len(valid_sql_candidates), execution_time)
                )
                seen_result_set.add(result_hash)

        top_k_sql_candidates = sorted(
            deduplicated_valid_sql_candidates,
            key=self._get_sql_candidate_sort_key,
            reverse=True,
        )[:self._stage_config.filter_top_k_sql]
        top_k_sql_candidates = self._refine_relevant_tied_candidate_timings(
            data_item,
            deduplicated_valid_sql_candidates,
            top_k_sql_candidates,
        )

        return [
            (
                sql_candidate,
                sql_map_to_execution_result[sql_candidate].result_table_str,
                consistency_score,
                execution_time,
            )
            for sql_candidate, _, consistency_score, execution_time in top_k_sql_candidates
        ]

    @staticmethod
    def _get_sql_candidate_sort_key(sql_candidate: Tuple[str, Any, float, float]) -> Tuple[float, float]:
        return (sql_candidate[2], -sql_candidate[3])

    def _refine_relevant_tied_candidate_timings(
        self,
        data_item: DataItem,
        all_sql_candidates: List[Tuple[str, Any, float, float]],
        provisional_top_k_sql_candidates: List[Tuple[str, Any, float, float]],
    ) -> List[Tuple[str, Any, float, float]]:
        if len(provisional_top_k_sql_candidates) <= 1:
            return provisional_top_k_sql_candidates

        relevant_consistency_scores = {sql_candidate[2] for sql_candidate in provisional_top_k_sql_candidates}
        consistency_score_counts = Counter(sql_candidate[2] for sql_candidate in all_sql_candidates)

        refined_sql_candidates = []
        did_refine = False
        for sql_candidate, result_hash, consistency_score, execution_time in all_sql_candidates:
            if consistency_score in relevant_consistency_scores and consistency_score_counts[consistency_score] > 1:
                execution_time = self._execution_service.measure_time(
                    data_item,
                    sql_candidate,
                    repeat=self._timing_refine_repeat,
                )
                did_refine = True
            refined_sql_candidates.append(
                (sql_candidate, result_hash, consistency_score, execution_time)
            )

        if not did_refine:
            return provisional_top_k_sql_candidates

        return sorted(
            refined_sql_candidates,
            key=self._get_sql_candidate_sort_key,
            reverse=True,
        )[:self._stage_config.filter_top_k_sql]
    
    def _get_pair_sqls_to_eval(self, top_k_sql_candidates: List[Tuple[str, str, float, float]]) -> List[Tuple[Tuple[str, str, float, float], Tuple[str, str, float, float]]]:
        """
        Get the pair of sqls to eval.
        """
        pair_sqls_to_eval = []
        for first_idx, first_sql in enumerate(top_k_sql_candidates):
            for second_idx, second_sql in enumerate(top_k_sql_candidates):
                if first_idx < second_idx:
                    pair_sqls_to_eval.append(
                        (first_sql, second_sql)
                    )
        return pair_sqls_to_eval
    
    def _compare_sqls(
        self,
        sql_a: str,
        execution_result_table_str_a: str,
        sql_b: str,
        execution_result_table_str_b: str,
        *,
        question: str,
        evidence: str,
        db_type: str | None,
        database_schema_profile: str,
    ) -> Tuple[List[str], Dict[str, int]]:
        """
        Compare the two sqls.
        """
        prompt = PromptFactory.format_br_pair_selection_prompt(
            database_schema_profile,
            question,
            evidence,
            sql_a,
            execution_result_table_str_a,
            sql_b,
            execution_result_table_str_b,
            db_type=db_type,
        )
        
        votes, total_token_usage = self._extractor.extract_with_retry(
            llm=self._llm,
            messages=[{"role": "user", "content": prompt}],
            rule_parser=self._parse_llm_response,
            fix_end_token=self._llm.llm_config.fix_end_token,
            end_token="</result>",
            n=self._stage_config.evaluator_sampling_budget
        )
        
        return votes, total_token_usage
    
    def _update_win_matrix(self, sql_a_idx: int, sql_b_idx: int, votes: List[str], win_matrix: np.ndarray) -> None:
        """
        Update the win matrix.
        """
        for voter_idx, vote in enumerate(votes):
            if vote == "A":
                win_matrix[sql_a_idx, sql_b_idx, voter_idx] = 1
                win_matrix[sql_b_idx, sql_a_idx, voter_idx] = 0
            elif vote == "B":
                win_matrix[sql_b_idx, sql_a_idx, voter_idx] = 1
                win_matrix[sql_a_idx, sql_b_idx, voter_idx] = 0
            elif vote == "TIE":
                win_matrix[sql_a_idx, sql_b_idx, voter_idx] = 0.5
                win_matrix[sql_b_idx, sql_a_idx, voter_idx] = 0.5
            else:
                logger.error(f"Invalid vote: {vote}")
                raise ValueError(f"Invalid vote: {vote}")
    
    def _compute_robust_win_matrix(self, win_matrix: np.ndarray) -> float:
        """
        Calculate the robust vote.
        """
        sql_count, _, _ = win_matrix.shape
        robust_win_matrix = np.zeros((sql_count, sql_count))
        for sql_a_idx in range(sql_count):
            for sql_b_idx in range(sql_count):
                if sql_a_idx != sql_b_idx:
                    win_prob = np.mean(win_matrix[sql_a_idx, sql_b_idx, :])
                    robust_win_matrix[sql_a_idx, sql_b_idx] = win_prob
        return robust_win_matrix

    def _select_with_evidence_listwise(
        self,
        data_item: DataItem,
        top_k_sql_candidates: List[Tuple[str, str, float, float]],
        database_schema_profile: str,
    ) -> Tuple[str, Dict[str, int], str]:
        total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        dialect = "sqlite" if getattr(data_item, "db_type", None) in (None, "sqlite") else data_item.db_type
        risk_findings = [
            audit_semantic_risks(
                candidate[0],
                data_item.question,
                data_item.evidence,
                dialect=dialect,
                schema=data_item.database_schema_after_schema_linking,
            )
            for candidate in top_k_sql_candidates
        ]
        logger.info(
            f"Semantic-risk audit for item {data_item.question_id}: "
            f"{dict(enumerate(risk_findings))}; warnings are advisory only"
        )

        vote_count = max(1, self._stage_config.evaluator_sampling_budget)
        orders = self._get_position_balanced_orders(len(top_k_sql_candidates), vote_count)

        future_to_order = {}
        for order in orders:
            ordered_candidates = [
                (
                    top_k_sql_candidates[index][0],
                    top_k_sql_candidates[index][1],
                    top_k_sql_candidates[index][2],
                    risk_findings[index],
                )
                for index in order
            ]
            prompt = PromptFactory.format_evidence_listwise_selection_prompt(
                database_schema_profile,
                data_item.question,
                data_item.evidence,
                ordered_candidates,
            )
            future = self._inner_thread_pool_executor.submit(
                self._extractor.extract_with_retry,
                llm=self._llm,
                messages=[{"role": "user", "content": prompt}],
                rule_parser=lambda response, size=len(order): self._parse_listwise_response(response, size),
                fix_end_token=self._llm.llm_config.fix_end_token,
                end_token="</result>",
                n=1,
            )
            future_to_order[future] = order

        global_votes = []
        for future in as_completed(future_to_order):
            order = future_to_order[future]
            try:
                local_votes, token_usage = future.result()
                for key in total_token_usage:
                    total_token_usage[key] += token_usage[key]
                if local_votes:
                    global_votes.append(order[local_votes[0]])
            except Exception as exc:
                logger.error(f"Evidence-listwise selection vote failed: {exc}")

        if not global_votes:
            return top_k_sql_candidates[0][0], total_token_usage, "listwise_fallback_no_votes"

        vote_counts = Counter(global_votes)
        max_votes = max(vote_counts.values())
        vote_agreement = max_votes / len(global_votes)
        if vote_agreement < self._stage_config.min_vote_agreement:
            logger.info(
                f"Evidence-listwise votes for item {data_item.question_id} did not reach "
                f"the agreement gate: {dict(sorted(vote_counts.items()))}; "
                f"agreement {vote_agreement:.3f} < {self._stage_config.min_vote_agreement:.3f}; "
                "falling back to the top-support cluster"
            )
            return (
                top_k_sql_candidates[0][0],
                total_token_usage,
                "listwise_low_agreement_fallback",
            )

        tied_indices = [index for index, count in vote_counts.items() if count == max_votes]
        selected_index = min(
            tied_indices,
            key=lambda index: (-top_k_sql_candidates[index][2], top_k_sql_candidates[index][3], index),
        )
        logger.info(
            f"Evidence-listwise votes for item {data_item.question_id}: "
            f"{dict(sorted(vote_counts.items()))}; selected cluster {selected_index}"
        )
        return (
            top_k_sql_candidates[selected_index][0],
            total_token_usage,
            "listwise_supermajority_selected",
        )
    
    def _select_best_sql(self, data_item: DataItem) -> None:
        """
        Select the best sql based on the eval scores.
        """
        start_time = time.time()
        
        # Track token usage for this specific data item
        total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                
        top_k_sql_candidates = self._get_top_k_sql_candidates(data_item)
        controller_decision = (
            self._meta_controller.assess_selection(
                data_item,
                top_k_sql_candidates,
                self._stage_config.shortcut_consistency_score_threshold,
                self._stage_config.evaluator_sampling_budget,
            )
            if self._meta_controller is not None
            else None
        )
        
        if len(top_k_sql_candidates) == 0:
            logger.warning("No valid SQL candidates, backing to top-1 SQL")
            data_item.final_selected_sql = data_item.sql_candidates_after_revision[0] if data_item.sql_candidates_after_revision else "Error"
            data_item.sql_selection_time = time.time() - start_time
            data_item.sql_selection_llm_cost = total_token_usage
            data_item.total_time += data_item.sql_selection_time
            data_item.total_llm_cost = {
                "prompt_tokens": data_item.total_llm_cost["prompt_tokens"] + data_item.sql_selection_llm_cost["prompt_tokens"],
                "completion_tokens": data_item.total_llm_cost["completion_tokens"] + data_item.sql_selection_llm_cost["completion_tokens"],
                "total_tokens": data_item.total_llm_cost["total_tokens"] + data_item.sql_selection_llm_cost["total_tokens"],
            }
            if controller_decision is not None:
                self._meta_controller.record_stage_outcome(
                    controller_decision,
                    status="fallback_no_executable_cluster",
                    selected_sql=data_item.final_selected_sql,
                    token_usage=total_token_usage,
                )
            return
        
        if len(top_k_sql_candidates) == 1:
            logger.info("Only one valid SQL candidate, directly select it")
            data_item.final_selected_sql = top_k_sql_candidates[0][0]
            data_item.sql_selection_time = time.time() - start_time
            data_item.sql_selection_llm_cost = total_token_usage
            data_item.total_time += data_item.sql_selection_time
            data_item.total_llm_cost = {
                "prompt_tokens": data_item.total_llm_cost["prompt_tokens"] + data_item.sql_selection_llm_cost["prompt_tokens"],
                "completion_tokens": data_item.total_llm_cost["completion_tokens"] + data_item.sql_selection_llm_cost["completion_tokens"],
                "total_tokens": data_item.total_llm_cost["total_tokens"] + data_item.sql_selection_llm_cost["total_tokens"],
            }
            if controller_decision is not None:
                self._meta_controller.record_stage_outcome(
                    controller_decision,
                    status="single_cluster_shortcut",
                    selected_sql=data_item.final_selected_sql,
                    token_usage=total_token_usage,
                )
            return
        
        # shortcut case
        # if the consistency score of top-1 SQL is larger than a threshold, directly select it
        dialect = "sqlite" if getattr(data_item, "db_type", None) in (None, "sqlite") else data_item.db_type
        shortcut_risks = [
            audit_semantic_risks(
                candidate[0],
                data_item.question,
                data_item.evidence,
                dialect=dialect,
                schema=data_item.database_schema_after_schema_linking,
            )
            for candidate in top_k_sql_candidates
        ]
        minimum_shortcut_risk = min(len(findings) for findings in shortcut_risks)
        top_candidate_has_minimum_risk = len(shortcut_risks[0]) == minimum_shortcut_risk
        if (
            top_k_sql_candidates[0][2] >= self._stage_config.shortcut_consistency_score_threshold
            and top_candidate_has_minimum_risk
        ):
        # if top_k_sql_candidates[0][2] - top_k_sql_candidates[1][2] >= self._stage_config.shortcut_consistency_score_threshold:
            logger.info(f"Top-1 SQL candidate has a large consistency score: {top_k_sql_candidates[0][2]}, directly select it")
            # logger.info(f"Top-1 SQL candidate has a larger consistency score than top-2 SQL candidate ({top_k_sql_candidates[0][2]} vs. {top_k_sql_candidates[1][2]}), directly select the top-1 SQL")
            data_item.final_selected_sql = top_k_sql_candidates[0][0]
            data_item.sql_selection_time = time.time() - start_time
            data_item.sql_selection_llm_cost = total_token_usage
            data_item.total_time += data_item.sql_selection_time
            data_item.total_llm_cost = {
                "prompt_tokens": data_item.total_llm_cost["prompt_tokens"] + data_item.sql_selection_llm_cost["prompt_tokens"],
                "completion_tokens": data_item.total_llm_cost["completion_tokens"] + data_item.sql_selection_llm_cost["completion_tokens"],
                "total_tokens": data_item.total_llm_cost["total_tokens"] + data_item.sql_selection_llm_cost["total_tokens"],
            }
            if controller_decision is not None:
                self._meta_controller.record_stage_outcome(
                    controller_decision,
                    status="consensus_shortcut",
                    selected_sql=data_item.final_selected_sql,
                    token_usage=total_token_usage,
                )
            return
        if top_k_sql_candidates[0][2] >= self._stage_config.shortcut_consistency_score_threshold:
            logger.info(
                f"Consistency shortcut bypassed for item {data_item.question_id}: "
                f"top risk {shortcut_risks[0]}, minimum finding count {minimum_shortcut_risk}"
            )
        
        if self._stage_config.selection_strategy == "evidence_listwise":
            database_schema_profile = get_schema_service().build_schema_profile(
                data_item.database_schema_after_schema_linking,
                include_value_statistics=True,
                include_value_examples=True,
            )
            selected_sql, listwise_token_usage, selection_status = self._select_with_evidence_listwise(
                data_item,
                top_k_sql_candidates,
                database_schema_profile,
            )
            data_item.final_selected_sql = selected_sql
            total_token_usage = listwise_token_usage
            data_item.sql_selection_time = time.time() - start_time
            data_item.sql_selection_llm_cost = total_token_usage
            data_item.total_time += data_item.sql_selection_time
            data_item.total_llm_cost = {
                "prompt_tokens": data_item.total_llm_cost["prompt_tokens"] + total_token_usage["prompt_tokens"],
                "completion_tokens": data_item.total_llm_cost["completion_tokens"] + total_token_usage["completion_tokens"],
                "total_tokens": data_item.total_llm_cost["total_tokens"] + total_token_usage["total_tokens"],
            }
            if controller_decision is not None:
                self._meta_controller.record_stage_outcome(
                    controller_decision,
                    status=selection_status,
                    selected_sql=data_item.final_selected_sql,
                    token_usage=total_token_usage,
                )
            return

        # using pair-wise comparison to select the best sql
        database_schema_profile = get_schema_service().build_schema_profile(
            data_item.database_schema_after_schema_linking,
            include_value_statistics=True,
            include_value_examples=True,
        )
        db_type = getattr(data_item, "db_type", None)
        win_matrix = np.zeros((len(top_k_sql_candidates), len(top_k_sql_candidates), self._stage_config.evaluator_sampling_budget))
        sql_to_idx = {sql[0]: idx for idx, sql in enumerate(top_k_sql_candidates)}
        pair_sqls_to_eval = self._get_pair_sqls_to_eval(top_k_sql_candidates)
        
        # Parallelize the pairwise comparisons
        has_failure = False
        future_to_pair = {
            self._inner_thread_pool_executor.submit(
                self._compare_sqls,
                sql_a[0],
                sql_a[1],
                sql_b[0],
                sql_b[1],
                question=data_item.question,
                evidence=data_item.evidence,
                db_type=db_type,
                database_schema_profile=database_schema_profile,
            ): (sql_a, sql_b)
            for sql_a, sql_b in pair_sqls_to_eval
        }
        
        for future in as_completed(future_to_pair):
            sql_a, sql_b = future_to_pair[future]
            try:
                votes, token_usage = future.result()
                self._update_win_matrix(sql_to_idx[sql_a[0]], sql_to_idx[sql_b[0]], votes, win_matrix)
                total_token_usage["prompt_tokens"] += token_usage["prompt_tokens"]
                total_token_usage["completion_tokens"] += token_usage["completion_tokens"]
                total_token_usage["total_tokens"] += token_usage["total_tokens"]
            except Exception as e:
                logger.error(f"Error comparing SQLs {sql_a[0]} and {sql_b[0]}: {e}")
                has_failure = True
        
        # If any comparison failed, set result to None
        if has_failure:
            logger.error(f"Some SQL comparisons failed for item {data_item.question_id}, setting final_selected_sql to None")
            data_item.final_selected_sql = None
        else:
            robust_win_matrix = self._compute_robust_win_matrix(win_matrix)
            ranking_scores = np.mean(robust_win_matrix, axis=1)
            score_weights = np.array([sql[2] for sql in top_k_sql_candidates]) / np.sum(np.array([sql[2] for sql in top_k_sql_candidates]))
            ranking_scores = ranking_scores * score_weights
            ranking = np.argsort(-ranking_scores)
            data_item.final_selected_sql = top_k_sql_candidates[ranking[0]][0]
        
        data_item.sql_selection_time = time.time() - start_time
        data_item.sql_selection_llm_cost = total_token_usage
        data_item.total_time += data_item.sql_selection_time
        data_item.total_llm_cost = {
            "prompt_tokens": data_item.total_llm_cost["prompt_tokens"] + data_item.sql_selection_llm_cost["prompt_tokens"],
            "completion_tokens": data_item.total_llm_cost["completion_tokens"] + data_item.sql_selection_llm_cost["completion_tokens"],
            "total_tokens": data_item.total_llm_cost["total_tokens"] + data_item.sql_selection_llm_cost["total_tokens"],
        }
        if controller_decision is not None:
            self._meta_controller.record_stage_outcome(
                controller_decision,
                status="pairwise_failed" if has_failure else "pairwise_selected",
                selected_sql=data_item.final_selected_sql,
                token_usage=total_token_usage,
            )
    
    def run(self):
        future_to_item = {}
        for data_item in self._dataset:
            if data_item.is_stage_complete("sql_selection"):
                logger.info(f"Skipping data item {data_item.question_id} because it has already been selected")
                continue
            future = self._thread_pool_executor.submit(self._select_best_sql, data_item)
            future_to_item[future] = data_item
        for idx, future in tqdm(enumerate(as_completed(future_to_item), start=1), total=len(future_to_item), desc="Selecting Best SQL"):
            future.result()
            self._artifact_store.record_item(future_to_item[future])
            log_progress("Selecting Best SQL", idx, len(future_to_item), self._progress_log_interval, previous_completed=idx - 1)
            if should_checkpoint(idx, self._checkpoint_interval):
                self.save_result()
        logger.info("Selecting Best SQL completed")
        
        # Validate that all required fields are filled
        self._artifact_store.flush()
        validate_pipeline_step(self._dataset, "sql_selection")
        self.save_result(materialize_snapshot=True)
        
        self._clean_up()

    def save_result(self, materialize_snapshot: bool = False):
        self._artifact_store.flush()
        if materialize_snapshot:
            save_dataset(self._dataset, self._stage_config.save_path)
            self._artifact_store.cleanup()

    def _clean_up(self):
        if self._thread_pool_executor is not None:
            self._thread_pool_executor.shutdown(wait=True)
            self._thread_pool_executor = None
        if self._inner_thread_pool_executor is not None:
            self._inner_thread_pool_executor.shutdown(wait=True)
            self._inner_thread_pool_executor = None
        if self._artifact_store is not None:
            self._artifact_store.close()
        reset_execution_service()
        reset_schema_service()
        self._llm = None
        self._extractor = None
        self._dataset = None
