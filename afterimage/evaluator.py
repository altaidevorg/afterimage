import json
import warnings
from collections import Counter
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from typing import List, Optional
from .types import (
    ConversationWithContext,
    EvaluatedConversationWithContext,
    GradeSchema,
)
from google import genai
from google.genai import types
from tqdm import tqdm
from threading import Lock
import time

from .providers import LLMProvider
from .prompts import default_evaluator_prompt
from .types import EvaluationSchema
from .key_management import SmartKeyPool
from .evaluation import (
    BaseEvaluator,
    CompositeEvaluator,
    CoherenceEvaluator,
    FactualityEvaluator,
    GroundingEvaluator,
    HelpfulnessEvaluator,
    RelevanceEvaluator,
    EvaluationMetric,
    EvaluationResult,
)
from .monitoring import GenerationMonitor


class SimpleSyntheticDatasetEvaluator:
    """Evaluates synthetic conversations.

    This is maintained for backward compatibility.
    Consider using the new evaluation system for new code.
    """

    def __init__(
        self,
        api_key: str | SmartKeyPool,
        model_name=None,
        safety_settings=None,
        max_retries: int = 3,
        monitor: Optional[GenerationMonitor] = None,
    ):
        self.key_pool = (
            SmartKeyPool.from_single_key(api_key)
            if isinstance(api_key, str)
            else api_key
        )
        self.model_name = model_name
        self.safety_settings = safety_settings
        self.max_retries = max_retries
        self.monitor = monitor

    def evaluate_row(
        self, row: ConversationWithContext
    ) -> EvaluatedConversationWithContext:
        """Evaluate a single row using the LLM."""
        start_time = time.time()
        try:
            api_key = self.key_pool.get_next_key()
            client = (
                genai.Client(vertexai=False, api_key=api_key)
                if api_key
                else genai.Client()
            )

            row_dict = (
                row.model_dump() if isinstance(row, ConversationWithContext) else row
            )

            # Initialize lists to store evaluations and scores for each turn
            evaluations = []
            final_scores = []
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_tokens = 0

            # Process each conversation turn pair (instruction-response)
            for i in range(0, len(row_dict["conversations"]), 2):
                if i + 1 >= len(row_dict["conversations"]):
                    break

                prompt = """Here's the instruction-response-context combination that you are asked to evaluate based on the aforementioned criteria.

## Instruction
{instruction}
---------------------

## Context
{context}
---------------------

## Response
{response}"""
                compiled_prompt = prompt.format(
                    instruction=row_dict["conversations"][i]["content"],
                    context=row_dict.get("response_context", None)
                    or row_dict.get("instruction_context"),
                    response=row_dict["conversations"][i + 1]["content"],
                )

                evaluation_output = client.models.generate_content(
                    model=self.model_name,
                    config=types.GenerateContentConfig(
                        system_instruction=default_evaluator_prompt,
                        response_mime_type="application/json",
                        response_schema=EvaluationSchema,
                    ),
                    contents=compiled_prompt,
                )

                usage = getattr(evaluation_output, "usage_metadata", None)
                if usage is not None:
                    total_prompt_tokens += getattr(usage, "prompt_token_count", 0) or 0
                    total_completion_tokens += (
                        getattr(usage, "candidates_token_count", 0) or 0
                    )
                    total_tokens += getattr(usage, "total_token_count", 0) or 0

                try:
                    evaluation = evaluation_output.parsed.dict()
                    assert len(evaluation) == 6
                except Exception as e:
                    if self.monitor:
                        self.monitor.track_evaluation(
                            duration=time.time() - start_time,
                            success=False,
                            evaluator_type=self.__class__.__name__,
                            scores={},
                            error=str(e) + ": " + evaluation_output.text,
                            error_type=e.__class__.__name__,
                        )
                        raise
                    return self.evaluate_row(row)
                else:
                    final_score = 0.0
                    for v in evaluation.values():
                        if isinstance(v, dict):
                            v["score"] += 0.5
                            final_score += v["score"]

                    final_score = final_score / 5

                    evaluations.append(evaluation)
                    final_scores.append(final_score)

            # Calculate the average final score and use the worst grade as overall grade
            avg_final_score = (
                sum(final_scores) / len(final_scores) if final_scores else 0
            )
            overall_evaluation = evaluations[0].copy()  # Use first evaluation as base
            overall_evaluation["final_score"] = avg_final_score

            # Use the worst grade among all turns as the overall grade
            grades = [eval["overall_grade"] for eval in evaluations]
            if GradeSchema.NOT_ACCEPTABLE in grades:
                overall_evaluation["overall_grade"] = GradeSchema.NOT_ACCEPTABLE
            elif GradeSchema.BAD in grades:
                overall_evaluation["overall_grade"] = GradeSchema.BAD
            elif GradeSchema.NEEDS_IMPROVEMENT in grades:
                overall_evaluation["overall_grade"] = GradeSchema.NEEDS_IMPROVEMENT
            elif GradeSchema.GOOD in grades:
                overall_evaluation["overall_grade"] = GradeSchema.GOOD
            else:
                overall_evaluation["overall_grade"] = GradeSchema.PERFECT

            if self.monitor:
                self.monitor.track_evaluation(
                    duration=time.time() - start_time,
                    success=True,
                    evaluator_type=self.__class__.__name__,
                    scores={
                        "overall": avg_final_score,
                        **{
                            k: v
                            for k, v in overall_evaluation.items()
                            if k != "overall_grade"
                        },
                    },
                    prompt_token_count=total_prompt_tokens,
                    completion_token_count=total_completion_tokens,
                    total_token_count=total_tokens,
                    model_name=self.model_name,
                )

            return EvaluatedConversationWithContext(
                evaluation=overall_evaluation, final_score=avg_final_score, **row.dict()
            )
        except Exception as e:
            self.key_pool.report_error(api_key)
            if self.monitor:
                self.monitor.track_evaluation(
                    duration=time.time() - start_time,
                    success=False,
                    evaluator_type=self.__class__.__name__,
                    scores={},
                    error=str(e),
                    error_type=e.__class__.__name__,
                )
            raise

    def evaluate_dataset(
        self,
        dataset_path: str,
        save_to: str = None,
        show_summary: bool = True,
        max_workers: int = 4,
    ) -> None:
        """Evaluate a dataset concurrently."""
        with open(dataset_path, encoding="utf8") as f:
            rows = [json.loads(line.strip()) for line in f]

        n_rows = len(rows)
        pbar = tqdm(total=n_rows, desc="Evaluating...", unit="row")
        evaluations = []
        total_score = 0

        counter_lock = Lock()

        def save_evaluations(evaluations):
            if save_to and evaluations:
                with open(save_to, "a+", encoding="utf8") as f:
                    for evaluation in evaluations:
                        f.write(evaluation.model_dump_json() + "\n")

        def update_stats(evaluated_conversation):
            with counter_lock:
                if show_summary:
                    evaluations.append(
                        evaluated_conversation.evaluation["overall_grade"]
                    )
                    nonlocal total_score
                    total_score += evaluated_conversation.evaluation["final_score"]
                pbar.update(1)

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self.evaluate_row, row) for row in rows]

                for future in as_completed(futures):
                    try:
                        evaluated_conversation = future.result()
                    except Exception as e:
                        if not isinstance(e, CancelledError):
                            warnings.warn(f"Exception in future: {e}")
                    else:
                        update_stats(evaluated_conversation)
                        if save_to:
                            save_evaluations([evaluated_conversation])

        except KeyboardInterrupt:
            warnings.warn("Interrupted! Waiting for graceful shutdown...")
        finally:
            pbar.close()

        if show_summary:
            avg_score = total_score / len(evaluations)
            counter = Counter(evaluations)
            print(f"avg. score: {avg_score:.2f}")
            for grade, count in counter.most_common():
                print(f"    - {grade}: {count}")


class HybridSyntheticDatasetEvaluator:
    """Modern evaluation system with SimpleSyntheticDatasetEvaluator-compatible interface."""

    def __init__(
        self,
        llm: LLMProvider,
        embedding_model: str = "altaidevorg/bge-m3-distill-8l",
        monitor: Optional[GenerationMonitor] = None,
    ):
        """Initialize evaluator with composite evaluation system.

        Args:
            llm: LLM provider for factuality and helpfulness checks
            embedding_model: Model name for embedding-based evaluations
            monitor: Monitoring instance
        """
        self.monitor = monitor
        self.evaluator = CompositeEvaluator(
            [
                (CoherenceEvaluator(embedding_model, monitor=monitor), 1.0),
                (FactualityEvaluator(llm, monitor=monitor), 1.0),
                (GroundingEvaluator(embedding_model, monitor=monitor), 1.0),
                (HelpfulnessEvaluator(llm, monitor=monitor), 1.0),
                (RelevanceEvaluator(embedding_model, monitor=monitor), 1.0),
            ]
        )

    def evaluate_row(
        self, conversation: ConversationWithContext
    ) -> EvaluatedConversationWithContext:
        """Evaluate a single conversation using the composite evaluator.

        Args:
            conversation: Conversation to evaluate

        Returns:
            Evaluated conversation with detailed metrics
        """
        start_time = time.time()
        try:
            result = self.evaluator.evaluate(conversation)
            # Convert evaluation results to EvaluationSchema format
            evaluation = {
                "coherence": {
                    "score": result.scores.get(EvaluationMetric.COHERENCE, 0),
                    "feedback": result.feedback.get(EvaluationMetric.COHERENCE, ""),
                },
                "grounding": {
                    "score": result.scores.get(EvaluationMetric.GROUNDING, 0),
                    "feedback": result.feedback.get(EvaluationMetric.GROUNDING, ""),
                },
                "relevance": {
                    "score": result.scores.get(EvaluationMetric.RELEVANCE, 0),
                    "feedback": result.feedback.get(EvaluationMetric.RELEVANCE, ""),
                },
                "factuality": {
                    "score": result.scores.get(EvaluationMetric.FACTUALITY, 0),
                    "feedback": result.feedback.get(EvaluationMetric.FACTUALITY, ""),
                },
                "helpfulness": {
                    "score": result.scores.get(EvaluationMetric.HELPFULNESS, 0),
                    "feedback": result.feedback.get(EvaluationMetric.HELPFULNESS, ""),
                },
                "overall_grade": (
                    GradeSchema.PERFECT
                    if result.overall_score >= 0.8
                    else GradeSchema.GOOD
                    if result.overall_score >= 0.05
                    else GradeSchema.NEEDS_IMPROVEMENT
                ),
            }

            if self.monitor:
                self.monitor.track_evaluation(
                    duration=time.time() - start_time,
                    success=True,
                    evaluator_type=self.__class__.__name__,
                    scores={
                        "overall": result.final_score,
                    },
                )

            return EvaluatedConversationWithContext(
                evaluation=evaluation,
                final_score=result.overall_score,
                **conversation.dict(),
            )

        except Exception as e:
            if self.monitor:
                self.monitor.track_evaluation(
                    duration=time.time() - start_time,
                    success=False,
                    evaluator_type=self.__class__.__name__,
                    scores={},
                    error=str(e),
                    error_type=e.__class__.__name__,
                )
            raise

    def evaluate_dataset(
        self, conversations: List[ConversationWithContext]
    ) -> List[EvaluatedConversationWithContext]:
        """Evaluate multiple conversations.

        Args:
            conversations: List of conversations to evaluate

        Returns:
            List of evaluated conversations
        """
        return [self.evaluate_row(conv) for conv in conversations]
