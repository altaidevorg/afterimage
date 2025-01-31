import json
import warnings
from collections import Counter
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from typing import Dict, List
from .types import (
    ConversationWithContext,
    EvaluatedConversationWithContext,
    GradeSchema,
)
import google.generativeai as genai
from tqdm import tqdm
from threading import Lock

from .common import default_model_name, default_safety_settings
from .prompts import default_evaluator_prompt
from .types import EvaluationSchema
from .key_management import SmartKeyPool


class SyntheticDatasetEvaluator:
    def __init__(
        self,
        api_key: str | SmartKeyPool,
        model_name: str = None,
        safety_settings: List = None,
    ):
        """Initialize the evaluator with model configurations.

        Args:
            api_key: Either a single API key string or a SmartKeyPool instance
            model_name: Model name to use
            safety_settings: Safety settings for the model
        """
        self.key_pool = (
            api_key
            if isinstance(api_key, SmartKeyPool)
            else SmartKeyPool.from_single_key(api_key)
        )
        self.model_name = model_name if model_name else default_model_name
        self.safety_settings = (
            safety_settings if safety_settings else default_safety_settings
        )

    def evaluate_row(
        self, row: ConversationWithContext
    ) -> EvaluatedConversationWithContext:
        """Evaluate a single row using the LLM."""
        try:
            api_key = self.key_pool.get_next_key()
            genai.configure(api_key=api_key)

            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=default_evaluator_prompt,
                safety_settings=self.safety_settings,
            )
            row_dict = (
                row.model_dump() if isinstance(row, ConversationWithContext) else row
            )

            # Initialize lists to store evaluations and scores for each turn
            evaluations = []
            final_scores = []

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
                    context=row_dict["context"],
                    response=row_dict["conversations"][i + 1]["content"],
                )

                evaluation_output = model.generate_content(
                    compiled_prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=EvaluationSchema,
                    ),
                ).text

                try:
                    evaluation = json.loads(evaluation_output)
                    assert len(evaluation) == 7
                except Exception as e:
                    print(e)
                    return self.evaluate_row(row)
                else:
                    final_score = 50 + sum(
                        [v["score"] for v in evaluation.values() if isinstance(v, dict)]
                    )
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

            return EvaluatedConversationWithContext(
                evaluation=overall_evaluation, final_score=avg_final_score, **row.dict()
            )
        except Exception as e:
            self.key_pool.report_error(api_key)
            raise e

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
