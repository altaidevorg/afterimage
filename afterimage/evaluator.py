import json
import warnings
from collections import Counter
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from typing import Any, Dict, List
from .types import ConversationWithContext, EvaluatedConversationWithContext
import google.generativeai as genai
from tqdm import tqdm

from .common import default_model_name, default_safety_settings
from .prompts import default_evaluator_prompt
from .types import EvaluationSchema


class SyntheticDatasetEvaluator:
    def __init__(
        self, api_key: str, model_name: str = None, safety_settings: List = None
    ):
        """Initialize the evaluator with model configurations."""
        assert api_key is not None, "You must provide an API key"
        self.api_key = api_key
        self.model_name = model_name if model_name else default_model_name
        self.safety_settings = (
            safety_settings if safety_settings else default_safety_settings
        )
        genai.configure(api_key=self.api_key)

    def evaluate_row(
        self, row: ConversationWithContext
    ) -> EvaluatedConversationWithContext:
        """Evaluate a single row using the LLM."""
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=default_evaluator_prompt,
            safety_settings=self.safety_settings,
        )
        row_dict = row.model_dump() if isinstance(row, ConversationWithContext) else row
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
            instruction=row_dict["conversations"][0]["content"],
            context=row_dict["context"],
            response=row_dict["conversations"][1]["content"],
        )
        evaluation_output = model.generate_content(
            compiled_prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json", response_schema=EvaluationSchema
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
                [v["score"] for v in evaluation.values() if isinstance(v, Dict)]
            )
            evaluation["final_score"] = final_score

            return EvaluatedConversationWithContext(
                evaluation=evaluation, final_score=final_score, **row.dict()
            )

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

        def save_evaluations(evaluations):
            if save_to and evaluations:
                with open(save_to, "a+", encoding="utf8") as f:
                    for evaluation in evaluations:
                        f.write(evaluation.model_dump_json() + "\n")

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
                        if show_summary:
                            evaluations.append(
                                evaluated_conversation.evaluation["overall_grade"]
                            )
                            total_score += evaluated_conversation.evaluation[
                                "final_score"
                            ]
                        pbar.update(1)
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
