import json
import warnings
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError
from collections import Counter
from tqdm import tqdm
from typing import Dict, TypedDict, Any
import google.generativeai as genai
from .common import default_model_name, default_safety_settings
from .prompts import default_evaluator_prompt


class GradeSchema(str, Enum):
    GOOD = "good"
    NEEDS_IMPROVEMENT = "needs_improvement"
    PERFECT = "perfect"


class EvaluationEntrySchema(TypedDict):
    comment: str
    score: int


class EvaluationSchema(TypedDict):
    relevance: EvaluationEntrySchema
    grounding: EvaluationEntrySchema
    correctness: EvaluationEntrySchema
    completeness: EvaluationEntrySchema
    coherence: EvaluationEntrySchema
    usefulness: EvaluationEntrySchema
    overall_grade: GradeSchema


class SyntheticDatasetEvaluator:
    def __init__(
        self, api_key: str, model_name: str = None, safety_settings: Dict = None
    ):
        """Initialize the evaluator with model configurations."""
        assert api_key is not None, "You must provide an API key"
        self.api_key = api_key
        self.model_name = model_name if model_name else default_model_name
        self.safety_settings = (
            safety_settings if safety_settings else default_safety_settings
        )
        genai.configure(api_key=self.api_key)

    def evaluate_row(self, row: Dict[str, Any]) -> EvaluationSchema:
        """Evaluate a single row using the LLM."""
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=default_evaluator_prompt,
            safety_settings=self.safety_settings,
        )
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
            instruction=row["conversations"][0]["content"],
            context=row["context"],
            response=row["conversations"][1]["content"],
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
        except Exception:
            return self.evaluate_row(row)
        else:
            overall_grade = evaluation.pop("overall_grade")
            final_score = 50 + sum(
                [v["score"] for v in evaluation.values() if isinstance(v, Dict)]
            )
            row_copy = row.copy()
            row_copy["evaluation"] = evaluation
            row_copy["overall_grade"] = overall_grade
            row_copy["final_score"] = final_score

            return row_copy

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
                        f.write(json.dumps(evaluation, ensure_ascii=False) + "\n")

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self.evaluate_row, row) for row in rows]

                for future in as_completed(futures):
                    try:
                        evaluation = future.result()
                    except Exception as e:
                        if not isinstance(e, CancelledError):
                            warnings.warn(f"Exception in future: {e}")
                    else:
                        if show_summary:
                            evaluations.append(evaluation["overall_grade"])
                            total_score += evaluation["final_score"]
                        pbar.update(1)
                        if save_to:
                            save_evaluations([evaluation])

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
