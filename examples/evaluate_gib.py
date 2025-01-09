import os
from afterimage import SyntheticDatasetEvaluator


if __name__ == "__main__":
    api_key = os.environ["GEMINI_API_KEY"]
    evaluator = SyntheticDatasetEvaluator(api_key)
    evaluator.evaluate_dataset(
        dataset_path="./gib-ds.jsonl",
        save_to="evaluations.jsonl",
        max_workers=4,
    )
