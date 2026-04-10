"""
Custom Dataset Split - Separate tool call samples for testing
"""

import json
import random
from datasets import Dataset
from utils import load_jsonl_dataset, format_chat


def prepare_dataset_with_tool_call_test(
    dataset_file: str, tokenizer, tools_schema, test_size=0.10, seed: int = 42
):
    """
    Load dataset and separate tool call samples for testing.

    Training: Full dataset (including non-tool-call samples)
    Test: Only tool call samples (test_size amount)

    Args:
        test_size: int or float. If float (0-1), use as percentage. If int, use as absolute number.
    """
    print(f"\n[INFO] Processing dataset: {dataset_file}")

    # Load raw dataset
    raw_dataset = load_jsonl_dataset(dataset_file)

    # Separate tool call and non-tool-call samples
    tool_call_indices = []
    no_tool_call_indices = []

    for idx, sample in enumerate(raw_dataset):
        output = sample.get("output", {})
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                output = {}

        tool_calls = output.get("tool_calls", [])
        if tool_calls and len(tool_calls) > 0:
            tool_call_indices.append(idx)
        else:
            no_tool_call_indices.append(idx)

    print(f"   With tool calls: {len(tool_call_indices)} samples")
    print(f"   Without tool calls: {len(no_tool_call_indices)} samples")

    # Calculate test size
    if isinstance(test_size, float) and 0 < test_size < 1:
        # Given as percentage
        actual_test_size = int(len(tool_call_indices) * test_size)
    else:
        # Given as absolute number
        actual_test_size = int(test_size)

    print(
        f"[INFO] Allocating {actual_test_size} tool call samples for testing ({test_size * 100 if isinstance(test_size, float) else actual_test_size}%)..."
    )

    # Separate test samples from tool call samples
    random.seed(seed)
    random.shuffle(tool_call_indices)

    test_indices = set(tool_call_indices[:actual_test_size])
    train_indices = set(range(len(raw_dataset))) - test_indices

    print(f"   Test allocated: {len(test_indices)} tool call samples")
    print(f"   Training allocated: {len(train_indices)} samples")

    # Create train and test datasets
    train_samples = [raw_dataset[i] for i in sorted(train_indices)]
    test_samples = [raw_dataset[i] for i in sorted(test_indices)]

    train_dataset = Dataset.from_list(train_samples)
    test_dataset = Dataset.from_list(test_samples)

    # Apply formatting
    train_dataset = train_dataset.map(
        lambda sample: format_chat(sample, tokenizer, tools_schema),
        desc="Formatting train dataset",
    )

    test_dataset = test_dataset.map(
        lambda sample: format_chat(sample, tokenizer, tools_schema),
        desc="Formatting test dataset",
    )

    print(f"[OK] Training Data: {len(train_dataset)} samples (all data)")
    print(f"[OK] Test Data: {len(test_dataset)} samples (tool calls only)")

    return {
        "train": train_dataset,
        "test": test_dataset,
        "test_raw": test_samples,  # Raw format for evaluation
    }
