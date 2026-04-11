"""
Dataset Analysis for training data
"""

import json
from collections import Counter
from typing import Dict, Any


def analyze_dataset(dataset_file: str, test_size: float = 0.10) -> Dict[str, Any]:
    """
    Analyze the uploaded dataset and return statistics.

    Returns:
        dict with keys:
        - total_samples: int
        - tool_call_samples: int
        - no_tool_call_samples: int
        - test_samples: int (how many will be reserved for test)
        - train_samples: int
        - tool_distribution: dict of tool_name -> count
        - sample_instructions: list of first 5 instructions
    """
    with open(dataset_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    samples = [json.loads(line) for line in lines if line.strip()]

    total = len(samples)
    tool_call_count = 0
    tool_counter = Counter()
    instructions = []

    for sample in samples:
        output = sample.get("output", {})
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except:
                output = {}

        # Get instruction
        instruction = sample.get("instruction", "")
        if instruction and len(instructions) < 5:
            instructions.append(
                instruction[:100] + "..." if len(instruction) > 100 else instruction
            )

        # Check tool calls
        tool_calls = output.get("tool_calls", [])
        if tool_calls and len(tool_calls) > 0:
            tool_call_count += 1
            # Count each tool
            for tc in tool_calls:
                if isinstance(tc, dict):
                    func = tc.get("function", {})
                    if isinstance(func, dict):
                        tool_name = func.get("name", "unknown")
                        tool_counter[tool_name] += 1

    no_tool_call_count = total - tool_call_count

    # Calculate test split
    if isinstance(test_size, float) and 0 < test_size < 1:
        test_count = int(tool_call_count * test_size)
    else:
        test_count = int(test_size)

    train_count = total - test_count

    return {
        "total_samples": total,
        "tool_call_samples": tool_call_count,
        "no_tool_call_samples": no_tool_call_count,
        "test_samples": test_count,
        "train_samples": train_count,
        "tool_distribution": dict(tool_counter),
        "sample_instructions": instructions,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        result = analyze_dataset(sys.argv[1])
        print(json.dumps(result, indent=2))
