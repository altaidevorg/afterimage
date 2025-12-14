# Evaluation Framework

Generating synthetic data is only half the battle. You also need to ensure that the data is high quality, faithful to your source material, and diverse. Afterimage provides a flexible **Evaluation Framework** to assess your datasets.

## Overview

There are two main approaches to evaluation in Afterimage:

1.  **Simple Evaluation (`SimpleSyntheticDatasetEvaluator`)**: Uses an LLM as a judge to rate conversations based on a rubric. Good for general quality checks.
2.  **Hybrid Evaluation (`HybridSyntheticDatasetEvaluator`)**: Combines embedding-based metrics (for semantic similarity) with LLM-based verification. This is more robust and cheaper for checking grounding.

## Metrics

Afterimage comes with several built-in evaluators that target different aspects of conversation quality.

### 1. Coherence (`CoherenceEvaluator`)
*   **Method**: Embedding alignment + LLM check.
*   **Goal**: Ensure the question and answer make sense together.
*   **Check**: Does the answer actually address the question asked?

### 2. Grounding (`GroundingEvaluator`)
*   **Method**: Semantic similarity (embeddings).
*   **Goal**: Ensure the answer is derived *only* from the provided context (RAG).
*   **Check**: Is the answer supported by the retrieved document chunks?
*   *Note: This is critical for preventing hallucinations.*

### 3. Relevance (`RelevanceEvaluator`)
*   **Method**: Embedding alignment.
*   **Goal**: Ensure the user's question is actually about the topic we wanted them to ask about.
*   **Check**: Does the generated question align with the source document?

### 4. Factuality (`FactualityEvaluator`)
*   **Method**: LLM-as-judge.
*   **Goal**: Verify factual accuracy against a gold standard or general knowledge.

### 5. Helpfulness (`HelpfulnessEvaluator`)
*   **Method**: LLM-as-judge.
*   **Goal**: Assess if the answer is useful, polite, and complete.

## Usage Guide

You can run evaluations on any `ConversationWithContext` object or a saved `.jsonl` dataset.

### Running Basic Evaluation

The simplest way is to use the `SimpleSyntheticDatasetEvaluator` which runs a standard "LLM-as-judge" prompt.

```python
import asyncio
import os
from afterimage.evaluation import SimpleSyntheticDatasetEvaluator

async def main():
    api_key = os.getenv("GEMINI_API_KEY")
    evaluator = SimpleSyntheticDatasetEvaluator(api_key=api_key)

    # Evaluate a JSONL file
    results = await evaluator.evaluate_dataset("my_conversations.jsonl")
    
    # Print summary
    print(f"Average Score: {results['average_score']}")
    print(f"Pass Rate: {results['pass_rate']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Running Hybrid Evaluation

For a more rigorous check, use the `HybridSyntheticDatasetEvaluator` and specify which metrics you want.

```python
from afterimage.evaluation import (
    HybridSyntheticDatasetEvaluator, 
    CoherenceEvaluator, 
    GroundingEvaluator
)

# Initialize specific evaluators
coherence = CoherenceEvaluator()
grounding = GroundingEvaluator()

# Create hybrid evaluator
evaluator = HybridSyntheticDatasetEvaluator(
    evaluators=[coherence, grounding]
)

# Run evaluation
results = await evaluator.evaluate_dataset("rag_dataset.jsonl")
```

### Interpreting reports
The evaluation results will typically contain:
*   **Score**: A numerical value (0.0 to 1.0 or 1 to 5).
*   **Feedback**: A text explanation of why that score was given.
*   **Needs Regeneration**: A boolean flag suggesting if this sample should be discarded.

You can use these signals to filter your dataset, keeping only the high-quality examples for fine-tuning.

---
[Previous: Persona Generation](persona_generation.md) | [Next: Structured Generation](structured_generation.md)
