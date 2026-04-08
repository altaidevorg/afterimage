# Evaluation Framework

Generating synthetic data is only half the battle. You also need to ensure that the data is high quality, faithful to your source material, and diverse. Afterimage provides an **async evaluation stack** built around **`ConversationJudge`**: embedding-based metrics plus LLM-as-judge rubrics, composed with configurable aggregation and grade thresholds.

## Overview

- **`ConversationJudge`** (`afterimage.evaluator`): Primary entry point. Runs `CoherenceEvaluator`, `GroundingEvaluator`, `RelevanceEvaluator` (via :class:`~afterimage.providers.embedding_providers.EmbeddingProvider`) and `FactualityEvaluator`, `HelpfulnessEvaluator` (via :class:`~afterimage.providers.llm_providers.LLMProvider` structured output). Returns :class:`~afterimage.types.EvaluatedConversationWithContext` with :class:`~afterimage.types.EvaluationSchema`.
- **`CompositeEvaluator`** (`afterimage.evaluation`): Combines async sub-evaluators in parallel; supports :class:`~afterimage.evaluation.base.AggregationMode` (`MEAN`, `WEIGHTED_MEAN`, `MIN`).
- **Generator integration**: `ConversationGenerator(..., auto_improve=True)` builds a judge automatically using `default_embedding_provider_config(model_provider_name)` when you do not pass `embedding_provider` or `embedding_provider_config`.

The legacy Gemini one-shot path and `evaluator_method="simple" | "hybrid"` have been removed in favor of this single pipeline.

## Metrics

### Coherence (`CoherenceEvaluator`)

*   **Method**: Cosine similarity between embeddings of each user turn and the following assistant turn.
*   **Goal**: Question and answer are semantically aligned.

### Grounding (`GroundingEvaluator`)

*   **Method**: Cosine similarity between `response_context` embedding and each assistant reply.
*   **Goal**: Answers stay close to the provided response context (RAG-style).

### Relevance (`RelevanceEvaluator`)

*   **Method**: Cosine similarity between `instruction_context` embedding and each user question.
*   **Goal**: Questions match the instruction-side context.

### Factuality / Helpfulness (`FactualityEvaluator`, `HelpfulnessEvaluator`)

*   **Method**: LLM structured output (`agenerate_structured`) with per-item scores in `[0, 1]` and short feedback.

## Usage

### Standalone judge

```python
import asyncio
from afterimage import ConversationJudge, LLMFactory, SmartKeyPool
from afterimage.providers import EmbeddingProviderFactory

async def main():
    pool = SmartKeyPool.from_single_key("YOUR_KEY")
    llm = LLMFactory.create("gemini", "gemini-2.0-flash", pool)
    embed = EmbeddingProviderFactory.create(
        {"type": "gemini", "model": "text-embedding-004"},
        key_pool=pool,
    )
    judge = ConversationJudge(llm=llm, embedding_provider=embed)
    row = ...  # ConversationWithContext
    out = await judge.aevaluate_row(row)
    print(out.evaluation.overall_grade, out.final_score)
    await judge.aclose()

asyncio.run(main())
```

### With `ConversationGenerator` (auto-improve)

Pass `auto_improve=True`. Optionally set `embedding_provider`, `embedding_provider_config`, or `judge_config` (:class:`~afterimage.evaluator.ConversationJudgeConfig`).

### Interpreting results

*   **Per-metric scores**: `0.0`–`1.0` in `evaluation.*.score`.
*   **overall_grade**: Derived from the composite overall score and configurable thresholds on `ConversationJudgeConfig`.
*   **needs_regeneration** (internal): Composite result when overall score is below `min_acceptable_score`; future hooks may use this with regeneration strategies.
