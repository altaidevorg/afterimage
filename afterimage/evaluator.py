"""Async conversation judging (LLM + embeddings)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .evaluation import (
    AggregationMode,
    CompositeEvaluator,
    CoherenceEvaluator,
    EvaluationMetric,
    EvaluationResult,
    FactualityEvaluator,
    GroundingEvaluator,
    HelpfulnessEvaluator,
    RelevanceEvaluator,
)
from .key_management import SmartKeyPool
from .monitoring import GenerationMonitor
from .providers import LLMProvider
from .providers.embedding_providers import EmbeddingProvider, EmbeddingProviderFactory
from .types import (
    ConversationWithContext,
    EvaluatedConversationWithContext,
    EvaluationEntrySchema,
    EvaluationSchema,
    GradeSchema,
    ModelProviderName,
)


_OPENROUTER_EMBEDDINGS_BASE = "https://openrouter.ai/api/v1"


def default_embedding_provider_config(
    model_provider_name: ModelProviderName,
) -> dict[str, Any]:
    """Default embedding backend for auto-improve when none is supplied.

    Uses the same API vendor as chat when possible; DeepSeek has no public
    embedding API in this stack, so a local SentenceTransformer worker pool is
    used. For ``local`` chat, the same process-based default applies.

    Args:
        model_provider_name: Active LLM provider for generation.

    Returns:
        Config dict for :class:`~afterimage.providers.embedding_providers.EmbeddingProviderFactory`.
    """
    if model_provider_name == "gemini":
        return {"type": "gemini", "model": "gemini-embedding-001"}
    if model_provider_name in ("deepseek", "local"):
        return {
            "type": "process",
            "model": "altaidevorg/bge-m3-distill-8l",
            "workers": 1,
        }
    if model_provider_name == "openrouter":
        return {
            "type": "openai",
            "model": "openai/text-embedding-3-small",
            "base_url": _OPENROUTER_EMBEDDINGS_BASE,
        }
    return {"type": "openai", "model": "text-embedding-3-small"}


@dataclass
class ConversationJudgeConfig:
    """Tuning knobs for :class:`ConversationJudge`.

    Grade bands (overall score): ``>= perfect_threshold`` → PERFECT;
    ``>= good_threshold`` → GOOD; ``>= needs_improvement_threshold`` → NEEDS_IMPROVEMENT;
    ``>= bad_threshold`` → BAD; else NOT_ACCEPTABLE.
    """

    min_acceptable_score: float = 0.58
    aggregation_mode: AggregationMode = AggregationMode.MEAN
    metric_weights: Optional[Dict[EvaluationMetric, float]] = None
    perfect_threshold: float = 0.88
    good_threshold: float = 0.72
    needs_improvement_threshold: float = 0.52
    bad_threshold: float = 0.32


class ConversationJudge:
    """Configurable async judge: embedding metrics + LLM rubrics.

    Produces :class:`~afterimage.types.EvaluatedConversationWithContext` with
    :class:`~afterimage.types.EvaluationSchema` suitable for storage and the
    generator auto-improve loop.
    """

    def __init__(
        self,
        llm: LLMProvider,
        embedding_provider: EmbeddingProvider,
        monitor: Optional[GenerationMonitor] = None,
        *,
        config: Optional[ConversationJudgeConfig] = None,
    ):
        """Build the composite judge.

        Args:
            llm: Provider for factuality and helpfulness (structured JSON).
            embedding_provider: Async embeddings for coherence, grounding, relevance.
            monitor: Optional metrics sink.
            config: Aggregation, thresholds, and grade cutoffs.
        """
        self._cfg = config or ConversationJudgeConfig()
        self.monitor = monitor
        self._llm = llm
        self._embedding = embedding_provider

        evaluators: list = [
            (CoherenceEvaluator(embedding_provider, monitor=monitor), 1.0),
            (FactualityEvaluator(llm, monitor=monitor), 1.0),
            (GroundingEvaluator(embedding_provider, monitor=monitor), 1.0),
            (HelpfulnessEvaluator(llm, monitor=monitor), 1.0),
            (RelevanceEvaluator(embedding_provider, monitor=monitor), 1.0),
        ]

        self._composite = CompositeEvaluator(
            evaluators,
            min_acceptable_score=self._cfg.min_acceptable_score,
            aggregation_mode=self._cfg.aggregation_mode,
            metric_weights=self._cfg.metric_weights,
        )

    @classmethod
    def from_factory(
        cls,
        llm: LLMProvider,
        *,
        key_pool: SmartKeyPool,
        model_provider_name: ModelProviderName,
        embedding_provider_config: Optional[dict[str, Any]] = None,
        monitor: Optional[GenerationMonitor] = None,
        config: Optional[ConversationJudgeConfig] = None,
    ) -> ConversationJudge:
        """Convenience: build embedding provider from config + shared key pool."""
        cfg = (
            embedding_provider_config
            if embedding_provider_config is not None
            else default_embedding_provider_config(model_provider_name)
        )
        embed = EmbeddingProviderFactory.create(cfg, key_pool=key_pool)
        return cls(llm, embed, monitor=monitor, config=config)

    def _score_to_grade(self, overall: float) -> GradeSchema:
        c = self._cfg
        if overall >= c.perfect_threshold:
            return GradeSchema.PERFECT
        if overall >= c.good_threshold:
            return GradeSchema.GOOD
        if overall >= c.needs_improvement_threshold:
            return GradeSchema.NEEDS_IMPROVEMENT
        if overall >= c.bad_threshold:
            return GradeSchema.BAD
        return GradeSchema.NOT_ACCEPTABLE

    def _result_to_evaluation_schema(
        self, result: EvaluationResult, grade: GradeSchema
    ) -> EvaluationSchema:
        def entry(metric: EvaluationMetric) -> EvaluationEntrySchema:
            return EvaluationEntrySchema(
                score=float(result.scores.get(metric, 0.0)),
                feedback=result.feedback.get(metric, ""),
            )

        return EvaluationSchema(
            coherence=entry(EvaluationMetric.COHERENCE),
            factuality=entry(EvaluationMetric.FACTUALITY),
            grounding=entry(EvaluationMetric.GROUNDING),
            helpfulness=entry(EvaluationMetric.HELPFULNESS),
            relevance=entry(EvaluationMetric.RELEVANCE),
            overall_grade=grade,
        )

    async def aevaluate_row(
        self, conversation: ConversationWithContext
    ) -> EvaluatedConversationWithContext:
        """Evaluate one conversation asynchronously."""
        start = time.time()
        try:
            result = await self._composite.aevaluate(conversation)
            grade = self._score_to_grade(result.overall_score)
            evaluation = self._result_to_evaluation_schema(result, grade)

            if self.monitor:
                self.monitor.track_evaluation(
                    duration=time.time() - start,
                    success=True,
                    evaluator_type=self.__class__.__name__,
                    scores={"overall": result.overall_score, **result.scores},
                )

            return EvaluatedConversationWithContext(
                evaluation=evaluation,
                final_score=result.overall_score,
                **conversation.model_dump(),
            )
        except Exception as e:
            if self.monitor:
                self.monitor.track_evaluation(
                    duration=time.time() - start,
                    success=False,
                    evaluator_type=self.__class__.__name__,
                    scores={},
                    error=str(e),
                    error_type=e.__class__.__name__,
                )
            raise

    async def aclose(self) -> None:
        """Release embedding provider resources when applicable."""
        await self._embedding.aclose()
