"""Core types and async composite evaluation."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

from ..types import ConversationWithContext


class EvaluationMetric(str, Enum):
    """Available evaluation metrics."""

    COHERENCE = "coherence"
    GROUNDING = "grounding"
    RELEVANCE = "relevance"
    FACTUALITY = "factuality"
    HELPFULNESS = "helpfulness"
    SAFETY = "safety"
    FORMATTING = "formatting"


class AggregationMode(str, Enum):
    """How per-metric scores are combined into a single overall score."""

    MEAN = "mean"
    """Unweighted arithmetic mean over all reported metrics."""

    WEIGHTED_MEAN = "weighted_mean"
    """Weighted average using :attr:`CompositeEvaluator.metric_weights` (default weight 1.0)."""

    MIN = "min"
    """Minimum of all metric scores (strictest)."""


@dataclass
class EvaluationResult:
    """Aggregated scores and feedback for one conversation."""

    scores: Dict[EvaluationMetric, float]
    feedback: Dict[EvaluationMetric, str]
    overall_score: float
    needs_regeneration: bool
    regeneration_strategy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def final_score(self) -> float:
        """Alias for :attr:`overall_score` (monitoring and legacy call sites)."""
        return self.overall_score


class BaseEvaluator(Protocol):
    """Protocol for async conversation evaluators."""

    async def aevaluate(
        self, conversation: ConversationWithContext
    ) -> EvaluationResult:
        """Score a single conversation."""
        ...


class CompositeEvaluator:
    """Runs multiple async evaluators in parallel and aggregates scores.

    Weighted combination: for each sub-evaluator ``(E, w)``, metric ``m`` receives
    ``score_m * w`` (only metrics that ``E`` returns). If multiple evaluators
    emit the same metric, contributions are summed. The overall score is then
    computed from the combined per-metric map using :attr:`aggregation_mode`.

    * ``MEAN``: ``sum(combined_scores.values()) / len(combined_scores)``
    * ``WEIGHTED_MEAN``: ``sum(s * metric_weights.get(m,1)) / sum(metric_weights.get(m,1) for m in combined)``
    * ``MIN``: ``min(combined_scores.values())`` (or 0 if empty)
    """

    def __init__(
        self,
        evaluators: List[tuple[BaseEvaluator, float]],
        min_acceptable_score: float = 0.6,
        aggregation_mode: AggregationMode = AggregationMode.MEAN,
        metric_weights: Optional[Dict[EvaluationMetric, float]] = None,
    ):
        self.evaluators = evaluators
        self.min_acceptable_score = min_acceptable_score
        self.aggregation_mode = aggregation_mode
        self.metric_weights = metric_weights or {}

    async def aevaluate(
        self, conversation: ConversationWithContext
    ) -> EvaluationResult:
        async def _run(
            pair: tuple[BaseEvaluator, float],
        ) -> tuple[EvaluationResult, float]:
            ev, w = pair
            return await ev.aevaluate(conversation), w

        pairs = await asyncio.gather(*[_run(p) for p in self.evaluators])

        combined_scores: dict[EvaluationMetric, float] = defaultdict(float)
        combined_feedback: dict[EvaluationMetric, list[str]] = defaultdict(list)

        for result, weight in pairs:
            for metric, score in result.scores.items():
                combined_scores[metric] += score * weight
            for metric, fb in result.feedback.items():
                combined_feedback[metric].append(fb)

        feedback_merged = {k: "; ".join(v) for k, v in combined_feedback.items()}

        overall = self._aggregate_overall(dict(combined_scores))

        return EvaluationResult(
            scores=dict(combined_scores),
            feedback=feedback_merged,
            overall_score=overall,
            needs_regeneration=overall < self.min_acceptable_score,
            regeneration_strategy=self._determine_regeneration_strategy(
                dict(combined_scores)
            ),
        )

    def _aggregate_overall(
        self, combined_scores: Dict[EvaluationMetric, float]
    ) -> float:
        if not combined_scores:
            return 0.0
        if self.aggregation_mode == AggregationMode.MIN:
            return min(combined_scores.values())
        if self.aggregation_mode == AggregationMode.WEIGHTED_MEAN:
            num = 0.0
            den = 0.0
            for m, s in combined_scores.items():
                w = self.metric_weights.get(m, 1.0)
                num += s * w
                den += w
            return num / den if den > 0 else 0.0
        # MEAN
        return sum(combined_scores.values()) / len(combined_scores)

    def _determine_regeneration_strategy(
        self, scores: Dict[EvaluationMetric, float]
    ) -> str:
        worst_metric = min(scores.items(), key=lambda x: x[1])
        strategies = {
            EvaluationMetric.COHERENCE: "improve_coherence",
            EvaluationMetric.GROUNDING: "improve_grounding",
            EvaluationMetric.RELEVANCE: "improve_relevance",
            EvaluationMetric.FACTUALITY: "verify_facts",
            EvaluationMetric.SAFETY: "ensure_safety",
        }
        return strategies.get(worst_metric[0], "general_improvement")
