from typing import Protocol, Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict

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


@dataclass
class EvaluationResult:
    """Detailed evaluation result for a conversation."""

    scores: Dict[EvaluationMetric, float]
    feedback: Dict[EvaluationMetric, str]
    overall_score: float
    needs_regeneration: bool
    regeneration_strategy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseEvaluator(Protocol):
    """Protocol for conversation evaluators."""

    def evaluate(self, conversation: ConversationWithContext) -> EvaluationResult:
        """Evaluate a conversation.

        Args:
            conversation: The conversation to evaluate

        Returns:
            Evaluation results
        """
        ...


class CompositeEvaluator:
    """Combines multiple evaluators with weights."""

    def __init__(
        self,
        evaluators: List[tuple[BaseEvaluator, float]],
        min_acceptable_score: float = 0.7,
    ):
        """Initialize composite evaluator.

        Args:
            evaluators: List of (evaluator, weight) tuples
            min_acceptable_score: Minimum score to consider acceptable
        """
        self.evaluators = evaluators
        self.min_acceptable_score = min_acceptable_score

    def evaluate(self, conversation: ConversationWithContext) -> EvaluationResult:
        """Evaluate using all evaluators and combine results."""
        results = []
        for evaluator, weight in self.evaluators:
            result = evaluator.evaluate(conversation)
            results.append((result, weight))

        # Combine scores and feedback
        combined_scores = defaultdict(float)
        combined_feedback = defaultdict(list)

        for result, weight in results:
            for metric, score in result.scores.items():
                combined_scores[metric] += score * weight
            for metric, feedback in result.feedback.items():
                combined_feedback[metric].append(feedback)

        overall_score = sum(combined_scores.values()) / len(combined_scores)

        return EvaluationResult(
            scores=dict(combined_scores),
            feedback={k: "; ".join(v) for k, v in combined_feedback.items()},
            overall_score=overall_score,
            needs_regeneration=overall_score < self.min_acceptable_score,
            regeneration_strategy=self._determine_regeneration_strategy(
                combined_scores
            ),
        )

    def _determine_regeneration_strategy(
        self, scores: Dict[EvaluationMetric, float]
    ) -> str:
        """Determine which aspect needs most improvement."""
        worst_metric = min(scores.items(), key=lambda x: x[1])

        strategies = {
            EvaluationMetric.COHERENCE: "improve_coherence",
            EvaluationMetric.RELEVANCE: "improve_relevance",
            EvaluationMetric.FACTUALITY: "verify_facts",
            EvaluationMetric.SAFETY: "ensure_safety",
        }

        return strategies.get(worst_metric[0], "general_improvement")
