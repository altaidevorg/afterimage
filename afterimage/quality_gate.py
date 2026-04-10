"""Quality gate for conversation generation.

Wraps ConversationJudge and implements accept/reject/retry logic for the
auto_improve workflow. Extracts evaluation retry logic from
ConversationGenerator.generate_single().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from .types import (
    ConversationWithContext,
    EvaluatedConversationWithContext,
    GradeSchema,
)

if TYPE_CHECKING:
    from .evaluator import ConversationJudge


# Grades that trigger a retry
_RETRY_GRADES = frozenset(
    {
        GradeSchema.NOT_ACCEPTABLE,
        GradeSchema.BAD,
        GradeSchema.NEEDS_IMPROVEMENT,
    }
)


@dataclass
class QualityResult:
    """Result of a quality gate evaluation.

    Attributes:
        conversation_row: The evaluated conversation (with evaluation if judge was used).
        accepted: Whether the conversation passed quality checks.
    """

    conversation_row: ConversationWithContext | EvaluatedConversationWithContext
    accepted: bool


class QualityGate:
    """Evaluates conversations and decides whether to accept or retry.

    When no evaluator is configured (auto_improve=False), all conversations
    are accepted immediately. When an evaluator is present, conversations are
    judged and only accepted if they meet the grade threshold.

    Attributes:
        evaluator: Optional ConversationJudge instance for quality evaluation.
    """

    def __init__(self, evaluator: Optional[ConversationJudge] = None):
        self._evaluator = evaluator

    @property
    def evaluator(self) -> Optional[ConversationJudge]:
        return self._evaluator

    @property
    def is_enabled(self) -> bool:
        """Whether quality gating is active."""
        return self._evaluator is not None

    async def evaluate(
        self, conversation_row: ConversationWithContext
    ) -> QualityResult:
        """Evaluate a conversation row.

        Args:
            conversation_row: The conversation to evaluate.

        Returns:
            QualityResult with the evaluated row and acceptance status.
        """
        if self._evaluator is None:
            return QualityResult(conversation_row=conversation_row, accepted=True)

        evaluated = await self._evaluator.aevaluate_row(conversation_row)
        accepted = evaluated.evaluation.overall_grade not in _RETRY_GRADES
        return QualityResult(conversation_row=evaluated, accepted=accepted)

    @staticmethod
    def should_retry(result: QualityResult) -> bool:
        """Whether the conversation should be regenerated.

        Args:
            result: A previous QualityResult.

        Returns:
            True if the conversation should be retried.
        """
        return not result.accepted
