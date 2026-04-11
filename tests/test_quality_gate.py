"""Tests for QualityGate."""

import pytest

from afterimage.quality_gate import QualityGate, QualityResult
from afterimage.types import (
    ConversationEntry,
    ConversationWithContext,
    EvaluatedConversationWithContext,
    EvaluationEntrySchema,
    EvaluationSchema,
    GradeSchema,
    Role,
)


def _make_conversation_row():
    return ConversationWithContext(
        conversations=[
            ConversationEntry(role=Role.USER, content="hello"),
            ConversationEntry(role=Role.ASSISTANT, content="hi there"),
        ],
        instruction_context="ctx",
        persona="curious user",
    )


def _make_evaluation(grade: GradeSchema) -> EvaluationSchema:
    entry = EvaluationEntrySchema(feedback="ok", score=0.8)
    return EvaluationSchema(
        coherence=entry,
        factuality=entry,
        grounding=entry,
        helpfulness=entry,
        relevance=entry,
        overall_grade=grade,
    )


class FakeEvaluator:
    """Mock evaluator that returns a fixed grade."""

    def __init__(self, grade: GradeSchema):
        self.grade = grade
        self.call_count = 0

    async def aevaluate_row(self, row):
        self.call_count += 1
        return EvaluatedConversationWithContext(
            **row.model_dump(),
            evaluation=_make_evaluation(self.grade),
        )


class TestQualityGateNoEvaluator:
    @pytest.mark.asyncio
    async def test_accepts_everything_without_evaluator(self):
        gate = QualityGate(evaluator=None)
        row = _make_conversation_row()
        result = await gate.evaluate(row)
        assert result.accepted is True
        assert result.conversation_row is row

    def test_is_enabled_false(self):
        gate = QualityGate(evaluator=None)
        assert gate.is_enabled is False


class TestQualityGateWithEvaluator:
    @pytest.mark.asyncio
    async def test_accepts_good_grade(self):
        evaluator = FakeEvaluator(GradeSchema.GOOD)
        gate = QualityGate(evaluator=evaluator)
        row = _make_conversation_row()
        result = await gate.evaluate(row)
        assert result.accepted is True
        assert evaluator.call_count == 1

    @pytest.mark.asyncio
    async def test_accepts_perfect_grade(self):
        evaluator = FakeEvaluator(GradeSchema.PERFECT)
        gate = QualityGate(evaluator=evaluator)
        row = _make_conversation_row()
        result = await gate.evaluate(row)
        assert result.accepted is True

    @pytest.mark.asyncio
    async def test_rejects_bad_grade(self):
        evaluator = FakeEvaluator(GradeSchema.BAD)
        gate = QualityGate(evaluator=evaluator)
        row = _make_conversation_row()
        result = await gate.evaluate(row)
        assert result.accepted is False

    @pytest.mark.asyncio
    async def test_rejects_not_acceptable_grade(self):
        evaluator = FakeEvaluator(GradeSchema.NOT_ACCEPTABLE)
        gate = QualityGate(evaluator=evaluator)
        row = _make_conversation_row()
        result = await gate.evaluate(row)
        assert result.accepted is False

    @pytest.mark.asyncio
    async def test_rejects_needs_improvement_grade(self):
        evaluator = FakeEvaluator(GradeSchema.NEEDS_IMPROVEMENT)
        gate = QualityGate(evaluator=evaluator)
        row = _make_conversation_row()
        result = await gate.evaluate(row)
        assert result.accepted is False

    def test_is_enabled_true(self):
        evaluator = FakeEvaluator(GradeSchema.GOOD)
        gate = QualityGate(evaluator=evaluator)
        assert gate.is_enabled is True


class TestShouldRetry:
    def test_should_retry_when_not_accepted(self):
        result = QualityResult(
            conversation_row=_make_conversation_row(), accepted=False
        )
        assert QualityGate.should_retry(result) is True

    def test_should_not_retry_when_accepted(self):
        result = QualityResult(conversation_row=_make_conversation_row(), accepted=True)
        assert QualityGate.should_retry(result) is False
