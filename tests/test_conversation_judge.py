"""Tests for async evaluation stack and ConversationJudge."""

import pytest

from afterimage.evaluation.base import (
    AggregationMode,
    CompositeEvaluator,
    EvaluationMetric,
    EvaluationResult,
)
from afterimage.evaluator import ConversationJudge, ConversationJudgeConfig
from afterimage.types import (
    ConversationEntry,
    ConversationWithContext,
    EvaluationEntrySchema,
    EvaluationSchema,
    GradeSchema,
    Role,
)


def _minimal_conv() -> ConversationWithContext:
    return ConversationWithContext(
        conversations=[
            ConversationEntry(role=Role.USER, content="What is 2+2?"),
            ConversationEntry(role=Role.ASSISTANT, content="Four."),
        ],
        instruction_context="math",
        response_context="math",
    )


class _FixedMetricEvaluator:
    def __init__(self, metric: EvaluationMetric, score: float):
        self._metric = metric
        self._score = score

    async def aevaluate(self, conversation: ConversationWithContext) -> EvaluationResult:
        return EvaluationResult(
            scores={self._metric: self._score},
            feedback={self._metric: "test"},
            overall_score=self._score,
            needs_regeneration=self._score < 0.5,
        )


@pytest.mark.asyncio
async def test_composite_evaluator_mean():
    conv = _minimal_conv()
    comp = CompositeEvaluator(
        [
            (_FixedMetricEvaluator(EvaluationMetric.COHERENCE, 0.2), 1.0),
            (_FixedMetricEvaluator(EvaluationMetric.FACTUALITY, 0.8), 1.0),
        ],
        aggregation_mode=AggregationMode.MEAN,
    )
    r = await comp.aevaluate(conv)
    assert abs(r.overall_score - 0.5) < 1e-6
    assert EvaluationMetric.COHERENCE in r.scores
    assert EvaluationMetric.FACTUALITY in r.scores


@pytest.mark.asyncio
async def test_composite_evaluator_min():
    conv = _minimal_conv()
    comp = CompositeEvaluator(
        [
            (_FixedMetricEvaluator(EvaluationMetric.COHERENCE, 0.2), 1.0),
            (_FixedMetricEvaluator(EvaluationMetric.FACTUALITY, 0.8), 1.0),
        ],
        aggregation_mode=AggregationMode.MIN,
    )
    r = await comp.aevaluate(conv)
    assert abs(r.overall_score - 0.2) < 1e-6


def test_evaluation_result_final_score_alias():
    r = EvaluationResult(
        scores={},
        feedback={},
        overall_score=0.73,
        needs_regeneration=False,
    )
    assert r.final_score == 0.73


@pytest.mark.asyncio
async def test_conversation_judge_grade_thresholds():
    from unittest.mock import AsyncMock

    from afterimage.evaluation.evaluators import LLMJudgeStructuredOutput
    from afterimage.providers.llm_providers import StructuredLLMResponse

    class StubLLM:
        async def agenerate_structured(self, prompt, schema, **kwargs):
            return StructuredLLMResponse(
                text="{}",
                parsed=LLMJudgeStructuredOutput(
                    scores=[0.95], feedback="ok", needs_improvement=False
                ),
                prompt_token_count=1,
                completion_token_count=1,
                total_token_count=2,
                finish_reason="stop",
                model_name="stub",
                raw_response=None,
            )

    embed = AsyncMock()
    embed.embed = AsyncMock(
        return_value=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    )
    embed.aclose = AsyncMock()

    judge = ConversationJudge(
        llm=StubLLM(),
        embedding_provider=embed,
        config=ConversationJudgeConfig(
            perfect_threshold=0.5,
            good_threshold=0.4,
            needs_improvement_threshold=0.3,
            bad_threshold=0.2,
        ),
    )
    out = await judge.aevaluate_row(_minimal_conv())
    assert out.evaluation is not None
    assert out.evaluation.overall_grade == GradeSchema.PERFECT
    await judge.aclose()
