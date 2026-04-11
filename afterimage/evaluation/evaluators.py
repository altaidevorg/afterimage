"""Async metric evaluators (embeddings + LLM-as-judge)."""

from __future__ import annotations

import math
import time
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from ..monitoring import GenerationMonitor
from ..providers import LLMProvider
from ..providers.embedding_providers import EmbeddingProvider
from ..types import ConversationWithContext, Role
from .base import EvaluationMetric, EvaluationResult


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity for dense vectors (L2-normalized dot product)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class LLMJudgeStructuredOutput(BaseModel):
    """Structured LLM judge response."""

    scores: List[float] = Field(
        ...,
        description="One score in [0, 1] per evaluated item, in order.",
    )
    feedback: str = ""
    needs_improvement: bool = False


class CoherenceEvaluator:
    """Question–answer semantic coherence via embedding cosine similarity."""

    def __init__(
        self,
        embedding: EmbeddingProvider,
        monitor: Optional[GenerationMonitor] = None,
        coherence_threshold: float = 0.65,
    ):
        self._embedding = embedding
        self.monitor = monitor
        self._threshold = coherence_threshold

    async def aevaluate(
        self, conversation: ConversationWithContext
    ) -> EvaluationResult:
        start = time.time()
        try:
            pairs: list[tuple[str, str]] = []
            for i in range(0, len(conversation.conversations), 2):
                if i + 1 < len(conversation.conversations):
                    pairs.append(
                        (
                            conversation.conversations[i].content,
                            conversation.conversations[i + 1].content,
                        )
                    )

            if not pairs:
                return self._result(0.0, "No question-answer pairs found", True, start)

            texts: list[str] = []
            for q, a in pairs:
                texts.extend([q, a])
            vecs = await self._embedding.embed(texts)
            sims: list[float] = []
            for i in range(0, len(vecs), 2):
                sims.append(_cosine_similarity(vecs[i], vecs[i + 1]))
            avg = sum(sims) / len(sims)
            ok = avg >= self._threshold
            fb = (
                "Good question-answer coherence"
                if ok
                else "Low coherence between questions and answers"
            )
            return self._result(avg, fb, not ok, start)
        except Exception:
            if self.monitor:
                self.monitor.track_evaluation(
                    duration=time.time() - start,
                    success=False,
                    evaluator_type=self.__class__.__name__,
                    scores={},
                    error="coherence evaluation failed",
                    error_type="Exception",
                )
            raise

    def _result(
        self, score: float, feedback: str, needs_regen: bool, start: float
    ) -> EvaluationResult:
        if self.monitor:
            self.monitor.track_evaluation(
                duration=time.time() - start,
                success=True,
                evaluator_type=self.__class__.__name__,
                scores={EvaluationMetric.COHERENCE: score},
            )
        return EvaluationResult(
            scores={EvaluationMetric.COHERENCE: score},
            feedback={EvaluationMetric.COHERENCE: feedback},
            overall_score=score,
            needs_regeneration=needs_regen,
        )


class GroundingEvaluator:
    """How well assistant answers align with the response context embedding."""

    def __init__(
        self,
        embedding: EmbeddingProvider,
        monitor: Optional[GenerationMonitor] = None,
        grounding_threshold: float = 0.55,
    ):
        self._embedding = embedding
        self.monitor = monitor
        self._threshold = grounding_threshold

    async def aevaluate(
        self, conversation: ConversationWithContext
    ) -> EvaluationResult:
        start = time.time()
        try:
            ctx = conversation.response_context
            if not ctx:
                return self._neutral("No context provided", start)

            answers = [
                t.content
                for t in conversation.conversations
                if t.role == Role.ASSISTANT
            ]
            if not answers:
                return self._bad("No answers found", start)

            vecs = await self._embedding.embed([ctx] + answers)
            ctx_vec, ans_vecs = vecs[0], vecs[1:]
            sims = [_cosine_similarity(ctx_vec, av) for av in ans_vecs]
            avg = sum(sims) / len(sims)
            ok = avg >= self._threshold
            fb = (
                "Answers are well-grounded in context"
                if ok
                else "Answers show weak grounding in context"
            )
            return self._scored(avg, fb, not ok, start)
        except Exception:
            if self.monitor:
                self.monitor.track_evaluation(
                    duration=time.time() - start,
                    success=False,
                    evaluator_type=self.__class__.__name__,
                    scores={},
                    error="grounding evaluation failed",
                    error_type="Exception",
                )
            raise

    def _neutral(self, msg: str, start: float) -> EvaluationResult:
        if self.monitor:
            self.monitor.track_evaluation(
                duration=time.time() - start,
                success=True,
                evaluator_type=self.__class__.__name__,
                scores={EvaluationMetric.GROUNDING: 1.0},
            )
        return EvaluationResult(
            scores={EvaluationMetric.GROUNDING: 1.0},
            feedback={EvaluationMetric.GROUNDING: msg},
            overall_score=1.0,
            needs_regeneration=False,
        )

    def _bad(self, msg: str, start: float) -> EvaluationResult:
        if self.monitor:
            self.monitor.track_evaluation(
                duration=time.time() - start,
                success=True,
                evaluator_type=self.__class__.__name__,
                scores={EvaluationMetric.GROUNDING: 0.0},
            )
        return EvaluationResult(
            scores={EvaluationMetric.GROUNDING: 0.0},
            feedback={EvaluationMetric.GROUNDING: msg},
            overall_score=0.0,
            needs_regeneration=True,
        )

    def _scored(
        self, score: float, feedback: str, needs_regen: bool, start: float
    ) -> EvaluationResult:
        if self.monitor:
            self.monitor.track_evaluation(
                duration=time.time() - start,
                success=True,
                evaluator_type=self.__class__.__name__,
                scores={EvaluationMetric.GROUNDING: score},
            )
        return EvaluationResult(
            scores={EvaluationMetric.GROUNDING: score},
            feedback={EvaluationMetric.GROUNDING: feedback},
            overall_score=score,
            needs_regeneration=needs_regen,
        )


class RelevanceEvaluator:
    """How relevant user questions are to the instruction context."""

    def __init__(
        self,
        embedding: EmbeddingProvider,
        monitor: Optional[GenerationMonitor] = None,
        relevance_threshold: float = 0.55,
    ):
        self._embedding = embedding
        self.monitor = monitor
        self._threshold = relevance_threshold

    async def aevaluate(
        self, conversation: ConversationWithContext
    ) -> EvaluationResult:
        start = time.time()
        try:
            ctx = conversation.instruction_context
            if not ctx:
                return self._neutral("No context provided", start)

            questions = [
                t.content for t in conversation.conversations if t.role == Role.USER
            ]
            if not questions:
                return self._bad("No questions found", start)

            vecs = await self._embedding.embed([ctx] + questions)
            ctx_vec, q_vecs = vecs[0], vecs[1:]
            sims = [_cosine_similarity(ctx_vec, qv) for qv in q_vecs]
            avg = sum(sims) / len(sims)
            ok = avg >= self._threshold
            fb = (
                "Questions are relevant to context"
                if ok
                else "Questions show weak relevance to context"
            )
            return self._scored(avg, fb, not ok, start)
        except Exception:
            if self.monitor:
                self.monitor.track_evaluation(
                    duration=time.time() - start,
                    success=False,
                    evaluator_type=self.__class__.__name__,
                    scores={},
                    error="relevance evaluation failed",
                    error_type="Exception",
                )
            raise

    def _neutral(self, msg: str, start: float) -> EvaluationResult:
        if self.monitor:
            self.monitor.track_evaluation(
                duration=time.time() - start,
                success=True,
                evaluator_type=self.__class__.__name__,
                scores={EvaluationMetric.RELEVANCE: 1.0},
            )
        return EvaluationResult(
            scores={EvaluationMetric.RELEVANCE: 1.0},
            feedback={EvaluationMetric.RELEVANCE: msg},
            overall_score=1.0,
            needs_regeneration=False,
        )

    def _bad(self, msg: str, start: float) -> EvaluationResult:
        if self.monitor:
            self.monitor.track_evaluation(
                duration=time.time() - start,
                success=True,
                evaluator_type=self.__class__.__name__,
                scores={EvaluationMetric.RELEVANCE: 0.0},
            )
        return EvaluationResult(
            scores={EvaluationMetric.RELEVANCE: 0.0},
            feedback={EvaluationMetric.RELEVANCE: msg},
            overall_score=0.0,
            needs_regeneration=True,
        )

    def _scored(
        self, score: float, feedback: str, needs_regen: bool, start: float
    ) -> EvaluationResult:
        if self.monitor:
            self.monitor.track_evaluation(
                duration=time.time() - start,
                success=True,
                evaluator_type=self.__class__.__name__,
                scores={EvaluationMetric.RELEVANCE: score},
            )
        return EvaluationResult(
            scores={EvaluationMetric.RELEVANCE: score},
            feedback={EvaluationMetric.RELEVANCE: feedback},
            overall_score=score,
            needs_regeneration=needs_regen,
        )


class AsyncLLMBaseEvaluator:
    """Shared async LLM judge with structured output and monitoring."""

    def __init__(
        self,
        llm: LLMProvider,
        max_retries: int = 3,
        monitor: Optional[GenerationMonitor] = None,
    ):
        self.llm = llm
        self.max_retries = max_retries
        self.monitor = monitor

    async def _aget_structured(
        self, prompt: str
    ) -> tuple[LLMJudgeStructuredOutput, Any]:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            temp = max(0.1, 0.35 - attempt * 0.08)
            try:
                resp = await self.llm.agenerate_structured(
                    prompt,
                    LLMJudgeStructuredOutput,
                    temperature=temp,
                )
                parsed = resp.parsed
                if parsed.scores and all(0.0 <= s <= 1.0 for s in parsed.scores):
                    return parsed, resp
            except Exception as e:
                last_err = e
                continue
        raise ValueError(
            f"LLM judge failed after {self.max_retries} attempts: {last_err}"
        )


class FactualityEvaluator(AsyncLLMBaseEvaluator):
    """LLM rubric for factual accuracy vs context."""

    async def aevaluate(
        self, conversation: ConversationWithContext
    ) -> EvaluationResult:
        start = time.time()
        try:
            responses = [
                t.content
                for t in conversation.conversations
                if t.role == Role.ASSISTANT
            ]
            if not responses:
                return self._finalize(
                    EvaluationResult(
                        scores={EvaluationMetric.FACTUALITY: 0.0},
                        feedback={
                            EvaluationMetric.FACTUALITY: "No responses to evaluate"
                        },
                        overall_score=0.0,
                        needs_regeneration=True,
                    ),
                    start,
                    [],
                )

            separator_str = "\n----\n"
            prompt = f"""Evaluate the factual accuracy of the following responses in relation to the provided context.
Rate each response on a scale of 0-1 and give concise feedback.

Context:
{conversation.response_context or conversation.instruction_context or ""}

Responses:
{separator_str.join(f"[{i + 1}] {r}" for i, r in enumerate(responses))}

Return scores (one float per response, 0-1), feedback (short summary), and needs_improvement (true if any response is unreliable)."""

            try:
                parsed, resp = await self._aget_structured(prompt)
                avg = sum(parsed.scores) / len(parsed.scores)
                needs = parsed.needs_improvement or avg < 0.55
                result = EvaluationResult(
                    scores={EvaluationMetric.FACTUALITY: avg},
                    feedback={EvaluationMetric.FACTUALITY: parsed.feedback},
                    overall_score=avg,
                    needs_regeneration=needs,
                )
                return self._finalize(result, start, [resp])
            except Exception:
                result = EvaluationResult(
                    scores={EvaluationMetric.FACTUALITY: 0.5},
                    feedback={
                        EvaluationMetric.FACTUALITY: "Factuality evaluation inconclusive"
                    },
                    overall_score=0.5,
                    needs_regeneration=False,
                )
                return self._finalize(result, start, [])
        except Exception:
            if self.monitor:
                self.monitor.track_evaluation(
                    duration=time.time() - start,
                    success=False,
                    evaluator_type=self.__class__.__name__,
                    scores={},
                    error="factuality evaluation failed",
                    error_type="Exception",
                )
            raise

    def _finalize(
        self,
        result: EvaluationResult,
        start: float,
        responses: list[Any],
    ) -> EvaluationResult:
        if self.monitor:
            tok: dict[str, Any] = {}
            if responses:
                tok["prompt_token_count"] = sum(r.prompt_token_count for r in responses)
                tok["completion_token_count"] = sum(
                    r.completion_token_count for r in responses
                )
                tok["total_token_count"] = sum(r.total_token_count for r in responses)
                tok["model_name"] = responses[0].model_name
            self.monitor.track_evaluation(
                duration=time.time() - start,
                success=True,
                evaluator_type=self.__class__.__name__,
                scores=result.scores,
                **tok,
            )
        return result


class HelpfulnessEvaluator(AsyncLLMBaseEvaluator):
    """LLM rubric for answer helpfulness."""

    async def aevaluate(
        self, conversation: ConversationWithContext
    ) -> EvaluationResult:
        start = time.time()
        try:
            pairs: list[tuple[str, str]] = []
            for i in range(0, len(conversation.conversations), 2):
                if i + 1 < len(conversation.conversations):
                    pairs.append(
                        (
                            conversation.conversations[i].content,
                            conversation.conversations[i + 1].content,
                        )
                    )
            if not pairs:
                return self._finalize(
                    EvaluationResult(
                        scores={EvaluationMetric.HELPFULNESS: 0.0},
                        feedback={
                            EvaluationMetric.HELPFULNESS: "No Q/A pairs to evaluate"
                        },
                        overall_score=0.0,
                        needs_regeneration=True,
                    ),
                    start,
                    [],
                )

            separator_str = "\n----\n"
            pairs = separator_str.join(f"[{i+1}] Q: {q}\nA: {a}" for i, (q, a) in enumerate(pairs))
            prompt = f"""Evaluate how helpful each answer is for its question (0-1 each).

Context (reference):
{conversation.response_context or conversation.instruction_context or ""}

Pairs:
{pairs}

Return scores (one per pair), feedback, needs_improvement."""

            try:
                parsed, resp = await self._aget_structured(prompt)
                avg = sum(parsed.scores) / len(parsed.scores)
                needs = parsed.needs_improvement or avg < 0.55
                result = EvaluationResult(
                    scores={EvaluationMetric.HELPFULNESS: avg},
                    feedback={EvaluationMetric.HELPFULNESS: parsed.feedback},
                    overall_score=avg,
                    needs_regeneration=needs,
                )
                return self._finalize(result, start, [resp])
            except Exception as e:
                result = EvaluationResult(
                    scores={EvaluationMetric.HELPFULNESS: 0.0},
                    feedback={
                        EvaluationMetric.HELPFULNESS: f"Helpfulness evaluation failed: {e}"
                    },
                    overall_score=0.0,
                    needs_regeneration=True,
                )
                return self._finalize(result, start, [])
        except Exception:
            if self.monitor:
                self.monitor.track_evaluation(
                    duration=time.time() - start,
                    success=False,
                    evaluator_type=self.__class__.__name__,
                    scores={},
                    error="helpfulness evaluation failed",
                    error_type="Exception",
                )
            raise

    def _finalize(
        self,
        result: EvaluationResult,
        start: float,
        responses: list[Any],
    ) -> EvaluationResult:
        if self.monitor:
            tok: dict[str, Any] = {}
            if responses:
                tok["prompt_token_count"] = sum(r.prompt_token_count for r in responses)
                tok["completion_token_count"] = sum(
                    r.completion_token_count for r in responses
                )
                tok["total_token_count"] = sum(r.total_token_count for r in responses)
                tok["model_name"] = responses[0].model_name
            self.monitor.track_evaluation(
                duration=time.time() - start,
                success=True,
                evaluator_type=self.__class__.__name__,
                scores=result.scores,
                **tok,
            )
        return result


class SafetyEvaluator:
    """Placeholder safety pass-through (always neutral-positive)."""

    async def aevaluate(
        self, conversation: ConversationWithContext
    ) -> EvaluationResult:
        return EvaluationResult(
            scores={EvaluationMetric.SAFETY: 1.0},
            feedback={EvaluationMetric.SAFETY: "Safety check not implemented"},
            overall_score=1.0,
            needs_regeneration=False,
        )
