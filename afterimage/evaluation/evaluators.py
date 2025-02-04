from sentence_transformers import SentenceTransformer
import json
from typing import List, TypedDict, Optional
from .base import BaseEvaluator, EvaluationMetric, EvaluationResult
from ..monitoring import GenerationMonitor
from ..providers import LLMProvider
from ..types import ConversationWithContext, Role
import time
import numpy as np


class EvaluationTypeHint(TypedDict):
    feedback: str
    scores: List[float]
    needs_improvement: bool


class CoherenceEvaluator(BaseEvaluator):
    """Evaluates question-answer coherence."""

    def __init__(
        self,
        embedding_model: str = "altaidevorg/bge-m3-distill-8l",
        monitor: Optional[GenerationMonitor] = None,
    ):
        self.model = SentenceTransformer(embedding_model)
        self.monitor = monitor

    def evaluate(self, conversation: ConversationWithContext) -> EvaluationResult:
        start_time = time.time()
        try:
            # Extract question-answer pairs
            pairs = []
            for i in range(0, len(conversation.conversations), 2):
                if i + 1 < len(conversation.conversations):
                    pairs.append(
                        (
                            conversation.conversations[i].content,
                            conversation.conversations[i + 1].content,
                        )
                    )

            if not pairs:
                result = self._create_result(
                    0.0, "No question-answer pairs found", needs_regeneration=True
                )

                if self.monitor:
                    self.monitor.track_evaluation(
                        duration=time.time() - start_time,
                        success=True,
                        evaluator_type=self.__class__.__name__,
                        scores=result.scores,
                    )

                return result

            # Calculate coherence scores
            coherence_scores = []
            for question, answer in pairs:
                embeddings = self.model.encode([question, answer])
                score = float(
                    self.model.similarity(embeddings[0], embeddings[1]).squeeze()
                )
                coherence_scores.append(score)

            avg_coherence = sum(coherence_scores) / len(coherence_scores)
            needs_regen = avg_coherence < 0.7

            feedback = (
                "Good question-answer coherence"
                if avg_coherence >= 0.7
                else "Low coherence between questions and answers"
            )

            result = self._create_result(
                avg_coherence, feedback, needs_regeneration=needs_regen
            )

            if self.monitor:
                self.monitor.track_evaluation(
                    duration=time.time() - start_time,
                    success=True,
                    evaluator_type=self.__class__.__name__,
                    scores=result.scores,
                )

            return result

        except Exception as e:
            if self.monitor:
                self.monitor.track_evaluation(
                    duration=time.time() - start_time,
                    success=False,
                    evaluator_type=self.__class__.__name__,
                    scores={},
                    error=str(e),
                    error_type=e.__class__.__name__,
                )
            raise

    def _create_result(
        self, score: float, feedback: str, needs_regeneration: bool
    ) -> EvaluationResult:
        return EvaluationResult(
            scores={EvaluationMetric.COHERENCE: score},
            feedback={EvaluationMetric.COHERENCE: feedback},
            overall_score=score,
            needs_regeneration=needs_regeneration,
        )


class GroundingEvaluator(BaseEvaluator):
    """Evaluates if answers are grounded in the provided context."""

    def __init__(
        self,
        embedding_model: str = "altaidevorg/bge-m3-distill-8l",
        monitor: Optional[GenerationMonitor] = None,
    ):
        self.model = SentenceTransformer(embedding_model)
        self.monitor = monitor

    def evaluate(self, conversation: ConversationWithContext) -> EvaluationResult:
        start_time = time.time()
        try:
            if not conversation.response_context:
                result = self._create_result(
                    1.0, "No context provided", needs_regeneration=False
                )

                if self.monitor:
                    self.monitor.track_evaluation(
                        duration=time.time() - start_time,
                        success=True,
                        evaluator_type=self.__class__.__name__,
                        scores=result.scores,
                    )
                return result

            # Get context embedding
            context_embedding = self.model.encode(conversation.response_context)

            # Only evaluate assistant responses
            answers = [
                turn.content
                for turn in conversation.conversations
                if turn.role == Role.ASSISTANT
            ]

            if not answers:
                result = self._create_result(
                    0.0, "No answers found", needs_regeneration=True
                )

                if self.monitor:
                    self.monitor.track_evaluation(
                        duration=time.time() - start_time,
                        success=True,
                        evaluator_type=self.__class__.__name__,
                        scores=result.scores,
                    )
                return result

            # Calculate grounding scores
            answer_embeddings = self.model.encode(answers)
            grounding_scores = [
                float(np.dot(context_embedding, ans_emb))
                for ans_emb in answer_embeddings
            ]

            avg_grounding = sum(grounding_scores) / len(grounding_scores)
            needs_regen = avg_grounding < 0.6

            feedback = (
                "Answers are well-grounded in context"
                if avg_grounding >= 0.6
                else "Answers show weak grounding in context"
            )

            result = self._create_result(
                avg_grounding, feedback, needs_regeneration=needs_regen
            )

            if self.monitor:
                self.monitor.track_evaluation(
                    duration=time.time() - start_time,
                    success=True,
                    evaluator_type=self.__class__.__name__,
                    scores=result.scores,
                )
            return result

        except Exception as e:
            if self.monitor:
                self.monitor.track_evaluation(
                    duration=time.time() - start_time,
                    success=False,
                    evaluator_type=self.__class__.__name__,
                    scores={},
                    error=str(e),
                    error_type=e.__class__.__name__,
                )
            raise

    def _create_result(
        self, score: float, feedback: str, needs_regeneration: bool
    ) -> EvaluationResult:
        return EvaluationResult(
            scores={EvaluationMetric.GROUNDING: score},
            feedback={EvaluationMetric.GROUNDING: feedback},
            overall_score=score,
            needs_regeneration=needs_regeneration,
        )


class RelevanceEvaluator(BaseEvaluator):
    """Evaluates if questions are relevant to the provided context."""

    def __init__(
        self,
        embedding_model: str = "altaidevorg/bge-m3-distill-8l",
        monitor: Optional[GenerationMonitor] = None,
    ):
        self.model = SentenceTransformer(embedding_model)
        self.monitor = monitor

    def evaluate(self, conversation: ConversationWithContext) -> EvaluationResult:
        start_time = time.time()
        try:
            if not conversation.instruction_context:
                result = self._create_result(
                    1.0, "No context provided", needs_regeneration=False
                )

                if self.monitor:
                    self.monitor.track_evaluation(
                        duration=time.time() - start_time,
                        success=True,
                        evaluator_type=self.__class__.__name__,
                        scores=result.scores,
                    )
                return result

            # Get context embedding
            context_embedding = self.model.encode(conversation.instruction_context)

            # Only evaluate user questions
            questions = [
                turn.content
                for turn in conversation.conversations
                if turn.role == Role.USER
            ]

            if not questions:
                result = self._create_result(
                    0.0, "No questions found", needs_regeneration=True
                )

                if self.monitor:
                    self.monitor.track_evaluation(
                        duration=time.time() - start_time,
                        success=True,
                        evaluator_type=self.__class__.__name__,
                        scores=result.scores,
                    )
                return result

            # Calculate relevance scores
            question_embeddings = self.model.encode(questions)
            relevance_scores = [
                float(np.dot(context_embedding, q_emb)) for q_emb in question_embeddings
            ]

            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            needs_regen = avg_relevance < 0.6

            feedback = (
                "Questions are relevant to context"
                if avg_relevance >= 0.6
                else "Questions show weak relevance to context"
            )

            result = self._create_result(
                avg_relevance, feedback, needs_regeneration=needs_regen
            )

            if self.monitor:
                self.monitor.track_evaluation(
                    duration=time.time() - start_time,
                    success=True,
                    evaluator_type=self.__class__.__name__,
                    scores=result.scores,
                )
            return result

        except Exception as e:
            if self.monitor:
                self.monitor.track_evaluation(
                    duration=time.time() - start_time,
                    success=False,
                    evaluator_type=self.__class__.__name__,
                    scores={},
                    error=str(e),
                    error_type=e.__class__.__name__,
                )
            raise

    def _create_result(
        self, score: float, feedback: str, needs_regeneration: bool
    ) -> EvaluationResult:
        return EvaluationResult(
            scores={EvaluationMetric.RELEVANCE: score},
            feedback={EvaluationMetric.RELEVANCE: feedback},
            overall_score=score,
            needs_regeneration=needs_regeneration,
        )


class LLMBaseEvaluator(BaseEvaluator):
    """Base class for LLM-based evaluators."""

    def __init__(
        self,
        llm: LLMProvider,
        max_retries: int = 3,
        monitor: Optional[GenerationMonitor] = None,
    ):
        self.llm = llm
        self.max_retries = max_retries
        self.monitor = monitor

    def _get_valid_json_response(self, prompt: str) -> dict:
        """Get valid JSON response with retry mechanism."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.llm.generate_content(
                    prompt=prompt,
                    temperature=0.3 - (attempt * 0.1),  # Reduce temperature on retries
                    max_tokens=500,
                    response_mime_type="application/json",
                    response_schema=EvaluationTypeHint,
                )

                evaluation = json.loads(response.text.strip())
                if isinstance(evaluation, dict):
                    if all(
                        key in evaluation
                        for key in ["scores", "feedback", "needs_improvement"]
                    ):
                        if (
                            isinstance(evaluation["scores"], list)
                            and evaluation["scores"]
                        ):
                            if all(
                                isinstance(score, (int, float)) and 0 <= score <= 1
                                for score in evaluation["scores"]
                            ):
                                return evaluation

            except Exception as e:
                last_error = e
                continue

        raise ValueError(
            f"Failed to get valid JSON response after {self.max_retries} attempts. Last error: {last_error}"
        )

    def evaluate(self, conversation: ConversationWithContext) -> EvaluationResult:
        start_time = time.time()
        try:
            result = self._evaluate_impl(conversation)

            if self.monitor:
                self.monitor.track_evaluation(
                    duration=time.time() - start_time,
                    success=True,
                    evaluator_type=self.__class__.__name__,
                    scores=result.scores,
                )

            return result

        except Exception as e:
            if self.monitor:
                self.monitor.track_evaluation(
                    duration=time.time() - start_time,
                    success=False,
                    evaluator_type=self.__class__.__name__,
                    scores={},
                    error=str(e),
                    error_type=e.__class__.__name__,
                )
            raise


class FactualityEvaluator(LLMBaseEvaluator):
    """Evaluates factual accuracy using LLM."""

    def evaluate(self, conversation: ConversationWithContext) -> EvaluationResult:
        responses = [
            turn.content
            for turn in conversation.conversations
            if turn.role == Role.ASSISTANT
        ]

        if not responses:
            return self._create_result(
                0.0, "No responses to evaluate", needs_regeneration=True
            )

        prompt = f"""Evaluate the factual accuracy of the following responses in relation to the provided context.
Rate each statement's factual accuracy and provide specific feedback.

Context:
{conversation.response_context or conversation.instruction_context}

Responses to evaluate:
{"\n---------\n\n".join(f"[{i + 1}] {resp}" for i, resp in enumerate(responses))}

Evaluate each response's factual accuracy on a scale of 0-1, where:
0.0-0.3: Contains significant factual errors or unsupported claims
0.4-0.6: Contains minor factual errors or partially supported claims
0.7-0.9: Mostly factually accurate with slight imprecisions
1.0: Completely factually accurate and well-supported by context

Provide your evaluation in JSON format."""

        try:
            evaluation = self._get_valid_json_response(prompt)
            avg_score = sum(evaluation["scores"]) / len(evaluation["scores"])
            needs_regen = evaluation["needs_improvement"] or avg_score < 0.6

            return self._create_result(
                avg_score, evaluation["feedback"], needs_regeneration=needs_regen
            )
        except Exception as e:
            # Return neutral score if evaluation fails after retries
            return self._create_result(
                0.5,
                "Factuality evaluation inconclusive",
                needs_regeneration=False,  # Don't trigger regeneration on evaluation failure
            )

    def _create_result(
        self, score: float, feedback: str, needs_regeneration: bool
    ) -> EvaluationResult:
        return EvaluationResult(
            scores={EvaluationMetric.FACTUALITY: score},
            feedback={EvaluationMetric.FACTUALITY: feedback},
            overall_score=score,
            needs_regeneration=needs_regeneration,
        )


class HelpfulnessEvaluator(LLMBaseEvaluator):
    """Evaluates response helpfulness using LLM."""

    def evaluate(self, conversation: ConversationWithContext) -> EvaluationResult:
        # Extract question-answer pairs
        pairs = []
        for i in range(0, len(conversation.conversations), 2):
            if i + 1 < len(conversation.conversations):
                pairs.append(
                    (
                        conversation.conversations[i].content,
                        conversation.conversations[i + 1].content,
                    )
                )

        if not pairs:
            return self._create_result(
                0.0, "No question-answer pairs to evaluate", needs_regeneration=True
            )

        # Create evaluation prompt
        prompt = f"""Evaluate how helpful and comprehensive the answers are in relation to their questions.
Rate each answer's helpfulness and provide specific feedback.

Context (for reference):
{conversation.response_context or conversation.instruction_context}

Question-Answer pairs to evaluate:
{"\n---------\n\n".join(f"[{i + 1}] Q: {q}\nA: {a}" for i, (q, a) in enumerate(pairs))}

Evaluate each answer's helpfulness on a scale of 0-1, where:
0.0-0.3: Unhelpful, irrelevant, or incomplete
0.4-0.6: Partially helpful but missing key information
0.7-0.9: Helpful with minor omissions
1.0: Exceptionally helpful and comprehensive

Provide your evaluation in JSON format."""

        try:
            evaluation = self._get_valid_json_response(prompt)
            avg_score = sum(evaluation["scores"]) / len(evaluation["scores"])
            needs_regen = evaluation["needs_improvement"] or avg_score < 0.6

            return self._create_result(
                avg_score, evaluation["feedback"], needs_regeneration=needs_regen
            )
        except Exception as e:
            return self._create_result(
                0.0,
                f"Failed to evaluate helpfulness: {str(e)}",
                needs_regeneration=True,
            )

    def _create_result(
        self, score: float, feedback: str, needs_regeneration: bool
    ) -> EvaluationResult:
        return EvaluationResult(
            scores={EvaluationMetric.HELPFULNESS: score},
            feedback={EvaluationMetric.HELPFULNESS: feedback},
            overall_score=score,
            needs_regeneration=needs_regeneration,
        )


class SafetyEvaluator(BaseEvaluator):
    """Evaluates content safety."""

    def evaluate(self, conversation: ConversationWithContext) -> EvaluationResult:
        # Placeholder for safety evaluation
        return EvaluationResult(
            scores={EvaluationMetric.SAFETY: 1.0},
            feedback={EvaluationMetric.SAFETY: "Safety check not implemented"},
            overall_score=1.0,
            needs_regeneration=False,
        )
