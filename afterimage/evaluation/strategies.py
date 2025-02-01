from typing import Protocol, Dict, Any
from .base import EvaluationResult, EvaluationMetric


class RegenerationStrategy(Protocol):
    """Protocol for regeneration strategies."""

    def modify_generation_params(
        self, original_params: Dict[str, Any], evaluation_result: EvaluationResult
    ) -> Dict[str, Any]:
        """Modify generation parameters based on evaluation.

        Args:
            original_params: Original generation parameters
            evaluation_result: Evaluation results

        Returns:
            Modified parameters
        """
        ...


class CoherenceImprover(RegenerationStrategy):
    """Improves question-answer coherence."""

    def modify_generation_params(
        self, original_params: Dict[str, Any], evaluation_result: EvaluationResult
    ) -> Dict[str, Any]:
        return {
            **original_params,
            # Lower temperature for more focused and coherent responses
            "temperature": min(original_params.get("temperature", 0.7) * 0.8, 0.5),
            # Reduce randomness in token selection
            "top_p": min(original_params.get("top_p", 0.9) * 0.9, 0.7),
            # Add coherence reminder to system prompt
            "system_prompt": original_params.get("system_prompt", "")
            + "\nEnsure your responses directly address the questions asked.",
        }


class GroundingImprover(RegenerationStrategy):
    """Improves answer grounding in context."""

    def modify_generation_params(
        self, original_params: Dict[str, Any], evaluation_result: EvaluationResult
    ) -> Dict[str, Any]:
        return {
            **original_params,
            # Lower temperature for more factual responses
            "temperature": min(original_params.get("temperature", 0.7) * 0.7, 0.4),
            # Increase context window to provide more information
            "context_window": min(
                original_params.get("context_window", 1000) * 1.5, 4000
            ),
            # Retrieve more context chunks
            "retrieval_k": min(original_params.get("retrieval_k", 3) + 2, 10),
            # Add grounding reminder to system prompt
            "system_prompt": original_params.get("system_prompt", "")
            + "\nBase your responses strictly on the provided context.",
        }


class RelevanceImprover(RegenerationStrategy):
    """Improves question relevance to context."""

    def modify_generation_params(
        self, original_params: Dict[str, Any], evaluation_result: EvaluationResult
    ) -> Dict[str, Any]:
        return {
            **original_params,
            # Reduce creativity in question generation
            "temperature": min(original_params.get("temperature", 0.7) * 0.8, 0.5),
            # Increase context focus
            "context_window": min(
                original_params.get("context_window", 1000) * 1.2, 3000
            ),
            # Add relevance reminder to system prompt
            "system_prompt": original_params.get("system_prompt", "")
            + "\nGenerate questions that are directly relevant to the provided context.",
        }


class FactualityImprover(RegenerationStrategy):
    """Improves factual accuracy."""

    def modify_generation_params(
        self, original_params: Dict[str, Any], evaluation_result: EvaluationResult
    ) -> Dict[str, Any]:
        return {
            **original_params,
            "temperature": min(original_params.get("temperature", 0.7) * 0.7, 0.3),
        }


class SafetyImprover(RegenerationStrategy):
    """Improves content safety."""

    def modify_generation_params(
        self, original_params: Dict[str, Any], evaluation_result: EvaluationResult
    ) -> Dict[str, Any]:
        return {
            **original_params,
            "safety_settings": [
                {"category": cat, "threshold": "BLOCK_MEDIUM"}
                for cat in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH"]
            ],
        }


# Update strategy mapping in CompositeEvaluator._determine_regeneration_strategy
def _determine_regeneration_strategy(
    self, scores: Dict[EvaluationMetric, float]
) -> str:
    """Determine which aspect needs most improvement."""
    worst_metric = min(scores.items(), key=lambda x: x[1])

    strategies = {
        EvaluationMetric.COHERENCE: "improve_coherence",
        EvaluationMetric.GROUNDING: "improve_grounding",
        EvaluationMetric.RELEVANCE: "improve_relevance",
        EvaluationMetric.FACTUALITY: "verify_facts",
        EvaluationMetric.SAFETY: "ensure_safety",
    }

    return strategies.get(worst_metric[0], "general_improvement")
