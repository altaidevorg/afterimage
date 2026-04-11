"""Type definitions for preference pair generation (DPO/RLHF)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ScoredResponse(BaseModel):
    """A single LLM response with its quality score and variation label."""

    content: str = Field(..., description="The response text")
    score: float = Field(..., description="Quality score from 0.0 to 1.0")
    variation_label: str = Field(
        ...,
        description="How this response was generated: temperature_low, temperature_high, "
        "prompt_enhanced, prompt_degraded, model_primary, model_secondary, etc.",
    )
    messages: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Full message list for chat_dpo format (includes system + history)",
    )


class PreferencePair(BaseModel):
    """A (chosen, rejected) pair for preference training."""

    prompt: str = Field(
        ..., description="The user prompt that generated both responses"
    )
    chosen: ScoredResponse
    rejected: ScoredResponse
    shared_prefix: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Shared conversation history before the final turn (multi-turn only)",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PreferenceConfig(BaseModel):
    """Configuration for preference pair generation."""

    num_pairs: int = Field(
        default=10, description="Number of preference pairs to generate"
    )
    num_responses: int = Field(
        default=2, description="Responses generated per prompt (min 2)"
    )
    min_score_gap: float = Field(
        default=0.1,
        description="Minimum score gap between chosen and rejected; pairs below this are discarded",
    )
    strategy: str = Field(
        default="temperature",
        description="Variation strategy: temperature | prompt | model | combined",
    )
    secondary_model: Optional[str] = Field(
        default=None,
        description="Secondary model name for model variation strategy",
    )
    multi_turn: bool = Field(
        default=False,
        description="Whether to generate multi-turn conversations with branching at final turn",
    )
    max_concurrency: Optional[int] = Field(
        default=None,
        description="Max concurrent response generations",
    )
    output_format: str = Field(
        default="dpo",
        description="Output format: dpo | chat_dpo | ultrafeedback | anthropic_hh | orpo",
    )
    output_path: str = Field(
        default="./preferences.jsonl",
        description="Output file path",
    )
    save_log: bool = Field(
        default=False,
        description="Whether to save full generation log with all scored responses",
    )
    log_path: Optional[str] = Field(
        default=None,
        description="Path for full generation log (default: output_path with _log suffix)",
    )


@dataclass
class PreferenceAnalytics:
    """Analytics computed after preference generation."""

    total_attempted: int = 0
    total_valid: int = 0
    total_discarded: int = 0
    discard_rate: float = 0.0
    strategy_distribution: Dict[str, int] = field(default_factory=dict)
    avg_score_gap: float = 0.0
    avg_chosen_score: float = 0.0
    avg_rejected_score: float = 0.0
    warnings: List[str] = field(default_factory=list)
