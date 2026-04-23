"""Structured LLM response schemas for OpenSimula."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FactorsResponse(BaseModel):
    factors: list[str] = Field(
        ...,
        min_length=1,
        description="Prime factors of variation (short names).",
    )
    factor_descriptions: list[str | None] = Field(
        default_factory=list,
        description="Optional parallel descriptions; may be shorter than factors.",
    )


class ChildProposalsResponse(BaseModel):
    children: list[str] = Field(
        ...,
        description="Proposed child category labels for the current parent.",
    )


class CriticChildrenResponse(BaseModel):
    """Critic may merge/edit/add/remove children (paper Appendix B.4)."""

    refined_labels: list[str] = Field(..., min_length=1)
    refined_descriptions: list[str | None] = Field(default_factory=list)


class PlanNextLevelResponse(BaseModel):
    plan: str = Field(
        ...,
        description="Guidance for granularity/consistency at the next depth.",
    )


class StrategiesResponse(BaseModel):
    strategy_names: list[str] = Field(..., min_length=1)
    strategy_weights: list[float] = Field(..., min_length=1)
    strategy_factor_groups: list[list[str]] = Field(
        ...,
        description="Each inner list is factor ids jointly sampled under that strategy.",
    )


class ScenariosResponse(BaseModel):
    scenarios: list[str] = Field(
        ...,
        min_length=1,
        description="Distinct meta-prompts / scenarios.",
    )


class ComplexifyResponse(BaseModel):
    complexified_scenario: str


class SingleQAGenResponse(BaseModel):
    question: str
    answer: str


class MCQGenResponse(BaseModel):
    question: str
    choices: list[str] = Field(..., min_length=2)
    correct_index: int = Field(..., ge=0)


class RawGenerationResponse(BaseModel):
    """Generic JSON-capable payload for QA/MCQ generation."""

    content: str = Field(
        ...,
        description="JSON string of task-specific object (question/answer/choices).",
    )


class RequirementCritiqueResponse(BaseModel):
    satisfying: bool
    explanation: str


class DoubleProbeCorrect(BaseModel):
    """First independent probe: is the labeled answer correct?"""

    is_correct: bool
    rationale: str = ""


class DoubleProbeIncorrect(BaseModel):
    """Second independent probe: is the labeled answer incorrect?"""

    is_incorrect: bool
    rationale: str = ""


class TaxonomyAssignmentResponse(BaseModel):
    """Assign one datapoint to the most relevant leaf (or node) per factor."""

    assignments: list[str] = Field(
        ...,
        description="Parallel to factor order in prompt: chosen node_id per factor.",
    )


class PairwiseComparisonBatch(BaseModel):
    """Model ranks items in a batch by relative complexity (step toward Elo)."""

    ordering: list[int] = Field(
        ...,
        description="Positions 0..k-1 referring to the batch list order (easiest to hardest).",
    )
