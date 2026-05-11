"""Structured LLM response schemas for context-to-skill stages."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProbeSpec(BaseModel):
    task: str = Field(..., description="A context-grounded task for the respondent")
    rubrics: list[str] = Field(
        ..., min_length=1, description="Binary grading requirements"
    )


class ProbeGenerationResponse(BaseModel):
    probes: list[ProbeSpec] = Field(..., min_length=1)


class RubricJudgeResponse(BaseModel):
    rationale: str
    requirement_status: list[bool] = Field(default_factory=list)
    overall_score: float = Field(..., ge=0.0, le=1.0)


class SkillProposalResponse(BaseModel):
    action: str = "create"
    name: str
    description: str
    target_failure_modes: list[str] = Field(default_factory=list)
    proposed_guidance: str


class SkillContentResponse(BaseModel):
    name: str
    description: str
    content: str
