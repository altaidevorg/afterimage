"""Pydantic models for context-to-skill discovery."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SkillSide = Literal["reasoner", "challenger"]


class SkillProbe(BaseModel):
    """A context-grounded task with rubrics used to probe respondent behavior."""

    id: str
    context_id: str
    task: str
    rubrics: list[str]
    iteration: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillProbeResult(BaseModel):
    """Result of answering and judging one skill probe."""

    probe: SkillProbe
    answer: str
    score: float
    passed: bool
    rubric_status: list[bool] = Field(default_factory=list)
    judge_feedback: str = ""
    skill_version_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillProposal(BaseModel):
    """Structured analysis of what skill should be created or revised."""

    id: str
    context_id: str
    iteration: int
    side: SkillSide = "reasoner"
    name: str
    description: str
    target_failure_modes: list[str] = Field(default_factory=list)
    proposed_guidance: str
    action: Literal["create", "revise", "keep"] = "create"
    source_probe_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillVersion(BaseModel):
    """One candidate skill version for a context."""

    id: str
    context_id: str
    iteration: int
    side: SkillSide = "reasoner"
    name: str
    description: str
    content: str
    source_probe_ids: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillSelectionResult(BaseModel):
    """Selected best skill version and replay metrics."""

    context_id: str
    selected_version_id: str
    selected_iteration: int
    hard_score: float
    easy_score: float
    combined_score: float
    all_results: list[dict[str, Any]] = Field(default_factory=list)
