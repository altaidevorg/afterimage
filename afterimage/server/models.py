from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums / literals
# ---------------------------------------------------------------------------

JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]

GenerationPhase = Literal[
    "analyzing_document",
    "generating_personas",
    "initializing",
    "generating",
    "saving",
    "complete",
]


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class GenerationRequest(BaseModel):
    """Full configuration for a dataset generation job."""

    # Document input — at least one required
    document_text: str | None = None
    document_url: str | None = None
    document_chunks: list[str] | None = None
    chunk_size: int = Field(5000, gt=0)

    # Generation parameters
    num_dialogs: int = Field(10, gt=0, le=1000)
    max_turns: int = Field(1, gt=0)
    max_concurrency: int = Field(4, gt=0, le=32)

    # Prompt configuration
    respondent_prompt: str | None = None
    correspondent_prompt: str | None = None
    auto_generate_prompts: bool = True
    custom_instruction_prompt: str | None = None

    # Persona configuration
    use_personas: bool = True
    persona_iterations: int = Field(0, ge=0)

    # Model configuration
    model_name: str = "gemini-2.0-flash"
    model_provider_name: str = "gemini"

    # Output format
    output_format: Literal["jsonl", "json"] = "jsonl"
    include_system_prompt_parts: bool = True

    # Language enforcement — set to None to use AfterImage's default
    # ("same language as context"). Defaults to "english".
    force_language: str | None = "english"


class AnalyzeDocumentRequest(BaseModel):
    document_text: str
    excerpt_length: int = Field(4000, gt=0)


# ---------------------------------------------------------------------------
# Progress / result sub-models
# ---------------------------------------------------------------------------


class JobProgress(BaseModel):
    completed: int = 0
    total: int = 0
    percent: float = 0.0
    current_phase: GenerationPhase = "initializing"
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: float | None = None


class JobResult(BaseModel):
    num_conversations: int
    download_url: str
    output_format: str


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class GenerationJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    estimated_duration_seconds: float | None = None
    links: dict[str, str] = Field(default_factory=dict)


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: JobProgress | None = None
    result: JobResult | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    # Request parameters — useful for auditing which model/settings actually ran
    model_name: str | None = None
    model_provider_name: str | None = None
    num_dialogs: int | None = None
    output_format: str | None = None


class JobSummary(BaseModel):
    job_id: str
    status: JobStatus
    num_dialogs: int
    model_name: str
    model_provider_name: str
    created_at: datetime
    updated_at: datetime


class JobListResponse(BaseModel):
    jobs: list[JobSummary]
    total: int
    page: int
    per_page: int


class AnalyzeDocumentResponse(BaseModel):
    respondent_role: str
    correspondent_role: str
    instruction: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    active_jobs: int
