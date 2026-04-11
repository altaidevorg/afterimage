"""YAML configuration schema for AfterImage CLI.

Defines Pydantic models that map to a user-friendly YAML config file and
provides :func:`load_config` for validated loading.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, model_validator


class StoppingFixed(BaseModel):
    """Stop after *n* conversations (same semantics as ``FixedNumberStoppingCallback``)."""

    type: Literal["fixed"] = "fixed"
    n: int = Field(..., ge=1, description="Stop when this many conversations are saved")


class StoppingContextCoverage(BaseModel):
    """Stop when document contexts have been used enough (requires ``documents``)."""

    type: Literal["context_coverage"] = "context_coverage"
    target_visits: int = Field(
        default=1,
        ge=1,
        description="Each context id must appear at least this many times",
    )
    coverage_threshold: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of documents that must meet target_visits",
    )


class StoppingPersonaUsage(BaseModel):
    """Stop after *n* unique personas have appeared (requires ``personas.enabled``)."""

    type: Literal["persona_usage"] = "persona_usage"
    n_personas: int = Field(..., ge=1)


class StoppingBudget(BaseModel):
    """Stop when cumulative token usage crosses a threshold (uses generation monitor)."""

    type: Literal["budget"] = "budget"
    max_prompt_tokens: Optional[int] = Field(default=None, ge=1)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    max_total_tokens: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _at_least_one_limit(self):
        if (
            self.max_prompt_tokens is None
            and self.max_completion_tokens is None
            and self.max_total_tokens is None
        ):
            raise ValueError(
                "generation.stopping budget: set at least one of "
                "max_prompt_tokens, max_completion_tokens, max_total_tokens"
            )
        return self


class StoppingRateLimit(BaseModel):
    """Stop when recent error rate is too high."""

    type: Literal["rate_limit"] = "rate_limit"
    max_error_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    min_samples: int = Field(default=10, ge=1)


class StoppingAll(BaseModel):
    """AND-combine nested rules (maps to ``AndStoppingCallback``)."""

    type: Literal["all"] = "all"
    conditions: list["StoppingCriterionConfig"] = Field(
        ...,
        min_length=1,
        description="All nested rules must signal stop before this rule stops",
    )


StoppingCriterionConfig = Annotated[
    Union[
        StoppingFixed,
        StoppingContextCoverage,
        StoppingPersonaUsage,
        StoppingBudget,
        StoppingRateLimit,
        StoppingAll,
    ],
    Field(discriminator="type"),
]

StoppingAll.model_rebuild()


def _stopping_nesting_depth(
    rules: list[StoppingCriterionConfig], depth: int = 0
) -> int:
    if depth > 8:
        raise ValueError(
            "generation.stopping: nesting deeper than 8 levels is not allowed"
        )
    max_d = depth
    for rule in rules:
        if isinstance(rule, StoppingAll):
            max_d = max(max_d, _stopping_nesting_depth(rule.conditions, depth + 1))
    return max_d


def iter_stopping_rules(rules: list[StoppingCriterionConfig]):
    """Flatten nested ``all`` groups for validation."""
    for rule in rules:
        yield rule
        if isinstance(rule, StoppingAll):
            yield from iter_stopping_rules(rule.conditions)


class GenerationConfig(BaseModel):
    """Controls how many conversations to generate and concurrency."""

    num_dialogs: Optional[int] = Field(
        default=10,
        description=(
            "Adds a fixed-count stopping rule when set. Use null with generation.stopping "
            "to rely only on custom stopping callbacks (e.g. budget)."
        ),
    )
    max_turns: int = Field(default=1, description="Maximum turns per dialog")
    max_concurrency: Optional[int] = Field(
        default=None,
        description="Max concurrent generations (provider default if omitted)",
    )
    stopping: list[StoppingCriterionConfig] = Field(
        default_factory=list,
        description=(
            "Extra stopping rules (OR semantics: any rule can end the run). "
            "Use type 'all' to AND-combine nested rules."
        ),
    )

    @model_validator(mode="after")
    def _needs_a_stop_signal(self):
        if not self.stopping and self.num_dialogs is None:
            raise ValueError(
                "generation: set num_dialogs or add at least one rule under generation.stopping"
            )
        if self.stopping:
            _stopping_nesting_depth(self.stopping)
        return self


class ModelConfig(BaseModel):
    """LLM provider and model settings."""

    provider: str = Field(
        default="gemini", description="gemini | openai | deepseek | local"
    )
    model_name: str = Field(default="gemini-2.0-flash", description="Model identifier")
    api_key_env: Optional[str] = Field(
        default=None, description="Environment variable name holding the API key"
    )
    base_url: Optional[str] = Field(
        default=None, description="Base URL for local/OpenAI-compatible servers"
    )


class RespondentConfig(BaseModel):
    """System prompt for the respondent (the assistant being fine-tuned)."""

    system_prompt: Optional[str] = Field(
        default=None, description="Inline system prompt"
    )
    system_prompt_file: Optional[str] = Field(
        default=None, description="Path to a file containing the system prompt"
    )

    @model_validator(mode="after")
    def _check_prompt_source(self):
        if self.system_prompt is None and self.system_prompt_file is None:
            raise ValueError(
                "respondent: provide either 'system_prompt' or 'system_prompt_file'"
            )
        if self.system_prompt is not None and self.system_prompt_file is not None:
            raise ValueError(
                "respondent: provide only one of 'system_prompt' or 'system_prompt_file', not both"
            )
        return self


class DocumentsConfig(BaseModel):
    """Document provider settings for context-grounded generation."""

    provider: str = Field(
        default="directory", description="memory | file | directory | jsonl | qdrant"
    )
    path: Optional[str] = Field(
        default=None, description="Path to documents directory or file"
    )
    # Qdrant-specific
    collection: Optional[str] = Field(
        default=None, description="Qdrant collection name"
    )
    url: Optional[str] = Field(default=None, description="Qdrant server URL")
    content_key: str = Field(
        default="text", description="JSON key containing document text"
    )


class ContextConfig(BaseModel):
    """Context sampling settings."""

    enabled: bool = Field(
        default=True, description="Whether to use context-grounded generation"
    )
    num_random_contexts: int = Field(
        default=2, description="Contexts sampled per round"
    )
    n_instructions: int = Field(
        default=3, description="Instructions generated per round"
    )


class PersonasConfig(BaseModel):
    """Persona generation settings."""

    enabled: bool = Field(
        default=False, description="Whether to use persona-based generation"
    )
    n_iterations: Optional[int] = Field(
        default=None, description="Persona tree depth (null = auto)"
    )


class QualityConfig(BaseModel):
    """Quality gating settings."""

    auto_improve: bool = Field(
        default=False, description="Retry low-quality generations"
    )


class AnalyticsConfig(BaseModel):
    """Analytics dashboard settings."""

    auto_analyze: bool = Field(
        default=False, description="Generate analytics report after generation"
    )
    output_path: Optional[str] = Field(
        default=None,
        description="Report output path (default: dataset path with .html extension)",
    )


class ExportConfig(BaseModel):
    """Auto-export settings for post-generation format conversion."""

    formats: list[str] = Field(
        default_factory=list, description="Format names to export to"
    )
    output_dir: Optional[str] = Field(
        default=None, description="Output directory for exports"
    )
    split: Optional[float] = Field(default=None, description="Train/val split ratio")
    shuffle: bool = Field(default=True, description="Shuffle before splitting")
    seed: int = Field(default=42, description="Random seed for splits")


class OutputConfig(BaseModel):
    """Output storage settings."""

    path: str = Field(
        default="./afterimage_output.jsonl", description="Output file path"
    )
    storage: str = Field(default="jsonl", description="jsonl | sql")
    export: Optional[ExportConfig] = Field(
        default=None, description="Auto-export after generation"
    )


class PreferenceGenerationConfig(BaseModel):
    """Settings for DPO/RLHF preference pair generation (``afterimage preference`` command)."""

    num_pairs: int = Field(
        default=10, description="Number of preference pairs to generate"
    )
    num_responses: int = Field(default=2, description="Responses generated per prompt")
    min_score_gap: float = Field(
        default=0.1, description="Minimum score gap to keep a pair"
    )
    strategy: str = Field(
        default="temperature",
        description="Variation strategy: temperature | prompt | model | combined",
    )
    secondary_model: Optional[str] = Field(
        default=None, description="Secondary model for model-variation strategy"
    )
    multi_turn: bool = Field(
        default=False, description="Multi-turn with branching at final turn"
    )
    max_concurrency: Optional[int] = Field(
        default=None, description="Max concurrent generations"
    )
    output_format: str = Field(
        default="dpo",
        description="Output format: dpo | chat_dpo | ultrafeedback | anthropic_hh | orpo",
    )
    output_path: str = Field(
        default="./preferences.jsonl", description="Output file path"
    )
    save_log: bool = Field(default=False, description="Save full generation log")


class AfterImageConfig(BaseModel):
    """Top-level AfterImage configuration."""

    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    respondent: RespondentConfig
    documents: Optional[DocumentsConfig] = Field(default=None)
    context: ContextConfig = Field(default_factory=ContextConfig)
    personas: PersonasConfig = Field(default_factory=PersonasConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    preference: Optional[PreferenceGenerationConfig] = Field(
        default=None, description="Preference pair generation settings"
    )

    @model_validator(mode="after")
    def _documents_personas_and_stopping(self):
        if self.personas.enabled and self.documents is None:
            raise ValueError(
                "personas.enabled requires a documents section in the config"
            )

        if self.documents is not None and not self.context.enabled:
            raise ValueError(
                "documents are configured but context.enabled is false; "
                "set context.enabled: true for grounded generation, or remove documents "
                "to use simple non-grounded generation"
            )

        for rule in iter_stopping_rules(self.generation.stopping):
            if isinstance(rule, StoppingContextCoverage) and self.documents is None:
                raise ValueError(
                    "generation.stopping context_coverage requires a documents section"
                )
            if isinstance(rule, StoppingPersonaUsage) and not self.personas.enabled:
                raise ValueError(
                    "generation.stopping persona_usage requires personas.enabled: true"
                )
        return self


def load_config(path: str | Path) -> AfterImageConfig:
    """Read and validate an AfterImage YAML configuration file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated :class:`AfterImageConfig`.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML is invalid or fails validation.
    """
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Config file must be a YAML mapping, got {type(raw).__name__}"
        )

    config = AfterImageConfig.model_validate(raw)

    # Resolve system_prompt_file relative to config file location
    if config.respondent.system_prompt_file is not None:
        prompt_path = Path(config.respondent.system_prompt_file)
        if not prompt_path.is_absolute():
            prompt_path = config_path.parent / prompt_path
        prompt_path = prompt_path.resolve()
        if not prompt_path.is_file():
            raise FileNotFoundError(f"System prompt file not found: {prompt_path}")
        config.respondent.system_prompt = prompt_path.read_text(
            encoding="utf-8"
        ).strip()
        config.respondent.system_prompt_file = str(prompt_path)

    # Resolve documents.path relative to config file location
    if config.documents is not None and config.documents.path is not None:
        doc_path = Path(config.documents.path)
        if not doc_path.is_absolute():
            doc_path = config_path.parent / doc_path
        config.documents.path = str(doc_path.resolve())

    # output.path stays relative to cwd (resolved at runtime, not here)

    return config


def resolve_api_key(config: AfterImageConfig) -> str | None:
    """Read the API key from the environment variable specified in config.

    Returns:
        The API key string, or None if provider is local and no key is configured.

    Raises:
        ValueError: If the env var is required but not set.
    """
    env_var = config.model.api_key_env
    if env_var is None:
        if config.model.provider == "local":
            return None
        # Infer default env var from provider
        defaults = {
            "gemini": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
        }
        env_var = defaults.get(config.model.provider)
        if env_var is None:
            return None

    value = os.environ.get(env_var)
    if value is None and config.model.provider != "local":
        raise ValueError(
            f"API key not found. Set environment variable: export {env_var}=your-key"
        )
    return value
