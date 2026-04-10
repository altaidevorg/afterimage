"""YAML configuration schema for AfterImage CLI.

Defines Pydantic models that map to a user-friendly YAML config file and
provides :func:`load_config` for validated loading.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class GenerationConfig(BaseModel):
    """Controls how many conversations to generate and concurrency."""

    num_dialogs: int = Field(default=10, description="Number of dialogs to generate")
    max_turns: int = Field(default=1, description="Maximum turns per dialog")
    max_concurrency: Optional[int] = Field(
        default=None, description="Max concurrent generations (provider default if omitted)"
    )


class ModelConfig(BaseModel):
    """LLM provider and model settings."""

    provider: str = Field(default="gemini", description="gemini | openai | deepseek | local")
    model_name: str = Field(default="gemini-2.0-flash", description="Model identifier")
    api_key_env: Optional[str] = Field(
        default=None, description="Environment variable name holding the API key"
    )
    base_url: Optional[str] = Field(
        default=None, description="Base URL for local/OpenAI-compatible servers"
    )


class RespondentConfig(BaseModel):
    """System prompt for the respondent (the assistant being fine-tuned)."""

    system_prompt: Optional[str] = Field(default=None, description="Inline system prompt")
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

    provider: str = Field(default="directory", description="memory | file | directory | jsonl | qdrant")
    path: Optional[str] = Field(default=None, description="Path to documents directory or file")
    # Qdrant-specific
    collection: Optional[str] = Field(default=None, description="Qdrant collection name")
    url: Optional[str] = Field(default=None, description="Qdrant server URL")
    content_key: str = Field(default="text", description="JSON key containing document text")


class ContextConfig(BaseModel):
    """Context sampling settings."""

    enabled: bool = Field(default=True, description="Whether to use context-grounded generation")
    num_random_contexts: int = Field(default=2, description="Contexts sampled per round")
    n_instructions: int = Field(default=3, description="Instructions generated per round")


class PersonasConfig(BaseModel):
    """Persona generation settings."""

    enabled: bool = Field(default=False, description="Whether to use persona-based generation")
    n_iterations: Optional[int] = Field(
        default=None, description="Persona tree depth (null = auto)"
    )


class QualityConfig(BaseModel):
    """Quality gating settings."""

    auto_improve: bool = Field(default=False, description="Retry low-quality generations")


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

    formats: list[str] = Field(default_factory=list, description="Format names to export to")
    output_dir: Optional[str] = Field(default=None, description="Output directory for exports")
    split: Optional[float] = Field(default=None, description="Train/val split ratio")
    shuffle: bool = Field(default=True, description="Shuffle before splitting")
    seed: int = Field(default=42, description="Random seed for splits")


class OutputConfig(BaseModel):
    """Output storage settings."""

    path: str = Field(
        default="./afterimage_output.jsonl", description="Output file path"
    )
    storage: str = Field(default="jsonl", description="jsonl | sql")
    export: Optional[ExportConfig] = Field(default=None, description="Auto-export after generation")


class PreferenceGenerationConfig(BaseModel):
    """Settings for DPO/RLHF preference pair generation (``afterimage preference`` command)."""

    num_pairs: int = Field(default=10, description="Number of preference pairs to generate")
    num_responses: int = Field(default=2, description="Responses generated per prompt")
    min_score_gap: float = Field(default=0.1, description="Minimum score gap to keep a pair")
    strategy: str = Field(
        default="temperature",
        description="Variation strategy: temperature | prompt | model | combined",
    )
    secondary_model: Optional[str] = Field(
        default=None, description="Secondary model for model-variation strategy"
    )
    multi_turn: bool = Field(default=False, description="Multi-turn with branching at final turn")
    max_concurrency: Optional[int] = Field(default=None, description="Max concurrent generations")
    output_format: str = Field(
        default="dpo",
        description="Output format: dpo | chat_dpo | ultrafeedback | anthropic_hh | orpo",
    )
    output_path: str = Field(default="./preferences.jsonl", description="Output file path")
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
        raise ValueError(f"Config file must be a YAML mapping, got {type(raw).__name__}")

    config = AfterImageConfig.model_validate(raw)

    # Resolve system_prompt_file relative to config file location
    if config.respondent.system_prompt_file is not None:
        prompt_path = Path(config.respondent.system_prompt_file)
        if not prompt_path.is_absolute():
            prompt_path = config_path.parent / prompt_path
        prompt_path = prompt_path.resolve()
        if not prompt_path.is_file():
            raise FileNotFoundError(
                f"System prompt file not found: {prompt_path}"
            )
        config.respondent.system_prompt = prompt_path.read_text(encoding="utf-8").strip()
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
