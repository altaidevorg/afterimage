"""Build a ConversationGenerator from an AfterImageConfig.

Translates the declarative YAML config into the imperative Python API,
wiring up providers, callbacks, storage, and stopping criteria.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import BaseStoppingCallback
from .callbacks import (
    AndStoppingCallback,
    BudgetStoppingCallback,
    ContextCoverageStoppingCallback,
    FixedNumberStoppingCallback,
    PersonaUsageStoppingCallback,
    RateLimitStoppingCallback,
)
from .config import (
    AfterImageConfig,
    StoppingAll,
    StoppingBudget,
    StoppingContextCoverage,
    StoppingCriterionConfig,
    StoppingFixed,
    StoppingPersonaUsage,
    StoppingRateLimit,
    resolve_api_key,
)
from .conversation_generator import ConversationGenerator
from .storage import JSONLStorage


@dataclass(frozen=True)
class BuiltConversationRun:
    """Wired :class:`~afterimage.conversation_generator.ConversationGenerator` plus stopping."""

    generator: ConversationGenerator
    stopping_criteria: list[BaseStoppingCallback]
    num_requested: int | None


def _llm_create_extras(config: AfterImageConfig) -> dict[str, Any]:
    if config.model.base_url:
        return {"base_url": config.model.base_url}
    return {}


def _stopping_config_to_callback(
    rule: StoppingCriterionConfig,
    document_provider: Any | None,
) -> BaseStoppingCallback:
    if isinstance(rule, StoppingFixed):
        return FixedNumberStoppingCallback(n=rule.n)
    if isinstance(rule, StoppingContextCoverage):
        if document_provider is None:
            raise ValueError(
                "Context coverage stopping requires documents to be configured"
            )
        return ContextCoverageStoppingCallback(
            provider=document_provider,
            target_visits=rule.target_visits,
            coverage_threshold=rule.coverage_threshold,
        )
    if isinstance(rule, StoppingPersonaUsage):
        return PersonaUsageStoppingCallback(n_personas=rule.n_personas)
    if isinstance(rule, StoppingBudget):
        return BudgetStoppingCallback(
            max_prompt_tokens=rule.max_prompt_tokens,
            max_completion_tokens=rule.max_completion_tokens,
            max_total_tokens=rule.max_total_tokens,
        )
    if isinstance(rule, StoppingRateLimit):
        return RateLimitStoppingCallback(
            max_error_rate=rule.max_error_rate,
            min_samples=rule.min_samples,
        )
    if isinstance(rule, StoppingAll):
        inner = [
            _stopping_config_to_callback(r, document_provider) for r in rule.conditions
        ]
        return AndStoppingCallback(inner)
    raise TypeError(f"Unsupported stopping rule type: {type(rule)!r}")


def build_stopping_criteria(
    config: AfterImageConfig,
    document_provider: Any | None,
) -> tuple[list[BaseStoppingCallback], int | None]:
    """Translate YAML ``generation.stopping`` plus ``num_dialogs`` into runtime callbacks."""
    callbacks: list[BaseStoppingCallback] = []
    for rule in config.generation.stopping:
        callbacks.append(_stopping_config_to_callback(rule, document_provider))
    if config.generation.num_dialogs is not None:
        callbacks.append(FixedNumberStoppingCallback(n=config.generation.num_dialogs))

    if not callbacks:
        callbacks.append(FixedNumberStoppingCallback(n=5))

    num_requested: int | None = None
    for cb in callbacks:
        if isinstance(cb, FixedNumberStoppingCallback):
            num_requested = cb.n
            break

    return callbacks, num_requested


def build_conversation_run(config: AfterImageConfig) -> BuiltConversationRun:
    """Build generator, stopping list, and progress-bar hint from *config*."""
    api_key = resolve_api_key(config)

    if api_key is None and config.model.provider == "local":
        api_key = "not-needed"

    provider_name = config.model.provider
    llm_extras = _llm_create_extras(config)

    document_provider = None
    if config.documents is not None:
        document_provider = _build_document_provider(config)

    if document_provider is not None and config.context.enabled:
        instruction_callback = _build_instruction_callback(
            config, api_key, provider_name, document_provider, llm_extras
        )
    else:
        from .callbacks import SimpleInstructionGeneratorCallback

        instruction_callback = SimpleInstructionGeneratorCallback(
            api_key=api_key,
            model_name=config.model.model_name,
            model_provider_name=provider_name,
            n_instructions=config.context.n_instructions,
            llm_create_extras=llm_extras,
        )

    respondent_prompt_modifier = None
    if document_provider is not None and config.context.enabled:
        from .callbacks import WithContextRespondentPromptModifier

        respondent_prompt_modifier = WithContextRespondentPromptModifier()

    if config.skills.enabled:
        from .skills import DirectorySkillStore, SkillRespondentPromptModifier

        respondent_prompt_modifier = SkillRespondentPromptModifier(
            skill_store=DirectorySkillStore(config.skills.path),
            base_modifier=respondent_prompt_modifier,
        )

    storage = _build_storage(config)

    auto_improve = config.quality.auto_improve
    if auto_improve and provider_name == "local":
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            raise ValueError(
                "Install local embeddings for quality checking: "
                'pip install "afterimage[embeddings-local]"'
            )

    gen = ConversationGenerator(
        respondent_prompt=config.respondent.system_prompt,
        api_key=api_key,
        model_name=config.model.model_name,
        model_provider_name=provider_name,
        llm_factory_kwargs=llm_extras if llm_extras else None,
        auto_improve=auto_improve,
        storage=storage,
        instruction_generator_callback=instruction_callback,
        respondent_prompt_modifier=respondent_prompt_modifier,
    )

    stopping_criteria, num_requested = build_stopping_criteria(
        config, document_provider
    )

    return BuiltConversationRun(
        generator=gen,
        stopping_criteria=stopping_criteria,
        num_requested=num_requested,
    )


def build_generator(config: AfterImageConfig) -> ConversationGenerator:
    """Return only the generator (backward compatible)."""
    return build_conversation_run(config).generator


def _build_document_provider(config: AfterImageConfig):
    """Instantiate the correct DocumentProvider from config."""
    docs = config.documents
    provider_type = docs.provider.lower()

    if provider_type == "directory":
        from .providers.document_providers import DirectoryDocumentProvider

        if docs.path is None:
            raise ValueError("documents.path is required for directory provider")
        return DirectoryDocumentProvider(directory=docs.path)

    if provider_type == "file":
        from .providers.document_providers import FileSystemDocumentProvider

        if docs.path is None:
            raise ValueError("documents.path is required for file provider")
        return FileSystemDocumentProvider(path_pattern=docs.path)

    if provider_type == "jsonl":
        from .providers.document_providers import JSONLDocumentProvider

        if docs.path is None:
            raise ValueError("documents.path is required for jsonl provider")
        return JSONLDocumentProvider(
            path_pattern=docs.path,
            content_key=docs.content_key,
            preserve_ids=docs.preserve_ids,
            include_metadata=docs.include_metadata,
        )

    if provider_type == "qdrant":
        from .providers.document_providers import QdrantDocumentProvider

        if QdrantDocumentProvider is None:
            raise ValueError(
                "Qdrant support requires qdrant-client: pip install qdrant-client"
            )
        from qdrant_client import QdrantClient

        url = docs.url or "http://localhost:6333"
        collection = docs.collection
        if collection is None:
            raise ValueError("documents.collection is required for qdrant provider")
        client = QdrantClient(url=url)
        return QdrantDocumentProvider(
            client=client,
            collection_name=collection,
            content_key=docs.content_key,
        )

    raise ValueError(f"Unknown document provider: {provider_type}")


def _build_instruction_callback(
    config: AfterImageConfig,
    api_key: str,
    provider_name: str,
    document_provider,
    llm_extras: dict[str, Any],
):
    """Build the appropriate instruction generator callback."""
    if config.personas.enabled:
        from .callbacks import PersonaInstructionGeneratorCallback

        return PersonaInstructionGeneratorCallback(
            api_key=api_key,
            documents=document_provider,
            model_name=config.model.model_name,
            model_provider_name=provider_name,
            num_random_contexts=config.context.num_random_contexts,
            n_instructions=config.context.n_instructions,
            llm_create_extras=llm_extras,
        )

    from .callbacks import ContextualInstructionGeneratorCallback

    return ContextualInstructionGeneratorCallback(
        api_key=api_key,
        documents=document_provider,
        model_name=config.model.model_name,
        model_provider_name=provider_name,
        num_random_contexts=config.context.num_random_contexts,
        n_instructions=config.context.n_instructions,
        llm_create_extras=llm_extras,
    )


def _build_storage(config: AfterImageConfig):
    """Build storage backend from config."""
    if config.output.storage.lower() == "jsonl":
        output_path = Path(config.output.path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return JSONLStorage(conversations_path=str(output_path))

    if config.output.storage.lower() == "sql":
        from .storage import SQLStorage

        return SQLStorage(url=config.output.path)

    raise ValueError(f"Unknown storage type: {config.output.storage}")
