"""Build a ConversationGenerator from an AfterImageConfig.

Translates the declarative YAML config into the imperative Python API,
wiring up providers, callbacks, and storage.
"""

from __future__ import annotations

from pathlib import Path

from .config import AfterImageConfig, resolve_api_key
from .conversation_generator import ConversationGenerator
from .storage import JSONLStorage


def build_generator(config: AfterImageConfig) -> ConversationGenerator:
    """Construct a fully configured :class:`ConversationGenerator` from *config*.

    Args:
        config: Validated configuration (typically from :func:`~afterimage.config.load_config`).

    Returns:
        A ready-to-use generator instance.

    Raises:
        ValueError: If required settings (API key, documents path) are missing.
    """
    api_key = resolve_api_key(config)

    # For local provider without an API key, use a placeholder
    if api_key is None and config.model.provider == "local":
        api_key = "not-needed"

    # Map config provider name to LLMFactory provider names
    provider_name = config.model.provider

    # --- Document provider ---
    document_provider = None
    if config.documents is not None:
        document_provider = _build_document_provider(config)

    # --- Instruction generator callback ---
    instruction_callback = None
    if document_provider is not None and config.context.enabled:
        instruction_callback = _build_instruction_callback(
            config, api_key, provider_name, document_provider
        )

    # --- Respondent prompt modifier ---
    respondent_prompt_modifier = None
    if document_provider is not None and config.context.enabled:
        from .callbacks import WithContextRespondentPromptModifier

        respondent_prompt_modifier = WithContextRespondentPromptModifier()

    # --- Storage ---
    storage = _build_storage(config)

    # --- Build generator ---
    # For local provider, temporarily disable auto_improve if it requires
    # an evaluator (which needs embeddings) — the user must have embeddings-local.
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
        auto_improve=auto_improve,
        storage=storage,
        instruction_generator_callback=instruction_callback,
        respondent_prompt_modifier=respondent_prompt_modifier,
    )

    # For local providers, store base_url so it flows through to LLMFactory calls.
    if config.model.base_url:
        gen._factory_kwargs["base_url"] = config.model.base_url

    return gen


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
            path_pattern=docs.path, content_key=docs.content_key
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


def _build_instruction_callback(config, api_key, provider_name, document_provider):
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
        )
    else:
        from .callbacks import ContextualInstructionGeneratorCallback

        return ContextualInstructionGeneratorCallback(
            api_key=api_key,
            documents=document_provider,
            model_name=config.model.model_name,
            model_provider_name=provider_name,
            num_random_contexts=config.context.num_random_contexts,
            n_instructions=config.context.n_instructions,
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
