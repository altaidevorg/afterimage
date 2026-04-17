"""Generate English-first dialogs grounded in court-opinion excerpts via Qdrant RAG.

This script is the **reference pattern** for vector RAG with AfterImage:

* :class:`~afterimage.callbacks.ContextualInstructionGeneratorCallback` samples
  briefing documents from your Qdrant collection.
* :class:`~afterimage.retrievers.QdrantRetriever` + ``WithRAGRespondentPromptModifier``
  injects additional relevant chunks for each generated user question, using an
  async :class:`~afterimage.providers.embedding_providers.EmbeddingProvider` so
  retrieval does not block the event loop.

**Prerequisites**

* ``GEMINI_API_KEY`` (default model provider below is Gemini).
* A Qdrant collection you control, whose payload includes a text field (default
  field name: ``content``) with opinion or judgment excerpts and optional citation
  metadata in the text.
* Query **embedding model must match** how the collection was indexed. The default
  uses a **process** embedding provider (local / worker pool). If you indexed with
  Gemini embeddings, use the commented block in ``main``.

**Configuration (environment)**

* ``QDRANT_URL`` — REST URL (default ``http://localhost:6333``).
* ``QDRANT_API_KEY`` — optional, for Qdrant Cloud.
* ``QDRANT_COLLECTION`` — collection name (default ``caselaw_chunks``).
* ``QDRANT_CONTENT_KEY`` — payload key holding text (default ``content``).
* ``QDRANT_MAX_DOCS`` — cap for document sampling (default ``50``).
* ``NUM_DIALOGS`` / ``MAX_TURNS`` — generation size (defaults ``5`` / ``1``).

Run from the repository root::

    uv run examples/generate_caselaw_rag.py

Synthetic datasets for research or model training only; outputs are not legal advice.
"""

from __future__ import annotations

import asyncio
import os
from datetime import timedelta

from afterimage import (
    ConversationGenerator,
    ContextualInstructionGeneratorCallback,
    EmbeddingProviderFactory,
    GenerationMonitor,
    WithRAGRespondentPromptModifier,
)
from afterimage.providers import QdrantDocumentProvider
from afterimage.retrievers import QdrantRetriever
from qdrant_client import QdrantClient

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Set GEMINI_API_KEY for the default Gemini model path.")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "caselaw_chunks")
CONTENT_KEY = os.getenv("QDRANT_CONTENT_KEY", "content")
MAX_DOCS = int(os.getenv("QDRANT_MAX_DOCS", "50"))
NUM_DIALOGS = int(os.getenv("NUM_DIALOGS", "5"))
MAX_TURNS = int(os.getenv("MAX_TURNS", "1"))
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def simple_alert_handler(alert) -> None:
    print(f"alert - {alert.name} - {alert.message}")


monitor = GenerationMonitor(
    log_dir="logs",
    alert_handlers=[simple_alert_handler],
    metrics_interval=60,
)

respondent_prompt = """You are a careful senior legal research assistant helping
produce synthetic training dialogues. Ground every substantive claim in the
**retrieved court opinions and excerpts** supplied in your context. When the
excerpts include neutral identifiers (court, docket or neutral citation, date),
use them faithfully in your explanation. Use plain English and define legal jargon
when it helps a lay reader. If the retrieved material is insufficient to answer,
say what is missing instead of inventing holdings or citations. This is educational
synthetic data only; you are not providing real-world legal advice."""


correspondent_prompt = """You are an experienced lawyer or a legally curious
client in a role-play. Your partner is a research assistant who answers from
retrieved case excerpts. Ask realistic questions and follow-ups about the legal
issues suggested by the case material you are given (for example contracts,
torts, criminal procedure, civil procedure, or administrative law). Stay in
character; do not break the fourth wall. For your first message, wait until you
are prompted to begin. Do not fabricate specific docket numbers or citations
unless they already appear in your briefing; otherwise stay at the level of
issues and facts inspired by the materials."""


def _qdrant_client() -> QdrantClient:
    kwargs: dict = {"url": QDRANT_URL, "timeout": 60.0}
    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY
    return QdrantClient(**kwargs)


conv_gen = ConversationGenerator(
    respondent_prompt=respondent_prompt,
    correspondent_prompt=correspondent_prompt,
    api_key=api_key,
    model_name=MODEL_NAME,
    monitor=monitor,
)

qd_client = _qdrant_client()
documents = QdrantDocumentProvider(
    client=qd_client,
    collection_name=QDRANT_COLLECTION,
    content_key=CONTENT_KEY,
    max_docs=MAX_DOCS,
)

instruction_generator_callback = ContextualInstructionGeneratorCallback(
    api_key=api_key,
    documents=documents,
    model_name=MODEL_NAME,
    num_random_contexts=1,
)

# Async embeddings for retrieval (matches common local indexes using BGE-style models).
embedding_provider = EmbeddingProviderFactory.create(
    {"type": "process", "model": "altaidevorg/bge-m3-distill-8l", "workers": 2},
)
# If your collection was built with Gemini embeddings, use instead:
# from afterimage import SmartKeyPool
# _pool = SmartKeyPool.from_single_key(api_key)
# embedding_provider = EmbeddingProviderFactory.create(
#     {"type": "gemini", "model": "gemini-embedding-001"},
#     key_pool=_pool,
# )

retriever = QdrantRetriever(
    client=qd_client,
    collection_name=QDRANT_COLLECTION,
    embedding_provider=embedding_provider,
    payload_key=CONTENT_KEY,
    limit=3,
)
respondent_prompt_modifier = WithRAGRespondentPromptModifier(retriever=retriever)


async def main() -> None:
    print(
        f"Qdrant: {QDRANT_URL!r} collection={QDRANT_COLLECTION!r} "
        f"content_key={CONTENT_KEY!r} num_dialogs={NUM_DIALOGS} max_turns={MAX_TURNS}"
    )
    try:
        await conv_gen.generate(
            num_dialogs=NUM_DIALOGS,
            max_turns=MAX_TURNS,
            instruction_generator_callback=instruction_generator_callback,
            respondent_prompt_modifier=respondent_prompt_modifier,
        )

        generation_time = monitor.get_metrics(
            "generation_time", window=timedelta(hours=1)
        )
        if generation_time.get("mean") is not None:
            print(f"Avg. generation time: {generation_time['mean']:.2f} secs")

        monitor.visualize_metrics(save_dir="plots")
    finally:
        await embedding_provider.aclose()
        monitor.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
