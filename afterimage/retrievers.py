from __future__ import annotations

import asyncio
import time
from abc import abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

from qdrant_client import AsyncQdrantClient, QdrantClient

from .providers.embedding_providers import EmbeddingProvider

# Canonical empty-hit message returned by built-in retrievers (single source of truth).
NO_RETRIEVAL_CONTEXT = "No relevant context found."

# Key used under :attr:`~afterimage.types.GeneratedResponsePrompt.metadata` for
# retriever diagnostics (e.g. hit ids and scores).
RETRIEVAL_METADATA_KEY = "retrieval"


def is_no_retrieval_context(text: str) -> bool:
    """Return True if *text* is empty or exactly :data:`NO_RETRIEVAL_CONTEXT`."""
    return not text or text == NO_RETRIEVAL_CONTEXT


@dataclass(frozen=True)
class RetrievalResult:
    """Retriever output: prompt-safe *context* string plus optional *metadata*."""

    context: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _require_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "SentenceTransformer-based QdrantRetriever requires sentence-transformers. "
            'Install with pip install "afterimage[embeddings-local]" '
            "or pip install sentence-transformers"
        ) from e
    return SentenceTransformer


@runtime_checkable
class ContextRetriever(Protocol):
    """Protocol for context retrieval used by :class:`~afterimage.callbacks.WithRAGRespondentPromptModifier`.

    **Required**

    * :meth:`get_context` — synchronous retrieval returning a string (may run network or
      local vector search). For async-only backends, document behavior; prefer implementing
      :meth:`aget_context` and calling it from generation code paths.

    **Optional (duck-typed, not enforced by structural typing)**

    * :meth:`aget_context` — async retrieval; preferred when the generator runs under
      ``asyncio`` so embeddings and I/O do not block the event loop.
    * :meth:`get_context_with_metadata` / :meth:`aget_context_with_metadata` — return a
      :class:`RetrievalResult` so citation-style fields (ids, scores) can flow into
      :attr:`afterimage.types.GeneratedResponsePrompt.metadata` under
      :data:`RETRIEVAL_METADATA_KEY`.

    Retrievers that need more than ``(query: str)`` (locale, budgets, auth) should be
    **configured objects** holding that state, not free functions.
    """

    @abstractmethod
    def get_context(self, query: str) -> str:
        """Retrieve context based on the query (sync).

        Implementations that only support async embedding backends should implement
        :meth:`aget_context` and may raise from here when called inside a running event loop.
        """
        pass


async def _aget_or_thread(retriever: ContextRetriever, query: str) -> str:
    """Use ``retriever.aget_context`` when present, else run sync ``get_context`` in a thread."""
    if hasattr(retriever, "aget_context"):
        return await retriever.aget_context(query)  # type: ignore[union-attr]
    return await asyncio.to_thread(retriever.get_context, query)


def get_retrieval_result_sync(
    retriever: ContextRetriever, query: str
) -> RetrievalResult:
    """Sync retrieval; prefers :meth:`get_context_with_metadata` when implemented."""
    if hasattr(retriever, "get_context_with_metadata"):
        out = retriever.get_context_with_metadata(query)  # type: ignore[union-attr]
        if not isinstance(out, RetrievalResult):
            raise TypeError(
                "get_context_with_metadata must return afterimage.retrievers.RetrievalResult"
            )
        return out
    return RetrievalResult(context=retriever.get_context(query), metadata={})


async def aget_retrieval_result(
    retriever: ContextRetriever, query: str
) -> RetrievalResult:
    """Async retrieval; prefers ``aget_context_with_metadata`` then ``get_context_with_metadata``."""
    if hasattr(retriever, "aget_context_with_metadata"):
        out = await retriever.aget_context_with_metadata(query)  # type: ignore[union-attr]
        if not isinstance(out, RetrievalResult):
            raise TypeError(
                "aget_context_with_metadata must return afterimage.retrievers.RetrievalResult"
            )
        return out
    if hasattr(retriever, "get_context_with_metadata"):
        out = await asyncio.to_thread(
            retriever.get_context_with_metadata,  # type: ignore[union-attr]
            query,
        )
        if not isinstance(out, RetrievalResult):
            raise TypeError(
                "get_context_with_metadata must return afterimage.retrievers.RetrievalResult"
            )
        return out
    text = await _aget_or_thread(retriever, query)
    return RetrievalResult(context=text, metadata={})


class StaticContextRetriever:
    """Returns a fixed context string (and optional metadata) for every query.

    Useful in tests and minimal tutorials without Qdrant or embeddings.
    """

    def __init__(
        self,
        context: str,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self._context = context
        self._metadata = dict(metadata) if metadata else {}

    def get_context(self, query: str) -> str:
        return self._context

    def get_context_with_metadata(self, query: str) -> RetrievalResult:
        return RetrievalResult(context=self._context, metadata=dict(self._metadata))

    async def aget_context(self, query: str) -> str:
        return self._context

    async def aget_context_with_metadata(self, query: str) -> RetrievalResult:
        return RetrievalResult(context=self._context, metadata=dict(self._metadata))


@dataclass
class CacheEntry:
    """Represents a cached context with timestamp."""

    context: str
    timestamp: float


class LRUCache:
    """Least Recently Used (LRU) cache implementation."""

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Optional[CacheEntry]:
        if key not in self.cache:
            return None

        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: CacheEntry):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class CacheRetriever(ContextRetriever):
    """Wraps any retriever with caching capabilities."""

    def __init__(
        self,
        base_retriever: ContextRetriever,
        cache_size: int = 1000,
        ttl: int = 3600,  # seconds
    ):
        self.retriever = base_retriever
        self.cache = LRUCache(cache_size)
        self.ttl = ttl

    def get_context(self, query: str) -> str:
        cached = self.cache.get(query)
        current_time = time.time()

        if cached and (current_time - cached.timestamp) < self.ttl:
            return cached.context

        context = self.retriever.get_context(query)
        self.cache.put(query, CacheEntry(context, current_time))
        return context

    async def aget_context(self, query: str) -> str:
        """Async cache; uses underlying ``aget_context`` when available."""
        cached = self.cache.get(query)
        current_time = time.time()

        if cached and (current_time - cached.timestamp) < self.ttl:
            return cached.context

        context = await _aget_or_thread(self.retriever, query)
        self.cache.put(query, CacheEntry(context, current_time))
        return context


class ChainedRetriever(ContextRetriever):
    """Combines multiple retrievers in sequence, using fallbacks."""

    def __init__(
        self,
        retrievers: List[ContextRetriever],
        min_context_length: int = 50,
        separator: str = "\n\n",
    ):
        assert len(retrievers) > 0, "Must provide at least one retriever"
        self.retrievers = retrievers
        self.min_length = min_context_length
        self.separator = separator

    def get_context(self, query: str) -> str:
        contexts = []

        for retriever in self.retrievers:
            context = retriever.get_context(query)
            if context and not is_no_retrieval_context(context):
                contexts.append(context)

            combined = self.separator.join(contexts)
            if len(combined) >= self.min_length:
                return combined

        return self.separator.join(contexts) if contexts else NO_RETRIEVAL_CONTEXT

    async def aget_context(self, query: str) -> str:
        """Sequential retrieval; each stage uses ``aget_context`` when available."""
        contexts: List[str] = []

        for retriever in self.retrievers:
            context = await _aget_or_thread(retriever, query)
            if context and not is_no_retrieval_context(context):
                contexts.append(context)

            combined = self.separator.join(contexts)
            if len(combined) >= self.min_length:
                return combined

        return self.separator.join(contexts) if contexts else NO_RETRIEVAL_CONTEXT


class EnsembleRetriever(ContextRetriever):
    """Combines results from multiple retrievers with weighted scoring."""

    def __init__(
        self,
        retrievers: List[Tuple[ContextRetriever, float]],
        aggregation_method: str = "weighted_merge",
        separator: str = "\n\n",
        max_contexts: int = 3,
    ):
        assert len(retrievers) > 0, "Must provide at least one retriever"
        assert all(w >= 0 for _, w in retrievers), "Weights must be non-negative"
        assert sum(w for _, w in retrievers) > 0, "At least one weight must be positive"

        self.retrievers = retrievers
        self.method = aggregation_method
        self.separator = separator
        self.max_contexts = max_contexts

    def get_context(self, query: str) -> str:
        all_contexts = []

        for retriever, weight in self.retrievers:
            if weight > 0:
                context = retriever.get_context(query)
                if context and not is_no_retrieval_context(context):
                    all_contexts.append((context, weight))

        if not all_contexts:
            return NO_RETRIEVAL_CONTEXT

        if self.method == "weighted_merge":
            sorted_contexts = sorted(all_contexts, key=lambda x: x[1], reverse=True)
            selected = [context for context, _ in sorted_contexts[: self.max_contexts]]
        else:
            selected = []
            while all_contexts and len(selected) < self.max_contexts:
                context, _ = all_contexts.pop(0)
                selected.append(context)

        return self.separator.join(selected)

    async def aget_context(self, query: str) -> str:
        """Parallel fetch from weighted retrievers, then same aggregation as :meth:`get_context`."""

        async def _fetch(
            retriever: ContextRetriever, weight: float
        ) -> Optional[Tuple[str, float]]:
            context = await _aget_or_thread(retriever, query)
            if context and not is_no_retrieval_context(context):
                return (context, weight)
            return None

        pairs = [(r, w) for r, w in self.retrievers if w > 0]
        results = await asyncio.gather(*[_fetch(r, w) for r, w in pairs])
        all_contexts = [x for x in results if x is not None]

        if not all_contexts:
            return NO_RETRIEVAL_CONTEXT

        if self.method == "weighted_merge":
            sorted_contexts = sorted(all_contexts, key=lambda x: x[1], reverse=True)
            selected = [context for context, _ in sorted_contexts[: self.max_contexts]]
        else:
            selected = []
            while all_contexts and len(selected) < self.max_contexts:
                context, _ = all_contexts.pop(0)
                selected.append(context)

        return self.separator.join(selected)


class QdrantRetriever(ContextRetriever):
    """Context retrieval from Qdrant using an async :class:`~afterimage.providers.embedding_providers.EmbeddingProvider` or a local SentenceTransformer.

    Pass exactly one of ``embedding_provider`` (recommended; API or process pool) or
    ``embedding_model`` (HuggingFace id, or loaded SentenceTransformer — requires
    ``embeddings-local`` extra when loading by id).

    For **async** generation (``aget_context`` / ``aget_context_with_metadata``), pass
    ``async_client`` — a :class:`qdrant_client.AsyncQdrantClient` — so vector search uses
    native async ``query_points`` and does not block the event loop on HTTP I/O. The
    sync ``client`` is still used for :meth:`get_context` / :meth:`get_context_with_metadata`
    when no event loop restriction applies; those calls use ``query_points`` on the sync
    client (or ``asyncio.to_thread`` when only the sync client is available inside async
    retrieval).
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding_model: Union[str, Any, None] = None,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        async_client: AsyncQdrantClient | None = None,
        payload_key: str = "text",
        limit: int = 3,
        score_threshold: float = 0.5,
        separator: str = "\n" + "-" * 80 + "\n\n",
    ):
        if embedding_provider is not None and embedding_model is not None:
            raise ValueError("Pass only one of embedding_provider or embedding_model")
        if embedding_provider is None and embedding_model is None:
            raise ValueError("Pass embedding_provider or embedding_model")

        self.client = client
        self._async_client = async_client
        self.collection_name = collection_name
        self.payload_key = payload_key
        self.limit = limit
        self.score_threshold = score_threshold
        self.separator = separator

        if embedding_provider is not None:
            self._embedding_provider = embedding_provider
            self._st_model = None
        else:
            self._embedding_provider = None
            ST = _require_sentence_transformers()
            if isinstance(embedding_model, str):
                self._st_model = ST(embedding_model)
            else:
                self._st_model = embedding_model

    def _points_to_retrieval(self, points: Sequence[Any] | None) -> RetrievalResult:
        if not points:
            return RetrievalResult(context=NO_RETRIEVAL_CONTEXT, metadata={})
        contexts: list[str] = []
        hits: list[dict[str, Any]] = []
        for result in points:
            payload = getattr(result, "payload", None) or {}
            if self.payload_key not in payload:
                continue
            raw = payload[self.payload_key]
            contexts.append(str(raw))
            hits.append(
                {
                    "id": getattr(result, "id", None),
                    "score": getattr(result, "score", None),
                }
            )
        joined = self.separator.join(contexts) if contexts else NO_RETRIEVAL_CONTEXT
        metadata: dict[str, Any] = (
            {"hits": hits, "collection_name": self.collection_name} if hits else {}
        )
        return RetrievalResult(context=joined, metadata=metadata)

    def _query_points_sync(self, query_vector: list[float]) -> RetrievalResult:
        resp = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=self.limit,
            score_threshold=self.score_threshold,
        )
        return self._points_to_retrieval(resp.points)

    async def _query_points_async(self, query_vector: list[float]) -> RetrievalResult:
        if self._async_client is not None:
            resp = await self._async_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=self.limit,
                score_threshold=self.score_threshold,
            )
            return self._points_to_retrieval(resp.points)
        return await asyncio.to_thread(self._query_points_sync, query_vector)

    async def aget_context_with_metadata(self, query: str) -> RetrievalResult:
        """Vector search returning context text plus hit ids and scores."""
        if self._embedding_provider is not None:
            vectors = await self._embedding_provider.embed([query])
            query_vector = vectors[0]
        else:
            query_vector = await asyncio.to_thread(
                lambda: self._st_model.encode(query).tolist()
            )
        return await self._query_points_async(query_vector)

    async def aget_context(self, query: str) -> str:
        """Retrieve context; uses ``embedding_provider.embed`` or encodes in a thread."""
        return (await self.aget_context_with_metadata(query)).context

    def get_context_with_metadata(self, query: str) -> RetrievalResult:
        """Sync vector search with structured metadata.

        When using ``embedding_provider``, uses :func:`asyncio.run` only if no event
        loop is running; inside ``async def`` code, use :meth:`aget_context_with_metadata`.
        """
        if self._embedding_provider is not None:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.aget_context_with_metadata(query))
            raise RuntimeError(
                "QdrantRetriever with embedding_provider cannot use get_context_with_metadata() "
                "while an asyncio event loop is running; await aget_context_with_metadata() instead."
            )

        query_vector = self._st_model.encode(query).tolist()
        return self._query_points_sync(query_vector)

    def get_context(self, query: str) -> str:
        """Sync retrieval; delegates to :meth:`get_context_with_metadata`."""
        return self.get_context_with_metadata(query).context
