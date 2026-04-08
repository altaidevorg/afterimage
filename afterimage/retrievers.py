from __future__ import annotations

import asyncio
import time
from abc import abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, Tuple, Union, runtime_checkable

from qdrant_client import QdrantClient

from .providers.embedding_providers import EmbeddingProvider


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
    """Protocol defining the interface for context retrieval strategies."""

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
            if context and context != "No relevant context found.":
                contexts.append(context)

            combined = self.separator.join(contexts)
            if len(combined) >= self.min_length:
                return combined

        return (
            self.separator.join(contexts) if contexts else "No relevant context found."
        )

    async def aget_context(self, query: str) -> str:
        """Sequential retrieval; each stage uses ``aget_context`` when available."""
        contexts: List[str] = []

        for retriever in self.retrievers:
            context = await _aget_or_thread(retriever, query)
            if context and context != "No relevant context found.":
                contexts.append(context)

            combined = self.separator.join(contexts)
            if len(combined) >= self.min_length:
                return combined

        return (
            self.separator.join(contexts) if contexts else "No relevant context found."
        )


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
                if context and context != "No relevant context found.":
                    all_contexts.append((context, weight))

        if not all_contexts:
            return "No relevant context found."

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
            if context and context != "No relevant context found.":
                return (context, weight)
            return None

        pairs = [(r, w) for r, w in self.retrievers if w > 0]
        results = await asyncio.gather(*[_fetch(r, w) for r, w in pairs])
        all_contexts = [x for x in results if x is not None]

        if not all_contexts:
            return "No relevant context found."

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
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding_model: Union[str, Any, None] = None,
        *,
        embedding_provider: EmbeddingProvider | None = None,
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

    def _search_with_vector(self, query_vector: list[float]) -> str:
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=self.limit,
            score_threshold=self.score_threshold,
        )

        contexts = []
        for result in search_results:
            if self.payload_key in result.payload:
                contexts.append(result.payload[self.payload_key])

        return (
            self.separator.join(contexts) if contexts else "No relevant context found."
        )

    async def aget_context(self, query: str) -> str:
        """Retrieve context; uses ``embedding_provider.embed`` or encodes in a thread."""
        if self._embedding_provider is not None:
            vectors = await self._embedding_provider.embed([query])
            query_vector = vectors[0]
        else:
            query_vector = await asyncio.to_thread(
                lambda: self._st_model.encode(query).tolist()
            )
        return self._search_with_vector(query_vector)

    def get_context(self, query: str) -> str:
        """Sync retrieval.

        When using ``embedding_provider``, this uses :func:`asyncio.run` only if no event
        loop is running; inside ``async def`` code, use :meth:`aget_context` instead.
        """
        if self._embedding_provider is not None:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.aget_context(query))
            raise RuntimeError(
                "QdrantRetriever with embedding_provider cannot use get_context() "
                "while an asyncio event loop is running; await aget_context() instead."
            )

        query_vector = self._st_model.encode(query).tolist()
        return self._search_with_vector(query_vector)
