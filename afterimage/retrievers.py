from abc import abstractmethod
from collections import OrderedDict
from typing import List, Optional, Protocol, Tuple, runtime_checkable
from dataclasses import dataclass
import time
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


@runtime_checkable
class ContextRetriever(Protocol):
    """Protocol defining the interface for context retrieval strategies."""

    @abstractmethod
    def get_context(self, query: str) -> str:
        """Retrieve context based on the query.

        Args:
            query: The query to search for relevant context

        Returns:
            str: Retrieved context as a string
        """
        pass


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

        # Move to end to show it was recently used
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
        """Initialize the cache retriever.

        Args:
            base_retriever: The underlying retriever to cache
            cache_size: Maximum number of entries to cache
            ttl: Time-to-live for cache entries in seconds
        """
        self.retriever = base_retriever
        self.cache = LRUCache(cache_size)
        self.ttl = ttl

    def get_context(self, query: str) -> str:
        # Check cache first
        cached = self.cache.get(query)
        current_time = time.time()

        if cached and (current_time - cached.timestamp) < self.ttl:
            return cached.context

        # Cache miss or expired - get fresh context
        context = self.retriever.get_context(query)
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
        """Initialize the chained retriever.

        Args:
            retrievers: List of retrievers to try in sequence
            min_context_length: Minimum acceptable context length
            separator: String to use when combining contexts
        """
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

            # If we have enough context, return it
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
        retrievers: List[Tuple[ContextRetriever, float]],  # (retriever, weight)
        aggregation_method: str = "weighted_merge",
        separator: str = "\n\n",
        max_contexts: int = 3,
    ):
        """Initialize the ensemble retriever.

        Args:
            retrievers: List of (retriever, weight) tuples
            aggregation_method: How to combine results ("weighted_merge" or "round_robin")
            separator: String to use when combining contexts
            max_contexts: Maximum number of contexts to include in final result
        """
        assert len(retrievers) > 0, "Must provide at least one retriever"
        assert all(w >= 0 for _, w in retrievers), "Weights must be non-negative"
        assert sum(w for _, w in retrievers) > 0, "At least one weight must be positive"

        self.retrievers = retrievers
        self.method = aggregation_method
        self.separator = separator
        self.max_contexts = max_contexts

    def get_context(self, query: str) -> str:
        all_contexts = []

        # Collect contexts from all retrievers
        for retriever, weight in self.retrievers:
            if weight > 0:  # Skip retrievers with zero weight
                context = retriever.get_context(query)
                if context and context != "No relevant context found.":
                    all_contexts.append((context, weight))

        if not all_contexts:
            return "No relevant context found."

        if self.method == "weighted_merge":
            # Sort by weight and take top contexts
            sorted_contexts = sorted(all_contexts, key=lambda x: x[1], reverse=True)
            selected = [context for context, _ in sorted_contexts[: self.max_contexts]]
        else:  # round_robin
            # Take one from each retriever until max_contexts
            selected = []
            while all_contexts and len(selected) < self.max_contexts:
                context, _ = all_contexts.pop(0)
                selected.append(context)

        return self.separator.join(selected)


class QdrantRetriever(ContextRetriever):
    """Implements context retrieval using Qdrant vector database."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding_model: SentenceTransformer | str,
        payload_key: str = "text",
        limit: int = 3,
        score_threshold: float = 0.5,
        separator: str = "\n" + "-" * 80 + "\n\n",
    ):
        """Initialize the Qdrant retriever.

        Args:
            client: Initialized QdrantClient for vector search
            collection_name: Name of the Qdrant collection to search
            embedding_model: SentenceTransformer model or name for embeddings
            payload_key: Key in the payload containing the text content
            limit: Maximum number of documents to retrieve
            score_threshold: Minimum similarity score to include results
            separator: String to use when combining multiple contexts
        """
        self.client = client
        self.collection_name = collection_name
        self.payload_key = payload_key
        self.limit = limit
        self.score_threshold = score_threshold
        self.separator = separator

        # Initialize embedding model
        if isinstance(embedding_model, str):
            self.model = SentenceTransformer(embedding_model)
        else:
            self.model = embedding_model

    def get_context(self, query: str) -> str:
        """Retrieve context from Qdrant using vector similarity search."""
        query_vector = self.model.encode(query).tolist()

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
