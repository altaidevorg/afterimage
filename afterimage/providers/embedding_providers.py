"""Async embedding providers (API and process-based).

This module defines a single async contract (:class:`EmbeddingProvider`) and
concrete backends for OpenAI-compatible APIs, Google Gemini, and local
SentenceTransformer models running in worker processes so the asyncio event
loop is not blocked by CPU/GPU embedding work.
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from google import genai
from openai import AsyncOpenAI

from ..key_management import SmartKeyPool


def _chunk_list(items: Sequence[str], size: int) -> list[list[str]]:
    """Split ``items`` into contiguous chunks of at most ``size`` elements.

    Args:
        items: Strings to partition.
        size: Maximum chunk length; must be positive.

    Returns:
        List of chunks, each a list of strings. Empty ``items`` yields ``[]``.

    Raises:
        ValueError: If ``size`` is not positive.
    """
    if size <= 0:
        raise ValueError("Batch size must be positive")
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


async def _aclose_genai_client(client: genai.Client) -> None:
    """Best-effort shutdown of async HTTP resources on a google-genai client.

    Args:
        client: Client instance that may have opened aiohttp/httpx clients.
    """
    try:
        if hasattr(client, "aio"):
            api_client = client.aio._api_client
            if hasattr(api_client, "_aiohttp_session") and api_client._aiohttp_session:
                await api_client._aiohttp_session.close()
            if (
                hasattr(api_client, "_async_httpx_client")
                and api_client._async_httpx_client
            ):
                await api_client._async_httpx_client.aclose()
    except Exception:
        pass


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for async text embedding backends.

    Implementations return one dense vector per input string, preserve order,
    and may batch requests internally. Call :meth:`aclose` when the provider
    is no longer needed (required for process-based providers).
    """

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed each string into a floating-point vector.

        Args:
            texts: Input strings. Empty list returns ``[]``.

        Returns:
            Embeddings in the same order as ``texts``; each embedding is a
            list of floats (dimension is model-specific).

        Note:
            Callers must not assume a single HTTP or IPC round trip; large
            inputs may be split into batches by the implementation.
        """
        ...

    async def aclose(self) -> None:
        """Release resources held by this provider (pools, clients).

        Implementations should make this idempotent (safe to call multiple
        times). API-only providers may use a no-op.
        """
        ...


class _NoOpAcloseMixin:
    """Mixin adding a no-op :meth:`aclose` for stateless API providers."""

    async def aclose(self) -> None:
        """See :meth:`EmbeddingProvider.aclose`."""
        return None


class OpenAIEmbeddingProvider(_NoOpAcloseMixin):
    """Embeddings via the OpenAI async client (``embeddings.create``).

    Supports OpenAI and OpenAI-compatible servers via ``base_url``. Uses
    :class:`~afterimage.key_management.SmartKeyPool` for key rotation and error
    reporting consistent with chat providers.
    """

    def __init__(
        self,
        api_key: str | SmartKeyPool,
        model: str = "text-embedding-3-small",
        *,
        base_url: Optional[str] = None,
        max_batch_size: int = 128,
        extra_create_kwargs: Optional[dict[str, Any]] = None,
    ):
        """Initialize the OpenAI embedding provider.

        Args:
            api_key: A single API key or a :class:`~afterimage.key_management.SmartKeyPool`.
            model: Embedding model id passed to ``embeddings.create``.
            base_url: Optional base URL for compatible APIs (e.g. proxies).
            max_batch_size: Maximum number of texts per ``embeddings.create`` call.
            extra_create_kwargs: Additional keyword arguments forwarded to
                ``embeddings.create`` (e.g. dimensions for matryoshka models).
        """
        self.key_pool = (
            api_key
            if isinstance(api_key, SmartKeyPool)
            else SmartKeyPool.from_single_key(api_key)
        )
        self.model = model
        self.base_url = base_url
        self.max_batch_size = max_batch_size
        self._extra_create_kwargs = extra_create_kwargs or {}

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for ``texts`` using the configured model.

        Args:
            texts: Non-empty list of strings to embed, or empty for no work.

        Returns:
            One embedding per input string, in order.

        Raises:
            Exception: Propagates API errors after reporting the key to the pool.
        """
        if not texts:
            return []

        api_key = await self.key_pool.aget_next_key()
        client = AsyncOpenAI(api_key=api_key, base_url=self.base_url)
        try:
            out: list[list[float]] = []
            for batch in _chunk_list(texts, self.max_batch_size):
                response = await client.embeddings.create(
                    model=self.model,
                    input=batch,
                    **self._extra_create_kwargs,
                )
                ordered = sorted(response.data, key=lambda d: d.index)
                out.extend([list(d.embedding) for d in ordered])
            return out
        except Exception:
            await self.key_pool.areport_error(api_key)
            raise


class GeminiEmbeddingProvider(_NoOpAcloseMixin):
    """Embeddings via Google Gemini ``client.aio.models.embed_content``.

    Uses the async Gemini client, closes transient HTTP resources after each
    :meth:`embed` call, and integrates with :class:`~afterimage.key_management.SmartKeyPool`.
    """

    def __init__(
        self,
        api_key: str | SmartKeyPool,
        model: str = "text-embedding-004",
        *,
        max_batch_size: int = 128,
    ):
        """Initialize the Gemini embedding provider.

        Args:
            api_key: A single API key or a :class:`~afterimage.key_management.SmartKeyPool`.
            model: Gemini embedding model resource name or id.
            max_batch_size: Maximum number of strings per ``embed_content`` call.
        """
        self.key_pool = (
            api_key
            if isinstance(api_key, SmartKeyPool)
            else SmartKeyPool.from_single_key(api_key)
        )
        self.model = model
        self.max_batch_size = max_batch_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for ``texts`` using the configured Gemini model.

        Args:
            texts: Non-empty list of strings to embed, or empty for no work.

        Returns:
            One embedding per input string, in order.

        Raises:
            ValueError: If the API returns an embedding without ``values``.
            Exception: Propagates API errors after reporting the key to the pool.
        """
        if not texts:
            return []

        api_key = await self.key_pool.aget_next_key()
        client = genai.Client(api_key=api_key, vertexai=False)
        try:
            out: list[list[float]] = []
            for batch in _chunk_list(texts, self.max_batch_size):
                resp = await client.aio.models.embed_content(
                    model=self.model,
                    contents=batch,
                )
                for emb in resp.embeddings or []:
                    values = emb.values
                    if values is None:
                        raise ValueError("Gemini embedding missing values")
                    out.append(list(values))
            return out
        except Exception:
            await self.key_pool.areport_error(api_key)
            raise
        finally:
            await _aclose_genai_client(client)


# --- Process pool workers (top-level for pickling on Windows) ---

_worker_model: Any = None


def _process_pool_init(model_name: str) -> None:
    """Load ``SentenceTransformer`` once in each worker process.

    Args:
        model_name: HuggingFace model id or local path.

    Raises:
        ImportError: If ``sentence_transformers`` is not installed.
    """
    global _worker_model
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "ProcessEmbeddingProvider requires sentence-transformers. "
            'Install with pip install "afterimage[embeddings-local]" or '
            "pip install sentence-transformers"
        ) from e
    _worker_model = SentenceTransformer(model_name)


def _process_pool_embed_batch(texts: list[str]) -> list[list[float]]:
    """Encode a batch in a worker process (must run after pool initializer).

    Args:
        texts: Batch of strings to encode.

    Returns:
        List of embedding vectors as plain Python floats.

    Raises:
        RuntimeError: If the worker model was not initialized.
    """
    global _worker_model
    if _worker_model is None:
        raise RuntimeError("Embedding worker model not initialized")
    import numpy as np

    arr = _worker_model.encode(texts, convert_to_numpy=True)
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return [row.tolist() for row in arr]


class ProcessEmbeddingProvider:
    """Local embeddings using SentenceTransformer in a process pool.

    Inference runs in child processes so the host asyncio loop is not blocked.
    The model is loaded once per worker via the pool initializer. Call
    :meth:`aclose` to shut down workers when finished.
    """

    def __init__(
        self,
        model_name: str,
        *,
        max_workers: int = 2,
        max_batch_size: int = 64,
    ):
        """Initialize configuration; the process pool starts on first :meth:`embed`.

        Args:
            model_name: HuggingFace model id or path passed to SentenceTransformer.
            max_workers: Number of worker processes.
            max_batch_size: Maximum strings passed to each worker call.

        Raises:
            ValueError: If ``max_workers`` is less than 1.
        """
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self._model_name = model_name
        self._max_workers = max_workers
        self._max_batch_size = max_batch_size
        self._executor: ProcessPoolExecutor | None = None
        self._closed = False

    def _get_executor(self) -> ProcessPoolExecutor:
        """Lazily create the process pool.

        Returns:
            Shared executor for this provider instance.

        Raises:
            RuntimeError: If :meth:`aclose` was already called.
        """
        if self._closed:
            raise RuntimeError("ProcessEmbeddingProvider is closed")
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self._max_workers,
                initializer=_process_pool_init,
                initargs=(self._model_name,),
            )
        return self._executor

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode texts in worker processes via :func:`asyncio.loop.run_in_executor`.

        Args:
            texts: Non-empty list of strings to embed, or empty for no work.

        Returns:
            One embedding per input string, in order.

        Raises:
            RuntimeError: If the provider was closed before or during use.
            ImportError: In workers if sentence-transformers is missing.
        """
        if not texts:
            return []

        loop = asyncio.get_running_loop()
        executor = self._get_executor()
        out: list[list[float]] = []
        for batch in _chunk_list(texts, self._max_batch_size):
            part: list[list[float]] = await loop.run_in_executor(
                executor,
                _process_pool_embed_batch,
                batch,
            )
            out.extend(part)
        return out

    async def aclose(self) -> None:
        """Shut down the process pool and mark this provider closed.

        Waits for workers to finish. Idempotent after the first call.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._closed = True


class EmbeddingProviderFactory:
    """Factory for constructing :class:`EmbeddingProvider` instances from config.

    Config dictionaries are intended to be JSON-serializable (aside from
    embedding-specific nested structures). Key names are matched case-insensitively.
    """

    @staticmethod
    def create(
        config: dict[str, Any],
        *,
        api_key: str | None = None,
        key_pool: SmartKeyPool | None = None,
    ) -> EmbeddingProvider:
        """Build a provider from a configuration mapping.

        Args:
            config: Must include ``type`` (``"openai"``, ``"gemini"``, or
                ``"process"``). Optional keys:

                * ``model`` — embedding model id for API providers.
                * ``model_path`` — HuggingFace id or path for ``process`` (also
                  ``model`` is accepted for process).
                * ``base_url`` — OpenAI-compatible base URL.
                * ``workers`` — process count for ``process`` (default ``2``).
                * ``max_batch_size`` — chunk size for batched calls.
                * ``api_key`` — inline secret when not using ``key_pool``.

            api_key: Optional default API key when ``key_pool`` is omitted
                (OpenAI/Gemini only).
            key_pool: Optional shared pool; takes precedence over ``api_key``
                and env vars for API providers.

        Returns:
            A concrete :class:`EmbeddingProvider`.

        Raises:
            ValueError: If ``type`` is missing, unknown, or required keys are absent.
        """
        cfg = {k.lower(): v for k, v in config.items()}
        provider_type = cfg.get("type")
        if not provider_type:
            raise ValueError("Embedding config requires 'type'")

        max_batch = cfg.get("max_batch_size")
        max_batch_int = int(max_batch) if max_batch is not None else None

        if provider_type == "openai":
            if key_pool is not None:
                pool = key_pool
            else:
                key = api_key or cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
                if not key:
                    raise ValueError(
                        "OpenAI embedding provider needs key_pool, api_key, config['api_key'], or OPENAI_API_KEY"
                    )
                pool = SmartKeyPool.from_single_key(str(key))
            model = cfg.get("model", "text-embedding-3-small")
            kwargs: dict[str, Any] = {
                "api_key": pool,
                "model": str(model),
                "base_url": cfg.get("base_url"),
            }
            if max_batch_int is not None:
                kwargs["max_batch_size"] = max_batch_int
            return OpenAIEmbeddingProvider(**kwargs)

        if provider_type == "gemini":
            if key_pool is not None:
                pool = key_pool
            else:
                key = api_key or cfg.get("api_key") or os.environ.get("GEMINI_API_KEY")
                if not key:
                    raise ValueError(
                        "Gemini embedding provider needs key_pool, api_key, config['api_key'], or GEMINI_API_KEY"
                    )
                pool = SmartKeyPool.from_single_key(str(key))
            model = cfg.get("model", "text-embedding-004")
            gkwargs: dict[str, Any] = {
                "api_key": pool,
                "model": str(model),
            }
            if max_batch_int is not None:
                gkwargs["max_batch_size"] = max_batch_int
            return GeminiEmbeddingProvider(**gkwargs)

        if provider_type == "process":
            model_name = cfg.get("model_path") or cfg.get("model")
            if not model_name:
                raise ValueError(
                    "Process embedding provider requires 'model_path' or 'model' (HuggingFace id or path)"
                )
            workers = int(cfg.get("workers", 2))
            pkwargs: dict[str, Any] = {
                "model_name": str(model_name),
                "max_workers": workers,
            }
            if max_batch_int is not None:
                pkwargs["max_batch_size"] = max_batch_int
            return ProcessEmbeddingProvider(**pkwargs)

        raise ValueError(f"Unknown embedding provider type: {provider_type!r}")
