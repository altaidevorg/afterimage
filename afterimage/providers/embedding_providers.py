"""Async embedding providers (API and process-based).

All providers implement :class:`EmbeddingProvider` with ``async def embed(texts) -> list[list[float]]``.
The asyncio event loop must not perform blocking local inference; use :class:`ProcessEmbeddingProvider`
for SentenceTransformer models.
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
    if size <= 0:
        raise ValueError("Batch size must be positive")
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


async def _aclose_genai_client(client: genai.Client) -> None:
    """Close async resources held by a google-genai client (mirrors GeminiProvider)."""
    try:
        if hasattr(client, "aio"):
            api_client = client.aio._api_client
            if (
                hasattr(api_client, "_aiohttp_session")
                and api_client._aiohttp_session
            ):
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
    """Unified async interface for text embeddings."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input string (same order).

        Empty ``texts`` returns ``[]``. Implementations may batch internally;
        callers should not assume a single HTTP request for large inputs.
        """
        ...

    async def aclose(self) -> None:
        """Release provider-held resources (pools, clients). Safe to call multiple times."""
        ...


class _NoOpAcloseMixin:
    async def aclose(self) -> None:
        return None


class OpenAIEmbeddingProvider(_NoOpAcloseMixin):
    """OpenAI (or OpenAI-compatible) embeddings via ``AsyncOpenAI.embeddings.create``."""

    def __init__(
        self,
        api_key: str | SmartKeyPool,
        model: str = "text-embedding-3-small",
        *,
        base_url: Optional[str] = None,
        max_batch_size: int = 128,
        extra_create_kwargs: Optional[dict[str, Any]] = None,
    ):
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
    """Google Gemini embeddings via ``client.aio.models.embed_content``."""

    def __init__(
        self,
        api_key: str | SmartKeyPool,
        model: str = "text-embedding-004",
        *,
        max_batch_size: int = 128,
    ):
        self.key_pool = (
            api_key
            if isinstance(api_key, SmartKeyPool)
            else SmartKeyPool.from_single_key(api_key)
        )
        self.model = model
        self.max_batch_size = max_batch_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
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
    global _worker_model
    if _worker_model is None:
        raise RuntimeError("Embedding worker model not initialized")
    import numpy as np

    arr = _worker_model.encode(texts, convert_to_numpy=True)
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return [row.tolist() for row in arr]


class ProcessEmbeddingProvider:
    """Local embeddings in worker processes (no event-loop blocking)."""

    def __init__(
        self,
        model_name: str,
        *,
        max_workers: int = 2,
        max_batch_size: int = 64,
    ):
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self._model_name = model_name
        self._max_workers = max_workers
        self._max_batch_size = max_batch_size
        self._executor: ProcessPoolExecutor | None = None
        self._closed = False

    def _get_executor(self) -> ProcessPoolExecutor:
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
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._closed = True


class EmbeddingProviderFactory:
    """Build an :class:`EmbeddingProvider` from JSON-friendly config dicts."""

    @staticmethod
    def create(
        config: dict[str, Any],
        *,
        api_key: str | None = None,
        key_pool: SmartKeyPool | None = None,
    ) -> EmbeddingProvider:
        """Instantiate a provider from ``config``.

        Expected keys:

        - ``type`` (required): ``"openai"`` | ``"gemini"`` | ``"process"``.
        - ``model`` (optional): embedding model id for API providers.
        - ``model_path`` (optional): HuggingFace id or path for ``process`` (alias of ``model``).
        - ``base_url`` (optional): OpenAI-compatible API base URL.
        - ``workers`` (optional): process pool size for ``process`` (default 2).
        - ``max_batch_size`` (optional): batch size for chunking (provider-specific default).
        - ``api_key`` (optional): inline key; otherwise ``api_key`` argument or env vars.

        Env fallbacks: ``OPENAI_API_KEY`` for openai, ``GEMINI_API_KEY`` for gemini.
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
                key = (
                    api_key
                    or cfg.get("api_key")
                    or os.environ.get("OPENAI_API_KEY")
                )
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
                key = (
                    api_key
                    or cfg.get("api_key")
                    or os.environ.get("GEMINI_API_KEY")
                )
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
