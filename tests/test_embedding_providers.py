"""Tests for async embedding providers."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from afterimage.key_management import SmartKeyPool
from afterimage.providers.embedding_providers import (
    EmbeddingProviderFactory,
    GeminiEmbeddingProvider,
    OpenAIEmbeddingProvider,
    ProcessEmbeddingProvider,
    _chunk_list,
)


def test_chunk_list():
    assert _chunk_list([], 3) == []
    assert _chunk_list(["a", "b", "c", "d"], 2) == [["a", "b"], ["c", "d"]]
    with pytest.raises(ValueError):
        _chunk_list(["a"], 0)


@pytest.mark.asyncio
async def test_openai_embedding_provider_embed_empty():
    p = OpenAIEmbeddingProvider(api_key="sk-test", max_batch_size=10)
    assert await p.embed([]) == []


@pytest.mark.asyncio
async def test_openai_embedding_provider_embed_batches():
    pool = SmartKeyPool.from_single_key("sk-test")

    mock_emb = MagicMock()
    mock_emb.index = 0
    mock_emb.embedding = [0.1, 0.2]
    mock_emb2 = MagicMock()
    mock_emb2.index = 1
    mock_emb2.embedding = [0.3, 0.4]

    mock_create = AsyncMock(
        return_value=MagicMock(data=[mock_emb, mock_emb2]),
    )
    mock_client_instance = MagicMock()
    mock_client_instance.embeddings.create = mock_create

    with patch(
        "afterimage.providers.embedding_providers.AsyncOpenAI",
        return_value=mock_client_instance,
    ):
        p = OpenAIEmbeddingProvider(api_key=pool, model="text-embedding-3-small", max_batch_size=10)
        out = await p.embed(["hello", "world"])

    assert len(out) == 2
    assert out[0] == [0.1, 0.2]
    assert out[1] == [0.3, 0.4]
    mock_create.assert_awaited_once()
    call_kw = mock_create.await_args.kwargs
    assert call_kw["model"] == "text-embedding-3-small"
    assert call_kw["input"] == ["hello", "world"]


@pytest.mark.asyncio
async def test_gemini_embedding_provider_embed_empty():
    p = GeminiEmbeddingProvider(api_key="g-test", max_batch_size=8)
    assert await p.embed([]) == []


@pytest.mark.asyncio
async def test_gemini_embedding_provider_embed():
    pool = SmartKeyPool.from_single_key("g-test")

    emb1 = MagicMock()
    emb1.values = [1.0, 0.0]
    emb2 = MagicMock()
    emb2.values = [0.0, 1.0]
    mock_resp = MagicMock()
    mock_resp.embeddings = [emb1, emb2]

    mock_embed_content = AsyncMock(return_value=mock_resp)
    mock_aio_models = MagicMock()
    mock_aio_models.embed_content = mock_embed_content
    mock_aio = MagicMock()
    mock_aio.models = mock_aio_models
    mock_client_instance = MagicMock()
    mock_client_instance.aio = mock_aio

    with patch(
        "afterimage.providers.embedding_providers.genai.Client",
        return_value=mock_client_instance,
    ), patch(
        "afterimage.providers.embedding_providers._aclose_genai_client",
        new_callable=AsyncMock,
    ):
        p = GeminiEmbeddingProvider(api_key=pool, model="text-embedding-004")
        out = await p.embed(["a", "b"])

    assert out == [[1.0, 0.0], [0.0, 1.0]]
    mock_embed_content.assert_awaited_once()
    assert mock_embed_content.await_args.kwargs["model"] == "text-embedding-004"
    assert mock_embed_content.await_args.kwargs["contents"] == ["a", "b"]


def test_embedding_provider_factory_openai_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OpenAI embedding"):
        EmbeddingProviderFactory.create({"type": "openai"})


def test_embedding_provider_factory_gemini_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Gemini embedding"):
        EmbeddingProviderFactory.create({"type": "gemini"})


def test_embedding_provider_factory_process_requires_model():
    with pytest.raises(ValueError, match="model_path"):
        EmbeddingProviderFactory.create({"type": "process"})


def test_embedding_provider_factory_unknown_type():
    with pytest.raises(ValueError, match="Unknown embedding"):
        EmbeddingProviderFactory.create({"type": "voodoo"})


def test_embedding_provider_factory_openai_with_pool():
    pool = SmartKeyPool.from_single_key("sk-x")
    p = EmbeddingProviderFactory.create({"type": "openai", "model": "text-embedding-3-small"}, key_pool=pool)
    assert isinstance(p, OpenAIEmbeddingProvider)


def test_embedding_provider_factory_gemini_with_pool():
    pool = SmartKeyPool.from_single_key("g-x")
    p = EmbeddingProviderFactory.create({"type": "gemini"}, key_pool=pool)
    assert isinstance(p, GeminiEmbeddingProvider)


def test_embedding_provider_factory_process():
    p = EmbeddingProviderFactory.create(
        {"type": "process", "model_path": "sentence-transformers/all-MiniLM-L6-v2", "workers": 1},
    )
    assert isinstance(p, ProcessEmbeddingProvider)


@pytest.mark.asyncio
async def test_process_embedding_provider_aclose_without_embed():
    p = ProcessEmbeddingProvider("sentence-transformers/all-MiniLM-L6-v2", max_workers=1)
    await p.aclose()


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("AFTERIMAGE_TEST_PROCESS_EMBED") != "1",
    reason="Set AFTERIMAGE_TEST_PROCESS_EMBED=1 to run (loads SentenceTransformer in subprocess)",
)
async def test_process_embedding_provider_embed_smoke():
    p = ProcessEmbeddingProvider(
        "sentence-transformers/all-MiniLM-L6-v2",
        max_workers=1,
        max_batch_size=8,
    )
    try:
        out = await p.embed(["hello", "world"])
        assert len(out) == 2
        assert len(out[0]) > 0
        assert len(out[1]) > 0
    finally:
        await p.aclose()
