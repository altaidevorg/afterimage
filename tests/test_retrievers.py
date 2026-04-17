"""Tests for retrievers and embedding-backed Qdrant retrieval."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from afterimage.retrievers import (
    NO_RETRIEVAL_CONTEXT,
    ChainedRetriever,
    EnsembleRetriever,
    QdrantRetriever,
)


@pytest.mark.asyncio
async def test_qdrant_retriever_aget_context_uses_embedding_provider():
    mock_client = MagicMock()
    hit = MagicMock()
    hit.payload = {"content": "retrieved chunk"}
    hit.id = "pt-1"
    hit.score = 0.91
    mock_resp = MagicMock()
    mock_resp.points = [hit]
    mock_client.query_points.return_value = mock_resp

    embed = MagicMock()
    embed.embed = AsyncMock(return_value=[[0.25, 0.5, 0.75]])

    r = QdrantRetriever(
        mock_client,
        "mycollection",
        embedding_provider=embed,
        payload_key="content",
        limit=2,
        score_threshold=0.1,
    )
    out = await r.aget_context("user question")

    assert "retrieved chunk" in out
    embed.embed.assert_awaited_once_with(["user question"])
    mock_client.query_points.assert_called()
    call_kw = mock_client.query_points.call_args.kwargs
    assert call_kw["collection_name"] == "mycollection"
    assert call_kw["query"] == [0.25, 0.5, 0.75]

    detail = await r.aget_context_with_metadata("user question")
    assert "retrieved chunk" in detail.context
    assert detail.metadata["hits"][0]["id"] == "pt-1"
    assert detail.metadata["hits"][0]["score"] == 0.91
    assert detail.metadata["collection_name"] == "mycollection"


@pytest.mark.asyncio
async def test_qdrant_retriever_aget_uses_async_client_when_provided():
    mock_sync = MagicMock()
    mock_async = MagicMock()
    hit = MagicMock()
    hit.payload = {"text": "async chunk"}
    hit.id = 42
    hit.score = 0.8
    mock_resp = MagicMock()
    mock_resp.points = [hit]
    mock_async.query_points = AsyncMock(return_value=mock_resp)

    embed = MagicMock()
    embed.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    r = QdrantRetriever(
        mock_sync,
        "col",
        embedding_provider=embed,
        async_client=mock_async,
        payload_key="text",
        limit=5,
        score_threshold=0.0,
    )
    out = await r.aget_context("q")
    assert "async chunk" in out
    mock_async.query_points.assert_awaited()
    mock_sync.query_points.assert_not_called()
    aq = mock_async.query_points.call_args.kwargs
    assert aq["collection_name"] == "col"
    assert aq["query"] == [0.1, 0.2, 0.3]


class _SyncShort:
    def get_context(self, query: str) -> str:
        return "x" * 15


class _AsyncLong:
    async def aget_context(self, query: str) -> str:
        return "y" * 40


@pytest.mark.asyncio
async def test_chained_retriever_aget_context_mixed_sync_async():
    chain = ChainedRetriever(
        [_SyncShort(), _AsyncLong()],
        min_context_length=50,
        separator=" | ",
    )
    out = await chain.aget_context("q")
    assert "x" * 15 in out
    assert "y" * 40 in out
    assert " | " in out


class _AsyncCtx:
    def __init__(self, text: str):
        self.text = text

    async def aget_context(self, query: str) -> str:
        return self.text


@pytest.mark.asyncio
async def test_ensemble_retriever_aget_context_parallel_weighted_merge():
    ens = EnsembleRetriever(
        [
            (_AsyncCtx("light context"), 1.0),
            (_AsyncCtx("heavy context"), 3.0),
        ],
        aggregation_method="weighted_merge",
        max_contexts=2,
    )
    out = await ens.aget_context("q")
    # Higher weight first in weighted_merge sort
    assert out.index("heavy context") < out.index("light context")


@pytest.mark.asyncio
async def test_ensemble_retriever_aget_context_all_empty():
    empty = MagicMock()
    empty.aget_context = AsyncMock(return_value=NO_RETRIEVAL_CONTEXT)

    ens = EnsembleRetriever([(empty, 1.0)])
    out = await ens.aget_context("q")
    assert out == NO_RETRIEVAL_CONTEXT
