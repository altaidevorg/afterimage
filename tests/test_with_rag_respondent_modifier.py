"""Tests for :class:`~afterimage.callbacks.WithRAGRespondentPromptModifier` + retrieval metadata."""

import pytest

from afterimage.callbacks.respondent_prompt_modifiers import (
    WithRAGRespondentPromptModifier,
)
from afterimage.retrievers import RETRIEVAL_METADATA_KEY, StaticContextRetriever


@pytest.mark.asyncio
async def test_with_rag_agenerate_propagates_retrieval_metadata():
    retriever = StaticContextRetriever(
        "retrieved chunk body",
        metadata={"hits": [{"id": "doc-1", "score": 0.88}]},
    )
    mod = WithRAGRespondentPromptModifier(retriever=retriever)
    out = await mod.agenerate(
        respondent_prompt="You are a helpful assistant.",
        context="Session briefing text.",
        instruction="What standard applies?",
    )
    assert "retrieved chunk body" in out.context
    assert "Session briefing" in out.context
    assert RETRIEVAL_METADATA_KEY in out.metadata
    assert out.metadata[RETRIEVAL_METADATA_KEY]["hits"][0]["id"] == "doc-1"


@pytest.mark.asyncio
async def test_with_rag_agenerate_omits_empty_retrieval_metadata():
    retriever = StaticContextRetriever("plain chunk", metadata={})
    mod = WithRAGRespondentPromptModifier(retriever=retriever)
    out = await mod.agenerate(
        respondent_prompt="You are helpful.", context="", instruction="Q?"
    )
    assert RETRIEVAL_METADATA_KEY not in out.metadata


def test_with_rag_generate_sync_metadata():
    retriever = StaticContextRetriever("x", metadata={"source": "static"})
    mod = WithRAGRespondentPromptModifier(retriever=retriever)
    out = mod.generate("Assistant.", "brief", "Ask me.")
    assert out.metadata[RETRIEVAL_METADATA_KEY]["source"] == "static"
