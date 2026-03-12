"""Tests for instruction generator callbacks."""
import pytest
from unittest.mock import MagicMock

from afterimage.callbacks.instruction_generator_callbacks import (
    ContextualInstructionGeneratorCallback,
    PersonaInstructionGeneratorCallback,
    ToolCallingInstructionGeneratorCallback,
    LLMFactory,
)
from afterimage.common import GeneratedInstructions
from afterimage.providers import InMemoryDocumentProvider
from afterimage.types import Document


class MockLLM:
    def generate_structured(self, prompt, schema):
        return MagicMock(
            parsed=MagicMock(instructions=["Test instruction"]),
            prompt_token_count=10,
            completion_token_count=10,
            total_token_count=20,
            finish_reason="stop",
            model_name="mock_model",
        )

    async def agenerate_structured(self, prompt, schema):
        return MagicMock(
            parsed=MagicMock(instructions=["Test instruction"]),
            prompt_token_count=10,
            completion_token_count=10,
            total_token_count=20,
            finish_reason="stop",
            model_name="mock_model",
        )


@pytest.fixture
def mock_llm_factory():
    mock_llm = MockLLM()
    factory = MagicMock()
    factory.create.return_value = mock_llm
    return factory, mock_llm


@pytest.fixture
def documents():
    return [Document(id="doc1", text="Context 1", personas=[])]


@pytest.fixture(autouse=True)
def patch_llm_factory(mock_llm_factory):
    factory, _ = mock_llm_factory
    original = LLMFactory.create
    LLMFactory.create = factory.create
    yield
    LLMFactory.create = original


def test_contextual_callback_generate(documents):
    callback = ContextualInstructionGeneratorCallback(
        api_key="test_key", documents=documents, num_random_contexts=1
    )
    result = callback.generate("Test prompt")
    assert result.instructions == ["Test instruction"]
    assert "Context 1" in result.context
    assert result.context_id == "doc1"
    assert result.context_ids == ["doc1"]


@pytest.mark.asyncio
async def test_contextual_callback_agenerate(documents):
    callback = ContextualInstructionGeneratorCallback(
        api_key="test_key", documents=documents, num_random_contexts=1
    )
    result = await callback.agenerate("Test prompt")
    assert result.instructions == ["Test instruction"]
    assert "Context 1" in result.context
    assert result.context_id == "doc1"
    assert result.context_ids == ["doc1"]


def test_persona_callback_generate(documents):
    callback = PersonaInstructionGeneratorCallback(
        api_key="test_key",
        documents=documents,
        num_random_contexts=1,
    )
    result = callback.generate("Test prompt {persona}")
    assert result.instructions == ["Test instruction"]
    assert result.persona == "A curious user"
    assert result.context_ids == ["doc1"]


@pytest.mark.asyncio
async def test_persona_callback_agenerate(documents):
    callback = PersonaInstructionGeneratorCallback(
        api_key="test_key", documents=documents, num_random_contexts=1
    )
    result = await callback.agenerate("Test prompt {persona}")
    assert result.instructions == ["Test instruction"]
    assert result.persona == "A curious user"
    assert result.context_ids == ["doc1"]


def test_contextual_callback_does_not_report_usage_before_generation_succeeds(documents):
    provider = InMemoryDocumentProvider(documents)
    callback = ContextualInstructionGeneratorCallback(
        api_key="test_key",
        documents=provider,
        num_random_contexts=1,
    )

    callback.generate("Test prompt")

    assert provider._doc_usage_counts["doc1"] == 0
    assert provider._doc_sampling_weights["doc1"] == 1.0


def test_persona_callback_does_not_report_usage_before_generation_succeeds(documents):
    provider = InMemoryDocumentProvider(documents)
    callback = PersonaInstructionGeneratorCallback(
        api_key="test_key",
        documents=provider,
        num_random_contexts=1,
    )

    callback.generate("Test prompt {persona}")

    assert provider._doc_usage_counts["doc1"] == 0
    assert provider._doc_sampling_weights["doc1"] == 1.0


def test_contextual_callback_returns_all_sampled_context_ids():
    provider = InMemoryDocumentProvider(
        [
            Document(id="doc1", text="Context 1", personas=[]),
            Document(id="doc2", text="Context 2", personas=[]),
        ]
    )
    callback = ContextualInstructionGeneratorCallback(
        api_key="test_key",
        documents=provider,
        num_random_contexts=2,
    )

    result = callback.generate("Test prompt")

    assert result.context_id == "doc1"
    assert result.context_ids == ["doc1", "doc2"]


def test_tool_calling_callback_returns_all_sampled_context_ids():
    provider = InMemoryDocumentProvider(
        [
            Document(id="doc1", text="Context 1", personas=[]),
            Document(id="doc2", text="Context 2", personas=[]),
        ]
    )
    callback = ToolCallingInstructionGeneratorCallback(
        api_key="test_key",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup_status",
                    "description": "Look up ticket status",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        documents=provider,
        num_random_contexts=2,
    )

    result = callback.generate("Test prompt")

    assert result.context_id == "doc1"
    assert result.context_ids == ["doc1", "doc2"]


@pytest.mark.asyncio
async def test_tool_calling_callback_agenerate_returns_all_sampled_context_ids():
    provider = InMemoryDocumentProvider(
        [
            Document(id="doc1", text="Context 1", personas=[]),
            Document(id="doc2", text="Context 2", personas=[]),
        ]
    )
    callback = ToolCallingInstructionGeneratorCallback(
        api_key="test_key",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup_status",
                    "description": "Look up ticket status",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        documents=provider,
        num_random_contexts=2,
    )

    result = await callback.agenerate("Test prompt")

    assert result.context_id == "doc1"
    assert result.context_ids == ["doc1", "doc2"]


def test_generated_instructions_context_ids_do_not_share_default_state():
    first = GeneratedInstructions(instructions=["One"], context="Context 1")
    second = GeneratedInstructions(instructions=["Two"], context="Context 2")

    first.context_ids.append("doc1")

    assert second.context_ids == []
