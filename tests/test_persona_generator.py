"""Tests for PersonaGenerator."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from afterimage.persona_generator import PersonaGenerator
from afterimage.providers import LLMProvider
from afterimage.storage import JSONLStorage
from afterimage.monitoring import GenerationMonitor
from afterimage.types import Document


@pytest.fixture
def persona_generator():
    storage_mock = MagicMock(spec=JSONLStorage)
    storage_mock.asave_documents = AsyncMock()
    monitor_mock = MagicMock(spec=GenerationMonitor)
    return PersonaGenerator(
        api_key="test_key",
        storage=storage_mock,
        monitor=monitor_mock,
    ), storage_mock, monitor_mock


def test_generate_success(persona_generator):
    generator, storage_mock, monitor_mock = persona_generator
    mock_response = MagicMock()
    mock_response.text = "Persona 1: A developer.\nPersona 2: A writer."
    mock_response.prompt_token_count = 10
    mock_response.completion_token_count = 20
    mock_response.total_token_count = 30
    mock_response.model_name = "mock_model"

    with patch("afterimage.persona_generator.LLMFactory") as mock_llm_factory:
        mock_llm = mock_llm_factory.create.return_value
        mock_llm.generate_content.return_value = mock_response

        personas = generator.generate_from_text("Sample text")

    assert len(personas) == 2
    assert personas[0] == "A developer."
    mock_llm.generate_content.assert_called_once()
    monitor_mock.track_generation.assert_called_once()
    _, kwargs = monitor_mock.track_generation.call_args
    assert kwargs["success"] is True
    assert kwargs["metadata"]["operation"] == "text_to_persona_generation"
    assert kwargs["prompt_token_count"] == 10
    assert kwargs["total_token_count"] == 30


@pytest.mark.asyncio
async def test_generate_for_documents_batching(persona_generator):
    generator, storage_mock, monitor_mock = persona_generator
    generator.agenerate_from_text = AsyncMock(return_value=["A persona."])

    await generator.generate_from_documents(["doc1", "doc2", "doc3"])

    assert generator.agenerate_from_text.call_count == 3
    assert storage_mock.asave_documents.call_count == 3
    args, _ = storage_mock.asave_documents.call_args
    assert isinstance(args[0][0], Document)
