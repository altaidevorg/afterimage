import pytest
from unittest.mock import MagicMock, AsyncMock
from pydantic import BaseModel

from afterimage.providers.llm_providers import (
    GeminiProvider,
    GeminiChatSession,
    AsyncGeminiChatSession,
    _extract_gemini_text_and_reasoning,
)


class SampleSchema(BaseModel):
    answer: str


def test_extract_gemini_text_and_reasoning_standard():
    response = MagicMock()
    part = MagicMock()
    part.thought = False
    part.text = "Hello world"
    candidate = MagicMock()
    candidate.content.parts = [part]
    response.candidates = [candidate]

    text, reasoning = _extract_gemini_text_and_reasoning(response)
    assert text == "Hello world"
    assert reasoning is None


def test_extract_gemini_text_and_reasoning_with_thought():
    response = MagicMock()
    thought_part = MagicMock()
    thought_part.thought = True
    thought_part.text = "Let me think step by step..."

    text_part = MagicMock()
    text_part.thought = False
    text_part.text = "Here is the final answer."

    candidate = MagicMock()
    candidate.content.parts = [thought_part, text_part]
    response.candidates = [candidate]

    text, reasoning = _extract_gemini_text_and_reasoning(response)
    assert text == "Here is the final answer."
    assert reasoning == "Let me think step by step..."


def test_extract_gemini_text_and_reasoning_fallback():
    response = MagicMock()
    response.candidates = None
    response.text = "Fallback text"

    text, reasoning = _extract_gemini_text_and_reasoning(response)
    assert text == "Fallback text"
    assert reasoning is None


def test_gemini_chat_session_send_message():
    mock_chat = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Chat answer"
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].finish_reason = "STOP"
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 5
    mock_response.usage_metadata.total_token_count = 15
    mock_chat.send_message.return_value = mock_response

    mock_client = MagicMock()
    session = GeminiChatSession(
        chat=mock_chat,
        client=mock_client,
        model_name="gemini-2.0-flash",
    )

    resp = session.send_message("Hello")
    assert resp.text == "Chat answer"
    assert resp.total_token_count == 15
    mock_chat.send_message.assert_called_once_with("Hello")


@pytest.mark.asyncio
async def test_async_gemini_chat_session_asend_message():
    mock_chat = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Async answer"
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].finish_reason = "STOP"
    mock_response.usage_metadata.prompt_token_count = 20
    mock_response.usage_metadata.candidates_token_count = 10
    mock_response.usage_metadata.total_token_count = 30
    mock_chat.send_message = AsyncMock(return_value=mock_response)

    mock_client = MagicMock()
    session = AsyncGeminiChatSession(
        chat=mock_chat,
        client=mock_client,
        model_name="gemini-2.0-flash",
    )

    resp = await session.asend_message("Hi async")
    assert resp.text == "Async answer"
    assert resp.total_token_count == 30
    mock_chat.send_message.assert_called_once_with("Hi async")


def test_gemini_provider_init():
    provider = GeminiProvider(api_key="test_key", model_name="gemini-2.0-flash")
    assert provider.model_name == "gemini-2.0-flash"
    assert provider.key_pool is not None
