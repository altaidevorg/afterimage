"""Tests for AsyncConversationGenerator."""
import pytest
from unittest.mock import MagicMock, AsyncMock

from afterimage.async_conversation_generator import AsyncConversationGenerator
from afterimage.providers.llm_providers import LLMResponse, ChatSession
from afterimage.common import GeneratedInstructions


class MockInstructionCallback:
    """Minimal instruction callback that returns one instruction."""

    monitor = None

    def set_monitor(self, monitor):
        self.monitor = monitor

    async def acreate_correspondent_prompt(self, respondent_prompt):
        return "You are a curious user."

    async def acall(self, correspondent_prompt):
        return GeneratedInstructions(
            instructions=["First question?"],
            context="",
            context_id="test",
            persona="A curious user",
        )


class MockChatSession(ChatSession):
    def __init__(self):
        super().__init__()
        self.history = []

    async def asend_message(self, message, temperature=0.7, **kwargs) -> LLMResponse:
        self.history.append(message)
        return LLMResponse(
            text="mocked response",
            prompt_token_count=10,
            completion_token_count=5,
            total_token_count=15,
            finish_reason="stop",
            model_name="mock_model",
            raw_response=None,
        )


class MockLLMProvider:
    async def agenerate_content(self, prompt, **kwargs) -> LLMResponse:
        return LLMResponse(
            text="mocked correspondent prompt",
            prompt_token_count=10,
            completion_token_count=5,
            total_token_count=15,
            finish_reason="stop",
            model_name="mock_model",
            raw_response=None,
        )

    async def astart_chat(self, **kwargs) -> ChatSession:
        return MockChatSession()


@pytest.mark.asyncio
async def test_async_conversation_generator_generate():
    from afterimage.providers import llm_providers

    original_create = llm_providers.LLMFactory.create
    llm_providers.LLMFactory.create = MagicMock(return_value=MockLLMProvider())

    try:
        generator = AsyncConversationGenerator(
            respondent_prompt="You are a helpful assistant.",
            api_key="mock_key",
            instruction_generator_callback=MockInstructionCallback(),
        )
        await generator.generate(num_dialogs=1)
    finally:
        llm_providers.LLMFactory.create = original_create
