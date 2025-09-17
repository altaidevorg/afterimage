import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock

from afterimage.async_conversation_generator import AsyncConversationGenerator
from afterimage.providers.llm_providers import LLMResponse, ChatSession


class MockChatSession(ChatSession):
    def __init__(self):
        super().__init__()
        self.history = []

    async def asend_message(self, message, temperature=0.7, **kwargs) -> LLMResponse:
        self.history.append(message)
        return LLMResponse(
            text="mocked response",
            tokens_used=10,
            finish_reason="stop",
            model_name="mock_model",
            raw_response=None,
        )


class MockLLMProvider:
    async def agenerate_content(self, prompt, **kwargs) -> LLMResponse:
        return LLMResponse(
            text="mocked correspondent prompt",
            tokens_used=10,
            finish_reason="stop",
            model_name="mock_model",
            raw_response=None,
        )

    async def astart_chat(self, **kwargs) -> ChatSession:
        return MockChatSession()


class TestAsyncConversationGenerator(unittest.TestCase):
    def test_generate(self):
        async def run_test():
            # Mock LLMFactory to return our mock provider
            from afterimage.providers import llm_providers
            llm_providers.LLMFactory.create = MagicMock(return_value=MockLLMProvider())

            # Initialize the generator
            generator = AsyncConversationGenerator(
                respondent_prompt="You are a helpful assistant.",
                api_key="mock_key",
            )
            await generator.initialize()

            # Run the generator
            await generator.generate(num_dialogs=1)

            # Check that the storage has been called
            # (in a real scenario, we would mock the storage and check calls to it)
            # For this simple test, we just check that it runs without errors.

        asyncio.run(run_test())

if __name__ == "__main__":
    unittest.main()
