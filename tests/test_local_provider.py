"""Tests for LocalLLMProvider."""

import pytest

from afterimage.providers.local_provider import LocalLLMProvider, _REQUEST_TIMEOUT
from afterimage.providers.llm_providers import LLMFactory, OpenRouterProvider


class TestLocalLLMProviderInit:
    def test_default_api_key(self):
        provider = LocalLLMProvider(base_url="http://localhost:8000/v1")
        assert provider.api_key == "not-needed"

    def test_custom_api_key(self):
        provider = LocalLLMProvider(
            base_url="http://localhost:8000/v1", api_key="custom-key"
        )
        assert provider.api_key == "custom-key"

    def test_no_smart_key_pool(self):
        provider = LocalLLMProvider(base_url="http://localhost:8000/v1")
        assert not hasattr(provider, "key_pool")

    def test_stores_base_url(self):
        provider = LocalLLMProvider(base_url="http://localhost:1234/v1")
        assert provider.base_url == "http://localhost:1234/v1"

    def test_timeout_is_extended(self):
        provider = LocalLLMProvider(base_url="http://localhost:8000/v1")
        client = provider._get_client()
        assert client.timeout == _REQUEST_TIMEOUT

    def test_async_client_timeout(self):
        provider = LocalLLMProvider(base_url="http://localhost:8000/v1")
        client = provider._get_async_client()
        assert client.timeout == _REQUEST_TIMEOUT


class TestLocalLLMProviderConnectionError:
    def test_connection_refused_gives_clear_message(self):
        provider = LocalLLMProvider(base_url="http://localhost:8000/v1")
        exc = ConnectionError("Connection refused")
        wrapped = provider._wrap_connection_error(exc)
        assert isinstance(wrapped, ConnectionRefusedError)
        assert "Is your model server running?" in str(wrapped)

    def test_non_connection_error_passes_through(self):
        provider = LocalLLMProvider(base_url="http://localhost:8000/v1")
        exc = ValueError("some other error")
        wrapped = provider._wrap_connection_error(exc)
        assert wrapped is exc


class TestLLMFactoryLocal:
    def test_factory_creates_local_provider(self):
        provider = LLMFactory.create(
            provider="local",
            model_name="test-model",
            base_url="http://localhost:8000/v1",
        )
        assert isinstance(provider, LocalLLMProvider)
        assert provider.model_name == "test-model"
        assert provider.base_url == "http://localhost:8000/v1"

    def test_factory_local_default_api_key(self):
        provider = LLMFactory.create(
            provider="local",
            base_url="http://localhost:8000/v1",
        )
        assert provider.api_key == "not-needed"

    def test_factory_local_ignores_smart_key_pool(self):
        from afterimage.key_management import SmartKeyPool

        pool = SmartKeyPool.from_single_key("test-key")
        provider = LLMFactory.create(
            provider="local",
            api_key=pool,
            base_url="http://localhost:8000/v1",
        )
        assert isinstance(provider, LocalLLMProvider)
        assert provider.api_key == "not-needed"

    def test_factory_local_with_system_instruction(self):
        provider = LLMFactory.create(
            provider="local",
            model_name="my-model",
            base_url="http://localhost:8000/v1",
            system_instruction="You are a helper.",
        )
        assert provider.system_instruction == "You are a helper."


class TestLLMFactoryOpenRouter:
    def test_factory_creates_openrouter_provider(self):
        provider = LLMFactory.create(
            provider="openrouter",
            model_name="openai/gpt-4o-mini",
            api_key="test-key",
        )
        assert isinstance(provider, OpenRouterProvider)
        assert provider.base_url == "https://openrouter.ai/api/v1"
        assert provider.model_name == "openai/gpt-4o-mini"


class TestLocalLLMProviderChatSession:
    def test_start_chat_returns_session(self):
        provider = LocalLLMProvider(
            base_url="http://localhost:8000/v1",
            model_name="test-model",
            system_instruction="Test prompt",
        )
        # start_chat creates an OpenAIChatSession which just stores the client
        # It shouldn't make any network calls
        session = provider.start_chat()
        assert session is not None
        assert session.model_name == "test-model"

    @pytest.mark.asyncio
    async def test_astart_chat_returns_session(self):
        provider = LocalLLMProvider(
            base_url="http://localhost:8000/v1",
            model_name="test-model",
            system_instruction="Test prompt",
        )
        session = await provider.astart_chat()
        assert session is not None
        assert session.model_name == "test-model"
