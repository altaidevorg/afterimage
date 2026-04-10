"""Local model provider for OpenAI-compatible endpoints (vLLM, Ollama, llama.cpp).

Wraps :class:`OpenAIProvider` with local-friendly defaults:
- No API key required (uses ``"not-needed"`` placeholder)
- No SmartKeyPool rate-limiting
- Longer timeouts for local/CPU inference
- Clear error messages for connection failures
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from openai import AsyncOpenAI, OpenAI

from .llm_providers import (
    AsyncOpenAIChatSession,
    ChatSession,
    LLMResponse,
    OpenAIChatSession,
    OpenAIProvider,
    StructuredLLMResponse,
    T,
    _extract_reasoning_content,
)
from ..key_management import SmartKeyPool


# Timeouts for local inference (seconds)
_CONNECT_TIMEOUT = 30.0
_REQUEST_TIMEOUT = 300.0


class LocalLLMProvider:
    """LLM provider for local OpenAI-compatible servers.

    Unlike :class:`OpenAIProvider`, this provider:
    - Defaults ``api_key`` to ``"not-needed"`` (most local servers accept any string)
    - Does NOT use :class:`SmartKeyPool` — no rate limiting
    - Uses extended timeouts (30s connect, 300s request) for slow local inference
    - Raises clear messages on connection errors
    """

    def __init__(
        self,
        base_url: str,
        model_name: str = "default",
        api_key: str = "not-needed",
        system_instruction: str | None = None,
        **kwargs,
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self.system_instruction = system_instruction
        self.kwargs = {k: v for k, v in kwargs.items() if k != "safety_settings"}

    def _get_client(self) -> OpenAI:
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=_REQUEST_TIMEOUT,
        )

    def _get_async_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=_REQUEST_TIMEOUT,
        )

    def _wrap_connection_error(self, exc: Exception) -> Exception:
        """Wrap connection errors with a helpful message."""
        msg = str(exc).lower()
        if "connect" in msg or "refused" in msg or "unreachable" in msg:
            return ConnectionRefusedError(
                f"Could not connect to {self.base_url}. Is your model server running?"
            )
        return exc

    def generate_content(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        client = self._get_client()
        try:
            messages = []
            if self.system_instruction:
                messages.append({"role": "system", "content": self.system_instruction})
            messages.append({"role": "user", "content": prompt})

            current_kwargs = {**self.kwargs, **kwargs}

            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                **current_kwargs,
            )
            assistant_message = response.choices[0].message

            return LLMResponse(
                text=assistant_message.content or "",
                prompt_token_count=response.usage.prompt_tokens if response.usage else 0,
                completion_token_count=response.usage.completion_tokens if response.usage else 0,
                total_token_count=response.usage.total_tokens if response.usage else 0,
                finish_reason=response.choices[0].finish_reason,
                model_name=self.model_name,
                raw_response=response,
                reasoning_content=_extract_reasoning_content(assistant_message),
            )
        except Exception as exc:
            raise self._wrap_connection_error(exc) from exc

    async def agenerate_content(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        client = self._get_async_client()
        try:
            messages = []
            if self.system_instruction:
                messages.append({"role": "system", "content": self.system_instruction})
            messages.append({"role": "user", "content": prompt})

            current_kwargs = {**self.kwargs, **kwargs}

            response = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                **current_kwargs,
            )
            assistant_message = response.choices[0].message

            return LLMResponse(
                text=assistant_message.content or "",
                prompt_token_count=response.usage.prompt_tokens if response.usage else 0,
                completion_token_count=response.usage.completion_tokens if response.usage else 0,
                total_token_count=response.usage.total_tokens if response.usage else 0,
                finish_reason=response.choices[0].finish_reason,
                model_name=self.model_name,
                raw_response=response,
                reasoning_content=_extract_reasoning_content(assistant_message),
            )
        except Exception as exc:
            raise self._wrap_connection_error(exc) from exc

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        temperature: float = 0.7,
        **kwargs,
    ) -> StructuredLLMResponse[T]:
        client = self._get_client()
        try:
            messages = []
            if self.system_instruction:
                messages.append({"role": "system", "content": self.system_instruction})
            messages.append({"role": "user", "content": prompt})

            current_kwargs = {**self.kwargs, **kwargs}

            response = client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=schema,
                temperature=temperature,
                **current_kwargs,
            )
            assistant_message = response.choices[0].message

            return StructuredLLMResponse(
                text=assistant_message.content or "",
                parsed=assistant_message.parsed,
                prompt_token_count=response.usage.prompt_tokens if response.usage else 0,
                completion_token_count=response.usage.completion_tokens if response.usage else 0,
                total_token_count=response.usage.total_tokens if response.usage else 0,
                finish_reason=response.choices[0].finish_reason,
                model_name=self.model_name,
                raw_response=response,
                reasoning_content=_extract_reasoning_content(assistant_message),
            )
        except Exception as exc:
            raise self._wrap_connection_error(exc) from exc

    async def agenerate_structured(
        self,
        prompt: str,
        schema: Type[T],
        temperature: float = 0.7,
        **kwargs,
    ) -> StructuredLLMResponse[T]:
        client = self._get_async_client()
        try:
            messages = []
            if self.system_instruction:
                messages.append({"role": "system", "content": self.system_instruction})
            messages.append({"role": "user", "content": prompt})

            current_kwargs = {**self.kwargs, **kwargs}

            response = await client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=schema,
                temperature=temperature,
                **current_kwargs,
            )
            assistant_message = response.choices[0].message

            return StructuredLLMResponse(
                text=assistant_message.content or "",
                parsed=assistant_message.parsed,
                prompt_token_count=response.usage.prompt_tokens if response.usage else 0,
                completion_token_count=response.usage.completion_tokens if response.usage else 0,
                total_token_count=response.usage.total_tokens if response.usage else 0,
                finish_reason=response.choices[0].finish_reason,
                model_name=self.model_name,
                raw_response=response,
                reasoning_content=_extract_reasoning_content(assistant_message),
            )
        except Exception as exc:
            raise self._wrap_connection_error(exc) from exc

    def start_chat(
        self,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> ChatSession:
        client = self._get_client()
        try:
            return OpenAIChatSession(
                client=client,
                model_name=self.model_name,
                system_instruction=self.system_instruction,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_sequences=stop_sequences,
                **self.kwargs,
                **kwargs,
            )
        except Exception as exc:
            raise self._wrap_connection_error(exc) from exc

    async def astart_chat(
        self,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> ChatSession:
        client = self._get_async_client()
        try:
            return AsyncOpenAIChatSession(
                client=client,
                model_name=self.model_name,
                system_instruction=self.system_instruction,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_sequences=stop_sequences,
                **self.kwargs,
                **kwargs,
            )
        except Exception as exc:
            raise self._wrap_connection_error(exc) from exc
