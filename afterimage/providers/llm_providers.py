import json
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Protocol, Type, TypeVar

from google import genai
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from ..common import default_safety_settings
from ..key_management import SmartKeyPool
from ..types import ConversationEntry

T = TypeVar("T", bound=BaseModel)


def _extract_reasoning_content(message: Any) -> str | None:
    """Best-effort extraction of reasoning/thinking text from OpenAI-compatible messages."""

    def _clean(value: Any) -> str | None:
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        return None

    for attr in ("reasoning_content", "reasoning", "thinking"):
        extracted = _clean(getattr(message, attr, None))
        if extracted:
            return extracted

    as_dict: dict[str, Any] | None = None
    if isinstance(message, dict):
        as_dict = message
    elif hasattr(message, "model_dump"):
        try:
            dumped = message.model_dump()
            if isinstance(dumped, dict):
                as_dict = dumped
        except Exception:
            as_dict = None

    if as_dict:
        for key in ("reasoning_content", "reasoning", "thinking"):
            extracted = _clean(as_dict.get(key))
            if extracted:
                return extracted

    return None


@dataclass
class CommonLLMResponse:
    """Standardized LLM response."""

    text: str
    prompt_token_count: int
    completion_token_count: int
    total_token_count: int
    finish_reason: str
    model_name: str
    raw_response: Any  # Provider-specific response


@dataclass
class LLMResponse(CommonLLMResponse):
    reasoning_content: str | None = None


@dataclass
class StructuredLLMResponse(CommonLLMResponse, Generic[T]):
    """Standardized LLM response with structured output."""

    parsed: T
    reasoning_content: str | None = None


class ChatSession:
    """Abstract chat session interface."""

    def __init__(self):
        self.token_count = 0

    def send_message(
        self, message: str | ConversationEntry, temperature: float = 0.7, **kwargs
    ) -> LLMResponse:
        """Send a message to the chat session."""
        raise NotImplementedError

    async def asend_message(
        self, message: str | ConversationEntry, temperature: float = 0.7, **kwargs
    ) -> LLMResponse:
        """Send a message to the chat session asynchronously."""
        raise NotImplementedError


class GeminiChatSession(ChatSession):
    """Gemini chat session implementation."""

    def __init__(self, chat, model_name: str):
        super().__init__()
        self.chat = chat
        self.model_name = model_name

    def send_message(
        self, message: str | ConversationEntry, temperature: float = 0.7, **kwargs
    ) -> LLMResponse:
        content = message if isinstance(message, str) else message.content

        response = self.chat.send_message(content)

        return LLMResponse(
            text=response.text,
            finish_reason=str(response.candidates[0].finish_reason),
            prompt_token_count=response.usage_metadata.prompt_token_count,
            completion_token_count=response.usage_metadata.candidates_token_count,
            total_token_count=response.usage_metadata.total_token_count,
            model_name=self.model_name,
            raw_response=response,
        )


class AsyncGeminiChatSession(ChatSession):
    """Asynchronous Gemini chat session implementation."""

    def __init__(self, chat, model_name: str):
        super().__init__()
        self.chat = chat
        self.model_name = model_name

    async def asend_message(
        self, message: str | ConversationEntry, temperature: float = 0.7, **kwargs
    ) -> LLMResponse:
        content = message if isinstance(message, str) else message.content

        response = await self.chat.send_message(content)

        total_token_count = response.usage_metadata.total_token_count
        self.token_count = total_token_count
        return LLMResponse(
            text=response.text,
            finish_reason=str(response.candidates[0].finish_reason),
            prompt_token_count=response.usage_metadata.prompt_token_count,
            completion_token_count=response.usage_metadata.candidates_token_count,
            total_token_count=total_token_count,
            model_name=self.model_name,
            raw_response=response,
        )


class OpenAIChatSession(ChatSession):
    """OpenAI chat session implementation."""

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__()
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop_sequences = stop_sequences
        self.kwargs = kwargs
        self.history = []
        if system_instruction:
            self.history.append({"role": "system", "content": system_instruction})

    def send_message(
        self, message: str | ConversationEntry, temperature: float = 0.7, **kwargs
    ) -> LLMResponse:
        content = message if isinstance(message, str) else message.content
        self.history.append({"role": "user", "content": content})

        current_kwargs = self.kwargs.copy()
        current_kwargs.update(kwargs)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.history,
            temperature=temperature,
            max_tokens=self.max_tokens,
            stop=self.stop_sequences,
            **current_kwargs,
        )

        assistant_message = response.choices[0].message
        assistant_reasoning = _extract_reasoning_content(assistant_message)
        self.history.append(
            {"role": assistant_message.role, "content": assistant_message.content}
        )

        total_token_count = response.usage.total_tokens
        self.token_count = total_token_count
        return LLMResponse(
            text=assistant_message.content or "",
            finish_reason=response.choices[0].finish_reason,
            prompt_token_count=response.usage.prompt_tokens,
            completion_token_count=response.usage.completion_tokens,
            total_token_count=total_token_count,
            model_name=self.model_name,
            raw_response=response,
            reasoning_content=assistant_reasoning,
        )


class AsyncOpenAIChatSession(ChatSession):
    """Asynchronous OpenAI chat session implementation."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model_name: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__()
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop_sequences = stop_sequences
        self.kwargs = kwargs
        self.history = []
        if system_instruction:
            self.history.append({"role": "system", "content": system_instruction})

    async def asend_message(
        self, message: str | ConversationEntry, temperature: float = 0.7, **kwargs
    ) -> LLMResponse:
        content = message if isinstance(message, str) else message.content
        self.history.append({"role": "user", "content": content})

        current_kwargs = self.kwargs.copy()
        current_kwargs.update(kwargs)

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=self.history,
            temperature=temperature,
            max_tokens=self.max_tokens,
            stop=self.stop_sequences,
            **current_kwargs,
        )

        assistant_message = response.choices[0].message
        assistant_reasoning = _extract_reasoning_content(assistant_message)
        self.history.append(
            {"role": assistant_message.role, "content": assistant_message.content}
        )

        total_token_count = response.usage.total_tokens
        self.token_count = total_token_count

        return LLMResponse(
            text=assistant_message.content or "",
            finish_reason=response.choices[0].finish_reason,
            prompt_token_count=response.usage.prompt_tokens,
            completion_token_count=response.usage.completion_tokens,
            total_token_count=total_token_count,
            model_name=self.model_name,
            raw_response=response,
            reasoning_content=assistant_reasoning,
        )


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    def generate_content(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion from prompt."""
        ...

    async def agenerate_content(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion from prompt asynchronously."""
        ...

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        temperature: float = 0.7,
        **kwargs,
    ) -> StructuredLLMResponse[T]:
        """Generate structured output that matches the given schema."""
        ...

    async def agenerate_structured(
        self,
        prompt: str,
        schema: Type[T],
        temperature: float = 0.7,
        **kwargs,
    ) -> StructuredLLMResponse[T]:
        """Generate structured output that matches the given schema asynchronously."""
        ...

    def start_chat(self, **kwargs) -> ChatSession:
        """Start a new chat session."""
        ...

    async def astart_chat(self, **kwargs) -> ChatSession:
        """Start a new chat session asynchronously."""
        ...


class GeminiProvider(LLMProvider):
    """Google Gemini implementation."""

    def _close_client(self, client: genai.Client):
        """Helper to close sync client resources."""
        try:
            # Close httpx client if it exists (private attribute)
            if hasattr(client, "_api_client"):
                api_client = client._api_client
                if hasattr(api_client, "_httpx_client") and api_client._httpx_client:
                    api_client._httpx_client.close()
        except Exception:
            pass

    async def _aclose_client(self, client: genai.Client):
        """Helper to close async client resources."""
        try:
            # Close aiohttp session if it exists (private attribute)
            # Accessing client.aio creates the async client wrappers,
            # so we check if _aio is already populated or if we can access the underlying api_client differently.
            # But client.aio corresponds to the AsyncClient wrapper.

            # If client.aio was used, it should be initialized.
            if hasattr(client, "aio"):
                api_client = client.aio._api_client
                if (
                    hasattr(api_client, "_aiohttp_session")
                    and api_client._aiohttp_session
                ):
                    await api_client._aiohttp_session.close()
                if (
                    hasattr(api_client, "_async_httpx_client")
                    and api_client._async_httpx_client
                ):
                    await api_client._async_httpx_client.aclose()
        except Exception:
            pass

    def __init__(
        self,
        api_key: str | SmartKeyPool,
        model_name: str = "gemini-2.0-flash",
        system_instruction: str | None = None,
        safety_settings: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ):
        self.key_pool = (
            api_key
            if isinstance(api_key, SmartKeyPool)
            else SmartKeyPool.from_single_key(api_key)
        )
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.safety_settings = safety_settings or default_safety_settings
        self.kwargs = kwargs

    def generate_content(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        api_key = self.key_pool.get_next_key()
        client = genai.Client(api_key=api_key, vertexai=False)

        try:
            generation_config = {
                "temperature": temperature,
                "system_instruction": self.system_instruction,
                "safety_settings": self.safety_settings,
                **self.kwargs,
            }
            if kwargs:
                generation_config.update(**kwargs)
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens
            if stop_sequences:
                generation_config["stop_sequences"] = stop_sequences

            response = client.models.generate_content(
                model=self.model_name, contents=prompt, config=generation_config
            )

            return LLMResponse(
                text=response.text,
                prompt_token_count=response.usage_metadata.prompt_token_count,
                completion_token_count=response.usage_metadata.candidates_token_count,
                total_token_count=response.usage_metadata.total_token_count,
                finish_reason=str(response.candidates[0].finish_reason),
                model_name=self.model_name,
                raw_response=response,
            )

        except Exception:
            self.key_pool.report_error(api_key)
            raise
        finally:
            self._close_client(client)

    async def agenerate_content(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        api_key = self.key_pool.get_next_key()
        client = genai.Client(api_key=api_key, vertexai=False)

        try:
            generation_config = {
                "temperature": temperature,
                "system_instruction": self.system_instruction,
                "safety_settings": self.safety_settings,
                **self.kwargs,
            }
            if kwargs:
                generation_config.update(**kwargs)
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens
            if stop_sequences:
                generation_config["stop_sequences"] = stop_sequences

            response = await client.aio.models.generate_content(
                model=self.model_name, contents=prompt, config=generation_config
            )

            return LLMResponse(
                text=response.text,
                prompt_token_count=response.usage_metadata.prompt_token_count,
                completion_token_count=response.usage_metadata.candidates_token_count,
                total_token_count=response.usage_metadata.total_token_count,
                finish_reason=str(response.candidates[0].finish_reason),
                model_name=self.model_name,
                raw_response=response,
            )

        except Exception:
            self.key_pool.report_error(api_key)
            raise
        finally:
            await self._aclose_client(client)

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        temperature: float = 0.7,
        **kwargs,
    ) -> StructuredLLMResponse[T]:
        api_key = self.key_pool.get_next_key()
        client = genai.Client(api_key=api_key, vertexai=False)

        try:
            generation_config = {
                "temperature": temperature,
                "system_instruction": self.system_instruction,
                "safety_settings": self.safety_settings,
                "response_mime_type": "application/json",
                "response_schema": schema,
                **self.kwargs,
            }
            if kwargs:
                generation_config.update(**kwargs)

            response = client.models.generate_content(
                model=self.model_name, contents=prompt, config=generation_config
            )

            return StructuredLLMResponse(
                text=response.text or "",
                parsed=response.parsed
                if hasattr(response, "parsed")
                else schema.model_validate_json(response.text),
                prompt_token_count=response.usage_metadata.prompt_token_count,
                completion_token_count=response.usage_metadata.candidates_token_count,
                total_token_count=response.usage_metadata.total_token_count,
                finish_reason=str(response.candidates[0].finish_reason),
                model_name=self.model_name,
                raw_response=response,
            )

        except Exception:
            self.key_pool.report_error(api_key)
            raise
        finally:
            self._close_client(client)

    async def agenerate_structured(
        self,
        prompt: str,
        schema: Type[T],
        temperature: float = 0.7,
        **kwargs,
    ) -> StructuredLLMResponse[T]:
        api_key = self.key_pool.get_next_key()
        client = genai.Client(api_key=api_key, vertexai=False)

        try:
            generation_config = {
                "temperature": temperature,
                "system_instruction": self.system_instruction,
                "safety_settings": self.safety_settings,
                "response_mime_type": "application/json",
                "response_schema": schema,
                **self.kwargs,
            }
            if kwargs:
                generation_config.update(**kwargs)

            response = await client.aio.models.generate_content(
                model=self.model_name, contents=prompt, config=generation_config
            )
            return StructuredLLMResponse(
                text=response.text or "",
                parsed=response.parsed
                if hasattr(response, "parsed")
                else schema.model_validate_json(response.text),
                prompt_token_count=response.usage_metadata.prompt_token_count,
                completion_token_count=response.usage_metadata.candidates_token_count,
                total_token_count=response.usage_metadata.total_token_count,
                finish_reason=str(response.candidates[0].finish_reason),
                model_name=self.model_name,
                raw_response=response,
            )

        except Exception:
            self.key_pool.report_error(api_key)
            raise
        finally:
            await self._aclose_client(client)

    def start_chat(
        self,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> ChatSession:
        api_key = self.key_pool.get_next_key()

        try:
            client = genai.Client(api_key=api_key)
            generation_config = {
                "temperature": temperature,
                "system_instruction": self.system_instruction,
                "safety_settings": self.safety_settings,
                **self.kwargs,
            }
            if kwargs:
                generation_config.update(**kwargs)
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens
            if stop_sequences:
                generation_config["stop_sequences"] = stop_sequences

            chat = client.chats.create(model=self.model_name, config=generation_config)

            return GeminiChatSession(chat, self.model_name)

        except Exception:
            self.key_pool.report_error(api_key)
            raise

    async def astart_chat(
        self,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> ChatSession:
        api_key = self.key_pool.get_next_key()

        try:
            client = genai.Client(api_key=api_key)
            generation_config = {
                "temperature": temperature,
                "system_instruction": self.system_instruction,
                "safety_settings": self.safety_settings,
                **self.kwargs,
            }
            if kwargs:
                generation_config.update(**kwargs)
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens
            if stop_sequences:
                generation_config["stop_sequences"] = stop_sequences

            chat = client.aio.chats.create(
                model=self.model_name, config=generation_config
            )

            return AsyncGeminiChatSession(chat, self.model_name)

        except Exception:
            self.key_pool.report_error(api_key)
            raise


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible API implementation."""

    def __init__(
        self,
        api_key: str | SmartKeyPool,
        model_name: str = "gpt-4o",
        base_url: Optional[str] = None,
        system_instruction: str | None = None,
        **kwargs,
    ):
        self.key_pool = (
            api_key
            if isinstance(api_key, SmartKeyPool)
            else SmartKeyPool.from_single_key(api_key)
        )
        self.model_name = model_name
        self.base_url = base_url
        self.system_instruction = system_instruction
        self.kwargs = {k: v for k, v in kwargs.items() if k != "safety_settings"}

    def _get_client(self) -> OpenAI:
        api_key = self.key_pool.get_next_key()
        return OpenAI(api_key=api_key, base_url=self.base_url)

    def _get_async_client(self) -> AsyncOpenAI:
        api_key = self.key_pool.get_next_key()
        return AsyncOpenAI(api_key=api_key, base_url=self.base_url)

    def generate_content(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        client = self._get_client()
        api_key = client.api_key

        try:
            messages = []
            if self.system_instruction:
                messages.append({"role": "system", "content": self.system_instruction})
            messages.append({"role": "user", "content": prompt})

            current_kwargs = self.kwargs.copy()
            current_kwargs.update(kwargs)

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
                prompt_token_count=response.usage.prompt_tokens,
                completion_token_count=response.usage.completion_tokens,
                total_token_count=response.usage.total_tokens,
                finish_reason=response.choices[0].finish_reason,
                model_name=self.model_name,
                raw_response=response,
                reasoning_content=_extract_reasoning_content(assistant_message),
            )

        except Exception:
            self.key_pool.report_error(api_key)
            raise

    async def agenerate_content(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        client = self._get_async_client()
        api_key = client.api_key

        try:
            messages = []
            if self.system_instruction:
                messages.append({"role": "system", "content": self.system_instruction})
            messages.append({"role": "user", "content": prompt})

            current_kwargs = self.kwargs.copy()
            current_kwargs.update(kwargs)

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
                prompt_token_count=response.usage.prompt_tokens,
                completion_token_count=response.usage.completion_tokens,
                total_token_count=response.usage.total_tokens,
                finish_reason=response.choices[0].finish_reason,
                model_name=self.model_name,
                raw_response=response,
                reasoning_content=_extract_reasoning_content(assistant_message),
            )

        except Exception:
            self.key_pool.report_error(api_key)
            raise

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        temperature: float = 0.7,
        **kwargs,
    ) -> StructuredLLMResponse[T]:
        client = self._get_client()
        api_key = client.api_key

        try:
            messages = []
            if self.system_instruction:
                messages.append({"role": "system", "content": self.system_instruction})
            messages.append({"role": "user", "content": prompt})

            current_kwargs = self.kwargs.copy()
            current_kwargs.update(kwargs)

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
                prompt_token_count=response.usage.prompt_tokens,
                completion_token_count=response.usage.completion_tokens,
                total_token_count=response.usage.total_tokens,
                finish_reason=response.choices[0].finish_reason,
                model_name=self.model_name,
                raw_response=response,
                reasoning_content=_extract_reasoning_content(assistant_message),
            )
        except Exception:
            self.key_pool.report_error(api_key)
            raise

    async def agenerate_structured(
        self,
        prompt: str,
        schema: Type[T],
        temperature: float = 0.7,
        **kwargs,
    ) -> StructuredLLMResponse[T]:
        client = self._get_async_client()
        api_key = client.api_key

        try:
            messages = []
            if self.system_instruction:
                messages.append({"role": "system", "content": self.system_instruction})
            messages.append({"role": "user", "content": prompt})

            current_kwargs = self.kwargs.copy()
            current_kwargs.update(kwargs)

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
                prompt_token_count=response.usage.prompt_tokens,
                completion_token_count=response.usage.completion_tokens,
                total_token_count=response.usage.total_tokens,
                finish_reason=response.choices[0].finish_reason,
                model_name=self.model_name,
                raw_response=response,
                reasoning_content=_extract_reasoning_content(assistant_message),
            )
        except Exception:
            self.key_pool.report_error(api_key)
            raise

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
        except Exception:
            self.key_pool.report_error(client.api_key)
            raise

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
        except Exception:
            self.key_pool.report_error(client.api_key)
            raise


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek implementation using OpenAI-compatible API."""

    BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        api_key: str | SmartKeyPool,
        model_name: str = "deepseek-chat",
        system_instruction: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            base_url=self.BASE_URL,
            system_instruction=system_instruction,
            **kwargs,
        )

    def _parse_structured_response(
        self, response: ChatCompletion, schema: Type[T]
    ) -> StructuredLLMResponse[T]:
        assistant_message = response.choices[0].message
        text = assistant_message.content or ""
        parsed = schema.model_validate_json(text)
        return StructuredLLMResponse(
            text=text,
            parsed=parsed,
            prompt_token_count=response.usage.prompt_tokens,
            completion_token_count=response.usage.completion_tokens,
            total_token_count=response.usage.total_tokens,
            finish_reason=response.choices[0].finish_reason,
            model_name=self.model_name,
            raw_response=response,
            reasoning_content=_extract_reasoning_content(assistant_message),
        )

    def _build_structured_messages(
        self, prompt: str, schema: Type[T]
    ) -> List[Dict[str, str]]:
        schema_str = json.dumps(schema.model_json_schema(), indent=2)
        system_content = (
            self.system_instruction or ""
        ) + f"\nRespond with a valid JSON object matching this schema:\n{schema_str}"
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        temperature: float = 0.7,
        **kwargs,
    ) -> StructuredLLMResponse[T]:
        client = self._get_client()
        api_key = client.api_key

        try:
            messages = self._build_structured_messages(prompt, schema)
            current_kwargs = {**self.kwargs, **kwargs}

            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                **current_kwargs,
            )

            return self._parse_structured_response(response, schema)
        except Exception:
            self.key_pool.report_error(api_key)
            raise

    async def agenerate_structured(
        self,
        prompt: str,
        schema: Type[T],
        temperature: float = 0.7,
        **kwargs,
    ) -> StructuredLLMResponse[T]:
        client = self._get_async_client()
        api_key = client.api_key

        try:
            messages = self._build_structured_messages(prompt, schema)
            current_kwargs = {**self.kwargs, **kwargs}

            response = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                **current_kwargs,
            )

            return self._parse_structured_response(response, schema)
        except Exception:
            await self.key_pool.areport_error(api_key)
            raise


class LLMFactory:
    """Factory for creating LLM providers."""

    @staticmethod
    def create(
        provider: str,
        model_name: Optional[str] = None,
        api_key: Optional[str | SmartKeyPool] = None,
        system_instruction: Optional[str] = None,
        **kwargs,
    ) -> LLMProvider:
        from .local_provider import LocalLLMProvider

        providers = {
            "gemini": GeminiProvider,
            "openai": OpenAIProvider,
            "deepseek": DeepSeekProvider,
            "local": LocalLLMProvider,
        }

        if provider not in providers:
            raise ValueError(f"Unknown provider: {provider}")

        provider_cls = providers[provider]

        if provider == "local":
            # LocalLLMProvider has a different constructor signature
            base_url = kwargs.pop("base_url", None) or "http://localhost:8000/v1"
            local_api_key = api_key if isinstance(api_key, str) else "not-needed"
            if isinstance(api_key, SmartKeyPool):
                local_api_key = "not-needed"
            init_kwargs = {
                "base_url": base_url,
                "api_key": local_api_key,
                "system_instruction": system_instruction,
                **kwargs,
            }
            if model_name is not None:
                init_kwargs["model_name"] = model_name
            return provider_cls(**init_kwargs)

        init_kwargs = {
            "api_key": api_key,
            "system_instruction": system_instruction,
            **kwargs,
        }
        if model_name is not None:
            init_kwargs["model_name"] = model_name
        return provider_cls(**init_kwargs)
