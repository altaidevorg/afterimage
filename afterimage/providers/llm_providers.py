from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Protocol, Type, TypeVar

from google import genai
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from ..common import default_safety_settings
from ..key_management import SmartKeyPool
from ..types import ConversationEntry

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResponse:
    """Standardized LLM response."""

    text: str
    prompt_token_count: int
    completion_token_count: int
    total_token_count: int
    finish_reason: str
    model_name: str
    raw_response: Any  # Provider-specific response


@dataclass
class StructuredLLMResponse(LLMResponse, Generic[T]):
    """Standardized LLM response with structured output."""

    parsed: T


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
        self.history.append(
            {"role": assistant_message.role, "content": assistant_message.content}
        )

        total_token_count = response.usage.total_tokens
        self.token_count = total_token_count
        return LLMResponse(
            text=assistant_message.content,
            finish_reason=response.choices[0].finish_reason,
            prompt_token_count=response.usage.prompt_tokens,
            completion_token_count=response.usage.completion_tokens,
            total_token_count=total_token_count,
            model_name=self.model_name,
            raw_response=response,
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
        self.history.append(
            {"role": assistant_message.role, "content": assistant_message.content}
        )

        total_token_count = response.usage.total_tokens
        self.token_count = total_token_count

        return LLMResponse(
            text=assistant_message.content,
            finish_reason=response.choices[0].finish_reason,
            prompt_token_count=response.usage.prompt_tokens,
            completion_token_count=response.usage.completion_tokens,
            total_token_count=total_token_count,
            model_name=self.model_name,
            raw_response=response,
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
        self.kwargs = kwargs

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

            return LLMResponse(
                text=response.choices[0].message.content,
                prompt_token_count=response.usage.prompt_tokens,
                completion_token_count=response.usage.completion_tokens,
                total_token_count=response.usage.total_tokens,
                finish_reason=response.choices[0].finish_reason,
                model_name=self.model_name,
                raw_response=response,
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

            return LLMResponse(
                text=response.choices[0].message.content,
                prompt_token_count=response.usage.prompt_tokens,
                completion_token_count=response.usage.completion_tokens,
                total_token_count=response.usage.total_tokens,
                finish_reason=response.choices[0].finish_reason,
                model_name=self.model_name,
                raw_response=response,
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

            return StructuredLLMResponse(
                text=response.choices[0].message.content or "",
                parsed=response.choices[0].message.parsed,
                prompt_token_count=response.usage.prompt_tokens,
                completion_token_count=response.usage.completion_tokens,
                total_token_count=response.usage.total_tokens,
                finish_reason=response.choices[0].finish_reason,
                model_name=self.model_name,
                raw_response=response,
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

            return StructuredLLMResponse(
                text=response.choices[0].message.content or "",
                parsed=response.choices[0].message.parsed,
                prompt_token_count=response.usage.prompt_tokens,
                completion_token_count=response.usage.completion_tokens,
                total_token_count=response.usage.total_tokens,
                finish_reason=response.choices[0].finish_reason,
                model_name=self.model_name,
                raw_response=response,
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
        providers = {
            "gemini": GeminiProvider,
            "openai": OpenAIProvider,
        }

        if provider not in providers:
            raise ValueError(f"Unknown provider: {provider}")

        provider_cls = providers[provider]
        return provider_cls(
            api_key=api_key,
            model_name=model_name,
            system_instruction=system_instruction,
            **kwargs,
        )
