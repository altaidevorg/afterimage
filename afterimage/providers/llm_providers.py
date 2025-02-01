from typing import Protocol, List, Optional, Any, Dict
from dataclasses import dataclass
import google.generativeai as genai

from ..types import ConversationEntry
from ..key_management import SmartKeyPool
from ..common import default_safety_settings


@dataclass
class LLMResponse:
    """Standardized LLM response."""

    text: str
    tokens_used: int
    finish_reason: str
    model_name: str
    raw_response: Any  # Provider-specific response


class ChatSession:
    """Abstract chat session interface."""

    def __init__(self):
        self.token_count = 0

    def send_message(
        self, message: str | ConversationEntry, temperature: float = 0.7, **kwargs
    ) -> LLMResponse:
        """Send a message to the chat session."""
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

        response = self.chat.send_message(
            content, generation_config={"temperature": temperature, **kwargs}
        )

        tokens_used = response.candidates[0].token_count
        self.token_count += tokens_used
        return LLMResponse(
            text=response.text,
            finish_reason=response.candidates[0].finish_reason,
            tokens_used=tokens_used,
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

    def start_chat() -> ChatSession:
        """Start a new chat session."""
        ...


class GeminiProvider(LLMProvider):
    """Google Gemini implementation."""

    def __init__(
        self,
        api_key: str | SmartKeyPool,
        model_name: str = "gemini-1.5-pro",
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
        genai.configure(api_key=api_key)

        try:
            model = genai.GenerativeModel(
                self.model_name,
                system_instruction=self.system_instruction,
                safety_settings=self.safety_settings,
                **self.kwargs,
            )

            generation_config = {"temperature": temperature, **kwargs}
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens
            if stop_sequences:
                generation_config["stop_sequences"] = stop_sequences

            response = model.generate_content(
                prompt, generation_config=generation_config
            )

            return LLMResponse(
                text=response.text,
                tokens_used=response.candidates[0].token_count,
                finish_reason=response.candidates[0].finish_reason,
                model_name=self.model_name,
                raw_response=response,
            )

        except Exception as e:
            self.key_pool.report_error(api_key)
            raise

    def start_chat(self) -> ChatSession:
        api_key = self.key_pool.get_next_key()
        genai.configure(api_key=api_key)

        try:
            model = genai.GenerativeModel(
                self.model_name,
                system_instruction=self.system_instruction,
                safety_settings=self.safety_settings,
                **self.kwargs,
            )

            chat = model.start_chat(history=[])

            return GeminiChatSession(chat, self.model_name)

        except Exception as e:
            self.key_pool.report_error(api_key)
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
            # Add more providers here
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
