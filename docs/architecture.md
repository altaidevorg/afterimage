# Architecture & Design

This document details the internal architecture of the Afterimage library. It is intended for advanced users who want to extend the library or understand its internals.

## System Overview

Afterimage is designed as a modular pipeline for synthetic data generation. The core philosophy is **composition over inheritance**—you build a generator by composing different strategies for prompts, instructions, and storage.

### Core Components

1.  **Generators (`BaseGenerator`)**: The orchestrators. They manage the main loop, concurrency, and state.
    *   `AsyncConversationGenerator`: Manages multi-turn dialogs.
    *   `AsyncStructuredGenerator`: Manages single-turn structured output.
2.  **Instruction Generators (`BaseInstructionGeneratorCallback`)**: Strategies for "What to ask".
    *   Responsible for producing the initial user instruction/question.
    *   Can have internal state (e.g., to ensure coverage of a document set).
3.  **Prompt Modifiers (`BaseRespondentPromptModifierCallback`)**: Strategies for "What to know".
    *   Responsible for modifying the system prompt of the assistant at runtime.
    *   Used for RAG (injecting context) or Persona adoption.
4.  **Storage (`BaseStorage`)**: Persistence layer.
    *   Decoupled from generation logic.
    *   Can be swapped (JSONL vs SQL) without changing the generator.
5.  **LLM Abstraction Layer (`afterimage.providers.llm_providers`)**:
    *   **Uniform Interface**: `LLMProvider` protocol normalizes interactions across models (Gemini, OpenAI, etc.).
    *   **Unified Responses**: Returns standardized `LLMResponse` or `StructuredLLMResponse` objects with consistent token counts and usage metadata.
    *   **Chat Abstraction**: `ChatSession` manages conversation history statefully, independent of the underlying API's specific mechanics.
    *   **Factory Creation**: `LLMFactory` allows dynamic instantiation of providers via strings.

## Extension Points

Afterimage is designed to be extended. Here are the common patterns:

### Custom Instruction Generator

If you want to generate instructions from a custom source (e.g., a live API or a specific algorithm), subclass `BaseInstructionGeneratorCallback`.

```python
from afterimage.base import BaseInstructionGeneratorCallback
from afterimage.common import GeneratedInstructions

class MyCustomInstructionGenerator(BaseInstructionGeneratorCallback):
    async def agenerate(self, original_prompt: str) -> GeneratedInstructions:
        # Your logic here
        return GeneratedInstructions(
            instruction="Tell me a joke about API limits.",
            context="System load is high."
        )
```

### Custom Storage

To save data to a custom backend (e.g., S3, Mongo, or a specific API endpoint), implement the `BaseStorage` protocol.

```python
from afterimage.storage import BaseStorage

class MyCloudStorage(BaseStorage):
    async def asave_conversations(self, conversations):
        # Push to cloud
        pass
        
    async def load_conversations(self, limit=None, offset=None):
        # Fetch from cloud
        return []
```

### Custom LLM Provider

To support a new model family (e.g., Anthropic, Mistral, or a local VLLM), implement the `LLMProvider` protocol. You must also implement a corresponding `ChatSession`.

```python
from afterimage.providers import LLMProvider, ChatSession, LLMResponse

class MyCustomChat(ChatSession):
    async def asend_message(self, message, **kwargs) -> LLMResponse:
        # Implement stateful chat logic
        pass

class MyCustomProvider(LLMProvider):
    def initialize(self, api_key: str):
        self.client = ...

    async def agenerate_content(self, prompt: str, **kwargs) -> LLMResponse:
        # Call your API
        return LLMResponse(
            text="response",
            prompt_token_count=10,
            completion_token_count=10,
            total_token_count=20,
            finish_reason="stop",
            model_name="my-model",
            raw_response={}
        )

    def start_chat(self, **kwargs) -> ChatSession:
        return MyCustomChat()
```

**Developer Tips for LLM Providers:**

*   **Async Support**: Always implement both sync and async methods. The library core relies heavily on `agenerate_content` for performance.
*   **Token Counting**: Ensure you populate token counts in `LLMResponse`. This is critical for the `GenerationMonitor` to track costs and throughput.
*   **Structured Output**: For `generate_structured`, leveraging Pydantic is highly recommended. If the underlying API doesn't support JSON schema natively, use a robust parser or instructor library.
*   **Error Handling**: Wrap your API calls in try/except blocks and use `SmartKeyPool.report_error(key)` if an API error occurs, so the pool can rotate keys or back off.

## Design Patterns

*   **Async-First**: The library is built from the ground up using `asyncio` for high throughput.
*   **Callback Pattern**: Logic is injected via callbacks rather than subclassing the generator itself.
*   **Pydantic Models**: All data exchange (config, inputs, outputs) is validated using Pydantic models for type safety.

---
[Previous: Advanced Configuration](advanced_usage.md)
