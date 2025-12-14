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

## Design Patterns

*   **Async-First**: The library is built from the ground up using `asyncio` for high throughput.
*   **Callback Pattern**: Logic is injected via callbacks rather than subclassing the generator itself.
*   **Pydantic Models**: All data exchange (config, inputs, outputs) is validated using Pydantic models for type safety.

---
[Previous: Advanced Configuration](advanced_usage.md)
