# Quickstart Guide for Afterimage

`afterimage` is a powerful Python library for generating synthetic conversation datasets using Large Language Models (LLMs). It allows you to simulate multi-turn conversations between a "Correspondent" (User) and a "Respondent" (Assistant) to create high-quality training or evaluation datasets.

## Installation

```bash
pip install afterimage
```

## Core Concepts

*   **Generator**: The core engine (`AsyncConversationGenerator`) that orchestrates the conversation flow. It manages the LLM sessions for both the user and the assistant.
*   **Correspondent**: The simulated user who asks questions. Its behavior is driven by an **Instruction Generator**.
*   **Respondent**: The assistant who answers questions. Its behavior is defined by a **System Prompt** and optional **Prompt Modifiers**.
*   **Document Provider**: A source of knowledge (text files, JSONL, memory) used to ground the conversation or generate relevant questions.
*   **Persona**: A specific character or role that the Correspondent adopts to make conversations more diverse and realistic.

## Table of Contents

0.  [Overview](overview.md)
1.  [Conversation Generation](conversation_generation.md)
2.  [Persona Generation](persona_generation.md)
3.  [Evaluation Framework](evaluation.md)
4.  [Structured Generation](structured_generation.md)
5.  [Monitoring & Observability](monitoring.md)
6.  [Advanced Configuration](advanced_usage.md)
7.  [Architecture & Design](architecture.md)

---

## 1. Basic Usage

The simplest way to use `afterimage` is to define a system prompt for the assistant (Respondent) and let the library automatically generate a persona for the user (Correspondent).

```python
import asyncio
import os
from afterimage import AsyncConversationGenerator

# Ensure your API key is set
api_key = os.getenv("GEMINI_API_KEY")

async def main():
    # 1. Define the Assistant's Persona (Respondent)
    respondent_prompt = """
    You are a helpful and polite customer support agent for a tech company.
    You answer questions about laptops, smartphones, and accessories.
    Be concise and professional.
    """

    # 2. Initialize the Generator
    # If you don't provide a correspondent_prompt, one is auto-generated based on the respondent_prompt.
    # We encourage to give a try to the auto-generated correspondent prompt first,
    # and then you can try to provide your own correspondent prompt.
    generator = AsyncConversationGenerator(
        respondent_prompt=respondent_prompt,
        api_key=api_key,
        model_name="gemini-2.0-flash",  # Default model
    )

    # 3. Generate Conversations
    print("Generating conversations...")
    await generator.generate(
        num_dialogs=3,      # Number of separate conversations to generate
        max_turns=1,        # Maximum number of turns (exchange pairs) per conversation. 1 is enough for most cases.
        max_concurrency=2   # Number of parallel generations
    )
    
    # 4. Access Generated Data (saved to JSONL by default)
    # You can also access the storage directly if needed, but default storage saves to disk.
    print("Done! Check the generated .jsonl file in your directory.")

if __name__ == "__main__":
    asyncio.run(main())
```

## 2. Context-Aware Generation (RAG-like)

To generate high-quality domain-specific datasets, you often want the "User" to ask questions based on specific documents. You can achieve this using `ContextualInstructionGeneratorCallback`.

```python
import asyncio
import os
from afterimage import (
    AsyncConversationGenerator,
    ContextualInstructionGeneratorCallback,
    InMemoryDocumentProvider,
    WithContextRespondentPromptModifier
)

api_key = os.getenv("GEMINI_API_KEY")

async def main():
    # 1. Provide Context Documents
    # You can use InMemoryDocumentProvider, JSONLDocumentProvider, or DirectoryDocumentProvider
    documents = InMemoryDocumentProvider([
        "The 'Afterimage' library is used for synthetic data generation.",
        "It supports both synchronous and asynchronous generation modes.",
        "Key components include Generators, Callbacks, and Document Providers."
    ])

    # 2. Setup Instruction Generator (The User's Brain)
    # This callback picks random documents and instructs the User to ask questions about them.
    instruction_callback = ContextualInstructionGeneratorCallback(
        api_key=api_key,
        documents=documents,
        num_random_contexts=1,  # How many docs to pick per conversation
        n_instructions=3        # How many questions to brainstorm internally
    )

    # 3. Setup Respondent Modifier (The Assistant's Context)
    # This injects the SAME context the user sees into the assistant's system prompt,
    # ensuring the assistant has the knowledge to answer correctly.
    prompt_modifier = WithContextRespondentPromptModifier()

    # 4. Initialize Generator
    generator = AsyncConversationGenerator(
        respondent_prompt="You are an expert on the Afterimage library. Answer questions based on the provided context.",
        api_key=api_key,
        instruction_generator_callback=instruction_callback,
        respondent_prompt_modifier=prompt_modifier
    )

    # 5. Generate
    await generator.generate(num_dialogs=5, max_turns=2)

if __name__ == "__main__":
    asyncio.run(main())
```

## 3. Persona-Based Generation

To add variety, you can generate specific "Personas" for your documents (e.g., "A confused beginner", "A skeptical expert") and have the User adopt these personas.

```python
import asyncio
import os
from afterimage import (
    AsyncConversationGenerator,
    PersonaInstructionGeneratorCallback,
    PersonaGenerator,
    InMemoryDocumentProvider
)

api_key = os.getenv("GEMINI_API_KEY")

async def main():
    # 1. Setup Documents
    texts = [
        "Espresso is brewed by forcing hot water under pressure through finely-ground coffee beans.",
        "Cold brew is made by steeping coarse grounds in cold water for 12-24 hours."
    ]
    documents = InMemoryDocumentProvider(texts)

    # 2. Generate Personas for Documents
    # This step analyzes the documents and creates suitable user personas (e.g., "Coffee Enthusiast", "Barista Student")
    print("Generating personas...")
    persona_gen = PersonaGenerator(api_key=api_key)
    await persona_gen.generate_from_documents(documents)

    # 3. Setup Persona-Aware Instruction Generator
    # This callback will now select a random persona along with the document
    instruction_callback = PersonaInstructionGeneratorCallback(
        api_key=api_key,
        documents=documents,
        num_random_contexts=1
    )

    # 4. Initialize Generator
    generator = AsyncConversationGenerator(
        respondent_prompt="You are a professional barista. Answer questions about coffee brewing methods.",
        api_key=api_key,
        instruction_generator_callback=instruction_callback
    )

    # 5. Generate
    await generator.generate(num_dialogs=4, max_turns=2)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## API Reference

### `AsyncConversationGenerator`

The main class for generating conversations asynchronously.

**Initialization Arguments:**

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `respondent_prompt` | `str` | **Required** | The system prompt for the assistant (Respondent). |
| `api_key` | `str` \| `SmartKeyPool` | **Required** | Your Google Gemini API key. |
| `correspondent_prompt` | `str` | `None` | System prompt for the user. If `None`, it is auto-generated from `respondent_prompt`. |
| `model_name` | `str` | `gemini-2.0-flash` | The LLM model to use for generation. |
| `instruction_generator_callback` | `BaseInstructionGeneratorCallback` | `None` | Callback to generate the User's first question/instruction. |
| `respondent_prompt_modifier` | `BaseRespondentPromptModifierCallback` | `None` | Callback to dynamically modify the Assistant's prompt (e.g., add context). |
| `storage` | `BaseStorage` | `JSONLStorage` | Handles saving generated conversations. Defaults to a local JSONL file. |
| `monitor` | `GenerationMonitor` | `None` | Optional monitor for tracking metrics. |

**`generate()` Method Arguments:**

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `num_dialogs` | `int` | `5` | Total number of conversations to generate. |
| `max_turns` | `int` | `3` | Maximum number of exchange pairs (User + Assistant) per conversation. |
| `max_concurrency` | `int` | `4` | Number of parallel generation tasks (Async only). |
| `seed_instructions` | `List` | `[]` | A list of initial questions to start conversations with (bypasses instruction generator). |

### Document Providers

Helpers to load text data for context-aware generation.

*   **`InMemoryDocumentProvider(texts: List[str])`**: Simple list of strings.
*   **`JSONLDocumentProvider(path_pattern: str, content_key: str = "text")`**: Reads from `.jsonl` files.
*   **`DirectoryDocumentProvider(directory: str, file_patterns: List[str])`**: Reads files (e.g., `*.txt`, `*.md`) from a folder.
*   **`QdrantDocumentProvider(...)`**: Reads from a Qdrant vector database (requires `qdrant-client`).
