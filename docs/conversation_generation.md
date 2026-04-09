# Conversation Generation

The core capability of Afterimage is generating rigorous synthetic conversations. This process involves simulating a dialogue between a **Correspondent** (User) and a **Respondent** (Assistant) to create training or evaluation data.

## `ConversationGenerator`

The `ConversationGenerator` class is the primary workhorse for this task. It orchestrates the multi-turn interaction, manages state, handles concurrency, and can even self-correct using an evaluator loop.

### Initialization

To start generating, you need to initialize the generator. The recommended pattern is to configure all strategy callbacks (for instructions and prompt modification) at initialization time.

```python
from afterimage import ConversationGenerator
import os

generator = ConversationGenerator(
    respondent_prompt="You are a helpful assistant.",
    api_key=os.getenv("GEMINI_API_KEY"),
    model_name="gemini-2.0-flash",
    # Strategies are now passed here
    instruction_generator_callback=my_instruction_gen,
    respondent_prompt_modifier=my_prompt_modifier,
    auto_improve=False,  # set to to True to enable auto-improvement
    evaluator_model_name="gemini-2.0-flash"
)
```

**Key Parameters:**

*   `respondent_prompt` (str): The system prompt that defines the behavior of the assistant.
*   `api_key` (str | SmartKeyPool): Your API key or a pool of keys for rotation.
*   `instruction_generator_callback` (BaseInstructionGeneratorCallback): Controls **what** the user asks (e.g., questions based on docs or personas).
*   `respondent_prompt_modifier` (BaseRespondentPromptModifierCallback, optional): Controls **context** (e.g., injecting RAG data into the system prompt).
*   `correspondent_prompt` (str, optional): A static system prompt for the user simulator. If neither this nor a callback is provided, one is auto-generated.
*   `auto_improve` (bool): If `True`, an internal evaluator checks each conversation. If quality is low, it regenerates the conversation automatically (up to a limit).
*   `storage` (BaseStorage, optional): Where to save the results. Defaults to `JSONLStorage`.

### Generating Conversations

Use the `generate` method to start the simulation.

```python
from afterimage.callbacks import PersonaUsageStoppingCallback

await generator.generate(
    num_dialogs=100,
    max_turns=3,
    max_concurrency=4,
    stopping_criteria=[
        PersonaUsageStoppingCallback(n_personas=50) # Stop if 50 unique personas are used
    ]
)
```

**Parameters:**

*   `num_dialogs` (int, optional): Number of conversations to generate.
*   `max_turns` (int): Maximum exchanges per conversation.
*   `max_concurrency` (int): Parallel generation limit.
*   `stopping_criteria` (List[BaseStoppingCallback], optional): Custom logic for when to stop generating (e.g., when all personas are covered). If `num_dialogs` is set, a `FixedNumberStoppingCallback` is automatically added.

## Strategies & Callbacks

Afterimage uses a callback system to modularize "User Behavior" and "Assistant Knowledge".

### 1. Instruction Generators (The "User")
These determine what the simulated user wants to talk about.
*   **`ContextualInstructionGeneratorCallback`**: Samples a document and generates a question based on it.
*   **`PersonaInstructionGeneratorCallback`**: Samples a document-aware persona ("Angry Customer", "Novice") and a document to generate a styled question. It prunes deeper persona layers when supply exceeds demand and uses depth-weighted reuse when more rows are needed than unique personas.
*   **`ToolCallingInstructionGeneratorCallback`**: Generates instructions specifically designed to trigger tool/function calls (requires a list of tools).

Persona-based generations also carry `persona_generation_depth` in row metadata so downstream analysis can see whether the selected persona came from the seed layer or an evolved layer.

### 2. Prompt Modifiers (The "Assistant")
These modify the assistant's system prompt at runtime, usually to inject context.
*   **`WithContextRespondentPromptModifier`**: Injects the text of the document selected by the instruction generator into the assistant's system prompt.
*   **`WithRAGRespondentPromptModifier`**: Uses a retriever to fetch relevant chunks based on the user's generated question (simulating a real RAG pipeline).

## Complete Example

Here is a full example showing how to generate a dataset for a technical support bot.

```python
import asyncio
import os
from afterimage import (
    ConversationGenerator,
    ContextualInstructionGeneratorCallback,
    InMemoryDocumentProvider,
    WithContextRespondentPromptModifier
)

async def main():
    api_key = os.getenv("GEMINI_API_KEY")

    # 1. Your Knowledge Base
    docs = InMemoryDocumentProvider([
        "Error 503 means the service is unavailable. Retry after 5 minutes.",
        "To reset your password, click 'Forgot Password' on the login screen.",
    ])

    # 2. Configure User Behavior (Ask questions about the docs)
    # We pass this to the generator constructor
    instruction_gen = ContextualInstructionGeneratorCallback(
        api_key=api_key,
        documents=docs
    )

    # 3. Configure Assistant Behavior (Have access to the docs)
    prompt_modifier = WithContextRespondentPromptModifier()

    # 4. Initialize Generator
    generator = ConversationGenerator(
        respondent_prompt="You are a Tier 1 Technical Support agent.",
        api_key=api_key,
        instruction_generator_callback=instruction_gen,
        respondent_prompt_modifier=prompt_modifier,
        auto_improve=True  # Ensure high quality
    )

    # 5. Run Generation
    print("Starting generation...")
    await generator.generate(
        num_dialogs=10,
        max_turns=3
    )
    print("Done. Conversation data saved to JSONL.")

if __name__ == "__main__":
    asyncio.run(main())
```
