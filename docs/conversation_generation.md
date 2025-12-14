# Conversation Generation

The core capability of Afterimage is generating rigorous synthetic conversations. This process involves simulating a dialogue between a **Correspondent** (User) and a **Respondent** (Assistant) to create training or evaluation data.

## `AsyncConversationGenerator`

The `AsyncConversationGenerator` is the primary workhorse for this task. It orchestrates the multi-turn interaction, manages state, and handles concurrent generation for high throughput.

### Initialization

To start generating, you need to initialize the generator with at least a `respondent_prompt` (the system instruction for the assistant you are simulating) and your API key.

```python
from afterimage import AsyncConversationGenerator
import os

generator = AsyncConversationGenerator(
    respondent_prompt="You are a helpful assistant.",
    api_key=os.getenv("GEMINI_API_KEY"),
    model_name="gemini-2.0-flash",
)
```

**Key Parameters:**

*   `respondent_prompt` (str): The system prompt that defines the behavior of the assistant.
*   `api_key` (str | SmartKeyPool): Your API key or a pool of keys for rotation.
*   `correspondent_prompt` (str, optional): The system prompt for the user simulator. If omitted, Afterimage automatically generates one based on the respondent prompt to ensure relevant questions are asked.
*   `instruction_generator_callback` (BaseInstructionGeneratorCallback, optional): A strategy to dynamically generate the first question/instruction. Essential for RAG or Persona-based generation.
*   `respondent_prompt_modifier` (BaseRespondentPromptModifierCallback, optional): A strategy to modify the respondent's prompt per conversation (e.g., to inject RAG context).
*   `storage` (BaseStorage, optional): Where to save the results. Defaults to `JSONLStorage` (local file).

### Generating Conversations

Use the `generate` method to start the simulation.

```python
await generator.generate(
    num_dialogs=100,
    max_turns=5,
    max_concurrency=4
)
```

**Parameters:**

*   `num_dialogs` (int): Total number of independent conversations to generate.
*   `max_turns` (int): THe maximum number of exchanges (User question + Assistant answer) per conversation. The actual number of turns is randomly sampled from a range `[1, max_turns]`.
*   `max_concurrency` (int): How many conversations to simulate in parallel. Increase this for higher throughput if your rate limits allow.
*   `seed_instructions` (List[str], optional): A list of specific starting questions to use. If provided, these will be used instead of the instruction generator for the first `N` dialogs.

## Customization with Callbacks

For realistic data, you rarely want a generic user. You want a user who asks about *your* specific data or acts like *your* specific customers.

### 1. Instruction Generators (The "What")
These determine **what** the user asks about.
*   **`ContextualInstructionGeneratorCallback`**: Reads documents and prompts the user to ask a question based on a specific random document.
*   **`PersonaInstructionGeneratorCallback`**: similar to above, but also assigns a specific persona to the user for that conversation.

### 2. Prompt Modifiers (The "Context")
These determine **what content** the assistant has access to.
*   **`WithContextRespondentPromptModifier`**: Injects the content of the document chosen by the instruction generator into the assistant's system prompt. This simulates a RAG setup where the assistant "knows" the answer.

## Complete Example

Here is a full example showing how to generate a dataset for a technical support bot using a manual document provider.

```python
import asyncio
import os
from afterimage import (
    AsyncConversationGenerator,
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
    instruction_gen = ContextualInstructionGeneratorCallback(
        api_key=api_key,
        documents=docs
    )

    # 3. Configure Assistant Behavior (Have access to the docs)
    prompt_modifier = WithContextRespondentPromptModifier()

    # 4. Initialize Generator
    generator = AsyncConversationGenerator(
        respondent_prompt="You are a Tier 1 Technical Support agent.",
        api_key=api_key,
        instruction_generator_callback=instruction_gen,
        respondent_prompt_modifier=prompt_modifier,
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

---
[Previous: Overview](overview.md) | [Next: Persona Generation](persona_generation.md)
