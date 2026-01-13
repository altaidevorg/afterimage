# Persona Generation

One of the challenges in synthetic data generation is lack of diversity. If every simulated user sounds the same, your model learns a limited distribution of language. Afterimage solves this with **Persona Generation**.

## Concept

A "Persona" in Afterimage is a short description of a user's background, tone, knowledge level, and intent. For example:
> **The Frustrated Novice**: A non-technical user who is angry because their workflow is broken. Uses short sentences, capitalization for emphasis, and non-technical terms.

By generating varied personas based on your actual data, you can test how your model handles different "vibes" and expertise levels.

## `PersonaGenerator`

The `PersonaGenerator` class is responsible for creating these personas. It can generate them from raw text (analyzing what kind of person would be interested in this text) or evolve existing personas.

### Initialization

```python
from afterimage import PersonaGenerator
import os

persona_gen = PersonaGenerator(api_key=os.getenv("GEMINI_API_KEY"))
```

### Methods

#### `generate_from_documents`

The most common workflow. It reads documents from a `DocumentProvider`, analyzes them, and creates a set of relevant personas.

```python
# Assuming you have a document provider 'docs'
await persona_gen.generate_from_documents(
    documents=docs,
    max_docs=10,       # Limit how many documents to analyze
    n_iterations=0     # (Experimental) Evolve personas further
)
```

This method doesn't return the personas directly. Instead, it saves them into the `DocumentProvider`'s internal memory (specifically, into the `Document` objects themselves). This allows the `PersonaInstructionGeneratorCallback` to later retrieve a document AND its associated personas together.

#### `generate_from_text` (Single)

Generate a single persona description based on a raw text string.

```python
text = "Advanced usage of the async/await pattern in Python 3.11."
persona = await persona_gen.generate_from_text(text)
print(persona)
# Output: "Senior Python Developer looking for performance optimizations..."
```

## Using Personas in Conversation

Once you have generated personas using `PersonaGenerator`, you need to tell the `AsyncConversationGenerator` to use them. You do this with the `PersonaInstructionGeneratorCallback`.

### Full Workflow

1.  **Load Documents**: Prepare your content.
2.  **Generate Personas**: Run `PersonaGenerator` over these documents to "enrich" them with potential user profiles.
3.  **Generate Conversations**: Use the `PersonaInstructionGeneratorCallback`. It will pick a document, randomly select one of the generated personas for that document, and then instruct the Correspondent to "act" like that person.

```python
import asyncio
import os
from afterimage import (
    AsyncConversationGenerator,
    PersonaGenerator,
    PersonaInstructionGeneratorCallback,
    InMemoryDocumentProvider
)

async def main():
    api_key = os.getenv("GEMINI_API_KEY")
    
    # 1. Load Documents
    docs = InMemoryDocumentProvider([
        "Our refund policy allows returns within 30 days.",
        "To enable 2FA, go to Settings > Security.",
    ])

    # 2. Generate Personas (Enrich the docs)
    print("Generating personas...")
    persona_gen = PersonaGenerator(api_key=api_key)
    await persona_gen.generate_from_documents(docs)

    # 3. Setup Callback
    instruction_callback = PersonaInstructionGeneratorCallback(
        api_key=api_key,
        documents=docs,
        num_random_contexts=1
    )

    # 4. Generate Conversations using Personas
    generator = AsyncConversationGenerator(
        respondent_prompt="You are a support agent.",
        api_key=api_key,
        instruction_generator_callback=instruction_callback
    )

    print("Generating persona-based conversations...")
    await generator.generate(num_dialogs=5)

if __name__ == "__main__":
    asyncio.run(main())
```
