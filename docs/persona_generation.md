# Persona Generation

One of the challenges in synthetic data generation is lack of diversity. If every simulated user sounds the same, your model learns a limited distribution of language. Afterimage solves this with **Persona Generation**.

## Concept

A "Persona" in Afterimage is a short description of a user's background, tone, knowledge level, and intent. For example:
> **The Frustrated Novice**: A non-technical user who is angry because their workflow is broken. Uses short sentences, capitalization for emphasis, and non-technical terms.

By generating varied personas based on your actual data, you can test how your model handles different "vibes" and expertise levels.

## `PersonaGenerator`

The `PersonaGenerator` class is responsible for creating these personas. It can generate them from raw text (analyzing what kind of person would be interested in this text) or evolve existing personas.

Each persona generation call now targets **exactly five personas** from the LLM. Outputs are whitespace-normalized, exact duplicates are dropped, and the call is retried up to three times before failing the persona enrichment step.

### Initialization

```python
from afterimage import PersonaGenerator
import os

persona_gen = PersonaGenerator(api_key=os.getenv("GEMINI_API_KEY"))
```

`PersonaGenerator` chat backends are **`gemini`**, **`openai`**, or **`deepseek`** only (there is no `local` option on this class today).

### Methods

#### `generate_from_documents`

The most common workflow. It reads documents from a `DocumentProvider`, analyzes them, and creates a set of relevant personas.

```python
# Assuming you have a document provider 'docs'
await persona_gen.generate_from_documents(
    documents=docs,
    max_docs=10,       # Limit how many documents to analyze
    n_iterations=None, # Auto-pick depth from target if available
    target_data_count=1000,
    num_random_contexts=1,
)
```

This method doesn't return the personas directly. Instead, it saves them into the `DocumentProvider`'s internal memory (specifically, into the `Document` objects themselves). This allows the `PersonaInstructionGeneratorCallback` to later retrieve a document AND its associated personas together.

When `n_iterations` is omitted, Afterimage tries to choose it automatically. The heuristic resolves an effective per-document persona target from either:

- `ceil(documents.target_context_usage_count / num_random_contexts)`, if the provider already exposes a context-usage target
- or `ceil(target_data_count / active_doc_count)`

It then chooses the depth whose expected persona pool is closest to that target. This avoids cases like generating `3905` personas for a `20`-row per-document demand.

Examples:

- target `20 / doc` -> `n_iterations = 1` -> `30` total personas
- target `1000 / doc` -> `n_iterations = 3` -> `780` total personas, then runtime oversampling covers the remainder
- target `3905 / doc` -> `n_iterations = 4` -> `3905` total personas

When `n_iterations > 0`, each successful generation step fans out by five. The expected per-document persona pool after `n_iterations = n` is:

```text
S(n) = sum(i=1..n+1) 5^i = 5(5^(n+1) - 1) / 4
```

For example, `n_iterations = 4` yields an expected full per-document persona pool of `3905`.

#### `generate_from_text` (Single)

Generate a single persona description based on a raw text string.

```python
text = "Advanced usage of the async/await pattern in Python 3.11."
persona = await persona_gen.generate_from_text(text)
print(persona)
# Output: "Senior Python Developer looking for performance optimizations..."
```

## Using Personas in Conversation

Once you have generated personas using `PersonaGenerator`, you configure a `ConversationGenerator` (the library also exports the same class as `AsyncConversationGenerator`) with `PersonaInstructionGeneratorCallback`.

### Full Workflow

1.  **Load Documents**: Prepare your content.
2.  **Generate Personas**: Run `PersonaGenerator` over these documents to "enrich" them with potential user profiles.
3.  **Generate Conversations**: Use the `PersonaInstructionGeneratorCallback`. It will pick a document, build a depth-aware persona pool for that document, and then instruct the Correspondent to "act" like that person.

At runtime, persona sampling is adaptive:
- If the target rows per document are lower than the available persona pool, Afterimage keeps the shallowest personas first and prunes deeper layers.
- If the target rows per document exceed the available pool, Afterimage reuses personas with a depth-based weight that favors upper layers.

```python
import asyncio
import os
from afterimage import (
    ConversationGenerator,
    PersonaGenerator,
    PersonaInstructionGeneratorCallback,
    InMemoryDocumentProvider,
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

    # 4. Generate conversations using personas
    generator = ConversationGenerator(
        respondent_prompt="You are a support agent.",
        api_key=api_key,
        instruction_generator_callback=instruction_callback,
    )

    print("Generating persona-based conversations...")
    await generator.generate(num_dialogs=5)

if __name__ == "__main__":
    asyncio.run(main())
```
