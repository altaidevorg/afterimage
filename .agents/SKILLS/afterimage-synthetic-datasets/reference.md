# Afterimage reference

Condensed from project afterimage-docs. Use for API details and patterns when the skill instructions are not enough.

## Overview

- **Generators**: `AsyncConversationGenerator` (multi-turn dialogs), `AsyncStructuredGenerator` (single-turn, Pydantic output).
- **Instruction Generator Callback**: Decides what the Correspondent asks (e.g. from documents or personas).
- **Respondent Prompt Modifier**: Injects context into the assistant system prompt (e.g. `WithContextRespondentPromptModifier`, `WithRAGRespondentPromptModifier`).
- **Document Providers**: InMemoryDocumentProvider(list of strings), JSONL, directory, or custom async iterator yielding `Document` objects.
- **Storage**: Default `JSONLStorage`; can use `SQLStorage` or custom `BaseStorage`.

## AsyncConversationGenerator

```python
from afterimage import AsyncConversationGenerator, ContextualInstructionGeneratorCallback, InMemoryDocumentProvider, WithContextRespondentPromptModifier

docs = InMemoryDocumentProvider(["doc text 1", "doc text 2"])
instruction_gen = ContextualInstructionGeneratorCallback(api_key=api_key, documents=docs, num_random_contexts=1)
prompt_modifier = WithContextRespondentPromptModifier()

generator = AsyncConversationGenerator(
    respondent_prompt="You are a helpful expert.",
    api_key=api_key,
    model_name="gemini-2.0-flash",
    instruction_generator_callback=instruction_gen,
    respondent_prompt_modifier=prompt_modifier,
    storage=JSONLStorage(conversations_path="out.jsonl", documents_path="docs.jsonl"),
)
await generator.generate(num_dialogs=10, max_turns=2, max_concurrency=2)
```

- **Persona-based**: First run `PersonaGenerator(api_key=...).generate_from_documents(documents)` to attach personas to documents. Then use `PersonaInstructionGeneratorCallback(api_key, documents, num_random_contexts=1)` and the same generator pattern.
- **Optional**: `auto_improve=True`, `evaluator_model_name`, `stopping_criteria` (e.g. `PersonaUsageStoppingCallback`).

## PersonaGenerator

- `PersonaGenerator(api_key=...)`.
- `await persona_gen.generate_from_documents(documents, max_docs=10, n_iterations=0)` — writes personas into the document provider’s documents; does not return them. Use with `PersonaInstructionGeneratorCallback` afterward.
- `persona_gen.generate_from_text(text)` — sync method, returns `list[str]` personas.
- `await persona_gen.agenerate_from_text(text)` — async method, returns `list[str]` personas.

## AsyncStructuredGenerator

- Single-turn: instruction + context → one structured output per sample.
- Define a Pydantic `BaseModel` as `output_schema`.
- Same callback pattern: `instruction_generator_callback`, optional `respondent_prompt_modifier`.
- `await generator.generate(num_samples=50, max_concurrency=4)`.

## Storage

- **JSONLStorage**: `conversations_path`, `documents_path`.
- **SQLStorage**: `url=` (e.g. postgresql://...), `conversations_table_name`.

## SmartKeyPool (rate limits)

- `SmartKeyPool(api_keys=[...], hourly_limit=1000, cooldown_period=600)`.
- Pass as `api_key=key_pool` to the generator.

## Custom extensions

- **Custom instruction generator**: Subclass `BaseInstructionGeneratorCallback`, implement `agenerate(self, original_prompt: str) -> GeneratedInstructions`.
- **Custom document provider**: Implement async iterator yielding `Document(content=..., metadata=...)`.
- **Custom storage**: Implement `BaseStorage` (e.g. `asave_conversations`, `load_conversations`).

## Terminology

- **Correspondent** = user simulator.
- **Respondent** = assistant.
- **Instruction** = what the user asks (from instruction generator).
- **Context** = text injected for RAG (via prompt modifier).
