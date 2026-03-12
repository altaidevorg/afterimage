---
name: afterimage-synthetic-datasets
description: Teaches agents how to generate synthetic conversation or structured datasets using the Afterimage Python library. Use when implementing or modifying synthetic dataset generation, Afterimage integration, persona or conversation generation, RAG-style datasets, or when the user mentions Afterimage, synthetic data, conversational datasets, PersonaGenerator, or AsyncConversationGenerator.
---

# Afterimage Synthetic Dataset Generation

## When to use this skill

Apply when the user or task involves:
- Generating synthetic training/evaluation data with LLMs
- Afterimage library (personas, conversation generation, RAG-style context)
- ALTAI worker synthetic_dataset job or related pipeline
- Document-grounded QA or support-style conversation datasets

## Core concepts (minimal)

- **Correspondent** = simulated user (asks questions). Behavior from **Instruction Generator** callbacks.
- **Respondent** = assistant (answers). Behavior from **respondent_prompt** and optional **Respondent Prompt Modifier** (e.g. inject context).
- **Document Provider** = source of knowledge (InMemoryDocumentProvider, JSONL, directory, etc.).
- **Persona** = character/role for the Correspondent (e.g. "Frustrated novice") for diversity.
- **Async-first**: Prefer async APIs; Afterimage is built on asyncio.

## Two main generators

| Generator | Use case | Key method |
|-----------|----------|------------|
| **AsyncConversationGenerator** | Multi-turn user ↔ assistant dialogs | `generate(num_dialogs=..., max_turns=..., max_concurrency=...)` |
| **AsyncStructuredGenerator** | Single-turn extraction into Pydantic schema | `generate(num_samples=..., max_concurrency=...)` |

## Standard workflow: document-grounded conversations (RAG-like)

1. **Documents** → `InMemoryDocumentProvider(texts)` or load from files/JSONL.
2. **Personas (optional but recommended)** → `PersonaGenerator(api_key=...).generate_from_documents(documents)` to enrich docs with user personas.
3. **Instruction callback** → For RAG-style: `ContextualInstructionGeneratorCallback(api_key, documents, num_random_contexts=1)` or with personas: `PersonaInstructionGeneratorCallback(api_key, documents, num_random_contexts=1)`.
4. **Respondent context** → So the assistant sees the same context: `WithContextRespondentPromptModifier()`.
5. **Generator** → `AsyncConversationGenerator(respondent_prompt=..., api_key=..., instruction_generator_callback=..., respondent_prompt_modifier=...)`.
6. **Run** → `await generator.generate(num_dialogs=..., max_turns=..., max_concurrency=...)`.
7. **Storage** → Default is JSONL; use `JSONLStorage(conversations_path=..., documents_path=...)` to control paths.

## Config (ALTAI worker)

For the synthetic_dataset run type, config comes from the run payload. Use constants; do not hardcode strings.

- **Respondent (assistant) prompt**: `config.tasks.taskOptions.assistantPrompt` or `config.respondent_prompt`.
- **Optional**: `num_dialogs`, `max_turns`, `max_concurrency`, `model_name` (defaults: 50, 1, 2, `gemini-2.0-flash`).
- **Documents**: From run temp dir (files downloaded from presigned URLs per `config.file_ids`). Build document list from parsed file contents (see worker `synthetic_dataset/documents` and `generation`).

## API key and env

- Prefer worker/ALTAI helpers if present (e.g. `worker.afterimage_utils.config.get_api_key()`).
- Otherwise: `GEMINI_API_KEY` or `OPENAI_API_KEY` from env. Require one for synthetic_dataset.

## Output and storage

- **Conversations**: JSONL with conversation turns. For fine-tuning, a processed format (e.g. `conversations`, `instruction_context`, `response_context`) may be written separately; see project `TASK_DATASET_COLUMNS` and `_build_processed_jsonl`-style logic if applicable.
- **HuggingFace**: If the project uploads to HF, use the same revision/branch names (e.g. raw_generated, docs_with_personas, main for task dataset) and repo type `dataset`.

## Rules

- Use **async** paths (`async def`, `await`) for generation and I/O.
- Use **f-strings** for logging, not `%s`-style.
- Do not guess Afterimage APIs; if the docs don’t cover the case, ask the user or read [reference.md](reference.md).
- Afterimage may be installed from a **local wheel** in this repo; do not assume a public pip package.

## Important files (ALTAI repo)

- Worker job: `worker/src/worker/jobs/synthetic_dataset/` (documents, generation, job).
- Afterimage helpers: `worker/src/worker/afterimage_utils/` (e.g. `create_generator`, HF utils).
- Config / queue: `worker/src/worker/config.py`, `worker/AGENTS.md`.

## Additional resources

- Full API and patterns: [reference.md](reference.md) (overview, conversation generation, personas, structured generation, storage, key pool).
