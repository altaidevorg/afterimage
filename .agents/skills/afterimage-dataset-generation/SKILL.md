---
name: afterimage-dataset-generation
description: >-
  Guides agents through AfterImage (afterimage) for synthetic conversational
  datasets, YAML/CLI generation, Python ConversationGenerator, document-grounded
  and persona flows, structured extraction, DPO preference pairs, export formats,
  and monitoring. Use when the user or task involves AfterImage, synthetic
  conversation data, JSONL datasets, afterimage generate/export/preference, or
  instruction generators and document providers in this repository.
---

# AfterImage dataset generation

## When to read this skill

Use for anything involving **AfterImage** in this repo: synthetic multi-turn data, **YAML configs**, **`afterimage` CLI**, **Python generators**, **personas**, **RAG-style context**, **structured Pydantic rows**, **DPO/preference** pipelines, or **export** to training formats.

For long-form API detail, open [reference.md](reference.md) after the quick path is clear.

## Mental model

| Piece | Role |
|--------|------|
| **Respondent** | Assistant side; fixed **system prompt** (`respondent_prompt` / config `respondent.system_prompt`). |
| **Correspondent** | User side; behavior from **instruction generator callback** and optional static correspondent prompt. |
| **Instruction generator** | Produces what the user asks (optionally grounded in **documents**). |
| **Respondent prompt modifier** | Injects **context** into the assistant system prompt (e.g. `WithContextRespondentPromptModifier`). |
| **Document provider** | Supplies `Document` objects (`InMemoryDocumentProvider`, `JSONLDocumentProvider`, `DirectoryDocumentProvider`, `QdrantDocumentProvider`, …). |
| **Storage** | Defaults to **JSONL**; optional **SQL** via `SQLStorage`. |

**Naming:** `AsyncConversationGenerator` and `AsyncStructuredGenerator` are **aliases** of `ConversationGenerator` and `StructuredGenerator`.

## Choose an entry path

1. **CLI + YAML** — fastest for standard runs; config lives in repo under `examples/configs/` as patterns.
2. **Python API** — full control (callbacks, pools, custom storage, embedding/judge wiring).

Do not invent API names: import from `afterimage` and `afterimage.callbacks` as in `docs/README.md` and `afterimage/__init__.py`.

## CLI quick path

```bash
afterimage generate -c path/to/config.yaml
afterimage export -i dataset.jsonl -f sharegpt -f messages
afterimage preference -c path/to/config.yaml
```

- **`afterimage generate`**: builds a `ConversationGenerator` from `AfterImageConfig` (`afterimage.config`, `afterimage.config_to_generator`).
- **Dry run**: `afterimage generate -c config.yaml --dry-run`.
- **Keys**: set env vars referenced by `model.api_key_env` (e.g. `GEMINI_API_KEY`, `OPENAI_API_KEY`).

Minimal config shape: `generation`, `model`, `respondent`, `output`; optional `documents`, `context`, `personas`, `quality`, `preference`, `export` under `output`. See `examples/configs/basic.yaml` and `docs/` tutorials.

## Python quick path

```python
import asyncio
import os
from afterimage import ConversationGenerator

async def main():
    gen = ConversationGenerator(
        respondent_prompt="You are a helpful assistant.",
        api_key=os.environ["GEMINI_API_KEY"],
        # model_name defaults to library default (see afterimage.common.default_model_name)
    )
    await gen.generate(num_dialogs=5, max_turns=1, max_concurrency=2)

asyncio.run(main())
```

### Document-grounded conversations (RAG-style)

- Instruction side: `ContextualInstructionGeneratorCallback(api_key=..., documents=provider, ...)`.
- Assistant context: `WithContextRespondentPromptModifier()` so the **same** sampled context appears for the respondent.
- Pass **`model_name`** explicitly when **`model_provider_name` is not `"gemini"`** — the default model id is Gemini-oriented; OpenAI/DeepSeek need an explicit chat model id on callbacks that build an LLM.

### Personas

1. `PersonaGenerator(api_key=...).generate_from_documents(documents, ...)` enriches documents with persona entries.
2. `PersonaInstructionGeneratorCallback(api_key=..., documents=..., num_random_contexts=...)` drives conversations.
3. **`PersonaGenerator`** supports **`gemini` | `openai` | `deepseek`** only (no **`local`** on that class).

### Structured rows (single-turn schema)

- `StructuredGenerator(output_schema=MyPydanticModel, respondent_prompt=..., api_key=..., instruction_generator_callback=...)`.
- `await generator.generate(num_samples=N, max_concurrency=...)`.

### Preference (DPO) data

- CLI: `afterimage preference -c config.yaml` with a `preference:` block (`docs/PREFERENCE_DATA.md`).
- Python: `ConversationGenerator.to_preference_generator(judge=..., config=PreferenceConfig(...))` then `await pref_gen.generate()`; import `PreferenceConfig` from `afterimage.preference`. Close the judge with `await judge.aclose()` when finished if you construct embeddings that need cleanup.

## Behaviors agents often get wrong

- **`max_turns`**: each dialog samples a turn count **uniformly from 1 through `max_turns`** (not always `max_turns`). See `ConversationGenerator.generate` docstring.
- **`GeneratedInstructions`**: uses **`instructions: list[str]`** and **`context`**, not a single `instruction` field.
- **Local models + `auto_improve`**: config path requires **`afterimage[embeddings-local]`** for sentence-transformers when using **local** chat provider with quality auto-improve (`config_to_generator` validates this).
- **Monitoring**: `GenerationMonitor` runs periodic **alert checks** when `metrics_interval > 0`; use **`metrics_interval=0`** to disable that thread only. **`check_alerts()`** runs rules once. Handlers receive `Alert` dataclass instances.

## Exports and dataset shape

- **CLI**: `afterimage export`, `afterimage export --list-formats`; programmatic **`afterimage.integrations`** (`get_exporter`, `list_formats`).
- **Row shape**: `docs/EXPORT_DATA_SHAPE.md` for JSONL fields (`conversations`, `metadata`, contexts, optional `evaluation` / `final_score`).

## Where to look in the repo

| Need | Location |
|------|-----------|
| Quickstart / copy-paste examples | `docs/README.md` |
| Conversations, callbacks, stopping | `docs/conversation_generation.md` |
| Personas | `docs/persona_generation.md` |
| Structured extraction | `docs/structured_generation.md` |
| Evaluation / auto-improve judge | `docs/evaluation.md` |
| Monitoring / metrics | `docs/monitoring.md` |
| SQL / key pools | `docs/advanced_usage.md` |
| Export formats | `docs/EXPORT.md` |
| Preference CLI/API | `docs/PREFERENCE_DATA.md` |
| Local OpenAI-compatible servers | `docs/LOCAL_MODELS.md` |
| Config → generator wiring | `afterimage/config.py`, `afterimage/config_to_generator.py` |
| Public exports | `afterimage/__init__.py` |

## Checklist before handing off a recipe

- [ ] API keys / `api_key_env` documented for the chosen provider.
- [ ] Respondent system prompt matches the **downstream training** target.
- [ ] If using documents: instruction callback + (when needed) **respondent** modifier both aligned with the same context strategy.
- [ ] **`max_turns`** and **`max_concurrency`** match cost and latency constraints.
- [ ] Output path and optional **`output.export`** / separate **`afterimage export`** step confirmed.
