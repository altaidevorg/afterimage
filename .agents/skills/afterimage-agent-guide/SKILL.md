---
name: afterimage-agent-guide
description: >-
  Uses the installed AfterImage PyPI package and its CLI in the user's own project
  to generate synthetic multi-turn datasets, grounded conversations, DPO pairs,
  exports, and analytics—not to hack the upstream library repo. Use when the user
  pip-installed afterimage, mentions ConversationGenerator, YAML configs, personas,
  SmartKeyPool, afterimage generate/export, or afterimage.altai.dev.
---

# AfterImage — consumer agent guide

Assume the agent works in **the user's application or data pipeline directory**, with **afterimage installed** (`pip install afterimage` / `uv add afterimage`). Do not assume a checkout of [github.com/altaidevorg/afterimage](https://github.com/altaidevorg/afterimage) unless the user explicitly says they are developing the library.

AfterImage is a **Python 3.11+** library and a **Click CLI** entry point named `afterimage`. It simulates a **correspondent** (user side) and **respondent** (assistant side), optionally grounded on documents, then writes **JSONL** (default) or **SQL** storage.

**Authoritative sources for consumers:**

- [https://afterimage.altai.dev](https://afterimage.altai.dev) — main documentation
- [https://afterimage.altai.dev/llms.txt](https://afterimage.altai.dev/llms.txt) — install, links, and examples oriented to agents
- [https://afterimage.altai.dev/api/index.html](https://afterimage.altai.dev/api/index.html) — API reference (matches the published package)
- [https://pypi.org/project/afterimage](https://pypi.org/project/afterimage) — version, extras, metadata
- Optional: clone the GitHub repo **only** when the user needs upstream examples (`examples/`) or is contributing patches

**Anti-hallucination rule:** Prefer the Sphinx API pages and `help(...)` / `inspect.signature(...)` on the **installed** package over random snippets from blogs or third-party posts. If a detail must be exact (export names, YAML keys), use the runtime checks in [reference.md](reference.md) or the tables there, which are keyed by **public module paths** (`afterimage.*`), not paths on disk in a monorepo.

---

## Mental model

1. **Respondent** = model you want to imitate in training data (`respondent_prompt` / YAML `respondent.system_prompt`).
2. **Correspondent** = model that plays the user; driven by a **correspondent system prompt** and/or an **instruction generator callback** that emits first user turns and metadata (context, persona).
3. Each dialog samples an integer **turn count uniformly from `1` through `max_turns`** (see `ConversationGenerator.generate` in the API docs / docstring).
4. **CLI** builds a `ConversationGenerator` from validated YAML; when there are no documents or `context.enabled` is false, it uses `SimpleInstructionGeneratorCallback` internally (same behavior as documented under configuration → generation in the official docs).

---

## Installation

```bash
pip install afterimage
# optional extras (see PyPI “Optional extras” or llms.txt for current names):
pip install "afterimage[embeddings-local]"
pip install "afterimage[server]"
pip install "afterimage[training]"
```

Console scripts: `afterimage` (CLI). `afterimage-server` requires the `server` extra.

Pin versions in the user’s project (`requirements.txt` / `pyproject.toml`) if reproducibility matters.

---

## CLI workflow (no Python)

Use **paths in the user’s project** (not paths inside a library clone):

```bash
export GEMINI_API_KEY=your_key
afterimage validate -c ./configs/basic.yaml
afterimage generate -c ./configs/basic.yaml
afterimage generate -c ./configs/basic.yaml --dry-run
afterimage export -i ./output/dataset.jsonl -f sharegpt -f messages --split 0.9
afterimage export --list-formats
afterimage analyze -i ./output/dataset.jsonl -o ./reports/dataset.html
afterimage preference -c ./configs/preference.yaml
```

Starter YAML shapes appear in the official docs and in the upstream repo’s `examples/configs/` **on GitHub** if the user wants a template to copy into `./configs/`.

`generation` in YAML must provide a stop signal: either `num_dialogs` or at least one `generation.stopping` rule (see configuration reference on [afterimage.altai.dev](https://afterimage.altai.dev)).

---

## Python API — minimal valid generator

`ConversationGenerator` **requires** at least one of `correspondent_prompt` or `instruction_generator_callback`. `generate()` **requires** an instruction callback on the constructor in current best practice; passing it only to `generate()` is deprecated.

Minimal pattern (no documents): `SimpleInstructionGeneratorCallback`.

```python
import asyncio
import os

from afterimage import ConversationGenerator
from afterimage.callbacks import SimpleInstructionGeneratorCallback

async def main() -> None:
    api_key = os.environ["GEMINI_API_KEY"]
    instruction_cb = SimpleInstructionGeneratorCallback(
        api_key=api_key,
        model_name="gemini-2.5-flash",
        model_provider_name="gemini",
        n_instructions=3,
    )
    gen = ConversationGenerator(
        respondent_prompt="You are a helpful assistant. Be concise.",
        api_key=api_key,
        model_name="gemini-2.5-flash",
        model_provider_name="gemini",
        instruction_generator_callback=instruction_cb,
    )
    await gen.generate(num_dialogs=20, max_turns=4, max_concurrency=4)
    rows = gen.load_conversations()
    print(f"saved rows: {len(rows)}")

asyncio.run(main())
```

---

## Python API — document + persona grounding

1. Build a `DocumentProvider` (e.g. `InMemoryDocumentProvider(list[str])`) or `JSONLDocumentProvider` / `DirectoryDocumentProvider` from `afterimage.providers`.
2. **Optional:** `await PersonaGenerator(...).generate_from_documents(docs)` — mutates document objects in place with personas.
3. Pass `PersonaInstructionGeneratorCallback` (or `ContextualInstructionGeneratorCallback`) as `instruction_generator_callback`.
4. For respondent answers conditioned on the same context string, set `respondent_prompt_modifier=WithContextRespondentPromptModifier()` (exported from top-level `afterimage`).

```python
import asyncio
import os

from afterimage import (
    ConversationGenerator,
    InMemoryDocumentProvider,
    PersonaGenerator,
    PersonaInstructionGeneratorCallback,
    WithContextRespondentPromptModifier,
)

DOCUMENTS = [
    "Espresso is brewed under pressure and is the base for milk drinks.",
    "A pour-over uses a filter; control grind and pour rate for extraction.",
]

async def main() -> None:
    api_key = os.environ["GEMINI_API_KEY"]
    docs = InMemoryDocumentProvider(DOCUMENTS)

    persona_gen = PersonaGenerator(api_key=api_key, model_name="gemini-2.5-flash")
    await persona_gen.generate_from_documents(docs)

    instruction_cb = PersonaInstructionGeneratorCallback(
        api_key=api_key,
        documents=docs,
        model_name="gemini-2.5-flash",
        num_random_contexts=1,
        n_instructions=3,
    )

    gen = ConversationGenerator(
        respondent_prompt="You are a coffee educator. Ground answers in the provided context.",
        api_key=api_key,
        model_name="gemini-2.5-flash",
        instruction_generator_callback=instruction_cb,
        respondent_prompt_modifier=WithContextRespondentPromptModifier(),
    )
    await gen.generate(num_dialogs=50, max_turns=3, max_concurrency=5)

asyncio.run(main())
```

---

## API keys at scale

`SmartKeyPool` (`from afterimage import SmartKeyPool` or `afterimage.key_management`) accepts `api_keys: list[str]`, optional `hourly_limit`, `daily_limit`, `error_threshold`, `cooldown_period`. `ConversationGenerator` accepts `api_key: str | SmartKeyPool` and wraps a bare string with `SmartKeyPool.from_single_key`.

---

## DPO / preference pairs

Use `afterimage.preference.generator.PreferenceGenerator` with a configured `ConversationGenerator` and `ConversationJudge`.

`ConversationJudge` is constructed with an **`LLMProvider`** and an **`EmbeddingProvider`** (or use `ConversationJudge.from_factory(...)` with `key_pool` and `default_embedding_provider_config` — same pattern the `afterimage preference` CLI uses).

```python
from afterimage.preference.types import PreferenceConfig

pref = gen.to_preference_generator(
    judge=judge,
    config=PreferenceConfig(num_pairs=100, output_path="./out/prefs.jsonl"),
)
pairs, analytics = await pref.generate()
pref.save_pairs(pairs, analytics)
await judge.aclose()
```

Full judge wiring example: [reference.md](reference.md).

---

## Quality gate (`auto_improve`)

With `ConversationGenerator(..., auto_improve=True)`, a judge is created automatically. For `model_provider_name="local"`, local embeddings may be required; the package raises a clear `ValueError` suggesting `pip install "afterimage[embeddings-local]"` when needed.

---

## Name alias

`AsyncConversationGenerator` is the same class as `ConversationGenerator` (re-export). Prefer `ConversationGenerator` in new code.

---

## Best practices (consumers)

- **Validate before spend:** `afterimage validate -c ...` or `afterimage generate ... --dry-run`.
- **Match provider keys:** `model_provider_name` / YAML `model.provider` must be one of `gemini`, `openai`, `deepseek`, `local`, `openrouter` (`afterimage.types.MODEL_PROVIDER_NAMES`).
- **Instruction callback on the constructor:** avoids deprecation warnings and matches CLI behavior.
- **Stable output path:** pass `JSONLStorage(conversations_path=...)` into `ConversationGenerator` when you cannot rely on default timestamped files.
- **YAML `documents.provider`:** supported values for the CLI are those documented for configs (see official docs). For in-memory text corpora in **Python**, use `InMemoryDocumentProvider`; do not assume a YAML `memory` provider exists unless the docs for your installed version say so.

---

## Finding implementation details without a repo checkout

| Goal | What to run or open |
|------|---------------------|
| Installed version | `python -c "import afterimage; print(afterimage.__version__)"` |
| Package location | `python -c "import afterimage, inspect; print(inspect.getfile(afterimage))"` |
| CLI help | `afterimage --help`, `afterimage generate --help`, … |
| Symbol signatures | `help(afterimage.ConversationGenerator)`, `inspect.signature(...)` |
| Export IDs for this install | `python -c "from afterimage.integrations import list_formats; print([x['name'] for x in list_formats()])"` |
| Deep reference | [reference.md](reference.md) |

For contributors maintaining **upstream** AfterImage, use the GitHub repository and `DESIGN.md`; that workflow is out of scope for this skill.
