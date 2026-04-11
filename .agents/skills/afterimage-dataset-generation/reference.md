# AfterImage — agent reference sheet

Concise pointers into this repository. Prefer reading source when behavior is ambiguous.

## Public Python surface

Import stable symbols from **`afterimage`** (see `afterimage/__init__.py`): e.g. `ConversationGenerator`, `StructuredGenerator`, `PersonaGenerator`, `SmartKeyPool`, `ConversationJudge`, `GenerationMonitor`, document providers (`InMemoryDocumentProvider`, …), `LLMFactory`, `EmbeddingProviderFactory`.

Instruction generators and modifiers live under **`afterimage.callbacks`** (re-exported from top-level for common types): `ContextualInstructionGeneratorCallback`, `PersonaInstructionGeneratorCallback`, `WithContextRespondentPromptModifier`, `WithRAGRespondentPromptModifier`, stopping callbacks, etc.

Preference types: **`afterimage.preference`** — `PreferenceConfig`, `PreferenceGenerator`, …

Storage: **`afterimage.storage`** — `BaseStorage`, `JSONLStorage`, `SQLStorage`.

## CLI (`afterimage` script)

Defined in `afterimage/cli.py`. Common commands:

| Command | Purpose |
|---------|---------|
| `afterimage generate -c FILE.yaml` | Run generation from config |
| `afterimage export ...` | Convert JSONL to training formats |
| `afterimage preference -c FILE.yaml` | Preference / DPO pair generation |
| `afterimage validate` | Config / path validation (see `--help`) |

Always suggest `--help` on the relevant subcommand when options are unclear.

## Config model

`AfterImageConfig` in `afterimage/config.py` includes:

- **`generation`**: `num_dialogs`, `max_turns`, `max_concurrency`, …
- **`model`**: `provider` (`gemini` \| `openai` \| `deepseek` \| `local`), `model_name`, `api_key_env`, optional `base_url` for local/OpenAI-compatible.
- **`respondent`**: `system_prompt`.
- **`output`**: `path`, `storage` (`jsonl` \| `sql`), optional nested **`export`** (formats, split, seed).
- **`documents`**, **`context`**, **`personas`**, **`quality`**, **`preference`**, etc. as optional sections.

Load path: `afterimage.config.load_config`.

## Documentation map (`docs/`)

Sphinx toctree root: `docs/index.rst`. Markdown tutorials and guides live alongside `docs/api/` (RST autodoc).

## Tests and examples

- **`tests/`** — pytest; `tests/test_monitoring.py` for monitor / alert behavior.
- **`examples/configs/`** — YAML templates (`basic.yaml`, `local.yaml`, …).

## Terminology

Use **Correspondent** / **Respondent** consistently with user-facing docs; avoid mixing in “user/assistant” without mapping to those terms when explaining AfterImage-specific hooks.
