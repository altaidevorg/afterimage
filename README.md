# AfterImage

[![Tests](https://github.com/altaidevorg/afterimage/actions/workflows/tests.yml/badge.svg)](https://github.com/altaidevorg/afterimage/actions/workflows/tests.yml)
[![Ruff format](https://github.com/altaidevorg/afterimage/actions/workflows/ruff-format.yml/badge.svg)](https://github.com/altaidevorg/afterimage/actions/workflows/ruff-format.yml)
[![Ruff lint](https://github.com/altaidevorg/afterimage/actions/workflows/ruff-lint.yml/badge.svg)](https://github.com/altaidevorg/afterimage/actions/workflows/ruff-lint.yml)
[![Documentation](https://img.shields.io/badge/docs-afterimage.altai.dev-0066cc)](https://afterimage.altai.dev)

**AfterImage** is a Python library and CLI for generating **synthetic conversational datasets** with modern LLMs (Gemini, OpenAI-compatible APIs, DeepSeek, and local OpenAI-compatible servers). It is built so you can **start with a YAML file and one command**, then **compose** callbacks, document providers, storage, evaluation, and export pipelines as your needs grow—from quick experiments to large, production-style runs.

## Two ways to work (same engine)

**1. CLI and config — easy to begin**  
Describe generation in YAML, set your API key in the environment, and run `afterimage generate`. No boilerplate, no custom harness required to get JSONL on disk. Optional commands cover **export** to fine-tuning formats and **preference** (DPO-style) pair generation.

**2. Python API — composable and extensible**  
Use `ConversationGenerator`, `StructuredGenerator`, and `PersonaGenerator` with pluggable **instruction generators**, **respondent prompt modifiers**, **stopping criteria**, **storage** (JSONL or SQL), **quality judges**, and **monitoring**. The same abstractions power the CLI; you swap or combine pieces instead of forking the stack.

That split keeps onboarding shallow while leaving room for **scale** (concurrency, key pools, SQL storage) and **specialized flows** (RAG-style context, personas, structured extraction, preference data). Guides and API reference are on **[afterimage.altai.dev](https://afterimage.altai.dev)**.

---

## Installation

The package can be installed from PyPI as **`afterimage`**.

```bash
uv add afterimage
```

```bash
pip install afterimage
```

**Optional extras** (see `pyproject.toml` for exact dependency sets):

```bash
uv add "afterimage[embeddings-local]"
# or
pip install "afterimage[embeddings-local]"
```

| Extra | Purpose |
|--------|---------|
| `embeddings-local` | Local embeddings (`sentence-transformers`) for process-based embedding providers, Qdrant-style workflows, and quality checks that need a local model. |
| `server` | FastAPI app (`afterimage-server` entry point). |
| `training` | Torch / TRL stack, Gradio, and FastMCP for `examples/demo_ui` and the training scripts under `examples/`. |

---

## Start in minutes (CLI)

Requires **Python 3.11+** and an API key (e.g. `GEMINI_API_KEY` for the sample config).

```bash
afterimage generate -c examples/configs/basic.yaml
```

Dry-run the plan without calling the API:

```bash
afterimage generate -c examples/configs/basic.yaml --dry-run
```

Export a dataset to common fine-tuning formats:

```bash
afterimage export -i your_dataset.jsonl -f sharegpt -f messages
afterimage export --list-formats
```

Generate **preference** pairs from config:

```bash
afterimage preference -c your_config.yaml
```

More examples live under [`examples/configs/`](examples/configs/). In-depth guides (conversations, personas, structured generation, evaluation, export, preference data, local models) are on **[afterimage.altai.dev](https://afterimage.altai.dev)**.

---

## What you can build

- **Multi-turn synthetic chat** for SFT, evaluation sets, or simulation.  
- **Document-grounded** questions and answers (instruction side + optional respondent context).  
- **Persona-driven** diversity tied to your corpus.  
- **Structured outputs** via Pydantic schemas (single-turn extraction or generation).  
- **DPO / RLHF-style preference** data with multiple variation strategies.  
- **Quality loops** (async judge, optional auto-improve) and **observability** (metrics, periodic alert checks, exports to JSON/CSV/Parquet).

---

## Repository layout

| Path | Contents |
|------|-----------|
| [`docs/`](docs/) | Sphinx sources; mirrors and extends the hosted site. |
| [`examples/`](examples/) | YAML configs and demo flows. |
| [`DESIGN.md`](DESIGN.md) | Architecture and design notes for contributors. |
| [`afterimage/`](afterimage/) | Library and CLI implementation. |

**Source & issues:** [github.com/altaidevorg/afterimage](https://github.com/altaidevorg/afterimage)

---

## License

MIT (see PyPI package metadata and `pyproject.toml` classifiers).
