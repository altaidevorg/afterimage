# AfterImage Design and Architecture

This document provides an overview of the design and architecture of the AfterImage library.

## Core Concepts

The library is designed around a few core concepts:

- **ConversationGenerator**: The main entry point for generating conversations. Acts as a thin facade that delegates internally to focused components (Orchestrator, SamplingStrategy, QualityGate).
- **Orchestrator**: Manages async concurrency (semaphores, worker tasks, `asyncio.gather`), stopping criteria, and progress reporting. Coordinates SamplingStrategy and QualityGate during generation runs.
- **SamplingStrategy**: Encapsulates persona selection, context/document sampling, and instruction dispatch logic. Infers target usage counts from stopping callbacks and configures sampling for both context and persona pipelines.
- **QualityGate**: Wraps the evaluator (ConversationJudge) and retry logic for the `auto_improve` workflow. Determines whether a generated conversation meets quality thresholds or needs regeneration.
- **PersonaGenerator**: Analyzes documents to generate diverse user personas, enhancing dataset variety.
- **LLMProvider**: An abstraction over different language model providers (Gemini, OpenAI compatible).
- **EmbeddingProvider**: Async-first text embeddings (`async def embed(texts) -> list[list[float]]`). API backends (`OpenAIEmbeddingProvider`, `GeminiEmbeddingProvider`) use each vendor’s async client and `SmartKeyPool`; `ProcessEmbeddingProvider` runs SentenceTransformer in a `ProcessPoolExecutor` so the asyncio loop is not blocked by local inference (install the `embeddings-local` extra for `sentence-transformers`). Use `EmbeddingProviderFactory.create({...})` in `afterimage/providers/embedding_providers.py`.
- **DatasetStorage**: An abstraction for storing and loading generated conversations and documents. It supports JSONL and SQL backends.
- **Callbacks**: These allow for customization of the generation process.
    - **InstructionGeneratorCallback**: Generates the initial questions or instructions (e.g., `PersonaInstructionGeneratorCallback`).
    - **RespondentPromptModifier**: Modifies the prompt for the respondent based on context (e.g., `WithRAGRespondentPromptModifier`).
- **Evaluation**: Async `ConversationJudge` combines embedding metrics (via `EmbeddingProvider`) and LLM rubrics (`agenerate_structured`), with `CompositeEvaluator` aggregation (`MEAN`, `WEIGHTED_MEAN`, `MIN`). `ConversationGenerator(auto_improve=True)` builds a judge using `default_embedding_provider_config` when no embedding backend is passed explicitly.
- **Monitoring**: Real-time tracking of generation metrics (time, tokens, errors) with alert support.
- **Reasoning Capture**: OpenAI-compatible providers expose optional `reasoning_content`/`thinking` text; `AsyncConversationGenerator` persists assistant reasoning into `ConversationEntry.reasoning_content` when present.
- **OpenSimula (experimental)**: The `afterimage.simula` subpackage implements Simula-style mechanism design for synthetic data: reasoning-driven taxonomies (propose–critic–plan), weighted mix sampling, meta-prompt diversification with optional complexification, requirement critics with refinement, double-critic gating for MCQ, single-QA and MCQ task helpers, taxonomic coverage and Elo-style complexity scoring, and `SimulaInstructionGeneratorCallback` to feed precomputed scenarios into `ConversationGenerator`. See `afterimage/simula/README.md` and `examples/simula/README.md`. Import via `from afterimage.simula import OpenSimula` (not re-exported from the package root `__init__` to keep the default import surface stable).
- **Adaptive Context Sampling**: Document providers now keep per-document usage counts and sampling weights so instruction generation can bias toward underused contexts. When a context coverage stopping callback is present, its `target_visits` is propagated into provider weights; otherwise providers fall back to a soft-decay weighting strategy (`1 / (usage + 1)`). Usage is recorded only after a final row is produced successfully, and all sampled context ids are carried through row metadata for coverage accounting across contextual, persona, and tool-calling instruction callbacks. Metadata-to-context-id extraction is centralized so usage reporting and coverage counting share the same semantics. If a provider target is set explicitly, generator-side inference does not overwrite it, and generated instruction payloads keep their `context_ids` in per-instance state.
- **Fixed-Width Persona Generation**: Persona generation now enforces an exactly-five-persona contract per generation step. Persona outputs are LLM-generated, whitespace-normalized, deduplicated, and retried up to three times before a document-level enrichment failure is surfaced back to the caller. Batch enrichment now follows a gather-then-commit flow: documents are enriched on deep copies, successful results are saved together only after the whole batch succeeds, and failed batches leave in-memory documents and storage untouched.
- **Dynamic Persona Tree Depth**: `PersonaGenerator.generate_from_documents()` now treats `n_iterations=None` as auto mode. In auto mode it resolves an effective per-document persona target from provider usage targets or `target_data_count`, accounting for how many contexts are merged into each row, then chooses the depth whose expected pool is closest to that target instead of always building a large tree. Partial persona-chain failures are tolerated per branch so one bad expansion does not discard an entire document's persona tree.
- **Depth-Aware Persona Sampling**: Persona-based instruction callbacks flatten stored persona trees per document, preserve `generation_depth`, and derive a per-document persona target from provider coverage targets, explicit request size, or fixed-number stopping callbacks. If the effective target is unknown, they keep the full pool instead of pruning heuristically. When demand is smaller than supply, shallow layers are kept first via top-down pruning and round-robin reuse. When demand is larger than supply, personas are reused with layer-normalized depth weights so upper layers truly receive more total reuse despite deeper layers having many more nodes. Selection state is synchronized for threaded sync generation, and selected persona depth is propagated as `persona_generation_depth` in generation metadata. Sync and async evaluator retries rebuild the full row after regeneration and preserve any prompt-modifier-adjusted respondent prompt, so judges see the updated conversation under the same prompting conditions. Scenario coverage now includes auto-depth generation targets, small targets, layer-boundary targets, exact-pool matches, large oversampling targets, multi-context target inference, active-document inference cases, fixed-number stopping inference, and evaluator retry regressions.
- **Provider-Aware Concurrency**: Async generators and persona generation resolve concurrency defaults per provider, with a higher default for DeepSeek workloads.
- **Demo UI Storage Compatibility**: Demo `CaptureStorage` implements the full storage interface expected by generators (including `load_documents`) and keeps sync/async conversation save methods compatible with base storage contracts.
- **Demo UI Provider Consistency**: Tool-calling generation now builds personas with the same DeepSeek model/provider configuration used by demo generators to avoid cross-provider key mismatches.
- **Training Version Compatibility Guards**: Demo training requirements constrain `transformers` and `trl` to a compatible range (`transformers>=4.56.2,<5.0.0`, `trl>=0.29.1,<0.30.0`) to avoid runtime API mismatches during `SFTConfig` import and trainer startup. The same stack is declared as the `training` optional extra in `pyproject.toml` (aligned with `examples/demo_ui/training_scripts/requirements.txt`); install with `pip install -e ".[training]"` or `uv sync --extra training` so the Gradio-launched `train.py` subprocess can import `trl` and related packages.
- **Environment Template Files**: Repository root includes a minimal `.env.example` and local `.env` template for demo runtime and training credentials (`GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `HF_TOKEN`, optional `HF_HUB_DISABLE_XET`); `.env` is gitignored to keep secrets out of version control.
- **CLI Interface**: The `afterimage` command provides `generate`, `validate`, and `export` subcommands. Generation is driven by YAML config files that map to Pydantic models in `config.py`. The `export` command converts datasets to ShareGPT, Alpaca, or HuggingFace messages formats.
- **Agent Trace Dataset Generation**: The `afterimage.agent_trace` subpackage provides environment-free synthetic agent-trace dataset generation (combining ESAT methodology with a sub-millisecond local Declarative Tool Simulation Framework). Key components include `SchemaArchitect` (LLM Pydantic response schema generator with static AST verification feedback loop), `SchemaVerifier` (6 structural invariant checks), `DeclarativeEngine` (4-tier fallback generator with `SimulationContext` entity lookup pools), `GridTaskSynthesizer` (360-bucket grid + `InverseFrequencySampler` + procedural task rewriter), `ReActTrajectoryLoop` (multi-turn teacher execution against local tools), `TrajectoryJudge` (9-point LLM quality rubric), and `AsyncAgentTraceGenerator` facade. Model defaults follow `gemini-3.5-flash-lite` for execution/synthesis and `gemini-3.6-flash` for schema architecture and trajectory judging.
- **Local Model Support**: The `local` provider wraps the OpenAI-compatible API with local-friendly defaults: no API key required, no rate limiting via SmartKeyPool, extended timeouts (30s connect, 300s request), and clear connection error messages. Works with vLLM, Ollama, and llama.cpp servers.

## Directory Structure

The code is organized into the following directories and files:

- `afterimage/`: The main source code for the library.
    - `__init__.py`: Exposes the main classes and functions.
    - `conversation_generator.py`: Thin facade over Orchestrator, SamplingStrategy, and QualityGate. Maintains the full public API for backward compatibility.
    - `orchestrator.py`: Async concurrency management — worker tasks, semaphores, stopping criteria, and progress reporting.
    - `sampling.py`: Persona and context sampling coordination — target inference, coverage configuration, and usage recording.
    - `quality_gate.py`: Evaluator wrapper with retry logic for the auto-improve workflow.
    - `config.py`: Pydantic models for YAML config schema and `load_config()` loader.
    - `config_to_generator.py`: `build_generator()` translates config into a ConversationGenerator.
    - `cli.py`: Click-based CLI (`afterimage generate/validate/export`).
    - `exporters.py`: Dataset format converters (ShareGPT, Alpaca, HuggingFace messages).
    - `base.py`: Base classes for generators and callbacks. Sampling methods delegate to SamplingStrategy.
    - `callbacks.py`: Implements default callbacks for instructions and persona handling.
    - `common.py`: Common constants and data structures.
        It also holds provider-aware concurrency defaults.
    - `evaluator.py`: `ConversationJudge` and embedding defaults for auto-improve.
    - `key_management.py`: Smart API key management with rate limiting.
    - `monitoring.py`: Monitoring system implementation.
    - `persona_generator.py`: **[NEW]** Logic for generating personas from documents.
    - `prompts.py`: Default prompts (instruction generation, respondent persona etc.).
    - `quality.py`: Quality checking logic.
    - `retrievers.py`: Context retrieval strategies for RAG (`ContextRetriever`, `RetrievalResult`, optional `*_context_with_metadata`, `QdrantRetriever`, `StaticContextRetriever`, composite retrievers).
    - `storage.py`: Storage backends (JSONL, SQL).
    - `types.py`: Data models using Pydantic.
    - `simula/`: **OpenSimula (experimental)** — `OpenSimula` orchestrator, `taxonomy_builder` (optional `show_progress` + tqdm), `cli_logging` (`configure_example_console`, `silence_noisy_third_party_loggers`), `sampling`, `meta_prompt`, `critics`, `double_critic`, `evaluation`, `document_context`, and `tasks/` (single QA, MCQ, multiturn instruction callback).
    - `evaluation/`: The new evaluation framework.
        - `__init__.py`: Exposes evaluation classes.
        - `base.py`: Base classes for evaluators.
        - `evaluators.py`: Concrete evaluation metrics.
        - `strategies.py`: Quality improvement strategies.
    - `providers/`:
        - `__init__.py`: Exposes provider classes.
        - `document_providers.py`: Document source implementations (Memory, File, Directory, Qdrant).
            Providers expose weighted random sampling plus document usage reporting for context coverage management, with target usage counts inferred from stopping callbacks when available.
        - `llm_providers.py`: LLM provider abstractions (Gemini, OpenAI, DeepSeek, OpenRouter, local OpenAI-compatible) and `LLMFactory`.
        - `local_provider.py`: `LocalLLMProvider` for OpenAI-compatible local servers (vLLM, Ollama, llama.cpp).
        - `embedding_providers.py`: Async embedding providers (OpenAI, Gemini, process pool) and factory.
- `examples/configs/`: YAML config examples (`basic.yaml`, `rag.yaml`, `local.yaml`).
- `examples/demo_ui/`: Gradio demo application.
    - `README.md`: Page-by-page demo UI guide (routes, features, setup, troubleshooting).
    - `app.py`: Ensures the repository root is included in `sys.path` when the demo is run directly as `uv run examples/demo_ui/app.py`.
    - `core/storage.py`: Implements demo capture storage with full storage-protocol compatibility.
    - `pages/handlers/generation.py`: Aligns persona-generation provider/model with the demo generator provider defaults.
    - `training_scripts/requirements.txt`: Owns training stack compatibility bounds for `trl` and `transformers`.
    - **Demo training subprocess**: `train.py` accepts `--dataset` (resolved relative to `training_scripts/` cwd) so the UI-prepared merge/filter output always matches what SFT trains on; `training_config` loads `.env` via `find_dotenv` so `HF_TOKEN` is found from repo root when the training cwd is `training_scripts/`. On failure, the runner surfaces the last lines of subprocess output instead of a generic message only.

## Design Patterns

The library uses several design patterns to achieve its goals:

- **Facade Pattern**: `ConversationGenerator` presents a simple public API while delegating to `Orchestrator`, `SamplingStrategy`, and `QualityGate` internally.
- **Strategy Pattern**: `LLMProvider`, `DatasetStorage`, `ContextRetriever`, `SamplingStrategy`, and `BaseEvaluator` define swappable algorithms.
- **Factory Pattern**: `LLMFactory` creates `LLMProvider` instances dynamically.
- **Callback Pattern**: Customizes generation flow via `InstructionGeneratorCallback` and `RespondentPromptModifier`.
- **Composite Pattern**: `CompositeEvaluator` combines multiple evaluation metrics.
- **Template Method Pattern**: `BaseGenerator` and callbacks define algorithmic skeletons with overridable steps.
- **Async/Await Pattern**: All the generators utilizes Python's `asyncio` for high-throughput concurrent generation.
- **Single Responsibility**: Each extracted component owns one concern — `SamplingStrategy` handles sampling, `QualityGate` handles evaluation gating, `Orchestrator` handles concurrency.

## Architecture

The architecture of the library is modular and extensible. The core components are decoupled from each other, which makes it easy to replace or extend them.

### Component Decomposition

```
CLI (afterimage generate/validate/export)
└── Config (YAML) → build_generator()
    └── ConversationGenerator (facade)
        ├── Orchestrator          — async concurrency, worker tasks, stopping criteria
        │   ├── SamplingStrategy  — persona/context sampling, target inference
        │   └── QualityGate       — evaluator wrapper, retry decisions
        ├── LLMProvider           — model interaction (chat sessions, content generation)
        │   ├── GeminiProvider    — Google Gemini API
        │   ├── OpenAIProvider    — OpenAI / compatible APIs
        │   ├── DeepSeekProvider  — DeepSeek API
        │   ├── LocalLLMProvider  — local servers (vLLM, Ollama, llama.cpp)
        │   └── SmartKeyPool      — rate-limit management (hidden from upper layers)
        ├── Storage               — JSONL / SQL backends
        └── Callbacks             — instruction generation, prompt modification, stopping
```

**Data flow during generation:**
1. `ConversationGenerator.generate()` delegates to `Orchestrator.run()`
2. `Orchestrator` uses `SamplingStrategy` to configure persona/context sampling
3. `Orchestrator` spawns concurrent worker tasks with semaphore-bounded parallelism
4. Each worker calls `ConversationGenerator.generate_single()` for one conversation
5. `generate_single()` runs the multi-turn loop, then passes the result through `QualityGate`
6. `QualityGate` evaluates quality and signals retry if the grade is below threshold
7. Accepted conversations are saved to `Storage`

### Extensibility

You can easily add a new LLM provider by implementing the `LLMProvider` protocol and registering it with the `LLMFactory`. Similarly, you can add a new storage backend by implementing the `DatasetStorage` protocol.

The evaluation framework is also designed to be extensible. You can add new evaluation metrics by implementing the `BaseEvaluator` protocol and adding it to the `CompositeEvaluator`.

The monitoring system is also extensible. You can add new metric handlers by implementing the `MetricHandler` protocol and adding it to the `GenerationMonitor`.

## Scripts

- `scripts/generate_qa.py`: QA dataset generation script that uses `AsyncConversationGenerator` with a **dynamic system prompt parts** feature. Before generating QA pairs, it makes a single LLM API call (using `google-genai` with Pydantic structured output) to analyze the input document and generate context-appropriate system prompt "parts" (a role description and an answering instruction). These parts are used as the respondent prompt and also saved to the output JSON alongside QA pairs and formatted samples (following the `format_sample` pattern with `input`/`output` fields).
