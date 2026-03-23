# AfterImage Design and Architecture

This document provides an overview of the design and architecture of the AfterImage library.

## Core Concepts

The library is designed around a few core concepts:

- **ConversationGenerator & AsyncConversationGenerator**: The main entry points for generating conversations. `AsyncConversationGenerator` is the recommended high-performance engine for concurrent generation.
- **PersonaGenerator**: Analyzes documents to generate diverse user personas, enhancing dataset variety.
- **LLMProvider**: An abstraction over different language model providers (Gemini, OpenAI compatible).
- **DatasetStorage**: An abstraction for storing and loading generated conversations and documents. It supports JSONL and SQL backends.
- **Callbacks**: These allow for customization of the generation process.
    - **InstructionGeneratorCallback**: Generates the initial questions or instructions (e.g., `PersonaInstructionGeneratorCallback`).
    - **RespondentPromptModifier**: Modifies the prompt for the respondent based on context (e.g., `WithRAGRespondentPromptModifier`).
- **Evaluation**: Flexible evaluation framework supporting Simple (LLM-as-judge) and Hybrid (Embedding + LLM) approaches.
- **Monitoring**: Real-time tracking of generation metrics (time, tokens, errors) with alert support.
- **Reasoning Capture**: OpenAI-compatible providers expose optional `reasoning_content`/`thinking` text; `AsyncConversationGenerator` persists assistant reasoning into `ConversationEntry.reasoning_content` when present.
- **Adaptive Context Sampling**: Document providers now keep per-document usage counts and sampling weights so instruction generation can bias toward underused contexts. When a context coverage stopping callback is present, its `target_visits` is propagated into provider weights; otherwise providers fall back to a soft-decay weighting strategy (`1 / (usage + 1)`). Usage is recorded only after a final row is produced successfully, and all sampled context ids are carried through row metadata for coverage accounting across contextual, persona, and tool-calling instruction callbacks. Metadata-to-context-id extraction is centralized so usage reporting and coverage counting share the same semantics. If a provider target is set explicitly, generator-side inference does not overwrite it, and generated instruction payloads keep their `context_ids` in per-instance state.
- **Fixed-Width Persona Generation**: Persona generation now enforces an exactly-five-persona contract per generation step. Persona outputs are LLM-generated, whitespace-normalized, deduplicated, and retried up to three times before a document-level enrichment failure is surfaced back to the caller. Batch enrichment now follows a gather-then-commit flow: documents are enriched on deep copies, successful results are saved together only after the whole batch succeeds, and failed batches leave in-memory documents and storage untouched.
- **Dynamic Persona Tree Depth**: `PersonaGenerator.generate_from_documents()` now treats `n_iterations=None` as auto mode. In auto mode it resolves an effective per-document persona target from provider usage targets or `target_data_count`, accounting for how many contexts are merged into each row, then chooses the depth whose expected pool is closest to that target instead of always building a large tree. Partial persona-chain failures are tolerated per branch so one bad expansion does not discard an entire document's persona tree.
- **Depth-Aware Persona Sampling**: Persona-based instruction callbacks flatten stored persona trees per document, preserve `generation_depth`, and derive a per-document persona target from provider coverage targets, explicit request size, or fixed-number stopping callbacks. If the effective target is unknown, they keep the full pool instead of pruning heuristically. When demand is smaller than supply, shallow layers are kept first via top-down pruning and round-robin reuse. When demand is larger than supply, personas are reused with layer-normalized depth weights so upper layers truly receive more total reuse despite deeper layers having many more nodes. Selection state is synchronized for threaded sync generation, and selected persona depth is propagated as `persona_generation_depth` in generation metadata. Sync and async evaluator retries rebuild the full row after regeneration and preserve any prompt-modifier-adjusted respondent prompt, so judges see the updated conversation under the same prompting conditions. Scenario coverage now includes auto-depth generation targets, small targets, layer-boundary targets, exact-pool matches, large oversampling targets, multi-context target inference, active-document inference cases, fixed-number stopping inference, and evaluator retry regressions.
- **Provider-Aware Concurrency**: Async generators and persona generation resolve concurrency defaults per provider, with a higher default for DeepSeek workloads.

## Directory Structure

The code is organized into the following directories and files:

- `afterimage/`: The main source code for the library.
    - `__init__.py`: Exposes the main classes and functions.
    - `async_conversation_generator.py`: **[NEW]** Asynchronous implementation of the conversation generator.
    - `base.py`: Base classes for generators and callbacks.
    - `callbacks.py`: Implements default callbacks for instructions and persona handling.
    - `common.py`: Common constants and data structures.
        It also holds provider-aware concurrency defaults.
    - `conversation_generator.py`: Synchronous conversation generator (Legacy).
    - `evaluator.py`: Conversation evaluation logic.
    - `key_management.py`: Smart API key management with rate limiting.
    - `monitoring.py`: Monitoring system implementation.
    - `persona_generator.py`: **[NEW]** Logic for generating personas from documents.
    - `prompts.py`: Default prompts (instruction generation, respondent persona etc.).
    - `quality.py`: Quality checking logic.
    - `retrievers.py`: Context retrieval strategies for RAG.
    - `storage.py`: Storage backends (JSONL, SQL).
    - `types.py`: Data models using Pydantic.
    - `evaluation/`: The new evaluation framework.
        - `__init__.py`: Exposes evaluation classes.
        - `base.py`: Base classes for evaluators.
        - `evaluators.py`: Concrete evaluation metrics.
        - `strategies.py`: Quality improvement strategies.
    - `providers/`:
        - `__init__.py`: Exposes provider classes.
        - `document_providers.py`: Document source implementations (Memory, File, Directory, Qdrant).
            Providers expose weighted random sampling plus document usage reporting for context coverage management, with target usage counts inferred from stopping callbacks when available.
        - `llm_providers.py`: LLM provider abstractions.

## Design Patterns

The library uses several design patterns to achieve its goals:

- **Strategy Pattern**: `LLMProvider`, `DatasetStorage`, `ContextRetriever`, and `BaseEvaluator` define swappable algorithms.
- **Factory Pattern**: `LLMFactory` creates `LLMProvider` instances dynamically.
- **Callback Pattern**: Customizes generation flow via `InstructionGeneratorCallback` and `RespondentPromptModifier`.
- **Composite Pattern**: `CompositeEvaluator` combines multiple evaluation metrics.
- **Template Method Pattern**: `BaseGenerator` and callbacks define algorithmic skeletons with overridable steps.
- **Async/Await Pattern**: `AsyncConversationGenerator` utilizes Python's `asyncio` for high-throughput concurrent generation.

## Architecture

The architecture of the library is modular and extensible. The core components are decoupled from each other, which makes it easy to replace or extend them. For example, you can easily add a new LLM provider by implementing the `LLMProvider` protocol and registering it with the `LLMFactory`. Similarly, you can add a new storage backend by implementing the `DatasetStorage` protocol.

The evaluation framework is also designed to be extensible. You can add new evaluation metrics by implementing the `BaseEvaluator` protocol and adding it to the `CompositeEvaluator`.

The monitoring system is also extensible. You can add new metric handlers by implementing the `MetricHandler` protocol and adding it to the `GenerationMonitor`.
