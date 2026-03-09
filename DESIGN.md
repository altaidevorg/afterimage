# AfterImage Design and Architecture

This document provides an overview of the design and architecture of the AfterImage library.

## Core Concepts

The library is designed around a few core concepts:

- **ConversationGenerator & AsyncConversationGenerator**: The main entry points for generating conversations. `AsyncConversationGenerator` is the recommended high-performance engine for concurrent generation.
- **PersonaGenerator**: Analyzes documents to generate diverse user personas, enhancing dataset variety.
- **LLMProvider**: An abstraction over different language model providers (Gemini, OpenAI compatible).
- **DatasetStorage**: An abstraction for storing the generated conversations. It supports JSONL and SQL backends.
- **Callbacks**: These allow for customization of the generation process.
    - **InstructionGeneratorCallback**: Generates the initial questions or instructions (e.g., `PersonaInstructionGeneratorCallback`).
    - **RespondentPromptModifier**: Modifies the prompt for the respondent based on context (e.g., `WithRAGRespondentPromptModifier`).
- **Evaluation**: Flexible evaluation framework supporting Simple (LLM-as-judge) and Hybrid (Embedding + LLM) approaches.
- **Monitoring**: Real-time tracking of generation metrics (time, tokens, errors) with alert support.

## Directory Structure

The code is organized into the following directories and files:

- `afterimage/`: The main source code for the library.
    - `__init__.py`: Exposes the main classes and functions.
    - `async_conversation_generator.py`: **[NEW]** Asynchronous implementation of the conversation generator.
    - `base.py`: Base classes for generators and callbacks.
    - `callbacks.py`: Implements default callbacks for instructions and persona handling.
    - `common.py`: Common constants and data structures.
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

## Scripts

- `generate_qa.py`: QA dataset generation script that uses `AsyncConversationGenerator` with a **dynamic system prompt parts** feature. Before generating QA pairs, it makes a single LLM API call (using `google-genai` with Pydantic structured output) to analyze the input document and generate context-appropriate system prompt "parts" (a role description and an answering instruction). These parts are used as the respondent prompt and also saved to the output JSON alongside QA pairs and formatted samples (following the `format_sample` pattern with `input`/`output` fields).
