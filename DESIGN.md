# AfterImage Design and Architecture

This document provides an overview of the design and architecture of the AfterImage library.

## Core Concepts

The library is designed around a few core concepts:

- **ConversationGenerator**: The main entry point for generating conversations. It orchestrates the entire process, from generating questions to getting answers and evaluating the results.
- **LLMProvider**: An abstraction over different language model providers (currently only Gemini is supported). This allows for easy extension to other providers like OpenAI, Anthropic, etc.
- **DatasetStorage**: An abstraction for storing the generated conversations. It supports JSONL and SQL backends.
- **Callbacks**: These allow for customization of the generation process. There are two main types of callbacks:
    - **InstructionGeneratorCallback**: Generates the initial questions or instructions for the conversation.
    - **RespondentPromptModifier**: Modifies the prompt for the respondent (the one answering the questions) based on the context.
- **Evaluation**: The library provides a flexible evaluation framework to assess the quality of the generated conversations. It supports both simple LLM-based evaluation and a more advanced hybrid approach.
- **Monitoring**: A system for tracking various metrics during the generation process, such as generation time, success/error rates, and token usage.

## Directory Structure

The code is organized into the following directories and files:

- `afterimage/`: The main source code for the library.
    - `__init__.py`: Exposes the main classes and functions of the library.
    - `base.py`: Contains the base classes for generators and callbacks.
    - `callbacks.py`: Implements the default callbacks for instruction generation and prompt modification.
    - `common.py`: Contains common constants and data structures.
    - `conversation_generator.py`: The main class for generating conversations.
    - `evaluator.py`: Implements the conversation evaluation logic.
    - `key_management.py`: Manages API keys with rate limiting and error handling.
    - `monitoring.py`: Implements the monitoring system.
    - `prompts.py`: Contains the default prompts used by the library.
    - `quality.py`: Implements the quality checking logic.
    - `retrievers.py`: Implements different context retrieval strategies for RAG.
    - `storage.py`: Implements the storage backends.
    - `types.py`: Contains the data models for conversations, evaluations, etc.
    - `evaluation/`: The new evaluation framework.
        - `__init__.py`: Exposes the main classes of the evaluation framework.
        - `base.py`: Contains the base classes for evaluators.
        - `evaluators.py`: Implements the different evaluation metrics.
        - `strategies.py`: Implements strategies for improving the quality of the generated conversations.
    - `providers/`:
        - `__init__.py`: Exposes the main classes of the providers.
        - `document_providers.py`: Implements different ways to provide documents for context.
        - `llm_providers.py`: Implements the abstraction over LLM providers.

## Design Patterns

The library uses several design patterns to achieve its goals:

- **Strategy Pattern**: The `LLMProvider`, `DatasetStorage`, `ContextRetriever`, and `BaseEvaluator` are all examples of the strategy pattern. They define a common interface for a family of algorithms, and let the client choose which one to use.
- **Factory Pattern**: The `LLMFactory` is a factory for creating `LLMProvider` instances. This makes it easy to add new providers without changing the client code.
- **Callback Pattern**: The `InstructionGeneratorCallback` and `RespondentPromptModifier` are callbacks that allow the client to customize the generation process.
- **Composite Pattern**: The `CompositeEvaluator` is a composite of multiple `BaseEvaluator`s. This allows for combining different evaluation metrics into a single evaluation.
- **Template Method Pattern**: The `BaseInstructionGeneratorCallback` and `BaseRespondentPromptModifierCallback` use the template method pattern to define the skeleton of an algorithm, while letting subclasses override specific steps.

## Architecture

The architecture of the library is modular and extensible. The core components are decoupled from each other, which makes it easy to replace or extend them. For example, you can easily add a new LLM provider by implementing the `LLMProvider` protocol and registering it with the `LLMFactory`. Similarly, you can add a new storage backend by implementing the `DatasetStorage` protocol.

The evaluation framework is also designed to be extensible. You can add new evaluation metrics by implementing the `BaseEvaluator` protocol and adding it to the `CompositeEvaluator`.

The monitoring system is also extensible. You can add new metric handlers by implementing the `MetricHandler` protocol and adding it to the `GenerationMonitor`.
