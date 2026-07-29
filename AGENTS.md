# AfterImage: A framework for synthetic dataset generation based on your docs

**AfterImage** is a Python package for generating synthetic conversational datasets using state-of-the-art generative AI models. It is highly customizable, enabling tailored instruction generation, persona-based simulation, and fine-tuned conversational prompts.

## Features

- **Async & Parallel**: High-performance asynchronous generation using `AsyncConversationGenerator`.
- **Customizable Prompts**: Create bespoke respondent and correspondent prompts for generating realistic conversations.
- **Contextual Instruction Generation**: Leverage contextual documents to craft unique and relevant conversation starters.
- **Persona-Based Simulation**: Generate and adopt diverse user personas (e.g., "Novice", "Expert") for more realistic variation.
- **Smart API Key Management**: Handle multiple API keys with automatic rotation, rate limiting, and error handling.
- **Retrieval-Augmented Generation (RAG)**: Enhance responses with relevant context from vector databases.
- **Flexible Document Providers**: Support multiple document sources (files, JSONL, directories, Qdrant).
- **Multiple Storage Backends**: Store conversations in JSONL files or SQL databases (SQLite, PostgreSQL, MySQL).
- **Save in JSONL Format**: Export datasets directly for downstream applications.
- **Quality Analysis**: Comprehensive dataset quality checks with visualization support.
- **Environment-Free Agent Traces**: Generate synthetic agent execution trajectories combining ESAT methodology with sub-millisecond local declarative simulation tools (`afterimage.agent_trace`).
- **Generation Monitoring**: Real-time monitoring of generation metrics with alerts and visualization.

**See @DESIGN.md for the code design and architecture.**

### Evaluation

Afterimage evaluates conversations with **`ConversationJudge`** (`afterimage.evaluator`): embedding-based coherence, grounding, and relevance (via `EmbeddingProvider`) plus LLM-as-judge factuality and helpfulness (`agenerate_structured`). Scores are merged by **`CompositeEvaluator`** with configurable **`AggregationMode`**. With **`ConversationGenerator(..., auto_improve=True)`**, a judge is created automatically; default embeddings follow the chat provider (Gemini/OpenAI API) or a local process model for DeepSeek.

### Monitoring

Afterimage includes a comprehensive monitoring system for tracking generation metrics:

- **Metrics**: Generation time, success/error rates, token usage, conversation length.
- **Visualization**: Real-time plots and trend analysis.
- **Alerts**: Customizable alerts for anomalies (e.g., high error rate).

---

## Installation

To install AfterImage, clone the repository and install the dependencies:

```bash
pip install git+https://github.com/altaidevorg/afterimage.git

# Optional: Install SQL storage dependencies
pip install 'sqlalchemy>=2.0'
```

---

## Getting Started

Here's a step-by-step guide to start using **AfterImage**.

### 1. Setup

Make sure you have a valid API key for Google Gemini API. Set it as an environment variable:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

**Note**: We currently primarily support the Gemini API and OpenAI-compatible APIs.

### 2. Quickstart Script (Async & Personas)

The recommended way to use Afterimage is via `AsyncConversationGenerator`.

```python
import asyncio
import os
from afterimage import (
    AsyncConversationGenerator,
    PersonaInstructionGeneratorCallback,
    PersonaGenerator,
    InMemoryDocumentProvider,
)

# Get API key
api_key = os.getenv("GEMINI_API_KEY")

async def main():
    # Define the assistant (Respondent)
    respondent_prompt = "You are a helpful customer support agent for a coffee shop."

    # Prepare contextual documents
    docs = InMemoryDocumentProvider([
        "Our coffee is sourced from Ethiopia and Brazil.",
        "We offer free shipping on orders over $50."
    ])
    
    # 1. Generate Personas from documents (optional but recommended)
    # This creates varied user profiles like "Coffee Enthusiast" or "Price-Conscious Shopper"
    persona_gen = PersonaGenerator(api_key=api_key)
    await persona_gen.generate_from_documents(docs)

    # 2. Set up the instruction generator
    # This drives the 'User' (Correspondent) behavior
    instruction_callback = PersonaInstructionGeneratorCallback(
        api_key=api_key,
        documents=docs,
        num_random_contexts=1,
    )

    # 3. Initialize the AsyncConversationGenerator
    generator = AsyncConversationGenerator(
        respondent_prompt=respondent_prompt,
        api_key=api_key,
        model_name="gemini-2.0-flash",
        instruction_generator_callback=instruction_callback,
    )

    # 4. Generate conversations
    await generator.generate(
        num_dialogs=10,
        max_turns=3,
        max_concurrency=4  # Parallel generation
    )
    
    print("Conversation dataset generated successfully!")

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Advanced Usage with Multiple API Keys

```python
from afterimage import AsyncConversationGenerator, SmartKeyPool

# Initialize a pool of API keys with rate limits
key_pool = SmartKeyPool(
    api_keys=["key1", "key2", "key3"],
    hourly_limit=1000,
)

# Initialize generator with the key pool
generator = AsyncConversationGenerator(
    respondent_prompt="You are an expert assistant...",
    api_key=key_pool,
)

# Keys will be automatically rotated
await generator.generate(num_dialogs=1000)
```

### 4. Using Document Providers

AfterImage supports various document providers:

```python
from afterimage.providers import (
    JSONLDocumentProvider,
    DirectoryDocumentProvider,
    QdrantDocumentProvider,
)

# JSONL files
jsonl_provider = JSONLDocumentProvider(
    path_pattern="data/**/*.jsonl",
    content_key="content",
)

# Directory of text files
dir_provider = DirectoryDocumentProvider(
    directory="documents",
    file_patterns=["*.txt", "*.md"],
)

# Qdrant vector database
qdrant_provider = QdrantDocumentProvider(
    client=qdrant_client,
    collection_name="my_docs",
)
```

### 5. Storage Options

Default storage is JSONL, but SQL is also supported:

```python
from afterimage.storage import JSONLStorage, SQLStorage

# SQLite storage
sqlite_storage = SQLStorage(
    url="sqlite:///conversations.db",
    table_name="my_conversations",
)

generator = AsyncConversationGenerator(
    ...,
    storage=sqlite_storage
)
```

## Rules
- Whenever you need to write code that interacts with Gemini API, refer to https://googleapis.github.io/python-genai/ for the full API docs.
- Read before changing: Whenever you intend to modify a file, read it first to refresh the cash. The user may have changed it since you last read it.
- Whenever you make a change, always update the @DESIGN.md file to reflect the change.
