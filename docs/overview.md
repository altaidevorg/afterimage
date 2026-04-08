# Overview

Afterimage is a powerful Python framework designed for generating high-quality synthetic conversation datasets using Large Language Models (LLMs). It provides a structured, scalable, and customizable way to simulate interactions between a user (Correspondent) and an AI assistant (Respondent), enabling the creation of diverse training and evaluation datasets for various applications.

Unlike simple prompt loops, Afterimage offers a sophisticated engine that manages personas, context retrieval (RAG), structured data extraction, and comprehensive quality evaluation, all while running asynchronously for maximum performance.

## Use Cases

Afterimage is built to support a wide range of synthetic data needs:

*   **Domain-Specific QA**: Generate question-answer pairs based on your technical documentation to fine-tune models for expert-level support.
*   **Enterprise Knowledge Training**: Create safe, synthetic datasets from internal corporate wikis or knowledge bases to train models without exposing raw sensitive data.
*   **Customer Support Simulation**: Simulate realistic customer interaction scenarios—including angry, confused, or happy customers—to train support bots or human agents.
*   **Data Extraction & Structuring**: Convert unstructured text into strictly formatted JSON data for downstream processing or database population.
*   **Evaluation Datasets**: Produce "Golden Sets" of complex questions and ground-truth answers to benchmark the performance of your RAG pipelines.

## Key Features

*   **⚡ Async & Parallel**: Built on `asyncio` to generate hundreds of conversations concurrently, maximizing API throughput.
*   **🎭 Persona Simulation**: Automatically generate unique user personas (e.g., "Novice User", "Senior Engineer") to ensure dataset diversity.
*   **🧠 RAG-Native**: First-class support for injecting context from vector databases (like Qdrant) or local files into the generation process.
*   **📋 Structured Generation**: Enforce Pydantic schemas on outputs to ensure generated data is machine-readable and valid.
*   **⚖️ Evaluation Framework**: Built-in evaluators (both LLM-as-judge and embedding-based) to measure coherence, factuality, and grounding.
*   **📊 Monitoring**: Real-time observability into generation metrics like token usage, meaningfulness, and error rates.

## Key Concepts

To effectively use Afterimage, it helps to understand its core abstractions:

### 1. Correspondent vs. Respondent
*   **Correspondent (The User)**: The simulated agent that initiates the conversation. It asks questions, follows up, or gives instructions. Its behavior is driven by an **Instruction Generator**.
*   **Respondent (The Assistant)**: The agent that replies. Its behavior is defined by your system prompt. This is usually the model you are trying to emulate or train.

### 2. Generators
The engine that runs the simulation.
*   **`ConversationGenerator`**: The primary engine for multi-turn chat.
*   **`StructuredGenerator`**: Specialized engine for single-turn data extraction into JSON.

### 3. Callbacks
Hooks to customize the behavior of the agents:
*   **Instruction Generator Callback**: Decides *what* the Correspondent asks. It can read from documents to ask relevant questions.
*   **Respondent Prompt Modifier**: Dynamically changes the Respondent's system prompt (e.g., injecting the relevant context/chunks for RAG).

### 4. Document Providers
The source of knowledge for the generation. Afterimage supports reading from:
*   Local text/markdown files.
*   JSONL datasets.
*   Vector Databases (Qdrant).
*   In-memory lists.

### 5. Personas
Profiles that shape the Correspondent's tone, vocabulary, and expertise level. Instead of a generic "User", Afterimage can simulate specific demographics to stress-test your model.
