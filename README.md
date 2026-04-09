# AfterImage

**AfterImage** is a flexible Python framework for generating synthetic conversational datasets using State-of-the-Art Large Language Models (LLMs), primarily Google Gemini and OpenAI-compatible APIs.

It is designed to be highly customizable, enabling tailored instruction generation, persona-based simulation, diverse document ingestion, and context-aware conversation generation.

## 🚀 Getting Started

To get started with AfterImage, please refer to the **[Quickstart Guide](https://github.com/altaidevorg/afterimage/blob/main/docs/README.md)**. It covers:

*   **Installation**: How to set up the library.
*   **Basic Usage**: Generating simple conversations.
*   **RAG & Context**: Using documents to drive questions.
*   **Personas**: Creating varied user personas for realistic datasets.

## 📖 Documentation

*   **[Quickstart Guide](https://github.com/altaidevorg/afterimage/blob/main/docs/README.md)**: The best place to start.
*   **[Design & Architecture](DESIGN.md)**: Understanding the core concepts and codebase structure.
*   **[Examples](./examples)**: Examples of how to use AfterImage.

## 📦 Installation

```bash
pip install git+https://github.com/altaidevorg/afterimage.git
```

Optional extras (see `pyproject.toml`):

* **`embeddings-local`** — `sentence-transformers` for `ProcessEmbeddingProvider`, `QdrantRetriever` (by model name), and `QualityChecker` semantic checks.
* **`server`** — FastAPI app (`afterimage-server`).
* **`training`** — Torch/TRL stack for `examples/demo_ui/training_scripts/train.py`.

```bash
pip install "afterimage[embeddings-local]@git+https://github.com/altaidevorg/afterimage.git"
```

## ✨ Key Features

*   **Async-First**: High-performance parallel generation.
*   **Persona Simulation**: Realistic user diversity.
*   **Context-Aware**: Grounds conversations in your documents (RAG).
*   **Multi-Modal**: Support for text and potentially other modalities in future.
*   **Monitoring**: Real-time generation metrics and alerts.

