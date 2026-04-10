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
*   **DPO/RLHF Preference Data**: Generate (chosen, rejected) pairs for reward model training — no manual labeling.
*   **Multi-Modal**: Support for text and potentially other modalities in future.
*   **Monitoring**: Real-time generation metrics and alerts.

## 🎯 Generating Preference Data (DPO/RLHF)

AfterImage can generate preference pairs directly from your documents:

```bash
afterimage preference -c config.yaml
```

Add a `preference` block to your config:

```yaml
preference:
  num_pairs: 100
  strategy: temperature    # temperature | prompt | model | combined
  output_format: dpo       # dpo | chat_dpo | ultrafeedback | anthropic_hh | orpo
  output_path: ./preferences.jsonl
```

See [docs/PREFERENCE_DATA.md](docs/PREFERENCE_DATA.md) for the full guide including multi-turn preferences, training tool integrations, and Python API.

