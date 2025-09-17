# AfterImage:  A framework for synthetic dataset generation based on your docs

**AfterImage** is a Python package for generating synthetic conversational datasets using state-of-the-art generative AI models. It is highly customizable, enabling tailored instruction generation and fine-tuned conversational prompts.

## Features

- **Customizable Prompts**: Create bespoke respondent and correspondent prompts for generating realistic conversations.
- **Contextual Instruction Generation**: Leverage contextual documents to craft unique and relevant conversation starters.
- **Smart API Key Management**: Handle multiple API keys with automatic rotation, rate limiting, and error handling.
- **Retrieval-Augmented Generation (RAG)**: Enhance responses with relevant context from vector databases.
- **Flexible Document Providers**: Support multiple document sources (files, JSONL, directories, Qdrant).
- **Multiple Storage Backends**: Store conversations in JSONL files or SQL databases (SQLite, PostgreSQL, MySQL).
- **Parallel Execution**: Generate multiple conversations efficiently using multithreading.
- **Save in JSONL Format**: Export datasets directly for downstream applications.
- **Quality Analysis**: Comprehensive dataset quality checks with visualization support.
- **Generation Monitoring**: Real-time monitoring of generation metrics with alerts and visualization.

**Note**: This is the initial version, but I will add more generators and evaluators soon.

**See @DESIGN.md for the code design and architecture.**

### Evaluation

Afterimage provides two evaluation approaches:

1. **Simple LLM-based Evaluation** (SimpleSyntheticDatasetEvaluator)
   - Uses LLM as a judge to evaluate conversations
   - Single-model evaluation approach
   - Suitable for basic quality checks
   - Legacy support for older code

2. **Hybrid Evaluation System** (HybridSyntheticDatasetEvaluator)
   - Combines embedding models and LLMs for comprehensive evaluation
   - Multiple evaluation metrics:
     - Coherence: Measures question-answer semantic alignment
     - Grounding: Ensures responses are based on provided context
     - Relevance: Checks if questions are based on the provided context
     - Factuality: Verifies factual accuracy using LLM
     - Helpfulness: Assesses response usefulness using LLM
   - Extensible architecture for custom evaluators
   - Weighted combination of metrics
   - More robust and detailed evaluation

### Monitoring

Afterimage includes a comprehensive monitoring system for tracking generation metrics:

1. **Metrics Tracking**
   - Generation time
   - Success/error rates
   - Token usage
   - Conversation length
   - Custom metrics support

2. **Visualization**
   - Real-time metric plots
   - Success/error rate trends
   - Generation time distribution
   - Token usage patterns

3. **Handlers**
   - File-based metric logging
   - Custom metric handlers
   - Alert system for anomalies
   - Extensible handler architecture

4. **Export Options**
   - JSON/JSONL format
   - CSV/Excel export
   - Parquet support
   - Visualization export

Example usage with monitoring:

```python
from afterimage import ConversationGenerator, GenerationMonitor

# Initialize monitor
monitor = GenerationMonitor(log_dir="monitoring_logs")

# Create generator with monitoring
generator = ConversationGenerator(
    respondent_prompt="Your prompt here",
    api_key="your-api-key",
    monitor=monitor
)

# Generate conversations
generator.generate_dataset(num_dialogs=3)

# Get metrics
success_rate = monitor.get_metrics("success_rate")
print(f"Success rate: {success_rate['mean']:.2%}")

# Visualize metrics
figures = monitor.visualize_metrics(save_dir="plots")

# Export metrics
monitor.export_metrics("metrics.json", format="json")

# Graceful shutdown
monitor.shutdown()
```

---

## Installation

To install AfterImage, clone the repository and install the dependencies:

```bash
pip install git+https://github.com/altaidevorg/afterimage.git

# Optional: Install SQL storage dependencies
pip install 'sqlalchemy>=2.0'  # For SQL storage support
```

---

## Getting Started

Here's a step-by-step guide to start using **AfterImage**.

### 1. Setup

Make sure you have a valid API key for Google Gemini) API. Set it as an environment variable:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

**Note**: We currently support only the Gemini API, but we will support other LLM providers soon. Feel free to let me know your choice.

### 2. Quickstart Script (Single API Key)

```python
import os
from afterimage import (
    ConversationGenerator,
    ContextualInstructionGeneratorCallback,
    WithContextRespondentPromptModifier,
)

# Get API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Set the GEMINI_API_KEY environment variable!")

# Define the respondent prompt
respondent_prompt = """Sen, dünya çapında tanınmış başarılı ve deneyimli bir avukatsın.
Görevin, diğer hukukçular ve sıradan vatandaşlar tarafından sorulan sorulara Türk hukuku kapsamında ve Yargıtay kararlarına dayanarak cevap vermek.
Her soruya dikkatle cevap ver ve yanıtlarının gerçek bir avukattan alınıyormuş gibi güvenilir ve detaylı olmasını sağla.
Cevaplarına her zaman mahkeme kararlarına dayandır ve uygun şekilde atıflarda bulun.
Hukuki kavramları doğru şekilde kullanmaya özen göster ama herkesin anlayabileceği bir şekilde sadeleştirerek açıkla.
Onlara asla 'gerçek bir avukata danışmalarını' ya da 'hukuki yardıma başvurmalarını' tavsiye etme çünkü sen, tüm vatandaşların ve hukukçuların güvendiği, saygıdeğer bir hukukçusun.
"""

# Initialize the ConversationGenerator with save path
conv_gen = ConversationGenerator(
    respondent_prompt=respondent_prompt, 
    api_key=api_key,
)

# Print the auto-generated correspondent prompt
print("Generated Correspondent Prompt:")
print(conv_gen.correspondent_prompt)

# Prepare contextual documents
docs = [
    "Hukuki örnek metin 1.",
    "Hukuki metin 2.",
    "Bir mahkeme kararından bir parça.",
    "Hukukla ilgili bir akademik makale.",
]

# Set up the instruction generator callback
instruction_generator_callback = ContextualInstructionGeneratorCallback(
    api_key=api_key,
    documents=docs,
    num_random_contexts=3,  # Experiment with different values
)

# Set up the respondent prompt modifier
respondent_prompt_modifier = WithContextRespondentPromptModifier()

# Generate conversations
conv_gen.generate(
    num_dialogs=100,                # Total dialogs to generate
    max_turns=3,                    # Max turns per conversation
    instruction_generator_callback=instruction_generator_callback,
    respondent_prompt_modifier=respondent_prompt_modifier,
)

print("Conversation dataset generated successfully!")
```

### 3. Advanced Usage with Multiple API Keys

```python
import os
from afterimage import ConversationGenerator, SmartKeyPool

# Initialize a pool of API keys with rate limits
key_pool = SmartKeyPool(
    api_keys=[
        "your-api-key-1",
        "your-api-key-2",
        "your-api-key-3"
    ],
    hourly_limit=1000,  # Optional: limit calls per hour per key
    daily_limit=10000,  # Optional: limit calls per day per key
    error_threshold=5,  # Optional: number of errors before key cooldown
    cooldown_period=300  # Optional: seconds to wait after errors
)

# Initialize generator with the key pool and save path
generator = ConversationGenerator(
    respondent_prompt="You are an expert assistant...",
    api_key=key_pool,
)

# Generate conversations (keys will be automatically rotated)
generator.generate(
    num_dialogs=1000,
    max_turns=3,
)

# Check key usage statistics
stats = key_pool.get_stats()
for key, key_stats in stats.items():
    print(f"Key {key[:8]}...")
    print(f"  Hourly calls: {key_stats['hourly_calls']}")
    print(f"  Daily calls: {key_stats['daily_calls']}")
    print(f"  Active: {key_stats['is_active']}")
    print(f"  Errors: {key_stats['error_count']}")
```

### 4. Using Document Providers

AfterImage supports various document providers for contextual instruction generation:

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
    recursive=True,
)

# Directory of text files
dir_provider = DirectoryDocumentProvider(
    directory="documents",
    file_patterns=["*.txt", "*.md"],
    recursive=True,
)

# Qdrant vector database
qdrant_provider = QdrantDocumentProvider(
    client=qdrant_client,
    collection_name="my_docs",
    content_key="text",
    filter={"must": [{"key": "language", "match": {"value": "en"}}]},
)

# Use in instruction generator
callback = ContextualInstructionGeneratorCallback(
    api_key="your-key",
    documents=qdrant_provider,
    num_random_contexts=3,
)
```

### 5. Retrieval-Augmented Generation (RAG)

AfterImage supports various retrieval strategies for enhancing responses:

```python
from afterimage.retrievers import (
    QdrantRetriever,
    ChainedRetriever,
    EnsembleRetriever,
    CacheRetriever,
)

# Basic Qdrant retriever
retriever = QdrantRetriever(
    client=qdrant_client,
    collection_name="knowledge_base",
    embedding_model="all-MiniLM-L6-v2",
)

# Ensemble of retrievers with weights
ensemble = EnsembleRetriever([
    (retriever1, 0.7),
    (retriever2, 0.3),
])

# Add caching for performance
cached_retriever = CacheRetriever(
    base_retriever=ensemble,
    cache_size=1000,
)

# Use in conversation generator
generator = ConversationGenerator(
    respondent_prompt="You are an expert assistant...",
    api_key="your-key",
    respondent_prompt_modifier=WithRAGRespondentPromptModifier(
        retriever=cached_retriever,
    ),
)
```

### 6. Storage Options

AfterImage supports multiple storage backends for saving generated conversations:

```python
from afterimage.storage import JSONLStorage, SQLStorage

# JSONL storage (default)
jsonl_storage = JSONLStorage(
    path="conversations.jsonl",  # Optional: uses datetime-based filename if not provided
    encoding="utf-8"
)

# SQLite storage
sqlite_storage = SQLStorage(
    url="sqlite:///conversations.db",
    table_name="my_conversations",
    metadata_fields=["language", "domain"]  # Optional: fields to index
)

# PostgreSQL storage
pg_storage = SQLStorage(
    url="postgresql://user:pass@localhost/afterimage",
    metadata_fields=["language", "quality_score"]
)

# Use in generator
generator = ConversationGenerator(
    respondent_prompt="You are an expert...",
    api_key="your-key",
    storage=sqlite_storage  # Specify storage backend
)

# Query conversations with filters
conversations = sqlite_storage.load_conversations(
    filters={
        "metadata.language": "tr",
        "metadata.domain": "legal"
    },
    order_by=[("timestamp", -1)],
    limit=100
)
```

### 7. Dataset Quality Analysis

AfterImage provides comprehensive quality analysis for generated datasets:

```python
from afterimage.quality import QualityChecker
from afterimage.storage import SQLStorage

# Initialize storage and checker
storage = SQLStorage("sqlite:///conversations.db")
checker = QualityChecker(
    storage=storage,
    min_length=50,
    max_length=2000,
    language="tr",  # Optional: check language consistency
    embedding_model="altaidevorg/bge-m3-distill-8l",  # Fast & efficient model
)

# Generate comprehensive report with visualizations
report = checker.generate_report(
    include_plots=True,
    save_dir="quality_report"
)

# Access specific metrics
length_stats = report["length_stats"]
print(f"Mean assistant response length: {length_stats['assistant']['mean']:.0f} chars")

coherence = report["coherence"]
print(f"Mean Q&A coherence: {coherence['mean_coherence']:.2f}")

if report["duplicates"]:
    print(f"Found {len(report['duplicates'])} near-duplicate responses")

# Check context utilization
context_stats = report["context_relevance"]
print(f"Mean context relevance: {context_stats['mean_relevance']:.2f}")
```

### 8. Tips for Effective Usage

1. **Experiment with Prompts**: Tailor respondent and correspondent prompts to your use case.
2. **Use Contextual Documents**: Provide domain-specific documents to enrich conversations.
3. **Parallelize**: Increase `max_workers` in `generate()` for faster dataset creation.

### 9. Generation Monitoring

AfterImage provides real-time monitoring of generation metrics with customizable alerts:

```python
from afterimage.monitoring import GenerationMonitor
from datetime import timedelta

# Custom alert handler
def slack_alert(alert):
    print(f"Alert: {alert.name} - {alert.message}")
    # Send to Slack, email, etc.

# Initialize monitor
monitor = GenerationMonitor(
    log_dir="monitoring_logs",
    alert_handlers=[slack_alert],
    metrics_interval=60  # seconds
)

# Use in generator
generator = ConversationGenerator(
    respondent_prompt="You are an expert...",
    api_key="your-key",
    monitor=monitor
)

# Generate conversations (metrics will be tracked)
generator.generate(num_dialogs=100)

# Get metrics for last 5 minutes
success_rate = monitor.get_metrics("success_rate", window=timedelta(minutes=5))
print(f"Success rate: {success_rate['mean']:.1%}")

# Generate visualizations
figures = monitor.visualize_metrics(
    window=timedelta(hours=1),
    save_dir="monitoring_plots"
)

# Export metrics data
monitor.export_metrics(
    "metrics_export.xlsx",
    format="excel",
    window=timedelta(hours=24)
)
```

#### Monitored Metrics

- **Generation Time**: Time taken for each operation
- **Success Rate**: Successful vs failed generations
- **Error Rate**: Generation errors and types
- **Token Usage**: Token consumption over time
- **Conversation Length**: Number of turns per conversation

#### Alert Conditions

- Low success rate (< 80%)
- High generation time (> 30s average)
- High error rate (> 20%)
- Token usage spikes (> 5000 tokens average)
- Short conversations (< 2 turns average)

#### Export Formats

- JSON: Complete metrics data
- CSV: Separate files for each metric
- Excel: Multiple metrics as sheets
- Parquet: Efficient columnar storage

## Rules
- Whenever you need to write code that interacts with Gemini API, refer to https://googleapis.github.io/python-genai/ for the full API docs.
- Read before changing: Whenever you intend to modify a file, read it first to refresh the cash. The user may have changed it since you last read it.
