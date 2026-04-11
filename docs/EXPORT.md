# Export & Integration Guide

AfterImage can export generated datasets to all major fine-tuning formats with a single command.

## Quick Start

```bash
# Export to ShareGPT format
afterimage export -i dataset.jsonl -f sharegpt

# Export to multiple formats at once
afterimage export -i dataset.jsonl -f sharegpt -f alpaca -f messages

# Export to all formats with train/val split
afterimage export -i dataset.jsonl --all --split 0.1

# List available formats
afterimage export --list-formats
```

## Available Formats

| Format         | Key                | Multi-turn | System Prompt | Used by                         |
|----------------|--------------------|:----------:|:-------------:|---------------------------------|
| ShareGPT       | `sharegpt`         | yes        | yes           | Unsloth, Axolotl, LLaMA-Factory|
| Alpaca         | `alpaca`           | no*        | no            | Stanford Alpaca, basic SFT      |
| Messages       | `messages`         | yes        | yes           | TRL SFTTrainer, HuggingFace     |
| Oumi           | `oumi`             | yes        | yes           | Oumi                            |
| LLaMA-Factory  | `llama_factory`    | yes        | yes           | LLaMA-Factory                   |
| OpenAI         | `openai`           | yes        | yes           | OpenAI fine-tuning API          |
| DPO            | `dpo`              | no         | yes           | TRL DPOTrainer, RLHF            |
| Raw            | `raw`              | yes        | yes           | AfterImage, custom pipelines    |

\* Alpaca supports multi-turn via `split_turns` mode (programmatic API).

## Output Schemas

### ShareGPT

```json
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
```

Roles are mapped: `user` → `human`, `assistant` → `gpt`. Consecutive same-role messages are merged with a newline separator.

### Alpaca

```json
{"instruction": "user question", "input": "optional context", "output": "assistant response"}
```

Uses first user/assistant pair. The `input` field is populated from `instruction_context` if present in the source row.

### Messages / Oumi

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Standard HuggingFace chat format. Oumi uses the same schema. Rows without an assistant message are skipped.

### LLaMA-Factory

```json
{
  "instruction": "last user message",
  "input": "context",
  "output": "last assistant response",
  "history": [["user msg 1", "assistant msg 1"], ["user msg 2", "assistant msg 2"]],
  "system": "optional system prompt"
}
```

All turns except the last pair go into `history` as `[user, assistant]` arrays.

### OpenAI Fine-Tuning

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Same as Messages format. Requires at least one assistant message (OpenAI API requirement).

### DPO (Preference Pairs)

```json
{"prompt": "user instruction", "chosen": "best response", "rejected": "worst response"}
```

Requires `final_score` in source data (enable `quality.auto_improve: true`). Groups conversations by first user message and pairs highest/lowest scored responses. Minimum score gap: 0.2 (configurable via API).

### Raw

```json
{"conversations": [...], "metadata": {...}, ...}
```

Passthrough — copies the entire source row as-is.

## CLI Reference

### `afterimage export`

```
Options:
  -i, --input PATH        Path to AfterImage JSONL dataset (required)
  -f, --format TEXT       Target format(s). Repeat for multiple
  --all                   Export to all available formats
  -o, --output-dir PATH   Output directory (default: same as input)
  --split FLOAT           Train/val split ratio (e.g. 0.1 = 10% validation)
  --shuffle/--no-shuffle  Shuffle before splitting (default: shuffle)
  --seed INT              Random seed for reproducible splits (default: 42)
  --system-prompt TEXT    System prompt to prepend
  --list-formats          Show all available formats
```

Output files are named `{input_stem}_{format}.jsonl`, or `{input_stem}_{format}_train.jsonl` / `{input_stem}_{format}_val.jsonl` when using `--split`.

### `afterimage push`

```
Options:
  -i, --input PATH   Path to AfterImage JSONL dataset (required)
  -f, --format TEXT   Export format before pushing (default: messages)
  --repo TEXT         HuggingFace repo: username/dataset-name (required)
  --private           Create as private dataset
  --split FLOAT       Train/val split ratio (default: 0.1)
```

`huggingface_hub` is a core dependency; no extra install is required for `push`.

## Auto-Export After Generation

Add an `export` section to your YAML config to automatically export after generation:

```yaml
output:
  path: output/dataset.jsonl
  export:
    formats:
      - sharegpt
      - messages
    output_dir: output/exports   # optional, defaults to same directory
    split: 0.1                   # optional train/val split
    shuffle: true                # default
    seed: 42                     # default
```

Auto-export is fail-safe — if export fails, your generated dataset is unaffected.

## Training Tool Integration

### Unsloth

```python
from datasets import load_dataset
dataset = load_dataset("json", data_files="dataset_sharegpt.jsonl")

from unsloth import FastLanguageModel
# ... load model ...
from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    dataset_text_field="conversations",
)
```

### Axolotl

```yaml
# axolotl config
datasets:
  - path: dataset_sharegpt.jsonl
    type: sharegpt
```

### TRL SFTTrainer

```python
from datasets import load_dataset
dataset = load_dataset("json", data_files={
    "train": "dataset_messages_train.jsonl",
    "validation": "dataset_messages_val.jsonl",
})

from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)
```

### TRL DPOTrainer

```python
from datasets import load_dataset
dataset = load_dataset("json", data_files="dataset_dpo.jsonl")

from trl import DPOTrainer
trainer = DPOTrainer(
    model=model,
    train_dataset=dataset["train"],
)
```

### LLaMA-Factory

```json
{
  "my_dataset": {
    "file_name": "dataset_llama_factory.jsonl",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "history": "history",
      "system": "system"
    }
  }
}
```

### Oumi

```bash
oumi train -c my_config.yaml \
  --training_data.datasets[0].dataset_name=dataset_oumi.jsonl
```

### OpenAI Fine-Tuning API

```bash
# Upload and create fine-tuning job
openai api fine_tuning.jobs.create \
  -t dataset_openai.jsonl \
  -m gpt-4o-mini-2024-07-18
```

### HuggingFace Hub

```bash
# Push directly from AfterImage
afterimage push -i dataset.jsonl --repo username/my-dataset --private

# Then use anywhere
from datasets import load_dataset
ds = load_dataset("username/my-dataset")
```

## Programmatic API

```python
from afterimage.integrations import get_exporter, list_formats

# List all formats
for fmt in list_formats():
    print(f"{fmt['name']}: {fmt['description']}")

# Export a single format
exporter = get_exporter("sharegpt")
result = exporter.export_file("input.jsonl", "output.jsonl")
print(f"Exported {result.total_output} rows, skipped {result.skipped}")

# Export with system prompt
result = exporter.export_file(
    "input.jsonl",
    "output.jsonl",
    system_prompt="You are a helpful assistant.",
)

# Convert a single conversation
row = {"conversations": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"},
]}
output_rows = exporter.convert_conversation(row)
```

## Design Notes

- **Streaming**: Exports process line-by-line. Memory usage is constant regardless of dataset size.
- **Graceful degradation**: Unconvertible rows are skipped with warnings, never crashing mid-export.
- **Deterministic**: Same input + seed always produces the same output.
- **No external dependencies**: All core exporters use only the Python standard library. `push` uses `huggingface_hub`, which is installed with the package.
