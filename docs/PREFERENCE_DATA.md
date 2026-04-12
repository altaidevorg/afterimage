# Generating DPO/RLHF Preference Data

AfterImage can generate preference pairs (chosen/rejected responses) directly from your documents — no manual labeling needed.

## Quick start

```bash
afterimage preference -c config.yaml
```

Add a `preference` section to your existing AfterImage config:

```yaml
# config.yaml
model:
  provider: openai
  model_name: gpt-4o-mini

respondent:
  system_prompt: "You are a helpful assistant."

preference:
  num_pairs: 100
  num_responses: 2
  strategy: temperature
  min_score_gap: 0.1
  output_format: dpo
  output_path: ./preferences.jsonl
```

## How it works

1. AfterImage generates a user prompt (with persona + document context, same as regular generation)
2. For the same prompt, it generates multiple responses with **controlled variation**
3. Each response is scored by the built-in quality judge
4. Highest and lowest scored responses become the (chosen, rejected) pair
5. Pairs with score gap below the threshold are discarded

## Variation strategies

### Temperature (default)

Generates responses at linearly-spaced temperatures. Lower temperature → more focused → usually higher quality.

```yaml
preference:
  strategy: temperature
  num_responses: 2   # 2 responses: low temp (0.1) and high temp (0.9)
```

Best for: quick setup, works with any model including local (vLLM, Ollama).

### Prompt variation

Modifies the system prompt: **enhanced** ("think step by step...") vs **degraded** ("answer briefly...").

```yaml
preference:
  strategy: prompt
  num_responses: 2
```

Best for: teaching instruction-following quality differences.

### Model variation

Uses different models for each response — one primary, one secondary.

```yaml
preference:
  strategy: model
  secondary_model: gpt-4o     # stronger model for chosen responses
  num_responses: 2
```

Best for: largest quality spread, most realistic preference data.

### Combined

Mixes temperature + prompt + model strategies for maximum diversity.

```yaml
preference:
  strategy: combined
  secondary_model: gpt-4o   # optional
  num_responses: 3
```

Best for: production datasets, reward model training.

## Output formats

| Format | Description | Used by |
|--------|-------------|---------|
| `dpo` | Standard prompt/chosen/rejected | TRL DPOTrainer |
| `chat_dpo` | Message lists with chat template | TRL with chat template |
| `ultrafeedback` | All responses with scores | UltraFeedback-style training |
| `anthropic_hh` | Human:/Assistant: format | Anthropic HH training |
| `orpo` | DPO schema + scores | ORPO training |

```bash
afterimage preference -c config.yaml --format chat_dpo
```

## Multi-turn preferences

Generate a shared conversation history and branch at the final turn:

```yaml
preference:
  multi_turn: true
  num_pairs: 50
```

This produces pairs where `shared_prefix` contains identical conversation history and `chosen`/`rejected` differ only in the final assistant response.

## Full generation log

Save all scored responses (not just chosen/rejected) for analysis:

```bash
afterimage preference -c config.yaml --save-log
```

Or in config:
```yaml
preference:
  save_log: true
  log_path: ./preferences_full_log.jsonl  # optional, defaults to output path + _log
```

## Using with training tools

### TRL DPOTrainer

```python
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

ds = load_dataset("json", data_files="preferences_dpo.jsonl", split="train")
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    train_dataset=ds,
    args=DPOConfig(output_dir="./dpo_output"),
)
trainer.train()
```

### TRL with chat template (chat_dpo format)

```python
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

ds = load_dataset("json", data_files="preferences_chat_dpo.jsonl", split="train")
# chat_dpo format already has message lists — works directly with DPOTrainer
trainer = DPOTrainer(model=model, train_dataset=ds, args=DPOConfig(...))
```

### Unsloth + DPO

Export as `chat_dpo` format and use Unsloth's DPO notebook directly.

## Python API

Set `OPENAI_API_KEY` in the environment (or switch the provider block to Gemini and `GEMINI_API_KEY` consistently).

```python
import asyncio
import os

from afterimage import ConversationGenerator
from afterimage.callbacks import ContextualInstructionGeneratorCallback
from afterimage.evaluator import ConversationJudge
from afterimage.key_management import SmartKeyPool
from afterimage.preference import PreferenceConfig
from afterimage.providers import InMemoryDocumentProvider, LLMFactory


async def main():
    pool = SmartKeyPool.from_single_key(os.environ["OPENAI_API_KEY"])
    llm = LLMFactory.create(provider="openai", model_name="gpt-4o-mini", api_key=pool)
    judge = ConversationJudge.from_factory(
        llm,
        key_pool=pool,
        model_provider_name="openai",
    )

    docs = InMemoryDocumentProvider(["Short context document text."])
    instruction_callback = ContextualInstructionGeneratorCallback(
        api_key=pool,
        documents=docs,
        model_provider_name="openai",
        model_name="gpt-4o-mini",
    )

    gen = ConversationGenerator(
        respondent_prompt="You are a helpful assistant.",
        api_key=pool,
        model_provider_name="openai",
        instruction_generator_callback=instruction_callback,
    )

    pref_gen = gen.to_preference_generator(
        judge=judge,
        config=PreferenceConfig(
            num_pairs=50,
            strategy="temperature",
            output_format="dpo",
            output_path="./preferences.jsonl",
        ),
    )

    pairs, analytics = await pref_gen.generate()
    pref_gen.save_pairs(pairs, analytics)
    print(f"Generated {len(pairs)} pairs, discard rate: {analytics.discard_rate:.0%}")
    await judge.aclose()


asyncio.run(main())
```

## CLI reference

```
afterimage preference [OPTIONS]

Options:
  -c, --config PATH       Path to YAML config file.  [required]
  --dry-run               Print plan without generating.
  --num-pairs INTEGER     Override preference.num_pairs from config.
  --format [dpo|chat_dpo|ultrafeedback|anthropic_hh|orpo]
                          Override output format.
  -o, --output PATH       Override output file path.
  --save-log              Save full generation log with all scored responses.
  --help                  Show this message and exit.
```

## Tips

- **Start small**: use `num_responses=2` to minimize API cost
- **Best dataset**: use `strategy=combined` for production datasets
- **High discard rate** (>40%): lower `min_score_gap` or switch to a stronger strategy
- **One variation always wins**: switch strategy — temperature may not create enough variation for your model
- **Local models**: temperature strategy works with vLLM, Ollama, and llama.cpp — temperature is passed through to the provider
- **Analytics**: check the stats printed after generation; warnings highlight degenerate patterns
