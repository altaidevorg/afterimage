# AfterImage:  A framework for synthetic dataset generation based on your docs

**AfterImage** is a Python package for generating synthetic conversational datasets using state-of-the-art generative AI models. It is highly customizable, enabling tailored instruction generation and fine-tuned conversational prompts.

## Features

- **Customizable Prompts**: Create bespoke respondent and correspondent prompts for generating realistic conversations.
- **Contextual Instruction Generation**: Leverage contextual documents to craft unique and relevant conversation starters.
- **Dynamic Conversation Flow**: Simulate back-and-forth dialogs with adjustable turns and behaviors.
- **Smart API Key Management**: Handle multiple API keys with automatic rotation, rate limiting, and error handling.
- **Parallel Execution**: Generate multiple conversations efficiently using multithreading.
- **Save in JSONL Format**: Export datasets directly for downstream applications.

**Note**: This is the initial version, but I will add more generators and evaluators soon.

---

## Installation

To install AfterImage, clone the repository and install the dependencies:

```bash
pip install git+https://github.com/altaidevorg/afterimage.git
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
    save_to="awesome_dataset.jsonl"  # Specify save path in constructor
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
    docs=docs,
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
    save_to="dataset.jsonl"  # Specify save path in constructor
)

# Generate conversations (keys will be automatically rotated)
generator.generate(
    num_dialogs=1000,
    max_turns=3
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

---

## Key Components

### 1. `ConversationGenerator`

The central class for managing dialog generation. Customize prompts, configure parameters, and manage output.

#### Initialization Parameters

- **`respondent_prompt`**: The primary prompt for the respondent model.
- **`api_key`**: Either a single API key string or a SmartKeyPool instance.
- **`correspondent_prompt`** (optional): Automatically generated if not provided.
- **`model_name`** (optional): Specify the AI model to use.
- **`save_to`** (optional): Path to save the generated dialogs in JSONL format.

#### Methods

- **`generate()`**: Main method for generating conversations.

### 2. `ContextualInstructionGeneratorCallback`

A callback for generating instructions based on contextual documents.

#### Initialization Parameters

- **`api_key`**: API key for the generative model.
- **`docs`**: List of documents to use for context.
- **`num_random_contexts`** (optional): Number of random contexts to include.

### 3. `WithContextRespondentPromptModifier`

A callback for modifying respondent prompts dynamically based on instructions and contexts.

### 4. `SmartKeyPool`

Manages multiple API keys with intelligent rotation and error handling.

#### Initialization Parameters

- **`api_keys`**: List of API keys to manage
- **`hourly_limit`** (optional): Maximum calls per hour per key
- **`daily_limit`** (optional): Maximum calls per day per key
- **`error_threshold`** (optional): Number of errors before key cooldown
- **`cooldown_period`** (optional): Seconds to wait after errors

#### Features

- Automatic key rotation based on usage and availability
- Optional rate limiting (hourly and daily)
- Error tracking and automatic key cooldown
- Usage statistics and monitoring
- Thread-safe operations for concurrent access

#### Methods

- **`get_next_key()`**: Get the next available API key
- **`report_error(key)`**: Report an error for a key
- **`get_stats()`**: Get usage statistics for all keys

---

## Saving the Dataset

The generated dataset is saved in **JSONL** format, making it easy to parse and use in various ML pipelines.

Each conversation entry includes:

- **Role**: Either `"user"` or `"assistant"`.
- **Content**: The corresponding dialog message.

Example output:

```json
{"conversations": [{"role": "user", "content": "What is the process for divorce?"}, {"role": "assistant", "content": "Under Turkish law, divorce involves..." }], "context": "The contextual document from which the conversation is synthesized"}
```

---

## Tips for Effective Usage

1. **Experiment with Prompts**: Tailor respondent and correspondent prompts to your use case.
2. **Use Contextual Documents**: Provide domain-specific documents to enrich conversations.
3. **Parallelize**: Increase `max_workers` in `generate()` for faster dataset creation.

---

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve the package.

---

## License

This package is licensed under the MIT License. See `LICENSE` for details.
