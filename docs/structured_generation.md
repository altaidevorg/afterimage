# Structured Generation

Sometimes you don't want a conversation. You want to extract specific information from a document or generate synthetic data that fits a strict schema (like a database row). Afterimage supports **Structured Generation** using `StructuredGenerator` and Pydantic models.

## Concept

Structured generation forces the LLM to output valid JSON that matches a schema you define. This is useful for:
*   **Data Extraction**: "Read this email and extract the sender, date, and sentiment."
*   **Synthetic Database Rows**: "Generate 100 fake user profiles with names, ages, and bios."
*   **Golden Sets for RAG**: "Generate a question, the correct answer, and the key facts" for evaluation.

## `StructuredGenerator` class

This generator works differently than the conversation generator. Instead of simulation loops (User <-> Assistant), it simulates a single-turn interaction: `Instruction + Context -> Structured Output`.

### Initialization

The strategy callbacks (for instructions and prompt modification) should be configured at initialization.

```python
from afterimage import StructuredGenerator
from pydantic import BaseModel, Field

# 1. Define your Output Schema
class CustomerFeedback(BaseModel):
    sentiment: str = Field(..., description="Positive, Negative, or Neutral")
    topics: list[str] = Field(..., description="List of topics mentioned (e.g., Pricing, UI)")
    summary: str = Field(..., description="One sentence summary")

# 2. Initialize Generator with Strategies
generator = StructuredGenerator(
    output_schema=CustomerFeedback,
    respondent_prompt="You are an expert data analyst. Extract insights from the feedback.",
    api_key=os.getenv("GEMINI_API_KEY"),
    # Strategies are passed here
    instruction_generator_callback=my_instruction_gen,
    respondent_prompt_modifier=my_prompt_modifier
)
```

**Key Parameters:**

*   `output_schema` (Type[BaseModel]): The Pydantic model defining the expected output structure.
*   `respondent_prompt` (str): System prompt for the generation model.
*   `instruction_generator_callback` (BaseInstructionGeneratorCallback, optional): Strategy to generate the input/instruction for each sample.
*   `respondent_prompt_modifier` (BaseRespondentPromptModifierCallback, optional): Strategy to modify the system prompt per sample.
*   `correspondent_prompt` (str, optional): A static prompt for the "user" side, if not using a callback.
*   `storage` (BaseStorage, optional): Where to save results. Defaults to `JSONLStorage`.

### Generating Data

Use the `generate` method to produce samples.

```python
await generator.generate(
    num_samples=50,
    max_concurrency=4,
)
```

**Parameters:**

*   `num_samples` (int, optional): Total number of samples to generate.
*   `max_concurrency` (int): Maximum concurrent generations.
*   `stopping_criteria` (List[BaseStoppingCallback], optional): Custom logic for stopping generation. If `num_samples` is set, a `FixedNumberStoppingCallback` is automatically added.

## Example: Data Extraction from Documents

Here is how to use `AsyncStructuredGenerator` to process a list of "raw" reviews and extract structured data from them.

```python
import asyncio
import os
from pydantic import BaseModel, Field
from afterimage import (
    StructuredGenerator,
    ContextualInstructionGeneratorCallback,
    InMemoryDocumentProvider
)

# 1. Schema
class ReviewAnalysis(BaseModel):
    product_name: str
    rating: int = Field(..., description="1-5 stars")
    is_spam: bool

# 2. Raw Data (The "Context")
raw_reviews = InMemoryDocumentProvider([
    "I loved the SuperWidget! 5 stars best purchase ever.",
    "Click here for free money! www.spam.com",
    "It broke after one day. Terrible quality. 1 star.",
])

async def main():
    api_key = os.getenv("GEMINI_API_KEY")

    # 3. Setup Instruction Generator
    # This will feed the raw reviews one by one as context
    instruction_gen = ContextualInstructionGeneratorCallback(
        api_key=api_key,
        documents=raw_reviews,
        num_random_contexts=1,
        # Just ask to analyze the context
        prompt="Analyze the review provided in the context." 
    )

    # 4. Initialize Generator
    generator = StructuredGenerator(
        output_schema=ReviewAnalysis,
        respondent_prompt="Analyze the provided review.",
        api_key=api_key,
        instruction_generator_callback=instruction_gen
    )

    # 5. Run Extraction
    print("Extracting data...")
    await generator.generate(num_samples=3)
    print("Done. Data saved to JSONL.")

if __name__ == "__main__":
    asyncio.run(main())
```

The output will be saved to a `.jsonl` file where each line is a valid JSON object matching your `ReviewAnalysis` schema.
