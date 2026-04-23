# OpenSimula examples

Runnable scripts (require `GEMINI_API_KEY` by default):

- [`minimal_pipeline.py`](minimal_pipeline.py) — taxonomy → mix → meta-prompt → single QA.
- [`mcq_pipeline.py`](mcq_pipeline.py) — same scaffold → MCQ with double-critic.

Minimal sketch (same flow inline; requires a valid API key for your chosen provider):

```python
import asyncio
import os

from afterimage.providers import InMemoryDocumentProvider, LLMFactory
from afterimage.simula import OpenSimula


async def main():
    api_key = os.environ["GEMINI_API_KEY"]
    llm = LLMFactory.create(
        provider="gemini",
        model_name="gemini-2.0-flash",
        api_key=api_key,
    )
    docs = InMemoryDocumentProvider(["Widget safety: avoid sharp edges."])
    sim = OpenSimula(llm, temperature=0.35)
    bundle = await sim.build_taxonomy(
        "Synthetic QA about widget safety for fine-tuning.",
        document_provider=docs,
        target_depth_D=2,
        proposal_N=2,
    )
    OpenSimula.validate_taxonomy_bundle(bundle)
    spec = await sim.infer_strategies(bundle)
    mix = sim.sample_mix(bundle, spec)
    meta = await sim.draw_meta_prompt(
        instruction_y=bundle.instruction_y,
        bundle=bundle,
        mix=mix,
        K=3,
        complexify_c=0.25,
    )
    row = await sim.generate_single_qa_datapoint(
        instruction_y=bundle.instruction_y,
        bundle=bundle,
        mix=mix,
        meta=meta,
    )
    print(row.model_dump(mode="json") if row else "rejected")


if __name__ == "__main__":
    asyncio.run(main())
```

### Multi-turn (third pattern)

`SimulaInstructionGeneratorCallback` yields one scenario per dialog. Match the list length to `num_dialogs` (or recycle / regenerate between runs):

```python
from afterimage import AsyncConversationGenerator
from afterimage.simula import SimulaInstructionGeneratorCallback

scenarios = [
    ("You are a customer asking about widget shipping times.", {"topic": "shipping"}),
    ("You are a customer asking about return policy.", {"topic": "returns"}),
]
callback = SimulaInstructionGeneratorCallback(scenarios)
# generator = AsyncConversationGenerator(
#     respondent_prompt="You are a helpful widget shop assistant.",
#     api_key=api_key,
#     model_name="gemini-2.0-flash",
#     instruction_generator_callback=callback,
# )
# await generator.generate(num_dialogs=len(scenarios), max_turns=3, max_concurrency=2)
```
