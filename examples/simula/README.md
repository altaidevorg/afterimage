# OpenSimula examples

These scripts exercise **OpenSimula** (`afterimage.simula`): an open implementation of ideas from Davidson et al., *Reasoning-Driven Synthetic Data Generation and Evaluation* (TMLR; [OpenReview PDF](https://openreview.net/pdf?id=NALsdGEPhB)), aligned with the mechanism-design framing in Google’s [research blog](https://research.google/blog/designing-synthetic-datasets-for-the-real-world-mechanism-design-and-reasoning-from-first-principles/). They are **not** a Google reference implementation.

## Model choice

Examples use **`gemini-2.5-flash`** as the default teacher: fast and inexpensive with acceptable quality for taxonomy expansion, strategies, meta-prompts, and critics. The paper’s downstream experiments used **Gemini 2.5 Flash** (non-thinking) as the teacher model; matching that family keeps behavior closer to the reported setup while staying cheap for local runs.

## How each knob maps to the paper

| Parameter / API | Paper role | What it controls | Sensible default |
|-------------------|------------|------------------|------------------|
| `instruction_y` | **y** | Global dataset intent: what domain, tone, and constraints the synthetic data should satisfy (§2.1–2.2). | One tight paragraph: audience, task, and non-goals. |
| `document_provider` (optional) | **S** | Reference material from the target domain so factors and taxonomies are grounded (§2.1: “and/or a sample S”). Use bounded excerpts (the library caps size in `document_context.py`). | Real policy snippets, product docs, or rubrics—not toy one-liners. |
| `target_depth_D` | **D** | Maximum taxonomy depth per factor; deeper trees sharpen **global** coverage control but cost more and risk missing branches (Fig. 1c, Appendix B.4). | **2–3** for prototypes; **4+** only when you need fine leaves and accept cost. |
| `proposal_N` | **N** (Best-of-N) | Independent child proposals per node before the critic merges them (Appendix B.4). Higher **N** widens the proposal distribution toward edge cases. | **2–4**; increase if leaves look repetitive. |
| `OpenSimula(..., temperature=…)` | — | Stochasticity for all structured LLM steps in this facade (taxonomy, strategies, scenarios, critics). Lower = more deterministic trees and stricter judges. | **0.35–0.45** for reproducible runs; raise slightly if diversity stalls. |
| `infer_strategies` → `sample_mix` | §2.2 **sampling strategies** | Strategies define **which factors are jointly sampled** and with what weights, avoiding incompatible mixes. `sample_mix` draws one **mix** (a tuple of taxonomy nodes = “requirements”). | Always run `infer_strategies` once per bundle (or hand-author `SamplingStrategySpec` for full control). |
| `draw_meta_prompt(..., K=…, complexify_c=…, sequential=…)` | **Local diversity** + **c** | **K** distinct **meta-prompts** (scenarios) for the same mix; one is subsampled (Algorithm 2). **complexify_c** is the probability **c** of complexifying that meta-prompt—orthogonal difficulty vs. coverage (§2.2). **sequential=True** uses the paper’s large-**N/V** regime: generate scenarios one-by-one with prior attempts in context to reduce mode collapse. | **K = 4–8**. **c**: the paper’s **Local** ablation uses **c = 0.5** (Table 1); that is aggressive. Examples use **~0.25–0.35** unless you want maximum difficulty skew. |
| `generate_*_datapoint` | Alg. 2 **generate + critic** | Requirement critic and optional refine loop; MCQ path adds **double-critic** (§2.2, §3.1) after the point satisfies the meta-prompt. | Leave `max_refine_rounds` at **4** unless traces show chronic rejection. |

**Evaluation helpers** (not run in these two scripts): `assign_datapoint_to_taxonomy`, `level_ratio_coverage`, and `elo_complexity_scores` implement §2.3 / Appendix E style signals. For Elo-style batch complexity, the paper reports a practical tradeoff around **batch size and repeat count ≈ 5** (Appendix E; “BS = N = 5”); pass `batch_size=5`, `repeats=5` into `elo_complexity_scores` when you wire evaluation.

**Scale note:** Experiments in the paper generated on the order of **512k** points per domain; examples generate **one** row to keep cost predictable.

---

## Runnable scripts

Both require `GEMINI_API_KEY` (or change `LLMFactory.create` to your provider).

- [`minimal_pipeline.py`](minimal_pipeline.py) — **y** + optional **S** → taxonomy → strategies → mix → meta-prompt → **single QA** (requirement critic + refine only).
- [`mcq_pipeline.py`](mcq_pipeline.py) — taxonomy without docs → **four-option MCQ** with **double-critic** after the requirement loop (verifiable label gate, §3.1).

---

## Inline sketch (same API as the scripts)

```python
import asyncio
import os

from afterimage.providers import InMemoryDocumentProvider, LLMFactory
from afterimage.simula import OpenSimula


async def main():
    api_key = os.environ["GEMINI_API_KEY"]
    llm = LLMFactory.create(
        provider="gemini",
        model_name="gemini-2.5-flash",
        api_key=api_key,
    )
    docs = InMemoryDocumentProvider(["""... realistic excerpt ..."""])
    sim = OpenSimula(llm, temperature=0.4)
    bundle = await sim.build_taxonomy(
        "... realistic y ...",
        document_provider=docs,
        target_depth_D=3,
        proposal_N=3,
    )
    OpenSimula.validate_taxonomy_bundle(bundle)
    spec = await sim.infer_strategies(bundle)
    mix = sim.sample_mix(bundle, spec)
    meta = await sim.draw_meta_prompt(
        instruction_y=bundle.instruction_y,
        bundle=bundle,
        mix=mix,
        K=6,
        complexify_c=0.3,
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

---

## Multi-turn (third pattern)

`SimulaInstructionGeneratorCallback` feeds **one first user message per dialog** into `ConversationGenerator` (see `conversation_generator.py`: one `acall` per dialog). Align `num_dialogs` with the number of `(instruction, metadata)` tuples.

Use **realistic** first turns: role, stakes, and what they already tried—so the respondent model gets a believable support or advisory thread.

```python
from afterimage import AsyncConversationGenerator
from afterimage.simula import SimulaInstructionGeneratorCallback

scenarios = [
    (
        "You are a SOC analyst on shift. You just escalated a possible BEC thread; "
        "the mailbox shows mixed legitimate invoices and one odd PDF request. "
        "Ask the security assistant what to verify first and why.",
        {"incident_class": "BEC", "urgency": "P2"},
    ),
    (
        "You are an HR business partner drafting parental leave for an EU employee. "
        "Ask the assistant for the minimum statutory angles you must not get wrong.",
        {"domain": "leave_policy", "region": "EU"},
    ),
]
callback = SimulaInstructionGeneratorCallback(scenarios)
# generator = AsyncConversationGenerator(
#     respondent_prompt="You are a senior security and compliance advisor...",
#     api_key=api_key,
#     model_name="gemini-2.5-flash",
#     instruction_generator_callback=callback,
# )
# await generator.generate(num_dialogs=len(scenarios), max_turns=4, max_concurrency=2)
```
