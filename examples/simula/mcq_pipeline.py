"""MCQ datapoint with double-critic (requires GEMINI_API_KEY)."""

from __future__ import annotations

import asyncio
import os
import sys

from afterimage.providers import LLMFactory
from afterimage.simula import OpenSimula


async def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY to run this example.", file=sys.stderr)
        sys.exit(1)
    llm = LLMFactory.create(
        provider="gemini",
        model_name="gemini-2.0-flash",
        api_key=api_key,
    )
    sim = OpenSimula(llm, temperature=0.35)
    bundle = await sim.build_taxonomy(
        "Four-option MCQs about basic Python syntax for students.",
        document_provider=None,
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
        complexify_c=0.3,
    )
    row = await sim.generate_mcq_datapoint(
        instruction_y=bundle.instruction_y,
        bundle=bundle,
        mix=mix,
        meta=meta,
        num_choices=4,
    )
    if row is None:
        print("No MCQ accepted.")
    else:
        print(row.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
