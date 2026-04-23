"""Runnable minimal OpenSimula pipeline (requires GEMINI_API_KEY)."""

from __future__ import annotations

import asyncio
import os
import sys

from afterimage.providers import InMemoryDocumentProvider, LLMFactory
from afterimage.simula import OpenSimula


async def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY to run this example.", file=sys.stderr)
        sys.exit(1)
    llm = LLMFactory.create(
        provider="gemini",
        model_name="gemini-2.5-flash",
        api_key=api_key,
    )
    docs = InMemoryDocumentProvider(["Widget safety: avoid sharp edges and small parts."])
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
    if row is None:
        print("No row accepted.")
    else:
        print(row.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
