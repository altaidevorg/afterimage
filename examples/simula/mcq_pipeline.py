"""
MCQ datapoint with requirement critic + **double-critic** (verifiable label gate).

Paper mapping:
  - Taxonomy / mix / meta-prompt: same global → local pipeline as §2.2 and Algorithm 2.
  - Double-critic: two independent structured probes (“correct” vs “incorrect”) to
    reduce sycophancy on labeled answers (§2.2, §3.1, Fig. 3). Runs **after** the
    requirement critic loop accepts the JSON (OpenSimula implementation choice).

This example targets **four-option** items in the style of technical reading comprehension
(similar *format* to multiple-choice benchmarks discussed in the paper, e.g. CTI-MCQ /
Global MMLU), without reproducing any benchmark text.

Model: **gemini-2.5-flash** (paper-aligned cheap teacher; see blog / §3.2).

Requires: GEMINI_API_KEY
"""

from __future__ import annotations

import asyncio
import os
import sys

from afterimage.providers import LLMFactory
from afterimage.simula import OpenSimula

INSTRUCTION_Y = """\
Generate synthetic **four-option multiple-choice questions** for an internal
assessment on **software supply chain and incident readiness** (SBOM basics, severity
triaging, secure defaults). Each question must test reading-comprehension of concepts
that appear in typical engineering handbooks—not trivia about version numbers.
Distractors should be plausible misconceptions. One correct option only.\
"""

TARGET_DEPTH_D = 3
PROPOSAL_N = 3
OPEN_SIMULA_TEMPERATURE = 0.4
META_PROMPT_K = 6
# Paper uses c=0.5 for Local; 0.32 here balances harder scenarios vs. rejections on MCQ.
COMPLEXIFY_C = 0.32
NUM_CHOICES = 4


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
    sim = OpenSimula(llm, temperature=OPEN_SIMULA_TEMPERATURE)

    # No document provider: pure y-driven taxonomies (still valid in §2.1: y and/or S).
    bundle = await sim.build_taxonomy(
        INSTRUCTION_Y,
        document_provider=None,
        target_depth_D=TARGET_DEPTH_D,
        proposal_N=PROPOSAL_N,
    )
    OpenSimula.validate_taxonomy_bundle(bundle)

    spec = await sim.infer_strategies(bundle)
    mix = sim.sample_mix(bundle, spec)
    meta = await sim.draw_meta_prompt(
        instruction_y=bundle.instruction_y,
        bundle=bundle,
        mix=mix,
        K=META_PROMPT_K,
        complexify_c=COMPLEXIFY_C,
        sequential=False,
    )

    row = await sim.generate_mcq_datapoint(
        instruction_y=bundle.instruction_y,
        bundle=bundle,
        mix=mix,
        meta=meta,
        num_choices=NUM_CHOICES,
    )
    if row is None:
        print("No MCQ accepted (requirement loop and/or double-critic).")
    else:
        print(row.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
