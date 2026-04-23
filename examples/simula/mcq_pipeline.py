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

Checkpointing: ``--checkpoint DIR``, ``--resume DIR``, ``--push-hf REPO_ID`` (same as
``minimal_pipeline.py``; see ``README.md``).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from tqdm.auto import tqdm

from afterimage.providers import LLMFactory
from afterimage.simula import (
    Checkpointer,
    OpenSimula,
    configure_example_console,
    load_checkpoint,
)

configure_example_console()

INSTRUCTION_Y = """\
Generate synthetic **four-option multiple-choice questions** for an internal
assessment on **software supply chain and incident readiness** (SBOM basics, severity
triaging, secure defaults). Each question must test reading-comprehension of concepts
that appear in typical engineering handbooks—not trivia about version numbers.
Distractors should be plausible misconceptions. One correct option only.\
"""

TARGET_DEPTH_D = 2
PROPOSAL_N = 3
OPEN_SIMULA_TEMPERATURE = 0.4
META_PROMPT_K = 6
# Paper uses c=0.5 for Local; 0.32 here balances harder scenarios vs. rejections on MCQ.
COMPLEXIFY_C = 0.32
NUM_CHOICES = 4

MODEL_NAME = "gemini-2.5-flash"
MAX_FACTORS = 4
MAX_CHILDREN_PER_NODE = 8
MAX_FRONTIER_PER_DEPTH = 12


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        metavar="DIR",
        help="Write opensimula/ after infer_strategies.",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="DIR",
        help="Load opensimula/; skip build_taxonomy and infer_strategies.",
    )
    p.add_argument(
        "--push-hf",
        default=None,
        metavar="REPO_ID",
        help="After --checkpoint save, upload opensimula/ to this Hub dataset repo.",
    )
    return p.parse_args()


async def main() -> None:
    args = _parse_args()
    if bool(args.checkpoint) and bool(args.resume):
        print("Use only one of --checkpoint and --resume.", file=sys.stderr)
        sys.exit(2)
    if args.push_hf and not args.checkpoint:
        print("--push-hf requires --checkpoint.", file=sys.stderr)
        sys.exit(2)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY to run this example.", file=sys.stderr)
        sys.exit(1)

    llm = LLMFactory.create(
        provider="gemini",
        model_name=MODEL_NAME,
        api_key=api_key,
    )
    sim = OpenSimula(llm, temperature=OPEN_SIMULA_TEMPERATURE)

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume.resolve()}\n", flush=True)
        ckpt = load_checkpoint(args.resume)
        bundle = ckpt.bundle
        spec = ckpt.sampling_strategy
        if spec is None:
            print("Checkpoint has no sampling_strategy.json; inferring strategies…", flush=True)
            spec = await sim.infer_strategies(bundle)
        tail = tqdm(
            total=3,
            desc="OpenSimula │ after taxonomy",
            unit="step",
            dynamic_ncols=True,
        )
        tail.set_postfix_str("sample mix")
    else:
        # No document provider: pure y-driven taxonomies (still valid in §2.1: y and/or S).
        print("Building taxonomy (tqdm; httpx/google_genai muted)…\n", flush=True)
        bundle = await sim.build_taxonomy(
            INSTRUCTION_Y,
            document_provider=None,
            target_depth_D=TARGET_DEPTH_D,
            proposal_N=PROPOSAL_N,
            max_factors=MAX_FACTORS,
            max_children_per_node=MAX_CHILDREN_PER_NODE,
            max_frontier_per_depth=MAX_FRONTIER_PER_DEPTH,
            show_progress=True,
        )
        print()
        OpenSimula.validate_taxonomy_bundle(bundle)

        tail = tqdm(
            total=4,
            desc="OpenSimula │ after taxonomy",
            unit="step",
            dynamic_ncols=True,
        )
        tail.set_postfix_str("infer strategies")
        spec = await sim.infer_strategies(bundle)
        tail.update(1)
        if args.checkpoint:
            run_cfg = {
                "example": "mcq_pipeline",
                "model": MODEL_NAME,
                "temperature": OPEN_SIMULA_TEMPERATURE,
                "target_depth_D": TARGET_DEPTH_D,
                "proposal_N": PROPOSAL_N,
                "meta_prompt_K": META_PROMPT_K,
                "complexify_c": COMPLEXIFY_C,
                "num_choices": NUM_CHOICES,
                "max_factors": MAX_FACTORS,
                "max_children_per_node": MAX_CHILDREN_PER_NODE,
                "max_frontier_per_depth": MAX_FRONTIER_PER_DEPTH,
            }
            with Checkpointer(args.checkpoint) as cp:
                bundle.save(cp)
                spec.save(cp)
                cp.write_run_config(run_cfg)
            man = cp.manifest
            assert man is not None
            print(
                f"Wrote OpenSimula checkpoint "
                f"({man.format} {man.format_version}) → {args.checkpoint / 'opensimula'}\n",
                flush=True,
            )
            if args.push_hf:
                url = cp.push_to_hub(args.push_hf)
                print(f"Pushed to Hub: {url}\n", flush=True)
        tail.set_postfix_str("sample mix")
    mix = sim.sample_mix(bundle, spec)
    tail.update(1)
    tail.set_postfix_str("meta-prompts")
    meta = await sim.draw_meta_prompt(
        instruction_y=bundle.instruction_y,
        bundle=bundle,
        mix=mix,
        K=META_PROMPT_K,
        complexify_c=COMPLEXIFY_C,
        sequential=False,
    )
    tail.update(1)
    tail.set_postfix_str("MCQ + critics")
    row = await sim.generate_mcq_datapoint(
        instruction_y=bundle.instruction_y,
        bundle=bundle,
        mix=mix,
        meta=meta,
        num_choices=NUM_CHOICES,
    )
    tail.update(1)
    tail.close()
    if row is None:
        print("No MCQ accepted (requirement loop and/or double-critic).")
    else:
        print(row.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
