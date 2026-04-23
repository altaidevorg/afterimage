"""
Single-QA OpenSimula pipeline (requirement critic + refine; no double-critic).

Paper mapping (Davidson et al., TMLR; Appendix B.4, §2.2, Algorithm 2):
  - instruction_y  →  **y**: dataset specification.
  - document text  →  optional **S**: domain grounding (§2.1); capped when passed via DocumentProvider.
  - target_depth_D →  **D** (taxonomy depth); deeper = finer global coverage, higher cost.
  - proposal_N     →  **N** in Best-of-N child proposals before the critic merges them.
  - infer_strategies / sample_mix → joint sampling strategies and one **mix** (§2.2).
  - K              →  number of **meta-prompt** candidates; one is randomly kept (local diversity).
  - complexify_c   →  probability **c** of complexifying that meta-prompt (§2.2). Table 1 uses **c=0.5**
    for the paper’s “Local” system; we use a lower default here to limit difficulty skew while iterating.

Model: **gemini-2.5-flash** — same family as the paper’s teacher (Gemini 2.5 Flash), cheap for loops.

Requires: GEMINI_API_KEY

Checkpointing: pass ``--checkpoint DIR`` to write ``opensimula/`` via ``Checkpointer``
(``bundle.save(cp)``, ``spec.save(cp)``, ``cp.write_run_config(...)``). ``--resume DIR`` skips taxonomy
and ``infer_strategies``. ``--push-hf REPO_ID`` uploads that tree (needs ``HF_TOKEN``). See ``README.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from tqdm.auto import tqdm

from afterimage.providers import InMemoryDocumentProvider, LLMFactory
from afterimage.simula import (
    Checkpointer,
    OpenSimula,
    configure_example_console,
    load_checkpoint,
)

configure_example_console()

# ---------------------------------------------------------------------------
# Scenario: grounded Q&A for security awareness training (not legal advice).
# Realistic y + multi-paragraph S mimic internal enablement material.
# ---------------------------------------------------------------------------
INSTRUCTION_Y = """\
You are generating synthetic **training Q&A** for enterprise employees (security
and acceptable-use awareness). Each item must be grounded in the provided policy
excerpts: answers should cite concrete controls or procedures implied by the text,
not invent vendor-specific products or laws not mentioned. Target length: question
≤120 words, answer ≤180 words, factual tone, no panic language.\
"""

POLICY_EXCERPTS = [
    """\
**Corporate Acceptable Use (excerpt).** Company systems may be monitored to ensure
compliance. Users must not disable endpoint protection, must report suspected phishing
within one hour via the security mailbox, and must not store customer personal data
on unapproved cloud drives. Remote access requires MFA on every session. Contractors
receive least-privilege accounts revoked within 24 hours of offboarding.\
""",
    """\
**Data classification (excerpt).** "Restricted" data includes credentials, live
customer PII, and unreleased financials. Restricted data may only transit over
approved encrypted channels. Incident severity P1/P2 requires paging the on-call SOC;
P3/P4 is next-business-day. Tabletop exercises for ransomware are mandatory annually
for all people managers.\
""",
]

# D=2: two expansion waves keeps first-run latency reasonable (depth 3 is fine if you
# raise cost caps and accept several minutes of API calls).
TARGET_DEPTH_D = 2
# N=3: moderate Best-of-N for child proposals (Appendix B.4).
PROPOSAL_N = 3
# LLM temperature for all OpenSimula steps in this script (not the paper's hyperparam).
OPEN_SIMULA_TEMPERATURE = 0.4
# K meta-prompts per mix before subsampling (§2.2 local diversity).
META_PROMPT_K = 6
# c: complexification probability. Paper Table 1 uses 0.5 for "Local"; 0.28 is gentler for demos.
COMPLEXIFY_C = 0.28

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
        help="Write opensimula/ (manifest + taxonomy + strategy) here after infer_strategies.",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="DIR",
        help="Load opensimula/ from DIR; skip build_taxonomy and infer_strategies.",
    )
    p.add_argument(
        "--push-hf",
        default=None,
        metavar="REPO_ID",
        help="After --checkpoint save, upload opensimula/ to this Hub dataset repo id.",
    )
    return p.parse_args()


async def main() -> None:
    args = _parse_args()
    if bool(args.checkpoint) and bool(args.resume):
        print("Use only one of --checkpoint and --resume.", file=sys.stderr)
        sys.exit(2)
    if args.push_hf and not args.checkpoint:
        print("--push-hf requires --checkpoint (nothing is saved on resume-only runs).", file=sys.stderr)
        sys.exit(2)

    print("Starting…", flush=True)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY to run this example.", file=sys.stderr)
        sys.exit(1)

    llm = LLMFactory.create(
        provider="gemini",
        model_name=MODEL_NAME,
        api_key=api_key,
    )
    docs = InMemoryDocumentProvider(POLICY_EXCERPTS)
    print("Document provider ready.", flush=True)

    sim = OpenSimula(llm, temperature=OPEN_SIMULA_TEMPERATURE)
    print("OpenSimula ready — taxonomy uses tqdm; httpx/google_genai logs are muted.\n", flush=True)

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
        bundle = await sim.build_taxonomy(
            INSTRUCTION_Y,
            document_provider=docs,
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
        tail.set_postfix_str("infer strategies (§2.2)")
        spec = await sim.infer_strategies(bundle)
        tail.update(1)
        if args.checkpoint:
            run_cfg = {
                "example": "minimal_pipeline",
                "model": MODEL_NAME,
                "temperature": OPEN_SIMULA_TEMPERATURE,
                "target_depth_D": TARGET_DEPTH_D,
                "proposal_N": PROPOSAL_N,
                "meta_prompt_K": META_PROMPT_K,
                "complexify_c": COMPLEXIFY_C,
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
                f"(format {man.format} {man.format_version}) → {args.checkpoint / 'opensimula'}\n",
                flush=True,
            )
            if args.push_hf:
                url = cp.push_to_hub(args.push_hf)
                print(f"Pushed to Hub: {url}\n", flush=True)
        tail.set_postfix_str("sample mix")
    mix = sim.sample_mix(bundle, spec)
    tail.update(1)
    tail.set_postfix_str(f"meta-prompts (K={META_PROMPT_K})")
    meta = await sim.draw_meta_prompt(
        instruction_y=bundle.instruction_y,
        bundle=bundle,
        mix=mix,
        K=META_PROMPT_K,
        complexify_c=COMPLEXIFY_C,
        sequential=False,
    )
    tail.update(1)
    tail.set_postfix_str("single QA + critic")
    row = await sim.generate_single_qa_datapoint(
        instruction_y=bundle.instruction_y,
        bundle=bundle,
        mix=mix,
        meta=meta,
    )
    tail.update(1)
    tail.close()

    if row is None:
        print("No row accepted (requirement critic or refine loop).", flush=True)
    else:
        print(row.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
