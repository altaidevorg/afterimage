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
"""

from __future__ import annotations

import asyncio
import os
import sys

from afterimage.providers import InMemoryDocumentProvider, LLMFactory
from afterimage.simula import OpenSimula

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

# D=3: enough depth for "topic → subtopic → leaf" without the cost of very deep trees.
TARGET_DEPTH_D = 3
# N=3: moderate Best-of-N for child proposals (Appendix B.4).
PROPOSAL_N = 3
# LLM temperature for all OpenSimula steps in this script (not the paper's hyperparam).
OPEN_SIMULA_TEMPERATURE = 0.4
# K meta-prompts per mix before subsampling (§2.2 local diversity).
META_PROMPT_K = 6
# c: complexification probability. Paper Table 1 uses 0.5 for "Local"; 0.28 is gentler for demos.
COMPLEXIFY_C = 0.28


async def main() -> None:
    print("Starting...")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY to run this example.", file=sys.stderr)
        sys.exit(1)

    llm = LLMFactory.create(
        provider="gemini",
        model_name="gemini-2.5-flash",
        api_key=api_key,
    )
    docs = InMemoryDocumentProvider(POLICY_EXCERPTS)
    print("Document provider created")

    sim = OpenSimula(llm, temperature=OPEN_SIMULA_TEMPERATURE)
    print("Simula created")

    bundle = await sim.build_taxonomy(
        INSTRUCTION_Y,
        document_provider=docs,
        target_depth_D=TARGET_DEPTH_D,
        proposal_N=PROPOSAL_N,
    )
    print("Bundle created")

    OpenSimula.validate_taxonomy_bundle(bundle)
    print("Bundle validated")

    spec = await sim.infer_strategies(bundle)
    print("Strategies infered")
    mix = sim.sample_mix(bundle, spec)
    print("Mix created")
    meta = await sim.draw_meta_prompt(
        instruction_y=bundle.instruction_y,
        bundle=bundle,
        mix=mix,
        K=META_PROMPT_K,
        complexify_c=COMPLEXIFY_C,
        sequential=False,
    )
    print("meta prompt created")

    row = await sim.generate_single_qa_datapoint(
        instruction_y=bundle.instruction_y,
        bundle=bundle,
        mix=mix,
        meta=meta,
    )
    print("single qa created")
    if row is None:
        print("No row accepted (requirement critic or refine loop).")
    else:
        print(row.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
