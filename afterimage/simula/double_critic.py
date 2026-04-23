"""Double-critic for verifiable tasks (paper §2.2, §3.1)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from ..providers.llm_providers import LLMProvider
from .schemas_llm import DoubleProbeCorrect, DoubleProbeIncorrect
from .types import DoubleCritiqueVerdict, MCQRow

logger = logging.getLogger(__name__)


@dataclass
class DoubleCritiqueContext:
    """Carries last verdict for lineage attachment."""

    last_verdict: DoubleCritiqueVerdict | None = None


def accept_double_critique(v: DoubleCritiqueVerdict) -> bool:
    """Accept only if model affirms correctness and does NOT affirm incorrectness."""
    return bool(v.claims_correct) and not bool(v.claims_incorrect)


async def double_critique_mcq(
    llm: LLMProvider,
    *,
    row: MCQRow,
    temperature: float = 0.15,
) -> DoubleCritiqueVerdict:
    """Two independent structured probes with different framings."""
    payload = json.dumps(
        {
            "question": row.question,
            "choices": row.choices,
            "correct_index": row.correct_index,
        },
        ensure_ascii=False,
    )
    prompt_a = (
        "You are an independent verifier (probe A). Given this multiple-choice item, "
        "is the marked correct_index factually correct for the question and choices? "
        "Answer conservatively.\n\n"
        f"{payload}"
    )
    ra = await llm.agenerate_structured(
        prompt=prompt_a,
        schema=DoubleProbeCorrect,
        temperature=temperature,
    )
    prompt_b = (
        "You are a different auditor (probe B). Focus on finding errors. "
        "For this MCQ, is the labeled correct_index WRONG or misleading relative to "
        "the question and choices?\n\n"
        f"{payload}"
    )
    rb = await llm.agenerate_structured(
        prompt=prompt_b,
        schema=DoubleProbeIncorrect,
        temperature=temperature,
    )
    return DoubleCritiqueVerdict(
        claims_correct=bool(ra.parsed.is_correct),
        claims_incorrect=bool(rb.parsed.is_incorrect),
        rationale_correct=ra.parsed.rationale,
        rationale_incorrect=rb.parsed.rationale,
    )


async def gate_mcq_with_double_critic(
    llm: LLMProvider,
    *,
    serialized: str,
    temperature: float = 0.15,
) -> bool:
    """Parse serialized JSON as MCQRow and run double critic."""
    try:
        data = json.loads(serialized)
        row = MCQRow.model_validate(data)
    except Exception as e:
        logger.warning("double_critic gate: invalid MCQ JSON: %s", e)
        return False
    v = await double_critique_mcq(llm, row=row, temperature=temperature)
    return accept_double_critique(v)


async def double_critique_mcq_with_context(
    llm: LLMProvider,
    *,
    ctx: DoubleCritiqueContext,
    row: MCQRow,
    temperature: float = 0.15,
) -> bool:
    v = await double_critique_mcq(llm, row=row, temperature=temperature)
    ctx.last_verdict = v
    return accept_double_critique(v)
