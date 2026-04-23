"""Single-turn QA generation (structured)."""

from __future__ import annotations

import json

from ...providers.llm_providers import LLMProvider
from ..schemas_llm import SingleQAGenResponse
from ..types import MetaPrompt, Mix, TaxonomyBundle
from ..meta_prompt import _mix_description


async def agenerate_single_qa_json(
    llm: LLMProvider,
    *,
    instruction_y: str,
    bundle: TaxonomyBundle,
    mix: Mix,
    meta: MetaPrompt,
    temperature: float = 0.55,
) -> str:
    """Return JSON string for a single QA row (question + answer)."""
    mix_desc = _mix_description(bundle, mix)
    prompt = (
        f"Dataset instructions y:\n{instruction_y}\n\n"
        f"Taxonomy mix:\n{mix_desc}\n\n"
        f"Meta-prompt / scenario:\n{meta.text}\n\n"
        "Generate ONE question and a concise correct answer suitable for supervision. "
        "The question must respect the taxonomy mix and meta-prompt."
    )
    resp = await llm.agenerate_structured(
        prompt=prompt,
        schema=SingleQAGenResponse,
        temperature=temperature,
    )
    row = resp.parsed
    return json.dumps(
        {"question": row.question.strip(), "answer": row.answer.strip()},
        ensure_ascii=False,
    )
