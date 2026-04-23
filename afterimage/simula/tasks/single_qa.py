"""Single-turn QA generation (structured)."""

from __future__ import annotations

import json

from ...monitoring import GenerationMonitor
from ...providers.llm_providers import LLMProvider
from ..llm_track import agenerate_structured_tracked
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
    monitor: GenerationMonitor | None = None,
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
    resp = await agenerate_structured_tracked(
        monitor,
        llm,
        operation="opensimula.task.single_qa_generate",
        metadata={"mix_id": mix.id, "meta_prompt_id": meta.id},
        prompt=prompt,
        schema=SingleQAGenResponse,
        temperature=temperature,
    )
    row = resp.parsed
    return json.dumps(
        {"question": row.question.strip(), "answer": row.answer.strip()},
        ensure_ascii=False,
    )
