"""Multiple-choice generation."""

from __future__ import annotations

import json

from ...monitoring import GenerationMonitor
from ...providers.llm_providers import LLMProvider
from ..llm_track import agenerate_structured_tracked
from ..schemas_llm import MCQGenResponse
from ..types import MetaPrompt, Mix, TaxonomyBundle
from ..meta_prompt import _mix_description


async def agenerate_mcq_json(
    llm: LLMProvider,
    *,
    instruction_y: str,
    bundle: TaxonomyBundle,
    mix: Mix,
    meta: MetaPrompt,
    num_choices: int = 4,
    temperature: float = 0.45,
    monitor: GenerationMonitor | None = None,
) -> str:
    """Return JSON string for MCQRow (exactly num_choices strings)."""
    mix_desc = _mix_description(bundle, mix)
    prompt = (
        f"Dataset instructions y:\n{instruction_y}\n\n"
        f"Taxonomy mix:\n{mix_desc}\n\n"
        f"Meta-prompt / scenario:\n{meta.text}\n\n"
        f"Generate ONE multiple-choice question with exactly {num_choices} answer choices. "
        "Exactly one choice must be correct; set correct_index 0-based."
    )
    resp = await agenerate_structured_tracked(
        monitor,
        llm,
        operation="opensimula.task.mcq_generate",
        metadata={
            "mix_id": mix.id,
            "meta_prompt_id": meta.id,
            "num_choices": num_choices,
        },
        prompt=prompt,
        schema=MCQGenResponse,
        temperature=temperature,
    )
    row = resp.parsed
    choices = [c.strip() for c in row.choices][:num_choices]
    while len(choices) < num_choices:
        choices.append("(placeholder)")
    ci = int(row.correct_index)
    if ci < 0 or ci >= num_choices:
        ci = 0
    payload = {
        "question": row.question.strip(),
        "choices": choices,
        "correct_index": ci,
    }
    return json.dumps(payload, ensure_ascii=False)
