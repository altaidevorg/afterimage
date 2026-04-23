"""Requirement critic and refinement loop (paper Algorithm 2, §2.2)."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Coroutine, Literal

from ..monitoring import GenerationMonitor
from ..providers.llm_providers import LLMProvider
from .double_critic import accept_double_critique, double_critique_mcq
from .llm_track import agenerate_structured_tracked
from .meta_prompt import _mix_description
from .schemas_llm import RawGenerationResponse, RequirementCritiqueResponse
from .types import (
    DataPointLineage,
    DataPointRecord,
    DatasetBatch,
    MCQRow,
    MetaPrompt,
    Mix,
    RequirementCritiqueVerdict,
    TaxonomyBundle,
)
from .sampling import mix_factor_paths

logger = logging.getLogger(__name__)

SimulaTask = Literal["single_qa", "mcq", "raw"]


async def requirement_critique(
    llm: LLMProvider,
    *,
    instruction_y: str,
    bundle: TaxonomyBundle,
    mix: Mix,
    meta: MetaPrompt,
    serialized_point: str,
    temperature: float = 0.2,
    monitor: GenerationMonitor | None = None,
) -> RequirementCritiqueVerdict:
    """Point-wise check that the datapoint fulfills meta-prompt and mix requirements."""
    mix_desc = _mix_description(bundle, mix)
    prompt = (
        f"Dataset instructions y:\n{instruction_y}\n\n"
        f"Taxonomy mix requirements:\n{mix_desc}\n\n"
        f"Meta-prompt:\n{meta.text}\n\n"
        "Generated datapoint (JSON or text):\n"
        f"{serialized_point}\n\n"
        "Does this datapoint satisfy ALL semantic and structural requirements implied "
        "by the meta-prompt and the taxonomy mix? Answer conservative false if unsure."
    )
    resp = await agenerate_structured_tracked(
        monitor,
        llm,
        operation="opensimula.critic.requirement",
        metadata={"mix_id": mix.id, "meta_prompt_id": meta.id},
        prompt=prompt,
        schema=RequirementCritiqueResponse,
        temperature=temperature,
    )
    return RequirementCritiqueVerdict(
        satisfying=resp.parsed.satisfying,
        explanation=resp.parsed.explanation,
    )


async def refine_serialized_point(
    llm: LLMProvider,
    *,
    instruction_y: str,
    bundle: TaxonomyBundle,
    mix: Mix,
    meta: MetaPrompt,
    serialized_point: str,
    explanation: str,
    temperature: float = 0.35,
    monitor: GenerationMonitor | None = None,
) -> str:
    """Agentic refinement from critic explanation (Algorithm 2)."""
    mix_desc = _mix_description(bundle, mix)
    prompt = (
        f"Dataset instructions y:\n{instruction_y}\n\n"
        f"Requirements:\n{mix_desc}\n\n"
        f"Meta-prompt:\n{meta.text}\n\n"
        "Current datapoint:\n"
        f"{serialized_point}\n\n"
        "Critic feedback:\n"
        f"{explanation}\n\n"
        "Rewrite the datapoint to fix the issues. Output ONLY valid JSON with the "
        "same top-level keys as the original if it was JSON; otherwise plain text."
    )
    resp = await agenerate_structured_tracked(
        monitor,
        llm,
        operation="opensimula.critic.refine",
        metadata={"mix_id": mix.id, "meta_prompt_id": meta.id},
        prompt=prompt,
        schema=RawGenerationResponse,
        temperature=temperature,
    )
    return resp.parsed.content.strip()


async def generate_with_requirement_loop(
    llm: LLMProvider,
    *,
    instruction_y: str,
    bundle: TaxonomyBundle,
    mix: Mix,
    meta: MetaPrompt,
    initial_serialized: str,
    max_refine_rounds: int = 4,
    temperature_critique: float = 0.2,
    temperature_refine: float = 0.35,
    monitor: GenerationMonitor | None = None,
) -> tuple[str, list[RequirementCritiqueVerdict]]:
    """Repeat critique → refine until satisfying or cap."""
    point = initial_serialized
    verdicts: list[RequirementCritiqueVerdict] = []
    for _ in range(max_refine_rounds + 1):
        v = await requirement_critique(
            llm,
            instruction_y=instruction_y,
            bundle=bundle,
            mix=mix,
            meta=meta,
            serialized_point=point,
            temperature=temperature_critique,
            monitor=monitor,
        )
        verdicts.append(v)
        if v.satisfying:
            return point, verdicts
        if _ == max_refine_rounds:
            break
        point = await refine_serialized_point(
            llm,
            instruction_y=instruction_y,
            bundle=bundle,
            mix=mix,
            meta=meta,
            serialized_point=point,
            explanation=v.explanation,
            temperature=temperature_refine,
            monitor=monitor,
        )
    return point, verdicts


def build_lineage(
    *,
    instruction_y: str,
    bundle: TaxonomyBundle,
    mix: Mix,
    meta: MetaPrompt,
    requirement_critiques: list[RequirementCritiqueVerdict],
    double_critique: Any = None,
) -> DataPointLineage:
    paths = mix_factor_paths(bundle, mix)
    return DataPointLineage(
        instruction_y=instruction_y,
        mix_id=mix.id,
        meta_prompt_id=meta.id,
        factor_paths={k: list(v) for k, v in paths.items()},
        expansion_trace_ids=[],
        requirement_critiques=requirement_critiques,
        double_critique=double_critique,
    )


async def run_generation_pipeline(
    llm: LLMProvider,
    *,
    instruction_y: str,
    bundle: TaxonomyBundle,
    mix: Mix,
    meta: MetaPrompt,
    generate_initial: Callable[
        [LLMProvider], Coroutine[Any, Any, str]
    ],
    task: SimulaTask,
    max_refine_rounds: int = 4,
    double_critic_temperature: float = 0.15,
    monitor: GenerationMonitor | None = None,
) -> DataPointRecord | None:
    """Generate → requirement critic refinements → optional double-critic (MCQ)."""
    initial = await generate_initial(llm)
    point, verdicts = await generate_with_requirement_loop(
        llm,
        instruction_y=instruction_y,
        bundle=bundle,
        mix=mix,
        meta=meta,
        initial_serialized=initial,
        max_refine_rounds=max_refine_rounds,
        monitor=monitor,
    )
    if not verdicts or not verdicts[-1].satisfying:
        return None

    double_v = None
    if task == "mcq":
        try:
            row = MCQRow.model_validate(json.loads(point))
        except Exception as e:
            logger.warning("MCQ parse failed after requirement loop: %s", e)
            return None
        double_v = await double_critique_mcq(
            llm,
            row=row,
            temperature=double_critic_temperature,
            monitor=monitor,
        )
        if not accept_double_critique(double_v):
            logger.info("Double-critic rejected MCQ after requirement loop")
            return None

    lineage = build_lineage(
        instruction_y=instruction_y,
        bundle=bundle,
        mix=mix,
        meta=meta,
        requirement_critiques=verdicts,
        double_critique=double_v,
    )
    try:
        payload = json.loads(point)
    except json.JSONDecodeError:
        payload = {"text": point}
    return DataPointRecord(task=task, payload=payload, lineage=lineage)


def append_batch(batch: DatasetBatch, rec: DataPointRecord | None) -> None:
    if rec is not None:
        batch.records.append(rec)
