"""Skill markdown generation stage."""

from __future__ import annotations

import uuid

from ..providers.llm_providers import LLMProvider
from .prompts import build_skill_bootstrap_prompt, build_skill_generation_prompt
from .schemas import SkillContentResponse
from .types import SkillProbeResult, SkillProposal, SkillSide, SkillVersion


class SkillGenerator:
    """Turn a skill proposal into a candidate skill version."""

    def __init__(self, llm: LLMProvider, *, temperature: float = 0.25):
        self.llm = llm
        self.temperature = temperature

    async def agenerate(
        self,
        *,
        context: str,
        proposal: SkillProposal,
        previous_skill: SkillVersion | None,
        side: SkillSide,
    ) -> SkillVersion:
        prompt = build_skill_generation_prompt(
            context=context,
            proposal=proposal,
            previous_skill=previous_skill,
            side=side,
        )
        response = await self.llm.agenerate_structured(
            prompt,
            SkillContentResponse,
            temperature=self.temperature,
        )
        parsed = response.parsed
        return SkillVersion(
            id=str(uuid.uuid4()),
            context_id=proposal.context_id,
            iteration=proposal.iteration,
            side=side,
            name=parsed.name,
            description=parsed.description,
            content=parsed.content,
            source_probe_ids=proposal.source_probe_ids,
        )

    async def agenerate_bootstrap(
        self,
        *,
        context: str,
        context_id: str,
        respondent_prompt: str,
        probe_results: list[SkillProbeResult],
        iteration: int,
    ) -> SkillVersion:
        prompt = build_skill_bootstrap_prompt(
            context=context,
            respondent_prompt=respondent_prompt,
            probe_results=probe_results,
        )
        response = await self.llm.agenerate_structured(
            prompt,
            SkillContentResponse,
            temperature=self.temperature,
        )
        parsed = response.parsed
        return SkillVersion(
            id=str(uuid.uuid4()),
            context_id=context_id,
            iteration=iteration,
            side="reasoner",
            name=parsed.name,
            description=parsed.description,
            content=parsed.content,
            source_probe_ids=[result.probe.id for result in probe_results],
            metadata={"generation_reason": "bootstrap_no_failed_probes"},
        )
