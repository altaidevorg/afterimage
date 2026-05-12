"""Skill proposal stage."""

from __future__ import annotations

import uuid

from ..providers.llm_providers import LLMProvider
from .prompts import build_skill_proposal_prompt
from .schemas import SkillProposalResponse
from .types import SkillProbeResult, SkillProposal, SkillVersion


class SkillProposer:
    """Analyze failed probes and propose a reusable skill update."""

    def __init__(self, llm: LLMProvider, *, temperature: float = 0.35):
        self.llm = llm
        self.temperature = temperature

    async def apropose(
        self,
        *,
        context: str,
        context_id: str,
        respondent_prompt: str,
        current_skill: SkillVersion | None,
        failed_results: list[SkillProbeResult],
        iteration: int,
    ) -> SkillProposal:
        prompt = build_skill_proposal_prompt(
            context=context,
            respondent_prompt=respondent_prompt,
            current_skill=current_skill,
            failed_results=failed_results,
            iteration=iteration,
        )
        response = await self.llm.agenerate_structured(
            prompt,
            SkillProposalResponse,
            temperature=self.temperature,
        )
        parsed = response.parsed
        action = (
            parsed.action if parsed.action in {"create", "revise", "keep"} else "create"
        )
        return SkillProposal(
            id=str(uuid.uuid4()),
            context_id=context_id,
            iteration=iteration,
            name=parsed.name,
            description=parsed.description,
            target_failure_modes=parsed.target_failure_modes,
            proposed_guidance=parsed.proposed_guidance,
            action=action,
            source_probe_ids=[r.probe.id for r in failed_results],
        )
