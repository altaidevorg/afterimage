"""Probe generation for context-to-skill discovery."""

from __future__ import annotations

import uuid

from ..providers.llm_providers import LLMProvider
from .prompts import build_probe_generation_prompt
from .schemas import ProbeGenerationResponse
from .types import SkillProbe, SkillVersion


class SkillProbeGenerator:
    """Generate context-grounded probes and rubrics with structured LLM output."""

    def __init__(self, llm: LLMProvider, *, temperature: float = 0.5):
        self.llm = llm
        self.temperature = temperature

    async def agenerate(
        self,
        *,
        context: str,
        context_id: str,
        respondent_prompt: str,
        challenger_skill: SkillVersion | None,
        n_probes: int,
        iteration: int,
        source_rubrics: list[str] | None = None,
    ) -> list[SkillProbe]:
        prompt = build_probe_generation_prompt(
            context=context,
            respondent_prompt=respondent_prompt,
            challenger_skill=challenger_skill,
            n_probes=n_probes,
            source_rubrics=source_rubrics,
        )
        response = await self.llm.agenerate_structured(
            prompt,
            ProbeGenerationResponse,
            temperature=self.temperature,
        )
        probes = []
        for spec in response.parsed.probes[:n_probes]:
            probes.append(
                SkillProbe(
                    id=str(uuid.uuid4()),
                    context_id=context_id,
                    task=spec.task,
                    rubrics=spec.rubrics,
                    iteration=iteration,
                )
            )
        return probes
