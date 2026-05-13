"""Rubric-specific judge for context-to-skill probes."""

from __future__ import annotations

from ..providers.llm_providers import LLMProvider
from .prompts import build_rubric_judge_prompt
from .schemas import RubricJudgeResponse
from .types import SkillProbe, SkillProbeResult


class RubricJudge:
    """Strictly grade one probe answer against binary rubrics."""

    def __init__(self, llm: LLMProvider, *, temperature: float = 0.1):
        self.llm = llm
        self.temperature = temperature

    async def aevaluate(
        self,
        *,
        probe: SkillProbe,
        answer: str,
        context: str | None = None,
        skill_version_id: str | None = None,
    ) -> SkillProbeResult:
        prompt = build_rubric_judge_prompt(
            context=context,
            task=probe.task,
            rubrics=probe.rubrics,
            answer=answer,
        )
        response = await self.llm.agenerate_structured(
            prompt,
            RubricJudgeResponse,
            temperature=self.temperature,
        )
        status = list(response.parsed.requirement_status)
        derived_passed = bool(status) and all(status)
        raw_score = float(response.parsed.overall_score)
        normalized_score = 1.0 if derived_passed else 0.0
        return SkillProbeResult(
            probe=probe,
            answer=answer,
            score=normalized_score,
            passed=derived_passed,
            rubric_status=status,
            judge_feedback=response.parsed.rationale,
            skill_version_id=skill_version_id,
            metadata={
                "raw_overall_score": raw_score,
                "score_normalized": raw_score != normalized_score,
            },
        )
