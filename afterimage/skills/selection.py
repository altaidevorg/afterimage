"""Skill version replay and selection."""

from __future__ import annotations

import asyncio

from ..providers.llm_providers import LLMProvider
from .judging import RubricJudge
from .prompts import build_reasoner_prompt
from .types import SkillProbeResult, SkillSelectionResult, SkillVersion


class SkillSelector:
    """Replay candidate skills on hard/easy probes and choose the best version."""

    def __init__(
        self,
        *,
        judge: RubricJudge,
        reasoner_llm: LLMProvider | None = None,
        hard_weight: float = 0.7,
        easy_weight: float = 0.3,
        scoring: str = "product",
        laplace_smoothing: bool = True,
    ):
        self.judge = judge
        self.reasoner_llm = reasoner_llm or judge.llm
        self.hard_weight = hard_weight
        self.easy_weight = easy_weight
        self.scoring = scoring
        self.laplace_smoothing = laplace_smoothing

    async def aselect(
        self,
        *,
        context: str,
        respondent_prompt: str,
        versions: list[SkillVersion],
        hard_results: list[SkillProbeResult],
        easy_results: list[SkillProbeResult],
    ) -> SkillSelectionResult | None:
        if not versions:
            return None

        scored = []
        for version in versions:
            hard_score, easy_score = await asyncio.gather(
                self._score_set(
                    context=context,
                    respondent_prompt=respondent_prompt,
                    version=version,
                    probe_results=hard_results,
                ),
                self._score_set(
                    context=context,
                    respondent_prompt=respondent_prompt,
                    version=version,
                    probe_results=easy_results,
                ),
            )
            if self.scoring == "weighted":
                combined = self.hard_weight * hard_score + self.easy_weight * easy_score
            else:
                combined = hard_score * easy_score
            scored.append(
                {
                    "version_id": version.id,
                    "iteration": version.iteration,
                    "hard_score": hard_score,
                    "easy_score": easy_score,
                    "combined_score": combined,
                }
            )

        best = max(scored, key=lambda row: row["combined_score"])
        return SkillSelectionResult(
            context_id=versions[0].context_id,
            selected_version_id=best["version_id"],
            selected_iteration=best["iteration"],
            hard_score=best["hard_score"],
            easy_score=best["easy_score"],
            combined_score=best["combined_score"],
            all_results=scored,
        )

    async def _score_set(
        self,
        *,
        context: str,
        respondent_prompt: str,
        version: SkillVersion,
        probe_results: list[SkillProbeResult],
    ) -> float:
        if not probe_results:
            return 1.0

        replay_results = await asyncio.gather(
            *[
                self._replay_probe(
                    context=context,
                    respondent_prompt=respondent_prompt,
                    version=version,
                    previous=previous,
                )
                for previous in probe_results
            ]
        )
        passed = sum(1 for result in replay_results if result.passed)
        if self.laplace_smoothing:
            return (passed + 1) / (len(probe_results) + 1)
        return passed / len(probe_results)

    async def _replay_probe(
        self,
        *,
        context: str,
        respondent_prompt: str,
        version: SkillVersion,
        previous: SkillProbeResult,
    ) -> SkillProbeResult:
        prompt = build_reasoner_prompt(
            context=context,
            respondent_prompt=respondent_prompt,
            skill=version,
            task=previous.probe.task,
        )
        response = await self.reasoner_llm.agenerate_content(prompt, temperature=0.2)
        return await self.judge.aevaluate(
            probe=previous.probe,
            answer=response.text,
            context=context,
            skill_version_id=version.id,
        )
