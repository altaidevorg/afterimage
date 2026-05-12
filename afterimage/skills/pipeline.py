"""High-level context-to-skill discovery pipeline."""

from __future__ import annotations

import asyncio
import uuid

from tqdm.auto import tqdm

from ..providers import DocumentProvider
from ..providers.llm_providers import LLMProvider
from ..types import Document
from .generation import SkillGenerator
from .judging import RubricJudge
from .probe_generation import SkillProbeGenerator
from .prompts import build_reasoner_prompt
from .proposal import SkillProposer
from .selection import SkillSelector
from .storage import DirectorySkillStore
from .types import (
    SkillProbe,
    SkillProbeResult,
    SkillSelectionResult,
    SkillSide,
    SkillVersion,
)


class SkillDiscoveryPipeline:
    """Discover context-specific skills from a document provider."""

    def __init__(
        self,
        *,
        document_provider: DocumentProvider,
        respondent_prompt: str,
        llm: LLMProvider,
        output_dir: str = "./skills",
        reasoner_llm: LLMProvider | None = None,
        probe_generator: SkillProbeGenerator | None = None,
        judge: RubricJudge | None = None,
        proposer: SkillProposer | None = None,
        skill_generator: SkillGenerator | None = None,
        selector: SkillSelector | None = None,
        reasoner_proposer: SkillProposer | None = None,
        challenger_proposer: SkillProposer | None = None,
        reasoner_skill_generator: SkillGenerator | None = None,
        challenger_skill_generator: SkillGenerator | None = None,
        use_source_rubrics: bool = False,
    ):
        self.document_provider = document_provider
        self.respondent_prompt = respondent_prompt
        self.llm = llm
        self.reasoner_llm = reasoner_llm or llm
        self.store = DirectorySkillStore(output_dir)
        self.probe_generator = probe_generator or SkillProbeGenerator(llm)
        self.judge = judge or RubricJudge(llm)
        shared_proposer = proposer or SkillProposer(llm)
        shared_generator = skill_generator or SkillGenerator(llm)
        self.reasoner_proposer = reasoner_proposer or shared_proposer
        self.challenger_proposer = challenger_proposer or SkillProposer(llm)
        self.reasoner_skill_generator = reasoner_skill_generator or shared_generator
        self.challenger_skill_generator = challenger_skill_generator or SkillGenerator(
            llm
        )
        self.selector = selector or SkillSelector(
            judge=self.judge,
            reasoner_llm=self.reasoner_llm,
        )
        self.use_source_rubrics = use_source_rubrics

    async def discover(
        self,
        *,
        iterations: int = 3,
        probes_per_context: int = 5,
        max_contexts: int | None = None,
        select_best: bool = True,
        bootstrap_when_no_failures: bool = False,
        show_progress: bool = False,
    ) -> list[SkillSelectionResult]:
        docs = self.document_provider.get_all()
        if max_contexts is not None:
            docs = docs[:max_contexts]

        selections = []
        document_iter = tqdm(
            docs,
            desc="Skill contexts",
            unit="context",
            disable=not show_progress,
        )
        for document in document_iter:
            if show_progress:
                document_iter.set_postfix_str(str(document.id)[:24])
            selection = await self.discover_for_document(
                document,
                iterations=iterations,
                probes_per_context=probes_per_context,
                select_best=select_best,
                bootstrap_when_no_failures=bootstrap_when_no_failures,
                show_progress=show_progress,
            )
            if selection is not None:
                selections.append(selection)
        return selections

    async def discover_for_document(
        self,
        document: Document,
        *,
        iterations: int = 3,
        probes_per_context: int = 5,
        select_best: bool = True,
        bootstrap_when_no_failures: bool = False,
        show_progress: bool = False,
    ) -> SkillSelectionResult | None:
        context_id = self.store.register_context(document)
        context = document.text or ""
        source_rubrics = (
            self._source_rubrics(document) if self.use_source_rubrics else None
        )
        reasoner_skill: SkillVersion | None = None
        challenger_skill: SkillVersion | None = None
        reasoner_versions: list[SkillVersion] = []
        hard_results: list[SkillProbeResult] = []
        easy_results: list[SkillProbeResult] = []
        per_iteration_steps = 1 + 2 * max(probes_per_context, 1) + 4
        progress = tqdm(
            total=max(iterations, 1) * per_iteration_steps + 4,
            desc=f"Context {context_id[:8]}",
            unit="step",
            leave=False,
            disable=not show_progress,
        )

        def advance(stage: str) -> None:
            if show_progress:
                progress.set_postfix_str(stage)
                progress.update(1)

        def advance_many(stage: str, count: int) -> None:
            if show_progress and count > 0:
                progress.set_postfix_str(stage)
                progress.update(count)

        def waiting(stage: str) -> None:
            if show_progress:
                progress.set_postfix_str(f"waiting: {stage}")
                progress.refresh()

        try:
            for iteration in range(1, iterations + 1):
                waiting(f"iteration {iteration}: challenger")
                probes = await self.probe_generator.agenerate(
                    context=context,
                    context_id=context_id,
                    respondent_prompt=self.respondent_prompt,
                    challenger_skill=challenger_skill,
                    n_probes=probes_per_context,
                    iteration=iteration,
                    source_rubrics=source_rubrics,
                )
                self.store.save_probes(context_id, probes)
                advance(f"iteration {iteration}: probes")

                waiting(f"iteration {iteration}: reasoner/judge")
                round_results = await asyncio.gather(
                    *[
                        self._evaluate_probe(
                            context=context,
                            probe=probe,
                            skill=reasoner_skill,
                        )
                        for probe in probes
                    ]
                )
                advance_many(f"iteration {iteration}: answer", len(probes))
                advance_many(f"iteration {iteration}: judge", len(probes))
                self.store.save_probe_results(context_id, round_results)

                failures = [result for result in round_results if not result.passed]
                successes = [result for result in round_results if result.passed]

                hardest_failure = self._select_hardest_failure(failures)
                if hardest_failure is not None:
                    hard_results.append(hardest_failure)
                easiest_success = self._select_easiest_success(successes)
                if easiest_success is not None:
                    easy_results.append(easiest_success)

                updated_reasoner = await self._update_skill_side(
                    context=context,
                    context_id=context_id,
                    iteration=iteration,
                    side="reasoner",
                    current_skill=reasoner_skill,
                    routed_results=failures,
                    proposer=self.reasoner_proposer,
                    generator=self.reasoner_skill_generator,
                    waiting=waiting,
                    advance=advance,
                )
                if updated_reasoner is not None:
                    reasoner_skill = updated_reasoner
                    self.store.save_version(reasoner_skill)
                    reasoner_versions.append(reasoner_skill)

                updated_challenger = await self._update_skill_side(
                    context=context,
                    context_id=context_id,
                    iteration=iteration,
                    side="challenger",
                    current_skill=challenger_skill,
                    routed_results=successes,
                    proposer=self.challenger_proposer,
                    generator=self.challenger_skill_generator,
                    waiting=waiting,
                    advance=advance,
                )
                if updated_challenger is not None:
                    challenger_skill = updated_challenger
                    self.store.save_version(challenger_skill)

            if not reasoner_versions and bootstrap_when_no_failures:
                waiting("bootstrap generator")
                reasoner_skill = (
                    await self.reasoner_skill_generator.agenerate_bootstrap(
                        context=context,
                        context_id=context_id,
                        respondent_prompt=self.respondent_prompt,
                        probe_results=easy_results,
                        iteration=iterations,
                    )
                )
                self.store.save_version(reasoner_skill)
                reasoner_versions.append(reasoner_skill)
                advance("bootstrap skill")

            if not reasoner_versions:
                return None

            if select_best:
                waiting("selection replay")
                selection = await self.selector.aselect(
                    context=context,
                    respondent_prompt=self.respondent_prompt,
                    versions=reasoner_versions,
                    hard_results=hard_results,
                    easy_results=easy_results,
                )
                advance("selection")
            else:
                latest = reasoner_versions[-1]
                selection = SkillSelectionResult(
                    context_id=context_id,
                    selected_version_id=latest.id,
                    selected_iteration=latest.iteration,
                    hard_score=0.0,
                    easy_score=0.0,
                    combined_score=0.0,
                    all_results=[],
                )
                advance("selection")

            if selection is not None:
                self.store.write_selection(selection)
                advance("write")
            return selection
        finally:
            progress.close()

    async def _update_skill_side(
        self,
        *,
        context: str,
        context_id: str,
        iteration: int,
        side: SkillSide,
        current_skill: SkillVersion | None,
        routed_results: list[SkillProbeResult],
        proposer: SkillProposer,
        generator: SkillGenerator,
        waiting,
        advance,
    ) -> SkillVersion | None:
        if routed_results:
            waiting(f"iteration {iteration}: {side} proposer")
            proposal = await proposer.apropose(
                context=context,
                context_id=context_id,
                respondent_prompt=self.respondent_prompt,
                current_skill=current_skill,
                routed_results=routed_results,
                iteration=iteration,
                side=side,
            )
            advance(f"iteration {iteration}: {side} proposal")
            if proposal.action == "keep" and current_skill is not None:
                next_skill = self._carry_forward_skill(
                    current_skill,
                    iteration=iteration,
                    reason=f"{side}_proposal_keep",
                )
            else:
                waiting(f"iteration {iteration}: {side} generator")
                next_skill = await generator.agenerate(
                    context=context,
                    proposal=proposal,
                    previous_skill=current_skill,
                    side=side,
                )
            advance(f"iteration {iteration}: {side} skill")
            return next_skill

        if current_skill is None:
            return None
        carried = self._carry_forward_skill(
            current_skill,
            iteration=iteration,
            reason=f"no_{side}_routed_results",
        )
        advance(f"iteration {iteration}: {side} carry")
        return carried

    async def _answer_probe(
        self,
        *,
        context: str,
        task: str,
        skill: SkillVersion | None,
    ) -> str:
        prompt = build_reasoner_prompt(
            context=context,
            respondent_prompt=self.respondent_prompt,
            skill=skill,
            task=task,
        )
        response = await self.reasoner_llm.agenerate_content(prompt, temperature=0.2)
        return response.text

    async def _evaluate_probe(
        self,
        *,
        context: str,
        probe: SkillProbe,
        skill: SkillVersion | None,
    ) -> SkillProbeResult:
        answer = await self._answer_probe(
            context=context,
            task=probe.task,
            skill=skill,
        )
        return await self.judge.aevaluate(
            probe=probe,
            answer=answer,
            context=context,
            skill_version_id=skill.id if skill else None,
        )

    @staticmethod
    def _source_rubrics(document: Document) -> list[str] | None:
        rubrics = document.metadata.get("rubrics")
        if not isinstance(rubrics, list):
            return None
        source_rubrics = [str(rubric) for rubric in rubrics if str(rubric).strip()]
        return source_rubrics or None

    @staticmethod
    def _select_hardest_failure(
        failures: list[SkillProbeResult],
    ) -> SkillProbeResult | None:
        if not failures:
            return None
        return min(failures, key=SkillDiscoveryPipeline._failure_priority)

    @staticmethod
    def _select_easiest_success(
        successes: list[SkillProbeResult],
    ) -> SkillProbeResult | None:
        if not successes:
            return None
        return min(successes, key=lambda result: len(result.probe.rubrics))

    @staticmethod
    def _failure_priority(result: SkillProbeResult) -> tuple[float, int]:
        status = result.rubric_status
        if status:
            rubric_pass_rate = sum(1 for passed in status if passed) / len(status)
        else:
            rubric_pass_rate = float(result.score)
        return (rubric_pass_rate, len(result.probe.rubrics))

    @staticmethod
    def _carry_forward_skill(
        previous_skill: SkillVersion,
        *,
        iteration: int,
        reason: str,
    ) -> SkillVersion:
        metadata = dict(previous_skill.metadata)
        metadata["carry_forward_reason"] = reason
        return SkillVersion(
            id=str(uuid.uuid4()),
            context_id=previous_skill.context_id,
            iteration=iteration,
            side=previous_skill.side,
            name=previous_skill.name,
            description=previous_skill.description,
            content=previous_skill.content,
            source_probe_ids=list(previous_skill.source_probe_ids),
            metrics=dict(previous_skill.metrics),
            metadata=metadata,
        )
