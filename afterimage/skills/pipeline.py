"""High-level context-to-skill discovery pipeline."""

from __future__ import annotations

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
from .types import SkillProbeResult, SkillSelectionResult, SkillVersion


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
    ):
        self.document_provider = document_provider
        self.respondent_prompt = respondent_prompt
        self.llm = llm
        self.reasoner_llm = reasoner_llm or llm
        self.store = DirectorySkillStore(output_dir)
        self.probe_generator = probe_generator or SkillProbeGenerator(llm)
        self.judge = judge or RubricJudge(llm)
        self.proposer = proposer or SkillProposer(llm)
        self.skill_generator = skill_generator or SkillGenerator(llm)
        self.selector = selector or SkillSelector(
            judge=self.judge,
            reasoner_llm=self.reasoner_llm,
        )

    async def discover(
        self,
        *,
        iterations: int = 3,
        probes_per_context: int = 5,
        max_contexts: int | None = None,
        select_best: bool = True,
        bootstrap_when_no_failures: bool = True,
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
        bootstrap_when_no_failures: bool = True,
        show_progress: bool = False,
    ) -> SkillSelectionResult | None:
        context_id = self.store.register_context(document)
        context = document.text or ""
        source_rubrics = self._source_rubrics(document)
        current_skill: SkillVersion | None = None
        versions: list[SkillVersion] = []
        hard_results: list[SkillProbeResult] = []
        easy_results: list[SkillProbeResult] = []
        progress = tqdm(
            total=max(iterations, 1) * (1 + 2 * max(probes_per_context, 1)) + 4,
            desc=f"Context {context_id[:8]}",
            unit="step",
            leave=False,
            disable=not show_progress,
        )

        def advance(stage: str) -> None:
            if show_progress:
                progress.set_postfix_str(stage)
                progress.update(1)

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
                    current_skill=current_skill,
                    n_probes=probes_per_context,
                    iteration=iteration,
                    source_rubrics=source_rubrics,
                )
                self.store.save_probes(context_id, probes)
                advance(f"iteration {iteration}: probes")

                round_results = []
                for probe in probes:
                    waiting(f"iteration {iteration}: reasoner")
                    answer = await self._answer_probe(
                        context=context,
                        task=probe.task,
                        skill=current_skill,
                    )
                    advance(f"iteration {iteration}: answer")
                    waiting(f"iteration {iteration}: judge")
                    result = await self.judge.aevaluate(
                        probe=probe,
                        answer=answer,
                        context=context,
                        skill_version_id=current_skill.id if current_skill else None,
                    )
                    advance(f"iteration {iteration}: judge")
                    round_results.append(result)
                    if result.passed:
                        easy_results.append(result)
                    else:
                        hard_results.append(result)

                self.store.save_probe_results(context_id, round_results)
                failures = [r for r in round_results if not r.passed]
                if not failures:
                    continue

                waiting(f"iteration {iteration}: proposer")
                proposal = await self.proposer.apropose(
                    context=context,
                    context_id=context_id,
                    respondent_prompt=self.respondent_prompt,
                    current_skill=current_skill,
                    failed_results=failures,
                    iteration=iteration,
                )
                advance(f"iteration {iteration}: proposal")
                if proposal.action == "keep" and current_skill is not None:
                    continue

                waiting(f"iteration {iteration}: generator")
                current_skill = await self.skill_generator.agenerate(
                    context=context,
                    proposal=proposal,
                    previous_skill=current_skill,
                )
                self.store.save_version(current_skill)
                versions.append(current_skill)
                advance(f"iteration {iteration}: skill")

            if not versions and bootstrap_when_no_failures:
                waiting("bootstrap generator")
                current_skill = await self.skill_generator.agenerate_bootstrap(
                    context=context,
                    context_id=context_id,
                    respondent_prompt=self.respondent_prompt,
                    probe_results=easy_results,
                    iteration=iterations,
                )
                self.store.save_version(current_skill)
                versions.append(current_skill)
                advance("bootstrap skill")

            if not versions:
                return None

            if select_best:
                waiting("selection replay")
                selection = await self.selector.aselect(
                    context=context,
                    respondent_prompt=self.respondent_prompt,
                    versions=versions,
                    hard_results=hard_results,
                    easy_results=easy_results,
                )
                advance("selection")
            else:
                latest = versions[-1]
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

    @staticmethod
    def _source_rubrics(document: Document) -> list[str] | None:
        rubrics = document.metadata.get("rubrics")
        if not isinstance(rubrics, list):
            return None
        source_rubrics = [str(rubric) for rubric in rubrics if str(rubric).strip()]
        return source_rubrics or None
