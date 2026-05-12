from __future__ import annotations

import pytest

from afterimage.providers.llm_providers import LLMResponse, StructuredLLMResponse
from afterimage.skills import (
    DirectorySkillStore,
    SkillDiscoveryPipeline,
    SkillRespondentPromptModifier,
)
from afterimage.skills.judging import RubricJudge
from afterimage.skills.prompts import build_probe_generation_prompt
from afterimage.skills.schemas import (
    ProbeGenerationResponse,
    ProbeSpec,
    RubricJudgeResponse,
    SkillContentResponse,
    SkillProposalResponse,
)
from afterimage.skills.selection import SkillSelector
from afterimage.skills.types import (
    SkillProbe,
    SkillProbeResult,
    SkillSelectionResult,
    SkillVersion,
)
from afterimage.types import Document


def _structured(parsed):
    return StructuredLLMResponse(
        text="",
        prompt_token_count=0,
        completion_token_count=0,
        total_token_count=0,
        finish_reason="stop",
        model_name="fake",
        raw_response=None,
        parsed=parsed,
    )


class FakeSkillLLM:
    def __init__(self):
        self.judge_calls = 0

    async def agenerate_content(self, prompt, temperature=0.7, **kwargs):
        return LLMResponse(
            text="fake answer",
            prompt_token_count=0,
            completion_token_count=0,
            total_token_count=0,
            finish_reason="stop",
            model_name="fake",
            raw_response=None,
        )

    async def agenerate_structured(self, prompt, schema, temperature=0.7, **kwargs):
        if schema is ProbeGenerationResponse:
            return _structured(
                ProbeGenerationResponse(
                    probes=[
                        ProbeSpec(
                            task="Answer using the refund policy.",
                            rubrics=["Mentions the 24 hour refund limit."],
                        )
                    ]
                )
            )
        if schema is RubricJudgeResponse:
            self.judge_calls += 1
            passed = self.judge_calls > 1
            return _structured(
                RubricJudgeResponse(
                    rationale="ok" if passed else "missing limit",
                    requirement_status=[passed],
                    overall_score=1.0 if passed else 0.0,
                )
            )
        if schema is SkillProposalResponse:
            return _structured(
                SkillProposalResponse(
                    action="create",
                    name="refund-policy-check",
                    description="Use refund policy constraints before answering.",
                    target_failure_modes=["missed refund limit"],
                    proposed_guidance="Always verify refund timing constraints.",
                )
            )
        if schema is SkillContentResponse:
            if "Challenger Generator" in prompt:
                return _structured(
                    SkillContentResponse(
                        name="refund-probe-hardening",
                        description="Generate stricter refund probes.",
                        content="Use multi-constraint refund tasks with explicit binary rubrics.",
                    )
                )
            return _structured(
                SkillContentResponse(
                    name="refund-policy-check",
                    description="Use refund policy constraints before answering.",
                    content="Check whether the refund request is within 24 hours.",
                )
            )
        raise AssertionError(f"Unexpected schema: {schema}")


class AllPassingSkillLLM:
    def __init__(self):
        self.structured_prompts = []

    async def agenerate_content(self, prompt, temperature=0.7, **kwargs):
        return LLMResponse(
            text="DRAGON SPEAKS\nGRAAAH, little mortal... answer.",
            prompt_token_count=0,
            completion_token_count=0,
            total_token_count=0,
            finish_reason="stop",
            model_name="fake",
            raw_response=None,
        )

    async def agenerate_structured(self, prompt, schema, temperature=0.7, **kwargs):
        self.structured_prompts.append(prompt)
        if schema is ProbeGenerationResponse:
            assert "Source benchmark rubrics" in prompt
            return _structured(
                ProbeGenerationResponse(
                    probes=[
                        ProbeSpec(
                            task="Answer in the required persona.",
                            rubrics=["Begins with DRAGON SPEAKS."],
                        )
                    ]
                )
            )
        if schema is RubricJudgeResponse:
            return _structured(
                RubricJudgeResponse(
                    rationale="ok",
                    requirement_status=[True],
                    overall_score=1.0,
                )
            )
        if schema is SkillProposalResponse:
            assert "Challenger Proposer" in prompt
            return _structured(
                SkillProposalResponse(
                    action="create",
                    name="dragon-probe-hardening",
                    description="Generate stricter dragon persona probes.",
                    target_failure_modes=["tasks too easy"],
                    proposed_guidance="Require exact persona headers and stricter formatting rubrics.",
                )
            )
        if schema is SkillContentResponse:
            if "No failed probe was found" in prompt:
                return _structured(
                    SkillContentResponse(
                        name="persona-format-rules",
                        description="Preserve context persona and formatting rules.",
                        content="Use the required persona header and opening sentence.",
                    )
                )
            assert "Challenger Generator" in prompt
            return _structured(
                SkillContentResponse(
                    name="dragon-probe-hardening",
                    description="Generate stricter dragon persona probes.",
                    content="Generate tasks that force exact persona headers and opening lines.",
                )
            )
        raise AssertionError(f"Unexpected schema: {schema}")


class MixedOutcomeSkillLLM:
    def __init__(self):
        self.judge_calls = 0

    async def agenerate_content(self, prompt, temperature=0.7, **kwargs):
        return LLMResponse(
            text="mixed outcome answer",
            prompt_token_count=0,
            completion_token_count=0,
            total_token_count=0,
            finish_reason="stop",
            model_name="fake",
            raw_response=None,
        )

    async def agenerate_structured(self, prompt, schema, temperature=0.7, **kwargs):
        if schema is ProbeGenerationResponse:
            return _structured(
                ProbeGenerationResponse(
                    probes=[
                        ProbeSpec(
                            task="Task one.",
                            rubrics=["Requirement A", "Requirement B"],
                        ),
                        ProbeSpec(
                            task="Task two.",
                            rubrics=["Requirement C"],
                        ),
                    ]
                )
            )
        if schema is RubricJudgeResponse:
            self.judge_calls += 1
            if self.judge_calls == 1:
                return _structured(
                    RubricJudgeResponse(
                        rationale="missed one requirement",
                        requirement_status=[False, True],
                        overall_score=0.0,
                    )
                )
            return _structured(
                RubricJudgeResponse(
                    rationale="ok",
                    requirement_status=[True],
                    overall_score=1.0,
                )
            )
        if schema is SkillProposalResponse:
            if "Reasoner Proposer" in prompt:
                return _structured(
                    SkillProposalResponse(
                        action="create",
                        name="reasoner-fix",
                        description="Fix respondent failure.",
                        target_failure_modes=["missed requirement"],
                        proposed_guidance="Check every rubric constraint before answering.",
                    )
                )
            return _structured(
                SkillProposalResponse(
                    action="create",
                    name="challenger-fix",
                    description="Tighten challenger probes.",
                    target_failure_modes=["too easy"],
                    proposed_guidance="Use fewer but sharper success rubrics.",
                )
            )
        if schema is SkillContentResponse:
            if "Reasoner Generator" in prompt:
                return _structured(
                    SkillContentResponse(
                        name="reasoner-fix",
                        description="Fix respondent failure.",
                        content="Verify every required condition before committing to the answer.",
                    )
                )
            return _structured(
                SkillContentResponse(
                    name="challenger-fix",
                    description="Tighten challenger probes.",
                    content="Generate probes that separate partial from complete understanding.",
                )
            )
        raise AssertionError(f"Unexpected schema: {schema}")


class RecordingSelector:
    def __init__(self):
        self.hard_results = []
        self.easy_results = []
        self.versions = []

    async def aselect(
        self,
        *,
        context,
        respondent_prompt,
        versions,
        hard_results,
        easy_results,
    ):
        self.hard_results = list(hard_results)
        self.easy_results = list(easy_results)
        self.versions = list(versions)
        selected = versions[-1]
        return SkillSelectionResult(
            context_id=selected.context_id,
            selected_version_id=selected.id,
            selected_iteration=selected.iteration,
            hard_score=1.0,
            easy_score=1.0,
            combined_score=1.0,
        )


class ReplayReasonerLLM:
    async def agenerate_content(self, prompt, temperature=0.7, **kwargs):
        return LLMResponse(
            text="replay answer",
            prompt_token_count=0,
            completion_token_count=0,
            total_token_count=0,
            finish_reason="stop",
            model_name="fake",
            raw_response=None,
        )


class AlwaysFailReplayJudge:
    def __init__(self):
        self.llm = ReplayReasonerLLM()

    async def aevaluate(self, *, probe, answer, context=None, skill_version_id=None):
        return SkillProbeResult(
            probe=probe,
            answer=answer,
            score=0.0,
            passed=False,
            rubric_status=[False],
            judge_feedback="still failing",
            skill_version_id=skill_version_id,
        )


class InconsistentJudgeLLM:
    async def agenerate_structured(self, prompt, schema, temperature=0.7, **kwargs):
        assert schema is RubricJudgeResponse
        return _structured(
            RubricJudgeResponse(
                rationale="all rubrics satisfied but score set incorrectly",
                requirement_status=[True, True],
                overall_score=0.0,
            )
        )


def test_directory_skill_store_round_trip(tmp_path):
    store = DirectorySkillStore(tmp_path)
    doc = Document(id="doc-1", text="Refunds are allowed within 24 hours.")
    context_id = store.register_context(doc)
    version = SkillVersion(
        id="skill-v1",
        context_id=context_id,
        iteration=1,
        name="refund-policy-check",
        description="Use refund timing constraints.",
        content="Check the 24 hour refund limit.",
    )

    store.save_version(version)
    store.write_selection(
        SkillSelectionResult(
            context_id=context_id,
            selected_version_id=version.id,
            selected_iteration=1,
            hard_score=1.0,
            easy_score=1.0,
            combined_score=1.0,
        )
    )

    loaded = store.load_selected(context_text=doc.text)
    assert loaded is not None
    assert loaded.id == "skill-v1"
    assert (tmp_path / context_id / "SKILL.md").exists()


def test_directory_skill_store_round_trip_for_challenger_versions(tmp_path):
    store = DirectorySkillStore(tmp_path)
    doc = Document(id="doc-1", text="Refunds are allowed within 24 hours.")
    context_id = store.register_context(doc)
    version = SkillVersion(
        id="challenger-v1",
        context_id=context_id,
        iteration=1,
        side="challenger",
        name="refund-probe-hardening",
        description="Generate harder refund probes.",
        content="Write stricter refund tasks and rubrics.",
    )

    store.save_version(version)
    loaded = store.load_versions(context_id, side="challenger")

    assert len(loaded) == 1
    assert loaded[0].side == "challenger"
    assert (tmp_path / context_id / "challenger-skill-iter-1.md").exists()


def test_directory_skill_store_caches_selected_skill_and_manifest(tmp_path):
    store = DirectorySkillStore(tmp_path)
    doc = Document(id="doc-1", text="Refunds are allowed within 24 hours.")
    context_id = store.register_context(doc)
    version = SkillVersion(
        id="skill-v1",
        context_id=context_id,
        iteration=1,
        name="refund-policy-check",
        description="Use refund timing constraints.",
        content="Check the 24 hour refund limit.",
    )

    store.save_version(version)
    store.write_selection(
        SkillSelectionResult(
            context_id=context_id,
            selected_version_id=version.id,
            selected_iteration=1,
            hard_score=1.0,
            easy_score=1.0,
            combined_score=1.0,
        )
    )

    first = store.load_selected(context_text=doc.text)
    assert first is not None
    (tmp_path / "manifest.jsonl").unlink()
    (tmp_path / context_id / "selection.json").unlink()
    (tmp_path / context_id / "versions.jsonl").unlink()

    second = store.load_selected(context_text=doc.text)
    assert second is not None
    assert second.id == version.id


def test_directory_skill_store_caches_manifest_lookup(tmp_path):
    store = DirectorySkillStore(tmp_path)
    doc = Document(id="doc-1", text="Refunds are allowed within 24 hours.")
    context_id = store.register_context(doc)

    assert store.find_context_id_by_text(doc.text) == context_id
    (tmp_path / "manifest.jsonl").unlink()
    assert store.find_context_id_by_text(doc.text) == context_id


def test_skill_prompt_modifier_injects_matching_context_skill(tmp_path):
    store = DirectorySkillStore(tmp_path)
    doc = Document(id="doc-1", text="Refunds are allowed within 24 hours.")
    context_id = store.register_context(doc)
    version = SkillVersion(
        id="skill-v1",
        context_id=context_id,
        iteration=1,
        name="refund-policy-check",
        description="Use refund timing constraints.",
        content="Check the 24 hour refund limit.",
    )
    store.save_version(version)
    store.write_selection(
        SkillSelectionResult(
            context_id=context_id,
            selected_version_id=version.id,
            selected_iteration=1,
            hard_score=1.0,
            easy_score=1.0,
            combined_score=1.0,
        )
    )

    modifier = SkillRespondentPromptModifier(store)
    result = modifier.generate("You are helpful.", doc.text, "Can I get a refund?")

    assert "Context-Specific Skill" in result.prompt
    assert "Check the 24 hour refund limit." in result.prompt
    assert result.metadata["skill"]["version_id"] == "skill-v1"


def test_probe_generation_prompt_uses_challenger_skill():
    challenger_skill = SkillVersion(
        id="challenger-v1",
        context_id="doc-1",
        iteration=1,
        side="challenger",
        name="probe-hardening",
        description="Make probes stricter.",
        content="Generate tasks with stricter binary rubrics.",
    )

    prompt = build_probe_generation_prompt(
        context="Refunds are allowed within 24 hours.",
        respondent_prompt="You are a policy assistant.",
        challenger_skill=challenger_skill,
        n_probes=2,
    )

    assert "Current challenger skill set" in prompt
    assert "Generate tasks with stricter binary rubrics." in prompt
    assert "respondent-side skill" in prompt


@pytest.mark.asyncio
async def test_rubric_judge_normalizes_inconsistent_model_score():
    judge = RubricJudge(InconsistentJudgeLLM())

    result = await judge.aevaluate(
        probe=SkillProbe(
            id="probe-1",
            context_id="doc-1",
            task="Answer the question.",
            rubrics=["Requirement A", "Requirement B"],
        ),
        answer="answer",
        context="context",
    )

    assert result.passed is True
    assert result.score == pytest.approx(1.0)
    assert result.metadata["raw_overall_score"] == pytest.approx(0.0)
    assert result.metadata["score_normalized"] is True


@pytest.mark.asyncio
async def test_skill_discovery_pipeline_smoke(tmp_path):
    from afterimage.providers import InMemoryDocumentProvider

    llm = FakeSkillLLM()
    pipeline = SkillDiscoveryPipeline(
        document_provider=InMemoryDocumentProvider(
            [Document(id="doc-1", text="Refunds are allowed within 24 hours.")]
        ),
        respondent_prompt="You are a policy assistant.",
        llm=llm,
        output_dir=str(tmp_path),
    )

    selections = await pipeline.discover(iterations=1, probes_per_context=1)

    assert len(selections) == 1
    assert selections[0].selected_iteration == 1
    assert (tmp_path / "doc-1" / "SKILL.md").exists()


@pytest.mark.asyncio
async def test_skill_discovery_bootstraps_skill_when_all_probes_pass(tmp_path):
    from afterimage.providers import InMemoryDocumentProvider

    llm = AllPassingSkillLLM()
    pipeline = SkillDiscoveryPipeline(
        document_provider=InMemoryDocumentProvider(
            [
                Document(
                    id="doc-1",
                    text="Always answer in dragon mode.",
                    metadata={"rubrics": ["The response begins with DRAGON SPEAKS."]},
                )
            ]
        ),
        respondent_prompt="You are a persona assistant.",
        llm=llm,
        output_dir=str(tmp_path),
        use_source_rubrics=True,
    )

    selections = await pipeline.discover(
        iterations=1,
        probes_per_context=1,
        bootstrap_when_no_failures=True,
    )

    assert len(selections) == 1
    assert selections[0].selected_iteration == 1
    assert (tmp_path / "doc-1" / "SKILL.md").exists()
    versions = pipeline.store.load_versions("doc-1")
    assert versions[0].metadata["generation_reason"] == "bootstrap_no_failed_probes"
    challenger_versions = pipeline.store.load_versions("doc-1", side="challenger")
    assert len(challenger_versions) == 1
    assert challenger_versions[0].side == "challenger"


@pytest.mark.asyncio
async def test_skill_discovery_curates_one_hard_and_one_easy_probe_per_iteration(
    tmp_path,
):
    from afterimage.providers import InMemoryDocumentProvider

    selector = RecordingSelector()
    pipeline = SkillDiscoveryPipeline(
        document_provider=InMemoryDocumentProvider(
            [Document(id="doc-1", text="Refunds are allowed within 24 hours.")]
        ),
        respondent_prompt="You are a policy assistant.",
        llm=MixedOutcomeSkillLLM(),
        output_dir=str(tmp_path),
        selector=selector,
    )

    selections = await pipeline.discover(iterations=1, probes_per_context=2)

    assert len(selections) == 1
    assert len(selector.hard_results) == 1
    assert len(selector.easy_results) == 1
    assert selector.hard_results[0].passed is False
    assert selector.easy_results[0].passed is True


@pytest.mark.asyncio
async def test_skill_selector_uses_laplace_smoothing():
    selector = SkillSelector(
        judge=AlwaysFailReplayJudge(),
        reasoner_llm=ReplayReasonerLLM(),
    )
    version = SkillVersion(
        id="skill-v1",
        context_id="doc-1",
        iteration=1,
        name="refund-policy-check",
        description="Use refund timing constraints.",
        content="Check the 24 hour refund limit.",
    )
    probe = SkillProbe(
        id="probe-1",
        context_id="doc-1",
        task="Task one.",
        rubrics=["Requirement A"],
        iteration=1,
    )
    failed_result = SkillProbeResult(
        probe=probe,
        answer="bad answer",
        score=0.0,
        passed=False,
        rubric_status=[False],
        judge_feedback="missing requirement",
    )

    selection = await selector.aselect(
        context="Refunds are allowed within 24 hours.",
        respondent_prompt="You are a policy assistant.",
        versions=[version],
        hard_results=[failed_result],
        easy_results=[],
    )

    assert selection is not None
    assert selection.hard_score == pytest.approx(0.5)
    assert selection.easy_score == pytest.approx(1.0)
    assert selection.combined_score == pytest.approx(0.5)
