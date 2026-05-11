from __future__ import annotations

import pytest

from afterimage.providers.llm_providers import LLMResponse, StructuredLLMResponse
from afterimage.skills import (
    DirectorySkillStore,
    SkillDiscoveryPipeline,
    SkillRespondentPromptModifier,
)
from afterimage.skills.schemas import (
    ProbeGenerationResponse,
    ProbeSpec,
    RubricJudgeResponse,
    SkillContentResponse,
    SkillProposalResponse,
)
from afterimage.skills.types import SkillSelectionResult, SkillVersion
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
        if schema is SkillContentResponse:
            assert "No failed probe was found" in prompt
            return _structured(
                SkillContentResponse(
                    name="persona-format-rules",
                    description="Preserve context persona and formatting rules.",
                    content="Use the required persona header and opening sentence.",
                )
            )
        if schema is SkillProposalResponse:
            raise AssertionError("No proposal should be needed when all probes pass")
        raise AssertionError(f"Unexpected schema: {schema}")


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
    )

    selections = await pipeline.discover(iterations=1, probes_per_context=1)

    assert len(selections) == 1
    assert selections[0].selected_iteration == 1
    assert (tmp_path / "doc-1" / "SKILL.md").exists()
    versions = pipeline.store.load_versions("doc-1")
    assert versions[0].metadata["generation_reason"] == "bootstrap_no_failed_probes"
