"""Requirement critic refinement loop."""

import pytest

from afterimage.providers.llm_providers import StructuredLLMResponse
from afterimage.simula.critics import generate_with_requirement_loop
from afterimage.simula.schemas_llm import RequirementCritiqueResponse
from afterimage.simula.types import MetaPrompt, Mix, MixEntry, TaxonomyBundle
from tests.test_simula_types import _toy_factor_and_tree


def _minimal_bundle() -> TaxonomyBundle:
    f, tax = _toy_factor_and_tree()
    return TaxonomyBundle(
        instruction_y="y",
        target_depth_D=2,
        proposal_N=2,
        factors=[f],
        taxonomies=[tax],
    )


@pytest.mark.asyncio
async def test_refinement_then_accept():
    bundle = _minimal_bundle()
    fid = bundle.factors[0].id
    leaf = [n for n in bundle.taxonomies[0].nodes.values() if n.label == "child1"][0]
    mix = Mix(entries=[MixEntry(factor_id=fid, node_id=leaf.id)])
    meta = MetaPrompt(text="ask about child1", mix_id=mix.id)

    verdicts_out = [
        RequirementCritiqueResponse(satisfying=False, explanation="fix json"),
        RequirementCritiqueResponse(satisfying=True, explanation="ok"),
    ]
    refine_out = '{"question":"q","answer":"a"}'
    it_v = iter(verdicts_out)
    refine_once = {"n": 0}

    class FakeLLM:
        async def agenerate_structured(self, prompt, schema, temperature=0.7, **kwargs):
            if schema is RequirementCritiqueResponse:
                return StructuredLLMResponse(
                    text="",
                    prompt_token_count=0,
                    completion_token_count=0,
                    total_token_count=0,
                    finish_reason="stop",
                    model_name="fake",
                    raw_response=None,
                    parsed=next(it_v),
                )
            from afterimage.simula.schemas_llm import RawGenerationResponse

            refine_once["n"] += 1
            return StructuredLLMResponse(
                text="",
                prompt_token_count=0,
                completion_token_count=0,
                total_token_count=0,
                finish_reason="stop",
                model_name="fake",
                raw_response=None,
                parsed=RawGenerationResponse(content=refine_out),
            )

    point, verdicts = await generate_with_requirement_loop(
        FakeLLM(),
        instruction_y=bundle.instruction_y,
        bundle=bundle,
        mix=mix,
        meta=meta,
        initial_serialized='{"question":"bad","answer":"no"}',
        max_refine_rounds=3,
    )
    assert verdicts[-1].satisfying
    assert "question" in point
    assert refine_once["n"] == 1
