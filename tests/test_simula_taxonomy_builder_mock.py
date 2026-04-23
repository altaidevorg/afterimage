"""TaxonomyBuilder with mocked LLM (Appendix B.4)."""

import pytest

from afterimage.providers.llm_providers import StructuredLLMResponse
from afterimage.simula.schemas_llm import (
    ChildProposalsResponse,
    CriticChildrenResponse,
    FactorsResponse,
    PlanNextLevelResponse,
)
from afterimage.simula.taxonomy_builder import TaxonomyBuilder
from afterimage.simula.types import validate_factor_taxonomy


@pytest.mark.asyncio
async def test_taxonomy_builder_deterministic_depth2():
    """D=2, N=1: minimal sequence of structured responses."""
    seq = [
        FactorsResponse(factors=["Axis"], factor_descriptions=["d"]),
        ChildProposalsResponse(children=["u", "v"]),
        CriticChildrenResponse(refined_labels=["u", "v"], refined_descriptions=[]),
        PlanNextLevelResponse(plan="deeper"),
        ChildProposalsResponse(children=["u1"]),
        CriticChildrenResponse(refined_labels=["u1"], refined_descriptions=[]),
        ChildProposalsResponse(children=["v1"]),
        CriticChildrenResponse(refined_labels=["v1"], refined_descriptions=[]),
    ]
    it = iter(seq)

    class FakeLLM:
        async def agenerate_structured(self, prompt, schema, temperature=0.7, **kwargs):
            item = next(it)
            return StructuredLLMResponse(
                text="",
                prompt_token_count=0,
                completion_token_count=0,
                total_token_count=0,
                finish_reason="stop",
                model_name="fake",
                raw_response=None,
                parsed=item,
            )

    b = TaxonomyBuilder(FakeLLM(), temperature=0.1)
    bundle = await b.build(
        "synthetic widgets",
        document_provider=None,
        target_depth_D=2,
        proposal_N=1,
    )
    assert len(bundle.factors) == 1
    assert len(bundle.taxonomies) == 1
    tax = bundle.taxonomies[0]
    validate_factor_taxonomy(tax)
    assert len(tax.per_depth_plans) == 1
    leaves = [n for n in tax.nodes.values() if n.depth == 2]
    assert len(leaves) == 2
