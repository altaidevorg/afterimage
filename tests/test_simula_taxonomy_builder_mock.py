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


def _sr(parsed):
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


@pytest.mark.asyncio
async def test_taxonomy_builder_deterministic_depth2():
    """D=2, N=1: minimal structured responses (prompt/schema dispatch; frontier is parallel)."""

    class FakeLLM:
        async def agenerate_structured(self, prompt, schema, temperature=0.7, **kwargs):
            if schema is FactorsResponse:
                return _sr(
                    FactorsResponse(factors=["Axis"], factor_descriptions=["d"]),
                )
            if schema is PlanNextLevelResponse:
                return _sr(PlanNextLevelResponse(plan="deeper"))
            if schema is ChildProposalsResponse:
                if "Expand children of node label: u" in prompt:
                    return _sr(ChildProposalsResponse(children=["u1"]))
                if "Expand children of node label: v" in prompt:
                    return _sr(ChildProposalsResponse(children=["v1"]))
                return _sr(ChildProposalsResponse(children=["u", "v"]))
            if schema is CriticChildrenResponse:
                raw_tail = prompt.split("Raw child proposals", 1)[-1]
                if "- u1" in raw_tail and "- v1" not in raw_tail:
                    return _sr(
                        CriticChildrenResponse(
                            refined_labels=["u1"], refined_descriptions=[]
                        ),
                    )
                if "- v1" in raw_tail and "- u1" not in raw_tail:
                    return _sr(
                        CriticChildrenResponse(
                            refined_labels=["v1"], refined_descriptions=[]
                        ),
                    )
                return _sr(
                    CriticChildrenResponse(
                        refined_labels=["u", "v"], refined_descriptions=[]
                    ),
                )
            raise AssertionError(f"unexpected schema {schema}")

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
