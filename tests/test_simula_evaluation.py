"""Taxonomic coverage metrics."""

import pytest

from afterimage.simula.evaluation import level_ratio_coverage
from afterimage.simula.types import (
    FactorTaxonomy,
    SimulaFactor,
    TaxonomyBundle,
    TaxonomyNode,
)


def _tiny_bundle() -> TaxonomyBundle:
    f = SimulaFactor(name="F")
    root = TaxonomyNode(id="r", factor_id=f.id, parent_id=None, depth=0, label="root")
    a = TaxonomyNode(id="a", factor_id=f.id, parent_id=root.id, depth=1, label="A")
    b = TaxonomyNode(id="b", factor_id=f.id, parent_id=root.id, depth=1, label="B")
    tax = FactorTaxonomy(
        factor_id=f.id,
        root_id=root.id,
        nodes={root.id: root, a.id: a, b.id: b},
    )
    return TaxonomyBundle(
        instruction_y="y",
        target_depth_D=2,
        proposal_N=2,
        factors=[f],
        taxonomies=[tax],
    )


def test_level_ratio_coverage_full():
    bundle = _tiny_bundle()
    fid = bundle.factors[0].id
    assignments = [{fid: "a"}, {fid: "b"}]
    cov = level_ratio_coverage(bundle, assignments)
    assert cov[fid][1] == 1.0


def test_level_ratio_coverage_partial():
    bundle = _tiny_bundle()
    fid = bundle.factors[0].id
    assignments = [{fid: "a"}, {fid: "a"}]
    cov = level_ratio_coverage(bundle, assignments)
    assert cov[fid][1] == 0.5


@pytest.mark.asyncio
async def test_elo_scores_mock():
    from afterimage.providers.llm_providers import StructuredLLMResponse
    from afterimage.simula.evaluation import elo_complexity_scores
    from afterimage.simula.schemas_llm import PairwiseComparisonBatch

    class Fake:
        n = 0

        async def agenerate_structured(self, prompt, schema, temperature=0.7, **kwargs):
            Fake.n += 1
            return StructuredLLMResponse(
                text="",
                prompt_token_count=0,
                completion_token_count=0,
                total_token_count=0,
                finish_reason="stop",
                model_name="fake",
                raw_response=None,
                parsed=PairwiseComparisonBatch(ordering=[0, 1, 2]),
            )

    scores = await elo_complexity_scores(
        Fake(),
        instruction_y="math",
        texts=["easy", "mid", "hard"],
        batch_size=3,
        repeats=1,
    )
    assert scores[2] > scores[0]
