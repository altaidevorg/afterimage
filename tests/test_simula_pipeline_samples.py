"""OpenSimula batched single-QA helpers (no LLM)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from afterimage.simula.pipeline import OpenSimula
from afterimage.simula.types import (
    FactorTaxonomy,
    SamplingStrategySpec,
    SimulaFactor,
    StrategyMixRule,
    TaxonomyBundle,
    TaxonomyNode,
)


def _bundle_and_spec() -> tuple[TaxonomyBundle, SamplingStrategySpec]:
    f = SimulaFactor(name="f", description="d")
    root = TaxonomyNode(
        factor_id=f.id,
        parent_id=None,
        depth=0,
        label="r",
        description="",
    )
    tree = FactorTaxonomy(factor_id=f.id, root_id=root.id, nodes={root.id: root})
    bundle = TaxonomyBundle(
        instruction_y="y",
        target_depth_D=1,
        proposal_N=1,
        factors=[f],
        taxonomies=[tree],
    )
    spec = SamplingStrategySpec(
        strategies=[StrategyMixRule(name="s", weight=1.0, factor_ids=[f.id])],
    )
    return bundle, spec


@pytest.mark.asyncio
async def test_agenerate_single_qa_samples_zero() -> None:
    llm = MagicMock()
    sim = OpenSimula(llm, temperature=0.4)
    bundle, spec = _bundle_and_spec()
    out = await sim.agenerate_single_qa_samples(
        instruction_y=bundle.instruction_y,
        bundle=bundle,
        spec=spec,
        n=0,
    )
    assert out == []


@pytest.mark.asyncio
async def test_agenerate_single_qa_samples_negative_raises() -> None:
    llm = MagicMock()
    sim = OpenSimula(llm, temperature=0.4)
    bundle, spec = _bundle_and_spec()
    with pytest.raises(ValueError, match="non-negative"):
        await sim.agenerate_single_qa_samples(
            instruction_y=bundle.instruction_y,
            bundle=bundle,
            spec=spec,
            n=-1,
        )
