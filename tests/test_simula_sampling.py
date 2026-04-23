"""Mix sampling from strategies."""

import random

import pytest

from afterimage.simula.sampling import sample_mix
from afterimage.simula.types import (
    FactorTaxonomy,
    SamplingStrategySpec,
    SimulaFactor,
    StrategyMixRule,
    TaxonomyBundle,
    TaxonomyNode,
)


def _bundle_two_factors() -> TaxonomyBundle:
    fa = SimulaFactor(name="A")
    fb = SimulaFactor(name="B")
    ta = _linear_tax(fa.id, "Aroot", ["a1", "a2"])
    tb = _linear_tax(fb.id, "Broot", ["b1"])
    return TaxonomyBundle(
        instruction_y="y",
        target_depth_D=2,
        proposal_N=2,
        factors=[fa, fb],
        taxonomies=[ta, tb],
    )


def _linear_tax(factor_id: str, root_label: str, leaf_labels: list[str]) -> FactorTaxonomy:
    root = TaxonomyNode(
        id="r",
        factor_id=factor_id,
        parent_id=None,
        depth=0,
        label=root_label,
    )
    nodes: dict[str, TaxonomyNode] = {root.id: root}
    for i, lab in enumerate(leaf_labels):
        n = TaxonomyNode(
            id=f"l{i}",
            factor_id=factor_id,
            parent_id=root.id,
            depth=1,
            label=lab,
        )
        nodes[n.id] = n
    return FactorTaxonomy(
        factor_id=factor_id,
        root_id=root.id,
        nodes=nodes,
    )


def test_sample_mix_weights():
    bundle = _bundle_two_factors()
    fa_id = bundle.factors[0].id
    fb_id = bundle.factors[1].id
    spec = SamplingStrategySpec(
        strategies=[
            StrategyMixRule(name="only_a", weight=1.0, factor_ids=[fa_id]),
            StrategyMixRule(name="both", weight=3.0, factor_ids=[fa_id, fb_id]),
        ]
    )
    rng = random.Random(0)
    counts = {"only_a": 0, "both": 0}
    for _ in range(2000):
        m = sample_mix(bundle, spec, rng=rng)
        assert m.strategy_name in ("only_a", "both")
        counts[m.strategy_name] += 1
    assert counts["both"] > counts["only_a"]


def test_forbidden_pair_resample():
    bundle = _bundle_two_factors()
    fa_id = bundle.factors[0].id
    fb_id = bundle.factors[1].id
    spec = SamplingStrategySpec(
        strategies=[
            StrategyMixRule(
                name="both",
                weight=1.0,
                factor_ids=[fa_id, fb_id],
                forbidden_label_pairs=[("a1", "b1")],
            ),
        ]
    )
    rng = random.Random(42)
    for _ in range(30):
        m = sample_mix(bundle, spec, rng=rng)
        labels = [
            bundle.taxonomies[0].nodes[e.node_id].label
            if e.factor_id == fa_id
            else bundle.taxonomies[1].nodes[e.node_id].label
            for e in m.entries
        ]
        joined = " ".join(labels).lower()
        assert not ("a1" in joined and "b1" in joined)
