"""Taxonomy bundle schema and tree validation."""

import pytest

from afterimage.simula.tree_utils import path_from_root
from afterimage.simula.types import (
    ChildProposalRaw,
    ExpansionStepTrace,
    FactorTaxonomy,
    SimulaFactor,
    TaxonomyBundle,
    TaxonomyNode,
    digest_documents_for_bundle,
    validate_factor_taxonomy,
)


def _toy_factor_and_tree() -> tuple[SimulaFactor, FactorTaxonomy]:
    f = SimulaFactor(name="topic", description=None)
    root = TaxonomyNode(
        id="r1",
        factor_id=f.id,
        parent_id=None,
        depth=0,
        label="topic",
    )
    c1 = TaxonomyNode(
        id="c1",
        factor_id=f.id,
        parent_id=root.id,
        depth=1,
        label="child1",
    )
    c2 = TaxonomyNode(
        id="c2",
        factor_id=f.id,
        parent_id=root.id,
        depth=1,
        label="child2",
    )
    nodes = {root.id: root, c1.id: c1, c2.id: c2}
    trace = ExpansionStepTrace(
        parent_node_id=root.id,
        depth=1,
        raw_proposals=[ChildProposalRaw(label="child1")],
        children_after_critic=[
            ChildProposalRaw(label="child1"),
            ChildProposalRaw(label="child2"),
        ],
    )
    tax = FactorTaxonomy(
        factor_id=f.id,
        root_id=root.id,
        nodes=nodes,
        expansion_traces=[trace],
        per_depth_plans=["next"],
    )
    return f, tax


def test_taxonomy_bundle_json_roundtrip():
    f, tax = _toy_factor_and_tree()
    b = TaxonomyBundle(
        instruction_y="test dataset",
        document_digests=digest_documents_for_bundle([("d1", "hello")]),
        target_depth_D=2,
        proposal_N=3,
        factors=[f],
        taxonomies=[tax],
    )
    b2 = b.model_dump_json_roundtrip()
    assert b2.instruction_y == b.instruction_y
    assert len(b2.taxonomies[0].nodes) == 3


def test_validate_factor_taxonomy_ok():
    _, tax = _toy_factor_and_tree()
    validate_factor_taxonomy(tax)


def test_validate_rejects_orphan():
    f = SimulaFactor(name="t")
    root = TaxonomyNode(
        id="r", factor_id=f.id, parent_id=None, depth=0, label="r"
    )
    bad = TaxonomyNode(
        id="x", factor_id=f.id, parent_id="missing", depth=1, label="x"
    )
    with pytest.raises(ValueError):
        validate_factor_taxonomy(
            FactorTaxonomy(
                factor_id=f.id,
                root_id=root.id,
                nodes={root.id: root, bad.id: bad},
            )
        )


def test_path_from_root():
    _, t = _toy_factor_and_tree()
    p = path_from_root(t, "c1")
    assert p[0] == t.root_id
    assert p[-1] == "c1"
