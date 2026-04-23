"""Reasoning-driven taxonomy construction (paper Appendix B.4)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..providers import DocumentProvider
from ..providers.llm_providers import LLMProvider
from .document_context import build_bounded_doc_context
from .schemas_llm import (
    ChildProposalsResponse,
    CriticChildrenResponse,
    FactorsResponse,
    PlanNextLevelResponse,
)
from .types import (
    ChildProposalRaw,
    ExpansionStepTrace,
    FactorTaxonomy,
    SimulaFactor,
    TaxonomyBundle,
    TaxonomyNode,
    digest_documents_for_bundle,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _ancestors(nodes: dict[str, TaxonomyNode], node_id: str) -> list[TaxonomyNode]:
    out: list[TaxonomyNode] = []
    cur: str | None = node_id
    while cur is not None:
        n = nodes[cur]
        out.append(n)
        cur = n.parent_id
    out.reverse()
    return out


def _sibling_labels(nodes: dict[str, TaxonomyNode], node_id: str) -> list[str]:
    n = nodes[node_id]
    if n.parent_id is None:
        return []
    return [
        nodes[cid].label
        for cid, cn in nodes.items()
        if cn.parent_id == n.parent_id and cid != node_id
    ]


class TaxonomyBuilder:
    """Builds :class:`~afterimage.simula.types.TaxonomyBundle` via propose–critic–plan loops."""

    def __init__(
        self,
        llm: LLMProvider,
        *,
        temperature: float = 0.4,
    ):
        self._llm = llm
        self._temperature = temperature

    async def build(
        self,
        instruction_y: str,
        *,
        document_provider: DocumentProvider | None = None,
        target_depth_D: int,
        proposal_N: int,
    ) -> TaxonomyBundle:
        doc_ctx = build_bounded_doc_context(document_provider)
        doc_block = doc_ctx.prompt_block()
        digests = digest_documents_for_bundle(list(doc_ctx.blocks))

        factors_out = await self._llm.agenerate_structured(
            prompt=self._prompt_propose_factors(instruction_y, doc_block),
            schema=FactorsResponse,
            temperature=self._temperature,
        )
        fr = factors_out.parsed
        descriptions = list(fr.factor_descriptions or [])
        factors: list[SimulaFactor] = []
        for i, name in enumerate(fr.factors):
            desc = descriptions[i] if i < len(descriptions) else None
            factors.append(SimulaFactor(name=name.strip(), description=desc))

        taxonomies: list[FactorTaxonomy] = []
        for fac in factors:
            if not fac.accepted:
                continue
            tax = await self._expand_factor_tree(
                instruction_y=instruction_y,
                doc_block=doc_block,
                factor=fac,
                target_depth_D=target_depth_D,
                proposal_N=proposal_N,
            )
            taxonomies.append(tax)

        return TaxonomyBundle(
            instruction_y=instruction_y,
            document_digests=digests,
            target_depth_D=target_depth_D,
            proposal_N=proposal_N,
            factors=factors,
            taxonomies=taxonomies,
        )

    def _prompt_propose_factors(self, y: str, doc_block: str) -> str:
        return (
            "You are designing coverage axes for a synthetic dataset.\n"
            f"Dataset instructions (y):\n{y}\n\n"
            "Optional reference excerpts (S):\n"
            f"{doc_block or '(none)'}\n\n"
            "Propose a small set of PRIME factors of variation (independent axes). "
            "Each factor will become its own hierarchical taxonomy. "
            "Return concise factor names (2–8 words each)."
        )

    async def _expand_factor_tree(
        self,
        *,
        instruction_y: str,
        doc_block: str,
        factor: SimulaFactor,
        target_depth_D: int,
        proposal_N: int,
    ) -> FactorTaxonomy:
        root = TaxonomyNode(
            factor_id=factor.id,
            parent_id=None,
            depth=0,
            label=factor.name,
            description=factor.description,
        )
        nodes: dict[str, TaxonomyNode] = {root.id: root}
        traces: list[ExpansionStepTrace] = []
        depth_plans: list[str] = []
        plan = (
            f"Expand the taxonomy for factor '{factor.name}' based on y and this factor. "
            "Prefer mutually exclusive children; cover the long tail."
        )
        q_curr: list[str] = [root.id]

        for depth in range(1, target_depth_D + 1):
            q_next: list[str] = []
            for nid in q_curr:
                anc = _ancestors(nodes, nid)
                sib_labels = _sibling_labels(nodes, nid)
                ctx_lines = [
                    f"Dataset instructions y:\n{instruction_y}",
                    f"Reference excerpts:\n{doc_block or '(none)'}",
                    f"Current factor: {factor.name} ({factor.description or ''})",
                    f"Expansion plan for this level: {plan}",
                    "Ancestors (root to parent): "
                    + " > ".join(a.label for a in anc[:-1])
                    if len(anc) > 1
                    else "Ancestors: (root)",
                    f"Expand children of node label: {nodes[nid].label}",
                    f"Sibling labels under same parent (avoid duplicates): {sib_labels}",
                ]
                context = "\n\n".join(ctx_lines)

                raw_all: list[ChildProposalRaw] = []
                for _ in range(proposal_N):
                    prop = await self._llm.agenerate_structured(
                        prompt=(
                            context
                            + "\n\nPropose a diverse set of child categories "
                            "(short labels). At least 3 children unless impossible."
                        ),
                        schema=ChildProposalsResponse,
                        temperature=min(0.9, self._temperature + 0.3),
                    )
                    for c in prop.parsed.children:
                        raw_all.append(ChildProposalRaw(label=c.strip(), description=None))

                crit = await self._llm.agenerate_structured(
                    prompt=(
                        context
                        + "\n\nRaw child proposals (merge near-duplicates, remove bad ones, "
                        "ensure completeness for this parent):\n"
                        + "\n".join(f"- {r.label}" for r in raw_all)
                    ),
                    schema=CriticChildrenResponse,
                    temperature=self._temperature,
                )
                refined_labels = [x.strip() for x in crit.parsed.refined_labels if x.strip()]
                descs = list(crit.parsed.refined_descriptions or [])
                refined_children: list[ChildProposalRaw] = []
                for i, lab in enumerate(refined_labels):
                    d = descs[i] if i < len(descs) else None
                    refined_children.append(ChildProposalRaw(label=lab, description=d))

                step = ExpansionStepTrace(
                    parent_node_id=nid,
                    depth=depth,
                    raw_proposals=raw_all,
                    children_after_critic=refined_children,
                    plan_for_next_level=None,
                )

                parent = nodes[nid]
                for ch in refined_children:
                    child = TaxonomyNode(
                        factor_id=factor.id,
                        parent_id=nid,
                        depth=parent.depth + 1,
                        label=ch.label,
                        description=ch.description,
                    )
                    nodes[child.id] = child
                    q_next.append(child.id)

                traces.append(step)

            if depth < target_depth_D and q_next:
                labels_block = "\n".join(f"- {nodes[cid].label}" for cid in q_next)
                pl = await self._llm.agenerate_structured(
                    prompt=(
                        f"Dataset instructions y:\n{instruction_y}\n\n"
                        f"Factor: {factor.name}\n\n"
                        "New category labels at this depth:\n"
                        f"{labels_block}\n\n"
                        "Write a short strategic plan for the NEXT depth: desired granularity, "
                        "what branches to refine, and what to avoid."
                    ),
                    schema=PlanNextLevelResponse,
                    temperature=self._temperature,
                )
                plan = pl.parsed.plan
                depth_plans.append(plan)

            q_curr = q_next
            if not q_curr:
                logger.warning(
                    "Taxonomy expansion stopped early: no children at depth %s for factor %s",
                    depth,
                    factor.name,
                )
                break

        return FactorTaxonomy(
            factor_id=factor.id,
            root_id=root.id,
            nodes=nodes,
            expansion_traces=traces,
            per_depth_plans=depth_plans,
        )
