"""Reasoning-driven taxonomy construction (paper Appendix B.4)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from ..monitoring import GenerationMonitor
from ..providers import DocumentProvider
from ..providers.llm_providers import LLMProvider
from .document_context import build_bounded_doc_context
from .llm_track import agenerate_structured_tracked
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

logger = logging.getLogger(__name__)


def _taxonomy_status(message: str, *, show_progress: bool) -> None:
    """INFO when not using tqdm; DEBUG when tqdm owns the console."""
    if show_progress:
        logger.debug(message)
    else:
        logger.info(message)


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
        monitor: GenerationMonitor | None = None,
    ):
        self._llm = llm
        self._temperature = temperature
        self._monitor = monitor

    async def build(
        self,
        instruction_y: str,
        *,
        document_provider: DocumentProvider | None = None,
        target_depth_D: int,
        proposal_N: int,
        max_factors: int = 4,
        max_children_per_node: int = 8,
        max_frontier_per_depth: int = 16,
        show_progress: bool = False,
    ) -> TaxonomyBundle:
        """Build taxonomies with bounded cost.

        Without caps, the model may return many factors and wide trees, producing
        hundreds of sequential LLM calls (appearing to "hang"). ``max_factors``,
        ``max_children_per_node``, and ``max_frontier_per_depth`` keep latency
        predictable; raise them for fuller coverage when you accept longer runs.

        When ``show_progress`` is True, renders a :mod:`tqdm` bar (requires tqdm
        installed; already an afterimage dependency) and demotes duplicate log
        lines to DEBUG.
        """
        tqdm_cls: Callable[..., Any] | None = None
        if show_progress:
            from tqdm.auto import tqdm as _tqdm

            tqdm_cls = _tqdm

        doc_ctx = build_bounded_doc_context(document_provider)
        doc_block = doc_ctx.prompt_block()
        digests = digest_documents_for_bundle(list(doc_ctx.blocks))

        _taxonomy_status(
            "OpenSimula taxonomy: proposing factors (1 LLM call)...",
            show_progress=show_progress,
        )
        p0 = (
            tqdm_cls(
                total=1,
                desc="OpenSimula │ propose factors (y,S→fᵢ)",
                unit="call",
                dynamic_ncols=True,
            )
            if tqdm_cls
            else None
        )
        factors_out = await agenerate_structured_tracked(
            self._monitor,
            self._llm,
            operation="opensimula.taxonomy.propose_factors",
            prompt=self._prompt_propose_factors(instruction_y, doc_block),
            schema=FactorsResponse,
            temperature=self._temperature,
        )
        fr = factors_out.parsed
        descriptions = list(fr.factor_descriptions or [])
        raw_names = [name.strip() for name in fr.factors if name.strip()]
        if len(raw_names) > max_factors:
            logger.warning(
                "Factor proposal returned %s factors; keeping first %s (max_factors).",
                len(raw_names),
                max_factors,
            )
            raw_names = raw_names[:max_factors]
        factors: list[SimulaFactor] = []
        for i, name in enumerate(raw_names):
            desc = descriptions[i] if i < len(descriptions) else None
            factors.append(SimulaFactor(name=name, description=desc))
        if p0 is not None:
            p0.update(1)
            p0.close()

        accepted_factors = [f for f in factors if f.accepted]
        _taxonomy_status(
            "OpenSimula taxonomy: expanding %s factor tree(s), D=%s N=%s "
            "max_children=%s max_frontier=%s"
            % (
                len(accepted_factors),
                target_depth_D,
                proposal_N,
                max_children_per_node,
                max_frontier_per_depth,
            ),
            show_progress=show_progress,
        )

        p_trees = (
            tqdm_cls(
                total=len(accepted_factors),
                desc="OpenSimula │ expand factor trees",
                unit="tree",
                dynamic_ncols=True,
            )
            if tqdm_cls and accepted_factors
            else None
        )

        taxonomies: list[FactorTaxonomy] = []
        for fac in factors:
            if not fac.accepted:
                continue
            if p_trees is not None:
                p_trees.set_postfix_str(fac.name[:45], refresh=False)
            tax = await self._expand_factor_tree(
                instruction_y=instruction_y,
                doc_block=doc_block,
                factor=fac,
                target_depth_D=target_depth_D,
                proposal_N=proposal_N,
                max_children_per_node=max_children_per_node,
                max_frontier_per_depth=max_frontier_per_depth,
                show_progress=show_progress,
                tqdm_cls=tqdm_cls,
            )
            taxonomies.append(tax)
            if p_trees is not None:
                p_trees.update(1)
        if p_trees is not None:
            p_trees.close()

        bundle = TaxonomyBundle(
            instruction_y=instruction_y,
            document_digests=digests,
            target_depth_D=target_depth_D,
            proposal_N=proposal_N,
            factors=factors,
            taxonomies=taxonomies,
        )
        _taxonomy_status(
            "OpenSimula taxonomy: finished %s factor tree(s)." % len(bundle.taxonomies),
            show_progress=show_progress,
        )
        return bundle

    def _prompt_propose_factors(self, y: str, doc_block: str) -> str:
        return (
            "You are designing coverage axes for a synthetic dataset.\n"
            f"Dataset instructions (y):\n{y}\n\n"
            "Optional reference excerpts (S):\n"
            f"{doc_block or '(none)'}\n\n"
            "Propose **3 to 5** PRIME factors of variation (independent axes). "
            "Each factor becomes its own hierarchical taxonomy (cost grows with count). "
            "Return concise factor names (2–8 words each). Never more than five factors."
        )

    async def _expand_factor_tree(
        self,
        *,
        instruction_y: str,
        doc_block: str,
        factor: SimulaFactor,
        target_depth_D: int,
        proposal_N: int,
        max_children_per_node: int,
        max_frontier_per_depth: int,
        show_progress: bool = False,
        tqdm_cls: Callable[..., Any] | None = None,
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
            frontier = list(q_curr)
            if len(frontier) > max_frontier_per_depth:
                logger.warning(
                    "OpenSimula taxonomy: factor=%r depth=%s frontier=%s nodes "
                    "(cap %s); expanding first nodes only for bounded runtime.",
                    factor.name,
                    depth,
                    len(frontier),
                    max_frontier_per_depth,
                )
                frontier = frontier[:max_frontier_per_depth]
            _taxonomy_status(
                "OpenSimula taxonomy: factor=%r depth=%s expanding %s node(s)"
                % (factor.name, depth, len(frontier)),
                show_progress=show_progress,
            )
            desc = (
                f"OpenSimula │ {str(factor.name)[:26]} │ depth {depth}/{target_depth_D}"
            )
            node_bar = (
                tqdm_cls(
                    total=len(frontier),
                    desc=desc,
                    unit="node",
                    leave=False,
                    dynamic_ncols=True,
                )
                if tqdm_cls and show_progress and frontier
                else None
            )
            q_next: list[str] = []
            for nid in frontier:
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

                proposal_prompt = (
                    context
                    + "\n\nPropose a diverse set of child categories "
                    "(short labels). At least 3 children unless impossible."
                )

                async def _one_proposal() -> ChildProposalsResponse:
                    resp = await agenerate_structured_tracked(
                        self._monitor,
                        self._llm,
                        operation="opensimula.taxonomy.propose_children",
                        metadata={
                            "factor_id": factor.id,
                            "factor_name": factor.name,
                            "parent_node_id": nid,
                            "depth": depth,
                        },
                        prompt=proposal_prompt,
                        schema=ChildProposalsResponse,
                        temperature=min(0.9, self._temperature + 0.3),
                    )
                    return resp.parsed

                proposal_parts = await asyncio.gather(
                    *(_one_proposal() for _ in range(proposal_N))
                )
                raw_all: list[ChildProposalRaw] = []
                for prop in proposal_parts:
                    for c in prop.children:
                        t = c.strip()
                        if t:
                            raw_all.append(ChildProposalRaw(label=t, description=None))

                crit = await agenerate_structured_tracked(
                    self._monitor,
                    self._llm,
                    operation="opensimula.taxonomy.critic_merge_children",
                    metadata={
                        "factor_id": factor.id,
                        "parent_node_id": nid,
                        "depth": depth,
                    },
                    prompt=(
                        context
                        + "\n\nRaw child proposals (merge near-duplicates, remove bad ones, "
                        "ensure completeness for this parent):\n"
                        + "\n".join(f"- {r.label}" for r in raw_all)
                    ),
                    schema=CriticChildrenResponse,
                    temperature=self._temperature,
                )
                refined_labels = [
                    x.strip() for x in crit.parsed.refined_labels if x.strip()
                ][:max_children_per_node]
                if len(crit.parsed.refined_labels) > max_children_per_node:
                    logger.debug(
                        "Critic returned %s children; truncated to max_children_per_node=%s",
                        len(crit.parsed.refined_labels),
                        max_children_per_node,
                    )
                descs = list(crit.parsed.refined_descriptions or [])[
                    :max_children_per_node
                ]
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
                if node_bar is not None:
                    node_bar.set_postfix_str(nodes[nid].label[:32], refresh=False)
                    node_bar.update(1)
            if node_bar is not None:
                node_bar.close()

            if depth < target_depth_D and q_next:
                if tqdm_cls and show_progress:
                    tqdm_cls.write(
                        f"  OpenSimula │ plan depth {depth}→{depth + 1} "
                        f"({factor.name[:40]}…, {len(q_next)} labels)"
                    )
                labels_block = "\n".join(f"- {nodes[cid].label}" for cid in q_next)
                pl = await agenerate_structured_tracked(
                    self._monitor,
                    self._llm,
                    operation="opensimula.taxonomy.plan_next_level",
                    metadata={"factor_id": factor.id, "depth": depth},
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
