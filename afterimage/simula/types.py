"""Pydantic models for OpenSimula (taxonomy bundle, mixes, meta-prompts, dataset rows).

Independent open implementation inspired by Davidson et al., TMLR (Simula); not affiliated with Google.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


def _new_id(prefix: str = "n") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class DocumentProvenance(BaseModel):
    """Audit: which document text supported a node or expansion."""

    document_id: str
    excerpt_sha256: str = Field(
        ...,
        description="SHA256 of the excerpt bytes included in the prompt for this provenance.",
    )
    char_start: int | None = None
    char_end: int | None = None


class SimulaFactor(BaseModel):
    """Prime factor of variation f_i (paper §2.1)."""

    id: str = Field(default_factory=lambda: _new_id("f"))
    name: str
    description: str | None = None
    accepted: bool = True


class TaxonomyNode(BaseModel):
    """One node in a factor taxonomy tree."""

    id: str = Field(default_factory=lambda: _new_id("node"))
    factor_id: str
    parent_id: str | None = None
    depth: int = 0
    label: str
    description: str | None = None
    provenance: list[DocumentProvenance] = Field(default_factory=list)


class ChildProposalRaw(BaseModel):
    """Single raw child label from Best-of-N proposal step."""

    label: str
    description: str | None = None


class ExpansionStepTrace(BaseModel):
    """Full trace for expanding one parent at one depth (Appendix B.4)."""

    parent_node_id: str
    depth: int
    raw_proposals: list[ChildProposalRaw] = Field(
        default_factory=list,
        description="Concatenation of up to N separate proposal calls (Best-of-N).",
    )
    children_after_critic: list[ChildProposalRaw] = Field(
        default_factory=list,
        description="Child set after critic merge/edit pass.",
    )
    plan_for_next_level: str | None = Field(
        None,
        description="Guidance M3 generates after finishing this depth (paper: optional per-level plan).",
    )


class FactorTaxonomy(BaseModel):
    """One tree T_i for factor f_i."""

    factor_id: str
    root_id: str
    nodes: dict[str, TaxonomyNode] = Field(default_factory=dict)
    expansion_traces: list[ExpansionStepTrace] = Field(
        default_factory=list,
        description="Ordered log of BFS expansion steps for auditability.",
    )
    per_depth_plans: list[str] = Field(
        default_factory=list,
        description="Plan text after completing each depth (length <= target depth).",
    )


def validate_factor_taxonomy(t: FactorTaxonomy) -> None:
    """Invariant checks: single root, consistent depths and parent links."""
    if not t.root_id or t.root_id not in t.nodes:
        raise ValueError("FactorTaxonomy: root_id must exist in nodes")
    root = t.nodes[t.root_id]
    if root.parent_id is not None:
        raise ValueError("Root node must have parent_id None")
    if root.depth != 0:
        raise ValueError("Root depth must be 0")
    if root.factor_id != t.factor_id:
        raise ValueError("Root factor_id must match taxonomy factor_id")
    seen: set[str] = set()
    stack = [t.root_id]
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        node = t.nodes.get(nid)
        if node is None:
            raise ValueError(f"Missing node {nid}")
        for cid, child in t.nodes.items():
            if child.parent_id == nid:
                stack.append(cid)
    if len(seen) != len(t.nodes):
        raise ValueError("Unreachable nodes present (not connected to root)")
    for nid, node in t.nodes.items():
        if node.factor_id != t.factor_id:
            raise ValueError(f"Node {nid} factor_id mismatch")
        if node.parent_id is None:
            if nid != t.root_id:
                raise ValueError("Multiple roots: only root may have parent_id None")
            continue
        parent = t.nodes.get(node.parent_id)
        if parent is None:
            raise ValueError(f"Node {nid} references missing parent {node.parent_id}")
        if node.depth != parent.depth + 1:
            raise ValueError(f"Node {nid} depth inconsistent with parent")


class TaxonomyBundle(BaseModel):
    """Serialized conceptual scaffold (paper §2.1)."""

    bundle_version: str = Field(
        "1",
        description="Format version for serialization compatibility.",
    )
    instruction_y: str = Field(..., description="Dataset / domain instructions y.")
    document_digests: list[str] = Field(
        default_factory=list,
        description="SHA256 hex digests of bounded doc excerpts fed into construction.",
    )
    target_depth_D: int = Field(..., ge=1, description="Target taxonomy depth D.")
    proposal_N: int = Field(
        ...,
        ge=1,
        description="Best-of-N: proposals per expansion step (paper N).",
    )
    factors: list[SimulaFactor] = Field(default_factory=list)
    taxonomies: list[FactorTaxonomy] = Field(default_factory=list)

    def model_dump_json_roundtrip(self) -> TaxonomyBundle:
        raw = self.model_dump(mode="json")
        return TaxonomyBundle.model_validate(raw)

    def save(self, checkpointer: object) -> None:
        """Write this bundle into ``checkpointer`` (a :class:`~afterimage.simula.checkpoint.Checkpointer`)."""
        from afterimage.simula.checkpoint import Checkpointer

        if not isinstance(checkpointer, Checkpointer):
            raise TypeError(f"expected Checkpointer, got {type(checkpointer).__name__}")
        checkpointer.write_taxonomy_bundle(self)


class StrategyMixRule(BaseModel):
    """One named sampling strategy: compatible factor subset + weight."""

    name: str
    weight: float = Field(..., gt=0)
    factor_ids: list[str] = Field(
        ...,
        min_length=1,
        description="Factors jointly sampled in this strategy (paper: compatible subsets).",
    )
    forbidden_label_pairs: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Optional (label_substring_a, label_substring_b) pairs to reject in a mix.",
    )


class SamplingStrategySpec(BaseModel):
    """Weighted strategies over taxonomies (paper §2.2)."""

    strategies: list[StrategyMixRule]

    def save(self, checkpointer: object) -> None:
        """Write this spec into ``checkpointer`` (a :class:`~afterimage.simula.checkpoint.Checkpointer`)."""
        from afterimage.simula.checkpoint import Checkpointer

        if not isinstance(checkpointer, Checkpointer):
            raise TypeError(f"expected Checkpointer, got {type(checkpointer).__name__}")
        checkpointer.write_sampling_strategy(self)


class MixEntry(BaseModel):
    factor_id: str
    node_id: str


class Mix(BaseModel):
    """One combination of sampled nodes (requirements for meta-prompts)."""

    id: str = Field(default_factory=lambda: _new_id("mix"))
    entries: list[MixEntry] = Field(default_factory=list)
    strategy_name: str | None = None


class MetaPrompt(BaseModel):
    """Scenario / meta-prompt derived from (y, mix) (paper §2.2)."""

    id: str = Field(default_factory=lambda: _new_id("meta"))
    text: str
    mix_id: str
    complexified: bool = False
    sequential_attempt_index: int | None = None


class RequirementCritiqueVerdict(BaseModel):
    satisfying: bool
    explanation: str


class DoubleCritiqueVerdict(BaseModel):
    """Independent correct / incorrect probes (paper §2.2, §3.1)."""

    claims_correct: bool
    claims_incorrect: bool
    rationale_correct: str = ""
    rationale_incorrect: str = ""


class SingleQARow(BaseModel):
    question: str
    answer: str


class MCQRow(BaseModel):
    question: str
    choices: list[str]
    correct_index: int = Field(..., ge=0)


class DataPointLineage(BaseModel):
    """Traceability (plan acceptance criterion)."""

    instruction_y: str
    mix_id: str
    meta_prompt_id: str
    factor_paths: dict[str, list[str]] = Field(
        default_factory=dict,
        description="factor_id -> list of node ids from root to chosen leaf.",
    )
    expansion_trace_ids: list[str] = Field(default_factory=list)
    requirement_critiques: list[RequirementCritiqueVerdict] = Field(
        default_factory=list
    )
    double_critique: DoubleCritiqueVerdict | None = None


class DataPointRecord(BaseModel):
    """One accepted synthetic datapoint with lineage."""

    task: Literal["single_qa", "mcq", "raw"]
    payload: dict[str, Any]
    lineage: DataPointLineage


class DatasetBatch(BaseModel):
    """In-memory batch of accepted points."""

    records: list[DataPointRecord] = Field(default_factory=list)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_documents_for_bundle(
    doc_parts: list[tuple[str, str]],
) -> list[str]:
    """Build stable digests from (document_id, excerpt) pairs."""
    out: list[str] = []
    for doc_id, excerpt in doc_parts:
        payload = json.dumps({"id": doc_id, "excerpt": excerpt}, sort_keys=True)
        out.append(hashlib.sha256(payload.encode("utf-8")).hexdigest())
    return out
