"""Global diversity: strategy table and mix sampling (paper §2.2)."""

from __future__ import annotations

import random
from collections import defaultdict

from ..providers.llm_providers import LLMProvider
from .schemas_llm import StrategiesResponse
from .types import (
    FactorTaxonomy,
    Mix,
    MixEntry,
    SamplingStrategySpec,
    StrategyMixRule,
    TaxonomyBundle,
)
from .tree_utils import path_from_root


def leaves_for_factor(tax: FactorTaxonomy) -> list[str]:
    """Node ids with no children (taxonomy leaves)."""
    children: dict[str, list[str]] = defaultdict(list)
    for nid, node in tax.nodes.items():
        if node.parent_id is not None:
            children[node.parent_id].append(nid)
    return [nid for nid in tax.nodes if not children[nid]]


def factor_taxonomy_map(bundle: TaxonomyBundle) -> dict[str, FactorTaxonomy]:
    return {tx.factor_id: tx for tx in bundle.taxonomies}


def mix_factor_paths(bundle: TaxonomyBundle, mix: Mix) -> dict[str, list[str]]:
    """factor_id -> node ids root..leaf for lineage."""
    m = factor_taxonomy_map(bundle)
    out: dict[str, list[str]] = {}
    for e in mix.entries:
        tx = m.get(e.factor_id)
        if tx is None:
            continue
        out[e.factor_id] = path_from_root(tx, e.node_id)
    return out


def sample_mix(
    bundle: TaxonomyBundle,
    spec: SamplingStrategySpec,
    rng: random.Random | None = None,
    *,
    max_resamples: int = 50,
) -> Mix:
    """Sample one mix: pick strategy by weight, then one leaf per listed factor."""
    rng = rng or random.Random()
    strategies = spec.strategies
    if not strategies:
        raise ValueError("SamplingStrategySpec.strategies must be non-empty")
    weights = [s.weight for s in strategies]
    total = sum(weights)
    if total <= 0:
        raise ValueError("Strategy weights must sum to > 0")
    r = rng.random() * total
    acc = 0.0
    chosen: StrategyMixRule | None = None
    for s in strategies:
        acc += s.weight
        if r <= acc:
            chosen = s
            break
    if chosen is None:
        chosen = strategies[-1]

    ftmap = factor_taxonomy_map(bundle)
    for fid in chosen.factor_ids:
        if fid not in ftmap:
            raise ValueError(f"Unknown factor_id in strategy: {fid}")

    for attempt in range(max_resamples):
        entries: list[MixEntry] = []
        ok = True
        for fid in chosen.factor_ids:
            tax = ftmap[fid]
            leaves = leaves_for_factor(tax)
            if not leaves:
                raise ValueError(f"No leaves for factor {fid}")
            nid = rng.choice(leaves)
            label = tax.nodes[nid].label
            entries.append(MixEntry(factor_id=fid, node_id=nid))
        if not chosen.forbidden_label_pairs:
            return Mix(entries=entries, strategy_name=chosen.name)
        labels = [ftmap[e.factor_id].nodes[e.node_id].label for e in entries]
        joined = " ".join(labels).lower()
        for a, b in chosen.forbidden_label_pairs:
            if a.lower() in joined and b.lower() in joined:
                ok = False
                break
        if ok:
            return Mix(entries=entries, strategy_name=chosen.name)

    raise RuntimeError("Could not sample mix satisfying constraints")


async def infer_sampling_strategies(
    llm: LLMProvider,
    bundle: TaxonomyBundle,
    *,
    temperature: float = 0.35,
) -> SamplingStrategySpec:
    """Use M3 to propose compatible joint-sampling strategies (paper §2.2)."""
    lines = []
    for f in bundle.factors:
        lines.append(f"- factor_id={f.id} name={f.name!r}")
    factor_block = "\n".join(lines)
    prompt = (
        f"Dataset instructions y:\n{bundle.instruction_y}\n\n"
        "Available factors (sample joint mixes only from compatible subsets):\n"
        f"{factor_block}\n\n"
        "Define 2–6 strategies. Each strategy lists factor_ids sampled together "
        "(order does not matter). Assign positive weights (not necessarily normalized). "
        "Avoid strategies that join incompatible axes (e.g. mutually exclusive themes)."
    )
    resp = await llm.agenerate_structured(
        prompt=prompt,
        schema=StrategiesResponse,
        temperature=temperature,
    )
    p = resp.parsed
    if len(p.strategy_names) != len(p.strategy_weights) or len(p.strategy_names) != len(
        p.strategy_factor_groups
    ):
        raise ValueError("Strategy arrays from model must have equal length")
    rules: list[StrategyMixRule] = []
    for name, w, group in zip(
        p.strategy_names, p.strategy_weights, p.strategy_factor_groups, strict=True
    ):
        rules.append(StrategyMixRule(name=name, weight=float(w), factor_ids=list(group)))
    return SamplingStrategySpec(strategies=rules)
