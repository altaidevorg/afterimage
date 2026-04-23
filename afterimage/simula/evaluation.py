"""Taxonomic coverage and calibrated complexity (paper §2.3, Appendix E)."""

from __future__ import annotations

import itertools
import logging
import random
from collections import defaultdict

from ..monitoring import GenerationMonitor
from ..providers.llm_providers import LLMProvider
from .llm_track import agenerate_structured_tracked
from .schemas_llm import PairwiseComparisonBatch, TaxonomyAssignmentResponse
from .types import TaxonomyBundle
from .sampling import factor_taxonomy_map, leaves_for_factor

logger = logging.getLogger(__name__)


def _factor_order(bundle: TaxonomyBundle) -> list[str]:
    return [f.id for f in bundle.factors if f.accepted]


async def assign_datapoint_to_taxonomy(
    llm: LLMProvider,
    *,
    bundle: TaxonomyBundle,
    datapoint_text: str,
    temperature: float = 0.2,
    monitor: GenerationMonitor | None = None,
) -> dict[str, str]:
    """Map each factor to the best-matching leaf node_id (paper §2.3)."""
    fo = _factor_order(bundle)
    ftmap = factor_taxonomy_map(bundle)
    lines = []
    for fid in fo:
        tax = ftmap[fid]
        leaves = leaves_for_factor(tax)
        leaf_labels = "\n".join(
            f"  id={lid} label={tax.nodes[lid].label!r}" for lid in leaves[:200]
        )
        lines.append(f"Factor {fid}:\n{leaf_labels}")
    prompt = (
        f"Dataset instructions y:\n{bundle.instruction_y}\n\n"
        "Taxonomy leaves per factor:\n"
        + "\n\n".join(lines)
        + "\n\nDatapoint to classify:\n"
        f"{datapoint_text}\n\n"
        f"Return exactly {len(fo)} node ids in the same factor order as listed above."
    )
    resp = await agenerate_structured_tracked(
        monitor,
        llm,
        operation="opensimula.eval.assign_datapoint_to_taxonomy",
        prompt=prompt,
        schema=TaxonomyAssignmentResponse,
        temperature=temperature,
    )
    ids = list(resp.parsed.assignments)
    if len(ids) != len(fo):
        raise ValueError(
            f"Expected {len(fo)} taxonomy assignments, got {len(ids)}"
        )
    return {fo[i]: ids[i] for i in range(len(fo))}


def level_ratio_coverage(
    bundle: TaxonomyBundle,
    assignments: list[dict[str, str]],
) -> dict[str, dict[int, float]]:
    """Per factor_id, depth level -> fraction of unique nodes at that depth covered."""
    ftmap = factor_taxonomy_map(bundle)
    out: dict[str, dict[int, float]] = {}
    for fid, tax in ftmap.items():
        by_depth: dict[int, set[str]] = defaultdict(set)
        for nid in tax.nodes.values():
            by_depth[nid.depth].add(nid.id)
        covered: dict[int, set[str]] = defaultdict(set)
        for assign in assignments:
            nid = assign.get(fid)
            if not nid or nid not in tax.nodes:
                continue
            d = tax.nodes[nid].depth
            covered[d].add(nid)
        ratios: dict[int, float] = {}
        for d, universe in by_depth.items():
            if not universe:
                continue
            ratios[d] = len(covered[d]) / len(universe)
        out[fid] = ratios
    return out


def _expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def _update_elo_pair(
    ratings: dict[int, float],
    winner: int,
    loser: int,
    k: float = 32.0,
) -> None:
    rw, rl = ratings[winner], ratings[loser]
    ew = _expected_score(rw, rl)
    el = _expected_score(rl, rw)
    ratings[winner] = rw + k * (1.0 - ew)
    ratings[loser] = rl + k * (0.0 - el)


async def elo_complexity_scores(
    llm: LLMProvider,
    *,
    instruction_y: str,
    texts: list[str],
    batch_size: int = 5,
    repeats: int = 3,
    temperature: float = 0.2,
    k_elo: float = 32.0,
    monitor: GenerationMonitor | None = None,
) -> dict[int, float]:
    """Batch-wise orderings → Elo ratings per item index (paper Appendix E style)."""
    n = len(texts)
    if n == 0:
        return {}
    ratings = {i: 1500.0 for i in range(n)}
    rng = random.Random(42)
    for repeat_idx in range(repeats):
        perm = list(range(n))
        rng.shuffle(perm)
        for start in range(0, n, batch_size):
            batch_idx = perm[start : start + batch_size]
            if len(batch_idx) < 2:
                continue
            block = "\n\n".join(f"[{j}] {texts[j]}" for j in batch_idx)
            k = len(batch_idx)
            prompt = (
                f"Dataset / task description:\n{instruction_y}\n\n"
                "Items are listed below in batch order as [0], [1], ... up to ["
                f"{k - 1}]. Order these positions from EASIEST to HARDEST by reasoning "
                "complexity for a capable student model.\n\n"
                f"{block}\n\n"
                f"Return exactly {k} integers, each in 0..{k - 1}, sorted easiest-to-hardest."
            )
            resp = await agenerate_structured_tracked(
                monitor,
                llm,
                operation="opensimula.eval.elo_complexity_batch",
                metadata={"batch_start": start, "repeat": repeat_idx},
                prompt=prompt,
                schema=PairwiseComparisonBatch,
                temperature=temperature,
            )
            order = [batch_idx[i] for i in resp.parsed.ordering if 0 <= i < k]
            if len(order) < 2:
                continue
            for a, b in zip(order[:-1], order[1:], strict=False):
                _update_elo_pair(ratings, b, a, k=k_elo)
    return ratings
