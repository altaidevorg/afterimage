"""Local diversity and complexification (paper §2.2, Algorithm 2 steps 2–3)."""

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING

from ..providers.llm_providers import LLMProvider
from .sampling import factor_taxonomy_map, mix_factor_paths
from .schemas_llm import ComplexifyResponse, ScenariosResponse
from .types import MetaPrompt, Mix, TaxonomyBundle

if TYPE_CHECKING:
    pass


def _mix_description(bundle: TaxonomyBundle, mix: Mix) -> str:
    ft = factor_taxonomy_map(bundle)
    parts: list[str] = []
    for e in mix.entries:
        tax = ft.get(e.factor_id)
        if tax is None:
            continue
        node = tax.nodes[e.node_id]
        parts.append(f"[{node.factor_id}] {node.label}")
    return "; ".join(parts)


async def generate_scenarios(
    llm: LLMProvider,
    *,
    instruction_y: str,
    bundle: TaxonomyBundle,
    mix: Mix,
    K: int,
    temperature: float = 0.65,
) -> list[MetaPrompt]:
    """Produce K distinct meta-prompts for the same mix (local diversity)."""
    if K < 1:
        raise ValueError("K must be >= 1")
    mix_desc = _mix_description(bundle, mix)
    prompt = (
        f"Dataset instructions y:\n{instruction_y}\n\n"
        "Sampled taxonomy requirements (mix):\n"
        f"{mix_desc}\n\n"
        f"Write exactly {K} DISTINCT scenario descriptions (meta-prompts). "
        "Each should be a concrete generation brief for one synthetic datapoint that "
        "satisfies ALL requirements. Vary framing, setting, and difficulty cues "
        "without changing the semantic coverage requirements."
    )
    resp = await llm.agenerate_structured(
        prompt=prompt,
        schema=ScenariosResponse,
        temperature=temperature,
    )
    scenarios = [s.strip() for s in resp.parsed.scenarios if s.strip()]
    if len(scenarios) < K:
        raise ValueError(f"Model returned fewer than K={K} scenarios")
    scenarios = scenarios[:K]
    return [
        MetaPrompt(text=t, mix_id=mix.id, complexified=False) for t in scenarios
    ]


async def complexify_meta_prompt(
    llm: LLMProvider,
    *,
    instruction_y: str,
    bundle: TaxonomyBundle,
    mix: Mix,
    meta: MetaPrompt,
    temperature: float = 0.45,
) -> MetaPrompt:
    """Orthogonal difficulty axis: elaborate while preserving semantic constraints."""
    mix_desc = _mix_description(bundle, mix)
    prompt = (
        f"Dataset instructions y:\n{instruction_y}\n\n"
        f"Requirements mix:\n{mix_desc}\n\n"
        "Meta-prompt to complexify:\n"
        f"{meta.text}\n\n"
        "Rewrite into a HARDER, more elaborate scenario that MUST still satisfy "
        "the same taxonomy requirements. Add distractors, multi-step constraints, "
        "or edge cases where appropriate."
    )
    resp = await llm.agenerate_structured(
        prompt=prompt,
        schema=ComplexifyResponse,
        temperature=temperature,
    )
    return MetaPrompt(
        id=meta.id,
        text=resp.parsed.complexified_scenario.strip(),
        mix_id=meta.mix_id,
        complexified=True,
        sequential_attempt_index=meta.sequential_attempt_index,
    )


def subsample_meta_prompts(
    metas: list[MetaPrompt],
    rng: random.Random | None = None,
) -> MetaPrompt:
    """Algorithm 2: choose one scenario at random."""
    if not metas:
        raise ValueError("metas must be non-empty")
    rng = rng or random.Random()
    return rng.choice(metas)


def _trigram_set(text: str) -> set[tuple[str, str, str]]:
    toks = re.findall(r"\w+", text.lower())
    return {tuple(toks[i : i + 3]) for i in range(max(0, len(toks) - 2))}


async def generate_scenarios_sequential(
    llm: LLMProvider,
    *,
    instruction_y: str,
    bundle: TaxonomyBundle,
    mix: Mix,
    K: int,
    temperature: float = 0.65,
) -> list[MetaPrompt]:
    """For large N/V: generate scenarios one-by-one with prior attempts in context (§2.2)."""
    if K < 1:
        raise ValueError("K must be >= 1")
    mix_desc = _mix_description(bundle, mix)
    prior: list[str] = []
    out: list[MetaPrompt] = []
    for i in range(K):
        prev_block = "\n".join(f"- {p}" for p in prior) or "(none yet)"
        prompt = (
            f"Dataset instructions y:\n{instruction_y}\n\n"
            "Sampled taxonomy requirements (mix):\n"
            f"{mix_desc}\n\n"
            "Previously generated meta-prompts for this SAME mix (avoid near-duplicates):\n"
            f"{prev_block}\n\n"
            "Write ONE new DISTINCT meta-prompt/scenario that still satisfies all requirements."
        )
        resp = await llm.agenerate_structured(
            prompt=prompt,
            schema=ScenariosResponse,
            temperature=temperature,
        )
        scenarios = [s.strip() for s in resp.parsed.scenarios if s.strip()]
        if not scenarios:
            raise ValueError("Sequential scenario generation returned empty")
        text = scenarios[0]
        prior.append(text)
        out.append(
            MetaPrompt(
                text=text,
                mix_id=mix.id,
                complexified=False,
                sequential_attempt_index=i,
            )
        )
    return out


def pairwise_trigram_overlap(a: str, b: str) -> float:
    """Jaccard-like overlap on word trigrams (stylized diversity check)."""
    sa, sb = _trigram_set(a), _trigram_set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    return inter / union
