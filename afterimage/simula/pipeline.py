"""OpenSimula: orchestrates taxonomy construction, sampling, meta-prompts, and datapoint pipelines.

Independent open implementation inspired by Davidson et al. (Simula, TMLR); not a Google product.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from ..providers import DocumentProvider
from ..providers.llm_providers import LLMProvider
from .critics import run_generation_pipeline
from .meta_prompt import (
    complexify_meta_prompt,
    generate_scenarios,
    generate_scenarios_sequential,
    subsample_meta_prompts,
)
from .sampling import infer_sampling_strategies, sample_mix
from .tasks.mcq import agenerate_mcq_json
from .tasks.single_qa import agenerate_single_qa_json
from .taxonomy_builder import TaxonomyBuilder
from .types import (
    DataPointRecord,
    MetaPrompt,
    Mix,
    SamplingStrategySpec,
    TaxonomyBundle,
    validate_factor_taxonomy,
)

if TYPE_CHECKING:
    pass


class OpenSimula:
    """High-level API for Simula-style synthetic dataset mechanisms (experimental)."""

    def __init__(
        self,
        llm: LLMProvider,
        *,
        temperature: float = 0.4,
    ):
        self._llm = llm
        self._temperature = temperature

    async def build_taxonomy(
        self,
        instruction_y: str,
        *,
        document_provider: DocumentProvider | None = None,
        target_depth_D: int = 3,
        proposal_N: int = 3,
    ) -> TaxonomyBundle:
        """Phase: global diversification — build factor taxonomies (Appendix B.4)."""
        builder = TaxonomyBuilder(self._llm, temperature=self._temperature)
        return await builder.build(
            instruction_y,
            document_provider=document_provider,
            target_depth_D=target_depth_D,
            proposal_N=proposal_N,
        )

    @staticmethod
    def validate_taxonomy_bundle(bundle: TaxonomyBundle) -> None:
        """Validate all factor trees (call after construction)."""
        for t in bundle.taxonomies:
            validate_factor_taxonomy(t)

    async def infer_strategies(self, bundle: TaxonomyBundle) -> SamplingStrategySpec:
        """Propose weighted joint-sampling strategies (paper §2.2)."""
        return await infer_sampling_strategies(
            self._llm,
            bundle,
            temperature=min(0.35, self._temperature + 0.1),
        )

    def sample_mix(
        self,
        bundle: TaxonomyBundle,
        spec: SamplingStrategySpec,
        rng: random.Random | None = None,
    ) -> Mix:
        """Sample one mix from strategies."""
        return sample_mix(bundle, spec, rng=rng)

    async def draw_meta_prompt(
        self,
        *,
        instruction_y: str,
        bundle: TaxonomyBundle,
        mix: Mix,
        K: int = 4,
        complexify_c: float = 0.0,
        sequential: bool = False,
        rng: random.Random | None = None,
    ) -> MetaPrompt:
        """Local diversification (+ optional complexification)."""
        rng = rng or random.Random()
        if sequential:
            metas = await generate_scenarios_sequential(
                self._llm,
                instruction_y=instruction_y,
                bundle=bundle,
                mix=mix,
                K=K,
                temperature=min(0.85, self._temperature + 0.35),
            )
        else:
            metas = await generate_scenarios(
                self._llm,
                instruction_y=instruction_y,
                bundle=bundle,
                mix=mix,
                K=K,
                temperature=min(0.85, self._temperature + 0.35),
            )
        meta = subsample_meta_prompts(metas, rng=rng)
        if complexify_c > 0.0 and rng.random() < complexify_c:
            meta = await complexify_meta_prompt(
                self._llm,
                instruction_y=instruction_y,
                bundle=bundle,
                mix=mix,
                meta=meta,
                temperature=min(0.5, self._temperature + 0.1),
            )
        return meta

    async def generate_single_qa_datapoint(
        self,
        *,
        instruction_y: str,
        bundle: TaxonomyBundle,
        mix: Mix,
        meta: MetaPrompt,
        max_refine_rounds: int = 4,
    ) -> DataPointRecord | None:
        """Single QA with requirement-critic loop (no double-critic)."""

        async def gen(llm: LLMProvider) -> str:
            return await agenerate_single_qa_json(
                llm,
                instruction_y=instruction_y,
                bundle=bundle,
                mix=mix,
                meta=meta,
                temperature=min(0.65, self._temperature + 0.2),
            )

        return await run_generation_pipeline(
            self._llm,
            instruction_y=instruction_y,
            bundle=bundle,
            mix=mix,
            meta=meta,
            generate_initial=gen,
            task="single_qa",
            max_refine_rounds=max_refine_rounds,
        )

    async def generate_mcq_datapoint(
        self,
        *,
        instruction_y: str,
        bundle: TaxonomyBundle,
        mix: Mix,
        meta: MetaPrompt,
        num_choices: int = 4,
        max_refine_rounds: int = 4,
    ) -> DataPointRecord | None:
        """MCQ with requirement critic + double-critic gate."""

        async def gen(llm: LLMProvider) -> str:
            return await agenerate_mcq_json(
                llm,
                instruction_y=instruction_y,
                bundle=bundle,
                mix=mix,
                meta=meta,
                num_choices=num_choices,
                temperature=min(0.55, self._temperature + 0.15),
            )

        return await run_generation_pipeline(
            self._llm,
            instruction_y=instruction_y,
            bundle=bundle,
            mix=mix,
            meta=meta,
            generate_initial=gen,
            task="mcq",
            max_refine_rounds=max_refine_rounds,
        )
