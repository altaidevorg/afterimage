"""OpenSimula: orchestrates taxonomy construction, sampling, meta-prompts, and datapoint pipelines.

Independent open implementation inspired by Davidson et al. (Simula, TMLR); not a Google product.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator
from ..monitoring import GenerationMonitor
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


class OpenSimula:
    """High-level API for Simula-style synthetic dataset mechanisms (experimental).

    All structured LLM stages (taxonomy construction, strategy inference, meta-prompt
    diversification and complexification, requirement critics, double-critic probes for
    MCQ, and task JSON generation) accept an optional :class:`~afterimage.monitoring.GenerationMonitor`.
    When ``monitor`` is set, each call is wrapped with ``track_generation`` and metadata
    including ``component="opensimula"`` and a dotted ``operation`` label. Call
    :meth:`~afterimage.monitoring.GenerationMonitor.shutdown` on the monitor when the run
    completes.

    Args:
        llm: Provider used for every structured generation in this pipeline.
        temperature: Base temperature; individual stages may clamp to their own ranges.
        monitor: Optional monitor instance, or ``None`` to disable metric collection.
    """

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

    async def build_taxonomy(
        self,
        instruction_y: str,
        *,
        document_provider: DocumentProvider | None = None,
        target_depth_D: int = 3,
        proposal_N: int = 3,
        max_factors: int = 4,
        max_children_per_node: int = 8,
        max_frontier_per_depth: int = 16,
        show_progress: bool = False,
    ) -> TaxonomyBundle:
        """Phase: global diversification — build factor taxonomies (Appendix B.4).

        ``max_factors``, ``max_children_per_node``, and ``max_frontier_per_depth``
        bound API cost. Without them, wide trees multiply into hundreds of sequential
        LLM calls (minutes of silence).
        """
        builder = TaxonomyBuilder(
            self._llm,
            temperature=self._temperature,
            monitor=self._monitor,
        )
        return await builder.build(
            instruction_y,
            document_provider=document_provider,
            target_depth_D=target_depth_D,
            proposal_N=proposal_N,
            max_factors=max_factors,
            max_children_per_node=max_children_per_node,
            max_frontier_per_depth=max_frontier_per_depth,
            show_progress=show_progress,
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
            monitor=self._monitor,
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
                monitor=self._monitor,
            )
        else:
            metas = await generate_scenarios(
                self._llm,
                instruction_y=instruction_y,
                bundle=bundle,
                mix=mix,
                K=K,
                temperature=min(0.85, self._temperature + 0.35),
                monitor=self._monitor,
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
                monitor=self._monitor,
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
                monitor=self._monitor,
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
            monitor=self._monitor,
        )

    async def agenerate_single_qa_samples(
        self,
        *,
        instruction_y: str,
        bundle: TaxonomyBundle,
        spec: SamplingStrategySpec,
        n: int,
        K: int = 6,
        complexify_c: float = 0.0,
        sequential: bool = False,
        max_concurrency: int = 2,
        rng: random.Random | None = None,
        max_refine_rounds: int = 4,
    ) -> list[DataPointRecord | None]:
        """Generate ``n`` single-QA datapoints with independent (mix, meta) draws each time.

        Results are ordered by sample index ``0 .. n-1``. Concurrency is bounded by
        ``max_concurrency`` (each task still performs its own mix, meta-prompt, and critic loop).
        Each concurrent task draws a fresh RNG stream from ``rng`` so subsampling stays
        deterministic under ``asyncio`` without sharing one :class:`random.Random` across tasks.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return []
        rng = rng or random.Random()
        sem = asyncio.Semaphore(max(1, max_concurrency))
        seed_lock = asyncio.Lock()

        async def one(i: int) -> tuple[int, DataPointRecord | None]:
            async with sem:
                async with seed_lock:
                    local_rng = random.Random(rng.randrange(2**31))
                mix = self.sample_mix(bundle, spec, rng=local_rng)
                meta = await self.draw_meta_prompt(
                    instruction_y=instruction_y,
                    bundle=bundle,
                    mix=mix,
                    K=K,
                    complexify_c=complexify_c,
                    sequential=sequential,
                    rng=local_rng,
                )
                rec = await self.generate_single_qa_datapoint(
                    instruction_y=instruction_y,
                    bundle=bundle,
                    mix=mix,
                    meta=meta,
                    max_refine_rounds=max_refine_rounds,
                )
                return (i, rec)

        pairs = await asyncio.gather(*(one(i) for i in range(n)))
        out: list[DataPointRecord | None] = [None] * n
        for i, rec in pairs:
            out[i] = rec
        return out

    async def aiter_single_qa_samples(
        self,
        *,
        instruction_y: str,
        bundle: TaxonomyBundle,
        spec: SamplingStrategySpec,
        n: int,
        K: int = 6,
        complexify_c: float = 0.0,
        sequential: bool = False,
        max_concurrency: int = 2,
        rng: random.Random | None = None,
        max_refine_rounds: int = 4,
    ) -> AsyncIterator[tuple[int, DataPointRecord | None]]:
        """Like :meth:`agenerate_single_qa_samples` but yield ``(index, record)`` as each task finishes.

        Useful for appending to JSONL as samples complete. If the consumer stops early,
        unfinished tasks are cancelled.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return
        rng = rng or random.Random()
        sem = asyncio.Semaphore(max(1, max_concurrency))
        seed_lock = asyncio.Lock()

        async def one(i: int) -> tuple[int, DataPointRecord | None]:
            async with sem:
                async with seed_lock:
                    local_rng = random.Random(rng.randrange(2**31))
                mix = self.sample_mix(bundle, spec, rng=local_rng)
                meta = await self.draw_meta_prompt(
                    instruction_y=instruction_y,
                    bundle=bundle,
                    mix=mix,
                    K=K,
                    complexify_c=complexify_c,
                    sequential=sequential,
                    rng=local_rng,
                )
                rec = await self.generate_single_qa_datapoint(
                    instruction_y=instruction_y,
                    bundle=bundle,
                    mix=mix,
                    meta=meta,
                    max_refine_rounds=max_refine_rounds,
                )
                return (i, rec)

        tasks = [asyncio.create_task(one(i)) for i in range(n)]
        try:
            for fut in asyncio.as_completed(tasks):
                yield await fut
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

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
                monitor=self._monitor,
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
            monitor=self._monitor,
        )
