"""Orchestrator for conversation generation.

Owns the turn loop (generate N dialogs x M turns), manages async concurrency
(semaphores, gather), and delegates to SamplingStrategy for sampling
configuration and QualityGate for quality decisions. Does NOT know about
personas, documents, or embeddings directly.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import TYPE_CHECKING, Any, List, Optional

from tqdm.asyncio import tqdm

from .common import resolve_generation_max_concurrency
from .sampling import SamplingStrategy
from .quality_gate import QualityGate
from .types import GenerationState

if TYPE_CHECKING:
    from .base import BaseStoppingCallback
    from .monitoring import GenerationMonitor
    from .storage import BaseStorage

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates concurrent generation workers, stopping criteria, and storage.

    The orchestrator is responsible for:
    - Creating and managing async worker tasks with concurrency control
    - Checking stopping criteria after each generated item
    - Saving generated items to storage
    - Updating the progress bar
    - Configuring sampling strategy before generation starts

    It delegates actual conversation generation to the generator's generate_single()
    method and quality evaluation to the QualityGate.
    """

    def __init__(
        self,
        sampling_strategy: SamplingStrategy,
        quality_gate: QualityGate,
        monitor: Optional[GenerationMonitor] = None,
    ):
        self._sampling = sampling_strategy
        self._quality_gate = quality_gate
        self._monitor = monitor

    @property
    def sampling_strategy(self) -> SamplingStrategy:
        return self._sampling

    @property
    def quality_gate(self) -> QualityGate:
        return self._quality_gate

    async def run(
        self,
        generator: Any,
        num_requested: int | None,
        max_turns: int,
        stopping_criteria: List[BaseStoppingCallback],
        instruction_generator_callback: Any,
        respondent_prompt_modifier: Any | None,
        storage: BaseStorage,
        model_provider_name: str,
        max_concurrency: int | None = None,
    ) -> None:
        """Execute the generation loop with concurrency control.

        Args:
            generator: The ConversationGenerator instance (for generate_single/go).
            num_requested: Target number of items (for progress bar).
            max_turns: Maximum conversation turns per dialog.
            stopping_criteria: List of stopping callbacks.
            instruction_generator_callback: Callback producing instructions.
            respondent_prompt_modifier: Optional callback to modify respondent prompts.
            storage: Storage backend for persisting results.
            model_provider_name: Provider name for concurrency defaults.
            max_concurrency: Override for concurrent worker count.
        """
        # Configure sampling
        self._sampling.configure_context_sampling(
            instruction_generator_callback,
            stopping_criteria,
        )
        self._sampling.configure_persona_sampling(
            instruction_generator_callback,
            num_requested=num_requested,
            stopping_criteria=stopping_criteria,
        )

        state = GenerationState(
            num_requested=num_requested or 0,
            monitor=self._monitor,
            stop_event=asyncio.Event(),
        )

        resolved_max_concurrency = resolve_generation_max_concurrency(
            model_provider_name,
            max_concurrency,
        )
        semaphore = asyncio.Semaphore(resolved_max_concurrency)

        async def save_conversations(conversations):
            if conversations:
                if hasattr(storage, "asave_conversations"):
                    await storage.asave_conversations(conversations)
                else:
                    await asyncio.to_thread(storage.save_conversations, conversations)

        async def worker_task():
            while not state.stop_event.is_set():
                async with semaphore:
                    if state.stop_event.is_set():
                        break

                    try:
                        async for conversation in generator.generate_single(
                            max_turns=max_turns,
                            instruction_generator_callback=instruction_generator_callback,
                            respondent_prompt_modifier=respondent_prompt_modifier,
                        ):
                            # Update state and check stopping criteria
                            state.update(conversation)
                            self._sampling.record_context_usage(
                                instruction_generator_callback,
                                conversation,
                            )

                            for criteria in stopping_criteria:
                                if await criteria.should_stop(state):
                                    if self._monitor:
                                        self._monitor.log_info(
                                            "Stopping criteria met, stopping generation...",
                                            criteria=criteria.__class__.__name__,
                                        )
                                    state.stop_event.set()
                                    break

                            await save_conversations([conversation])

                            if state.stop_event.is_set():
                                break

                    except Exception as e:
                        logger.error(f"Error in generation: {e}")
                        traceback.print_exc()
                        if self._monitor:
                            self._monitor.record_metric("error_rate", 1.0)
                        continue

        pbar = tqdm(
            total=num_requested if num_requested is not None else None,
            desc="Generating...",
            unit="conversation",
        )
        tasks: list[asyncio.Task] = []

        for _ in range(resolved_max_concurrency):
            tasks.append(asyncio.create_task(worker_task()))

        last_count = 0
        while not state.stop_event.is_set() or any(not t.done() for t in tasks):
            if state.num_generated > last_count:
                pbar.update(state.num_generated - last_count)
                last_count = state.num_generated

            if state.stop_event.is_set():
                for t in tasks:
                    if not t.done():
                        t.cancel()
                break

            await asyncio.sleep(0.1)

            if all(t.done() for t in tasks):
                break

        pbar.update(state.num_generated - last_count)
        pbar.close()

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            if self._monitor:
                self._monitor.log_error(
                    "Error while trying to finalize generation", error=e
                )
                self._monitor.record_metric("error_rate", 1.0)
            traceback.print_exc()
        finally:
            if self._monitor:
                self._monitor.log_info("Generation complete")
                self._monitor.visualize_metrics()
