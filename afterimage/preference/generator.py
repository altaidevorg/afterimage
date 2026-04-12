"""PreferenceGenerator: produces (chosen, rejected) pairs for DPO/RLHF training."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..base import (
    BaseInstructionGeneratorCallback,
    BaseRespondentPromptModifierCallback,
)
from ..evaluator import ConversationJudge
from ..providers import LLMFactory
from ..types import ConversationEntry, ConversationWithContext, Role
from .types import PreferenceAnalytics, PreferenceConfig, PreferencePair, ScoredResponse

if TYPE_CHECKING:
    from ..conversation_generator import ConversationGenerator

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CONCURRENCY = 4


class PreferenceGenerator:
    """Generates preference pairs (chosen/rejected) from a ConversationGenerator.

    Works by:
    1. Reusing the conversation generator's instruction callback to produce prompts.
    2. Generating ``num_responses`` responses per prompt via variation strategies.
    3. Scoring each response with the ConversationJudge.
    4. Selecting highest/lowest scored responses as chosen/rejected.
    5. Discarding pairs with insufficient score gap.

    Args:
        conversation_generator: Fully configured ConversationGenerator (prompts, callbacks, etc.).
        judge: ConversationJudge instance for scoring responses.
        config: Preference generation settings.
        secondary_llm_provider: Optional secondary LLM for model-variation strategy.
            Required when ``config.strategy`` is ``"model"`` or ``"combined"``.
    """

    def __init__(
        self,
        conversation_generator: "ConversationGenerator",
        judge: ConversationJudge,
        config: Optional[PreferenceConfig] = None,
        secondary_llm_provider=None,
    ):
        self._gen = conversation_generator
        self._judge = judge
        self._config = config or PreferenceConfig()
        self._secondary_llm = secondary_llm_provider

        # Build primary LLM from the generator's settings
        self._primary_llm = LLMFactory.create(
            provider=self._gen.model_provider_name,
            model_name=self._gen.model_name,
            api_key=self._gen.key_pool,
            system_instruction=self._gen.respondent_prompt,
            **self._gen.llm_factory_kwargs,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        num_pairs: Optional[int] = None,
        instruction_generator_callback: Optional[
            BaseInstructionGeneratorCallback
        ] = None,
        respondent_prompt_modifier: Optional[
            BaseRespondentPromptModifierCallback
        ] = None,
    ) -> tuple[List[PreferencePair], PreferenceAnalytics]:
        """Generate preference pairs.

        Args:
            num_pairs: Override config.num_pairs.
            instruction_generator_callback: Override the generator's callback.
            respondent_prompt_modifier: Override the generator's prompt modifier.

        Returns:
            Tuple of (pairs, analytics).
        """
        cfg = self._config
        target = num_pairs or cfg.num_pairs
        max_concurrency = cfg.max_concurrency or _DEFAULT_MAX_CONCURRENCY

        callback = (
            instruction_generator_callback or self._gen.instruction_generator_callback
        )
        modifier = respondent_prompt_modifier or self._gen.respondent_prompt_modifier

        if callback is None:
            raise ValueError(
                "No instruction_generator_callback available. Pass one or configure "
                "the ConversationGenerator with one."
            )

        # Initialise correspondent prompt if needed
        await self._gen.ainitialize(callback)

        pairs: List[PreferencePair] = []
        analytics = PreferenceAnalytics()
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _attempt_one_batch():
            """Generate one batch of instructions and produce pairs from them."""
            nonlocal pairs
            gen_instructions = await callback.acall(self._gen.correspondent_prompt)
            tasks = [
                self._process_instruction(
                    instruction=instruction,
                    gen_instructions=gen_instructions,
                    modifier=modifier,
                    semaphore=semaphore,
                )
                for instruction in gen_instructions.instructions
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.warning("Preference pair generation failed: %s", r)
                    analytics.total_attempted += 1
                    analytics.total_discarded += 1
                elif r is not None:
                    pairs.append(r)
                    analytics.total_attempted += 1
                    analytics.total_valid += 1
                else:
                    analytics.total_attempted += 1
                    analytics.total_discarded += 1

        # Keep generating until we have enough pairs, with a safety cap
        max_batches = max(target * 10, 50)
        batches = 0
        while len(pairs) < target and batches < max_batches:
            await _attempt_one_batch()
            batches += 1
        if len(pairs) < target:
            logger.warning(
                "Only generated %d/%d pairs after %d batches. "
                "Consider lowering min_score_gap or increasing num_responses.",
                len(pairs),
                target,
                batches,
            )

        # Trim to exact count
        pairs = pairs[:target]

        # Compute final analytics with all stats and warnings
        from .analytics import compute_analytics

        analytics = compute_analytics(pairs, analytics.total_attempted)

        return pairs, analytics

    async def _process_instruction(
        self,
        instruction: str,
        gen_instructions,
        modifier,
        semaphore: asyncio.Semaphore,
    ) -> Optional[PreferencePair]:
        """Generate and score responses for a single instruction, return a pair or None."""
        async with semaphore:
            try:
                return await self._build_pair(
                    instruction=instruction,
                    gen_instructions=gen_instructions,
                    modifier=modifier,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to build preference pair for instruction %r: %s",
                    instruction[:80],
                    exc,
                )
                return None

    async def _build_pair(
        self,
        instruction: str,
        gen_instructions,
        modifier,
    ) -> Optional[PreferencePair]:
        """Core logic: generate responses, score, filter, return PreferencePair."""
        cfg = self._config
        respondent_prompt = self._gen.respondent_prompt

        # Apply respondent prompt modifier if available
        response_context = None
        if modifier is not None:
            modified = await modifier.acall(
                respondent_prompt,
                context=gen_instructions.context,
                instruction=instruction,
            )
            respondent_prompt = modified.prompt
            response_context = modified.context

        # For multi-turn: generate shared conversation history first
        shared_prefix: Optional[List[Dict[str, Any]]] = None
        history: list = []
        if cfg.multi_turn:
            shared_prefix, history = await self._generate_shared_prefix(
                instruction=instruction,
                respondent_prompt=respondent_prompt,
            )

        # Generate multiple responses
        response_tuples = await self._generate_responses(
            system_prompt=respondent_prompt,
            user_turn=instruction,
            history=history,
        )

        if len(response_tuples) < 2:
            return None

        # Score responses
        scored = await self._score_responses(
            response_tuples=response_tuples,
            instruction=instruction,
            instruction_context=gen_instructions.context,
            response_context=response_context,
            persona=gen_instructions.persona,
        )

        if len(scored) < 2:
            return None

        # Sort by score descending
        scored.sort(key=lambda r: r.score, reverse=True)
        chosen = scored[0]
        rejected = scored[-1]

        # Filter by score gap
        if chosen.score - rejected.score < cfg.min_score_gap:
            return None

        # Build messages for chosen/rejected (for chat_dpo format)
        sys_msg = {"role": "system", "content": respondent_prompt}
        chosen_messages = _build_chat_messages(
            sys_msg, shared_prefix or [], instruction, chosen.content
        )
        rejected_messages = _build_chat_messages(
            sys_msg, shared_prefix or [], instruction, rejected.content
        )

        chosen = chosen.model_copy(update={"messages": chosen_messages})
        rejected = rejected.model_copy(update={"messages": rejected_messages})

        return PreferencePair(
            prompt=instruction,
            chosen=chosen,
            rejected=rejected,
            shared_prefix=shared_prefix,
            metadata={
                "context_id": gen_instructions.context_id,
                "context_ids": gen_instructions.context_ids,
                "persona": gen_instructions.persona,
                "instruction_context": gen_instructions.context,
                "response_context": response_context,
                "all_scores": [
                    {
                        "content": r.content[:100],
                        "score": r.score,
                        "label": r.variation_label,
                    }
                    for r in scored
                ],
            },
        )

    async def _generate_shared_prefix(
        self,
        instruction: str,
        respondent_prompt: str,
    ) -> tuple[list, list]:
        """Generate a shared multi-turn conversation prefix.

        Returns (shared_prefix as list of dicts, history as list of dicts for LLM).
        """
        conversation = await self._gen.go(
            turns=1,
            first_question=instruction,
            respondent_prompt=respondent_prompt,
        )

        # shared_prefix is all turns EXCEPT the last assistant turn
        # (we'll branch at the last turn)
        prefix_entries = conversation[:-1]  # everything but the last assistant response

        shared_prefix = [
            {"role": e.role.value, "content": e.content} for e in prefix_entries
        ]
        history = shared_prefix  # pass history to response generation

        return shared_prefix, history

    async def _generate_responses(
        self,
        system_prompt: str,
        user_turn: str,
        history: list,
    ) -> list:
        """Dispatch to the configured variation strategy.

        Returns list of (content, temperature, label) tuples.
        """
        cfg = self._config
        strategy = cfg.strategy

        from .strategies import (
            combined_strategy,
            model_strategy,
            prompt_strategy,
            temperature_strategy,
        )

        if strategy == "temperature":
            return await temperature_strategy(
                llm=self._primary_llm,
                system_prompt=system_prompt,
                user_turn=user_turn,
                history=history,
                num_responses=cfg.num_responses,
            )
        elif strategy == "prompt":
            return await prompt_strategy(
                llm=self._primary_llm,
                system_prompt=system_prompt,
                user_turn=user_turn,
                history=history,
                num_responses=cfg.num_responses,
            )
        elif strategy == "model":
            if self._secondary_llm is None:
                raise ValueError(
                    "strategy='model' requires secondary_llm_provider to be set"
                )
            return await model_strategy(
                primary_llm=self._primary_llm,
                secondary_llm=self._secondary_llm,
                system_prompt=system_prompt,
                user_turn=user_turn,
                history=history,
                num_responses=cfg.num_responses,
            )
        elif strategy == "combined":
            return await combined_strategy(
                primary_llm=self._primary_llm,
                secondary_llm=self._secondary_llm,
                system_prompt=system_prompt,
                user_turn=user_turn,
                history=history,
                num_responses=cfg.num_responses,
            )
        else:
            raise ValueError(
                f"Unknown strategy: {strategy!r}. "
                "Choose from: temperature, prompt, model, combined"
            )

    async def _score_responses(
        self,
        response_tuples: list,
        instruction: str,
        instruction_context: Optional[str],
        response_context: Optional[str],
        persona: Optional[str],
    ) -> List[ScoredResponse]:
        """Score all responses concurrently using the judge."""
        tasks = []
        for content, _temp, label in response_tuples:
            conv = ConversationWithContext(
                conversations=[
                    ConversationEntry(role=Role.USER, content=instruction),
                    ConversationEntry(role=Role.ASSISTANT, content=content),
                ],
                instruction_context=instruction_context,
                response_context=response_context,
                persona=persona,
            )
            tasks.append(self._score_one(conv, content, label))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        scored = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Scoring failed: %s", r)
            else:
                scored.append(r)
        return scored

    async def _score_one(
        self,
        conv: ConversationWithContext,
        content: str,
        label: str,
    ) -> ScoredResponse:
        evaluated = await self._judge.aevaluate_row(conv)
        return ScoredResponse(
            content=content,
            score=evaluated.final_score or 0.0,
            variation_label=label,
        )

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    def save_pairs(
        self,
        pairs: List[PreferencePair],
        analytics: Optional[PreferenceAnalytics] = None,
    ) -> None:
        """Save pairs to the configured output file in the configured format."""
        from .formats import format_preference_pairs

        cfg = self._config
        output_path = Path(cfg.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows = format_preference_pairs(pairs, fmt=cfg.output_format)
        with open(output_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if cfg.save_log:
            log_path = cfg.log_path
            if log_path is None:
                log_path = str(output_path.with_suffix("")) + "_log.jsonl"
            self._save_log(pairs, analytics, log_path)

    def _save_log(
        self,
        pairs: List[PreferencePair],
        analytics: Optional[PreferenceAnalytics],
        log_path: str,
    ) -> None:
        """Save full generation log (all responses + scores)."""
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                row = pair.model_dump()
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if analytics is not None:
                import dataclasses

                f.write(
                    json.dumps(
                        {"_analytics": dataclasses.asdict(analytics)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )


# ---------------------------------------------------------------------------
# Factory method on ConversationGenerator
# ---------------------------------------------------------------------------


def _to_preference_generator(
    self,
    judge: ConversationJudge,
    config: Optional[PreferenceConfig] = None,
    secondary_llm_provider=None,
) -> "PreferenceGenerator":
    """Create a :class:`PreferenceGenerator` from this generator.

    Args:
        judge: Scoring judge (ConversationJudge).
        config: Preference generation config. Defaults to PreferenceConfig().
        secondary_llm_provider: Optional secondary LLM for model-variation strategy.

    Returns:
        Configured :class:`PreferenceGenerator`.
    """
    return PreferenceGenerator(
        conversation_generator=self,
        judge=judge,
        config=config,
        secondary_llm_provider=secondary_llm_provider,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_chat_messages(
    system_msg: dict,
    prefix: list,
    user_turn: str,
    assistant_response: str,
) -> List[Dict[str, Any]]:
    """Build full chat message list for chat_dpo format."""
    msgs = [system_msg] + list(prefix)
    msgs.append({"role": "user", "content": user_turn})
    msgs.append({"role": "assistant", "content": assistant_response})
    return msgs
