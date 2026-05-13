"""Conversation generator — coordinates orchestration, sampling, and quality evaluation.

Delegates concurrency to :class:`~afterimage.orchestrator.Orchestrator`,
sampling to :class:`~afterimage.sampling.SamplingStrategy`, and quality
evaluation to :class:`~afterimage.quality_gate.QualityGate`.
"""

import asyncio
import logging
import random
import time
import warnings
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from .base import (
    BaseGenerator,
    BaseInstructionGeneratorCallback,
    BaseRespondentPromptModifierCallback,
    BaseStoppingCallback,
)
from .callbacks import FixedNumberStoppingCallback
from .conversation_turn_hooks import ConversationTurnContext, ConversationTurnHooks
from .common import (
    default_model_name,
    default_safety_settings,
    resolve_generation_max_concurrency,
)
from .evaluator import (
    ConversationJudge,
    ConversationJudgeConfig,
    default_embedding_provider_config,
)
from .key_management import SmartKeyPool
from .monitoring import GenerationMonitor
from .orchestrator import Orchestrator
from .prompts import get_correspondent_instruction_generation_prompt
from .providers import ChatSession, LLMFactory
from .providers.embedding_providers import EmbeddingProvider
from .quality_gate import QualityGate
from .sampling import SamplingStrategy
from .storage import BaseStorage, JSONLStorage
from .types import (
    Conversation,
    ConversationEntry,
    ConversationWithContext,
    EvaluatedConversationWithContext,
    ModelProviderName,
    Role,
)

logger = logging.getLogger(__name__)


def format_correspondent_followup_user_message(assistant_reply: str) -> str:
    """Shape the respondent's last reply for the correspondent chat on turn 2+.

    The correspondent model plays the *human user*. Feeding the raw assistant
    reply as the next user message often causes role drift (assistant voice,
    new persona, language switches). This wrapper keeps the contract explicit.
    """
    text = (assistant_reply or "").strip()
    return (
        "The assistant (the domain expert you are talking to) has just replied "
        "to you below.\n\n"
        "Stay the same human user you were in the previous turns of this chat. "
        "Write only your next short message to them: a natural follow-up question "
        "or request.\n\n"
        "Rules:\n"
        "- If your earlier user messages in this chat were not in English, "
        "write your next message in that same language. Do not switch to English "
        "unless your own previous user turns were already English.\n"
        "- Do not answer for the assistant, give bullet tutorials, or use an "
        "assistant voice.\n"
        "- Use the same natural language as your earlier user messages in this "
        "chat unless a code-switch is natural for that user.\n"
        "- Do not invent a new job, company, or scenario unless it truly follows "
        "from what you already said.\n"
        '- No preamble (e.g. no "As a user" or "Here is my question").\n\n'
        "<assistant_last_message>\n"
        f"{text}\n"
        "</assistant_last_message>\n"
        "---\n\n"
        "Your next message as the user (one turn only):"
    )


class ConversationGenerator(BaseGenerator):
    """Generates conversations between a correspondent (question generator) and a respondent (answer generator) asynchronously.

    Args:
        respondent_prompt: System prompt to the respondent, e.g., assistant that you want you fine-tune on this dataset
        api_key: Either a single API key string or a SmartKeyPool instance for LLM use
        correspondent_prompt: System prompt to the correspondent, e.g., model that roleplays a user of the assistant
            that you want to fine-tune on this dataset
        model_name: Model name to use
        safety_settings: Safety settings for the model
        auto_improve: Whether to try to improve low-quality generations
        evaluator_model_name: Model name for the evaluator LLM when auto_improve is True.
        model_provider_name: Chat LLM vendor: ``gemini``, ``openai``, ``deepseek``, ``local`` (OpenAI-compatible server), or ``openrouter``.
        llm_factory_kwargs: Extra keyword arguments passed to :meth:`~afterimage.providers.llm_providers.LLMFactory.create` for every LLM built by this generator (for example ``base_url`` for compatible endpoints).
        embedding_provider: Optional shared :class:`~afterimage.providers.embedding_providers.EmbeddingProvider` for embedding metrics.
        embedding_provider_config: JSON-style config for :class:`~afterimage.providers.embedding_providers.EmbeddingProviderFactory` when ``embedding_provider`` is omitted (defaults by chat provider).
        judge_config: Optional :class:`~afterimage.evaluator.ConversationJudgeConfig` (aggregation and grade thresholds).
        storage: Storage implementation for saving conversations.
                If `None`, creates JSONLStorage with datetime-based filename.
        monitor: GenerationMonitor instance for tracking generation metrics.
            If  `None`, a default one is created.
        instruction_generator_callback: Callback for instruction generation. Can also be passed to generate() method (deprecated).
        respondent_prompt_modifier: Callback to modify respondent prompts. Can also be passed to generate() method (deprecated).
    """

    @property
    def evaluator(self):
        """The conversation evaluator (ConversationJudge or None).

        Setting this also updates the internal QualityGate so that
        generate_single() uses the new evaluator for retry decisions.
        """
        return self._evaluator

    @evaluator.setter
    def evaluator(self, value):
        self._evaluator = value
        # Keep the quality gate in sync when evaluator is set after init
        if hasattr(self, "_quality_gate"):
            self._quality_gate._evaluator = value

    def __init__(
        self,
        respondent_prompt: str,
        api_key: str | SmartKeyPool,
        correspondent_prompt: str | None = None,
        model_name: str | None = None,
        safety_settings: List[Dict[str, str]] | None = None,
        auto_improve: bool = False,
        evaluator_model_name: str | None = None,
        model_provider_name: ModelProviderName = "gemini",
        llm_factory_kwargs: dict[str, Any] | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_provider_config: dict[str, Any] | None = None,
        judge_config: ConversationJudgeConfig | None = None,
        storage: Optional[BaseStorage] = None,
        monitor: Optional[GenerationMonitor] = None,
        instruction_generator_callback: BaseInstructionGeneratorCallback | None = None,
        respondent_prompt_modifier: BaseRespondentPromptModifierCallback | None = None,
        turn_hooks: ConversationTurnHooks | None = None,
    ):
        self.monitor: GenerationMonitor = (
            monitor or GenerationMonitor()
        )  # ensure it's always created

        self.key_pool: SmartKeyPool = (
            api_key
            if isinstance(api_key, SmartKeyPool)
            else SmartKeyPool.from_single_key(api_key)
        )

        self.llm_factory_kwargs: dict[str, Any] = dict(llm_factory_kwargs or ())
        self.model_provider_name = model_provider_name
        self.model_name = model_name if model_name is not None else default_model_name
        self.monitor.log_info(
            "Model info set",
            model_provider=self.model_provider_name,
            model_name=self.model_name,
            num_api_keys=len(self.key_pool),
        )
        self.safety_settings = (
            safety_settings if safety_settings is not None else default_safety_settings
        )

        # users should pass at least one of correspondent prompt and instruction generator callback.
        # if both are passed, the correspondent prompt will be used as is passed.
        # if neither are passed, raise an error.
        if correspondent_prompt is None and instruction_generator_callback is None:
            error = ValueError(
                "At least one of `correspondent_prompt` or `instruction_generator_callback` should be passed."
            )
            self.monitor.log_error(
                "failed to initialize because correspondent prompt and instruction generator callback are both None",
                error,
            )
            raise error

        if correspondent_prompt is None:
            warning_msg = "A correspondent prompt will be automatically created because you did not pass one."
            self.monitor.log_warning(warning_msg)
            warnings.warn(warning_msg)

        self.respondent_prompt = respondent_prompt
        self.monitor.log_info(
            "Respondent prompt set",
            respondent_prompt=respondent_prompt,
        )
        self.correspondent_prompt = correspondent_prompt

        self.instruction_generator_callback = instruction_generator_callback
        self.respondent_prompt_modifier = respondent_prompt_modifier
        self.turn_hooks = turn_hooks

        # --- Quality gate (wraps evaluator) ---
        self._quality_gate = QualityGate(evaluator=None)
        self.evaluator = None
        if auto_improve:
            evaluator_model_name = (
                evaluator_model_name
                if evaluator_model_name is not None
                else self.model_name
            )
            evaluator_llm = LLMFactory.create(
                provider=self.model_provider_name,
                model_name=evaluator_model_name,
                api_key=self.key_pool,
                safety_settings=self.safety_settings,
                **self.llm_factory_kwargs,
            )
            if embedding_provider is not None:
                self.evaluator = ConversationJudge(
                    llm=evaluator_llm,
                    embedding_provider=embedding_provider,
                    monitor=self.monitor,
                    config=judge_config,
                )
            else:
                embed_cfg = (
                    embedding_provider_config
                    if embedding_provider_config is not None
                    else default_embedding_provider_config(self.model_provider_name)
                )
                self.evaluator = ConversationJudge.from_factory(
                    evaluator_llm,
                    key_pool=self.key_pool,
                    model_provider_name=self.model_provider_name,
                    embedding_provider_config=embed_cfg,
                    monitor=self.monitor,
                    config=judge_config,
                )

        # --- Sampling strategy ---
        self._sampling_strategy = SamplingStrategy(monitor=self.monitor)

        # --- Orchestrator ---
        self._orchestrator = Orchestrator(
            sampling_strategy=self._sampling_strategy,
            quality_gate=self._quality_gate,
            monitor=self.monitor,
        )

        self.initiators = []
        self.storage = storage or JSONLStorage()

        if (
            self.instruction_generator_callback
            and self.instruction_generator_callback.monitor is None
        ):
            self.instruction_generator_callback.set_monitor(self.monitor)

    async def create_correspondent_prompt(self, assistant_prompt: str) -> str:
        """Create a correspondent prompt based on the assistant prompt."""
        start_time = time.time()
        api_key: str | None = None
        try:
            prompt = get_correspondent_instruction_generation_prompt(
                assistant_prompt=assistant_prompt
            )
            api_key = await self.key_pool.aget_next_key()
            model = LLMFactory.create(
                provider=self.model_provider_name,
                model_name=self.model_name,
                api_key=api_key,
                safety_settings=self.safety_settings,
                **self.llm_factory_kwargs,
            )

            response = await model.agenerate_content(prompt=prompt, temperature=0.7)
            prompt = (
                response.text.strip()
                .lstrip("<user_system_prompt>")
                .rstrip("</user_system_prompt>")
                .strip()
            )

            if self.monitor:
                self.monitor.track_generation(
                    duration=time.time() - start_time,
                    success=True,
                    prompt_token_count=response.prompt_token_count,
                    completion_token_count=response.completion_token_count,
                    total_token_count=response.total_token_count,
                    model_name=response.model_name,
                    metadata={"operation": "correspondent_prompt_generation"},
                )

            return prompt
        except Exception as e:
            if self.monitor:
                self.monitor.track_generation(
                    duration=time.time() - start_time,
                    success=False,
                    error=str(e),
                    metadata={
                        "operation": "correspondent_prompt_generation",
                        "error_type": e.__class__.__name__,
                    },
                )
            if api_key is not None:
                await self.key_pool.areport_error(api_key)
            raise

    async def create_model(self, prompt: str) -> ChatSession:
        """Creates and initializes a chat model with the given prompt."""
        start_time = time.time()
        api_key: str | None = None
        try:
            api_key = await self.key_pool.aget_next_key()
            model = LLMFactory.create(
                provider=self.model_provider_name,
                model_name=self.model_name,
                api_key=api_key,
                system_instruction=prompt,
                safety_settings=self.safety_settings,
                **self.llm_factory_kwargs,
            )

            chat = await model.astart_chat()

            if self.monitor:
                self.monitor.record_metric(
                    "model_creation_time",
                    time.time() - start_time,
                    metadata={"success": True},
                )

            return chat

        except Exception as e:
            if self.monitor:
                self.monitor.record_metric(
                    "model_creation_time",
                    time.time() - start_time,
                    metadata={
                        "success": False,
                        "error": str(e),
                        "error_type": e.__class__.__name__,
                    },
                )
            if api_key is not None:
                await self.key_pool.areport_error(api_key)
            raise

    async def ask(
        self, correspondent: ChatSession, answer: str | ConversationEntry
    ) -> str:
        """Generates a question from the correspondent based on the given answer."""
        start_time = time.time()
        try:
            response = await correspondent.asend_message(answer)
            question = response.text

            self.monitor.track_generation(
                duration=time.time() - start_time,
                success=True,
                prompt_token_count=response.prompt_token_count,
                completion_token_count=response.completion_token_count,
                total_token_count=response.total_token_count,
                finish_reason=response.finish_reason,
                model_name=response.model_name,
                metadata={
                    "operation": "question_generation",
                },
            )

            return question
        except Exception as e:
            self.monitor.track_generation(
                duration=time.time() - start_time,
                success=False,
                error=str(e),
                metadata={
                    "operation": "question_generation",
                    "error_type": e.__class__.__name__,
                },
            )
            raise

    async def answer(
        self, respondent: ChatSession, question: str | ConversationEntry
    ) -> ConversationEntry:
        """Generates an answer from the respondent based on the given question."""
        start_time = time.time()
        try:
            response = await respondent.asend_message(question)
            answer = response.text

            self.monitor.track_generation(
                duration=time.time() - start_time,
                success=True,
                prompt_token_count=response.prompt_token_count,
                completion_token_count=response.completion_token_count,
                total_token_count=response.total_token_count,
                finish_reason=response.finish_reason,
                model_name=response.model_name,
                metadata={
                    "operation": "answer_generation",
                },
            )

            return ConversationEntry(
                role=Role.ASSISTANT,
                content=answer,
                reasoning_content=response.reasoning_content,
            )
        except Exception as e:
            self.monitor.track_generation(
                duration=time.time() - start_time,
                success=False,
                error=str(e),
                metadata={
                    "operation": "answer_generation",
                    "error_type": e.__class__.__name__,
                },
            )
            raise

    def _turn_context(
        self,
        conversation: List[ConversationEntry],
        planned_turns: int,
        correspondent_prompt: str,
        respondent_prompt: str,
    ) -> ConversationTurnContext:
        completed = sum(1 for e in conversation if e.role == Role.ASSISTANT)
        return ConversationTurnContext(
            planned_turns=planned_turns,
            respondent_turns_completed=completed,
            conversation=tuple(conversation),
            respondent_system_prompt=respondent_prompt,
            correspondent_system_prompt=correspondent_prompt,
        )

    async def go(
        self,
        turns: int = 1,
        first_question: str | None = None,
        check_for_near_duplicates: bool = False,
        correspondent_prompt: str | None = None,
        respondent_prompt: str | None = None,
    ) -> List[ConversationEntry]:
        """Simulates a multi-turn conversation between the correspondent and respondent."""
        start_time = time.time()
        conversation = []
        th = self.turn_hooks
        first_correspondent_message = "Ask your first question."
        correspondent: ChatSession | None = None
        respondent: ChatSession | None = None

        try:
            if correspondent_prompt is None:
                correspondent_prompt = self.correspondent_prompt
                # If still None, create it using generator's method
                if correspondent_prompt is None:
                    correspondent_prompt = await self.create_correspondent_prompt(
                        self.respondent_prompt
                    )

            if respondent_prompt is None:
                respondent_prompt = self.respondent_prompt

            correspondent = await self.create_model(correspondent_prompt)
            respondent = await self.create_model(respondent_prompt)

            if first_question is None:
                if th:
                    await th.before_correspondent_completion(
                        self._turn_context(
                            conversation,
                            turns,
                            correspondent_prompt,
                            respondent_prompt,
                        ),
                        first_correspondent_message,
                    )
                question = await self.ask(correspondent, first_correspondent_message)
            else:
                question = first_question
            self.initiators.append(question)
            conversation.append(ConversationEntry(role=Role.USER, content=question))
            if th:
                await th.after_correspondent_completion(
                    self._turn_context(
                        conversation,
                        turns,
                        correspondent_prompt,
                        respondent_prompt,
                    ),
                    question,
                )

            for turn in range(turns):
                if th:
                    await th.before_respondent_completion(
                        self._turn_context(
                            conversation,
                            turns,
                            correspondent_prompt,
                            respondent_prompt,
                        ),
                        question,
                    )
                answer_entry = await self.answer(respondent, question)
                conversation.append(answer_entry)
                if th:
                    await th.after_respondent_completion(
                        self._turn_context(
                            conversation,
                            turns,
                            correspondent_prompt,
                            respondent_prompt,
                        ),
                        answer_entry,
                    )

                if (turn + 1) == turns:
                    break
                else:
                    followup = format_correspondent_followup_user_message(
                        answer_entry.content
                    )
                    if th:
                        await th.before_correspondent_completion(
                            self._turn_context(
                                conversation,
                                turns,
                                correspondent_prompt,
                                respondent_prompt,
                            ),
                            followup,
                        )
                    question = await self.ask(correspondent, followup)

                    conversation.append(
                        ConversationEntry(role=Role.USER, content=question)
                    )
                    self.initiators.append(question)
                    if th:
                        await th.after_correspondent_completion(
                            self._turn_context(
                                conversation,
                                turns,
                                correspondent_prompt,
                                respondent_prompt,
                            ),
                            question,
                        )

            self.monitor.track_generation(
                duration=time.time() - start_time,
                success=True,
                conversation_length=len(conversation) // 2,
                metadata={
                    "operation": "conversation_generation",
                    "planned_turns": turns,
                    "actual_turns": len(conversation) // 2,
                },
            )

            return conversation

        except Exception as e:
            self.monitor.track_generation(
                duration=time.time() - start_time,
                success=False,
                error=str(e),
                metadata={
                    "operation": "conversation_generation",
                    "error_type": e.__class__.__name__,
                    "completed_turns": len(conversation) // 2,
                },
            )
            raise
        finally:
            if correspondent is not None:
                await correspondent.aclose()
            if respondent is not None:
                await respondent.aclose()

    async def generate_single(
        self,
        max_turns: int,
        check_for_near_duplicates: bool = False,
        instruction_generator_callback: BaseInstructionGeneratorCallback | None = None,
        respondent_prompt_modifier: BaseRespondentPromptModifierCallback | None = None,
    ) -> AsyncGenerator[Union[EvaluatedConversationWithContext, Conversation], None]:
        """Generates conversations for a single session and yields them."""
        # ainitialize ensures correspondent_prompt is set
        await self.ainitialize(instruction_generator_callback)

        correspondent_prompt = self.correspondent_prompt
        respondent_prompt = self.respondent_prompt
        turns = random.randint(1, max_turns)

        if instruction_generator_callback:
            if hasattr(instruction_generator_callback, "acall"):
                gen_instructions = await instruction_generator_callback.acall(
                    correspondent_prompt
                )
            else:
                gen_instructions = await asyncio.to_thread(
                    instruction_generator_callback, correspondent_prompt
                )

            for instruction in gen_instructions.instructions:
                instruction_context = gen_instructions.context
                persona = gen_instructions.persona
                response_context = None
                response_metadata = {}
                current_respondent_prompt = respondent_prompt
                if respondent_prompt_modifier:
                    if hasattr(respondent_prompt_modifier, "acall"):
                        modified_respondent_prompt = (
                            await respondent_prompt_modifier.acall(
                                respondent_prompt,
                                context=instruction_context,
                                instruction=instruction,
                            )
                        )
                    else:
                        modified_respondent_prompt = await asyncio.to_thread(
                            respondent_prompt_modifier,
                            respondent_prompt,
                            context=instruction_context,
                            instruction=instruction,
                        )
                    current_respondent_prompt = modified_respondent_prompt.prompt
                    response_context = modified_respondent_prompt.context
                    response_metadata = modified_respondent_prompt.metadata

                conversation = await self.go(
                    turns=turns,
                    first_question=instruction,
                    check_for_near_duplicates=check_for_near_duplicates,
                    correspondent_prompt=correspondent_prompt,
                    respondent_prompt=current_respondent_prompt,
                )

                def build_conversation_row(
                    generated_conversation,
                ) -> ConversationWithContext:
                    return ConversationWithContext(
                        conversations=generated_conversation,
                        instruction_context=instruction_context,
                        response_context=response_context,
                        persona=persona,
                        metadata={
                            "context_id": gen_instructions.context_id,
                            "context_ids": gen_instructions.context_ids,
                            "persona_name": persona,
                            "persona_generation_depth": (
                                gen_instructions.persona_generation_depth
                            ),
                            **response_metadata,
                        },
                    )

                conversation_row = build_conversation_row(conversation)

                # --- Quality gate: evaluate and retry if needed ---
                result = await self._quality_gate.evaluate(conversation_row)
                while QualityGate.should_retry(result):
                    conversation = await self.go(
                        turns=turns,
                        first_question=instruction,
                        check_for_near_duplicates=check_for_near_duplicates,
                        correspondent_prompt=correspondent_prompt,
                        respondent_prompt=current_respondent_prompt,
                    )
                    conversation_row = build_conversation_row(conversation)
                    result = await self._quality_gate.evaluate(conversation_row)

                yield result.conversation_row
        else:
            raise ValueError("An `instruction_generator_callback` must be provided.")

    async def generate(
        self,
        num_dialogs: int | None = None,
        max_turns: int = 1,
        stopping_criteria: Optional[List[BaseStoppingCallback]] = None,
        instruction_generator_callback: BaseInstructionGeneratorCallback | None = None,
        respondent_prompt_modifier: BaseRespondentPromptModifierCallback | None = None,
        max_concurrency: int | None = None,
        num_requested: int | None = None,
    ) -> None:
        """Generates multiple conversation dialogs until stopping criteria is met.

        Args:
            num_dialogs: Number of dialogs to generate. Defaults to 5 if no other stopping criteria is specified.
            max_turns: Maximum number of turns per dialog. Actual number of turns is randomly sampled from 1 .. max_turns.
            stopping_criteria: A list of callbacks to determine when to stop generation. If num_dialogs is specified, FixedNumberStoppingCallback is added to this list automatically.
            instruction_generator_callback: Callback for instruction generation.
                Deprecated: Pass this to the constructor instead. Defaults to None.
            respondent_prompt_modifier: Callback to modify respondent prompts.
                Deprecated: Pass this to the constructor instead. Defaults to None.
            max_concurrency: Number of concurrent generations. Defaults to 8 for
                DeepSeek and 4 for other providers.
            num_requested: Hint for progress reporting (e.g. tqdm total). When omitted,
                the first :class:`~afterimage.callbacks.FixedNumberStoppingCallback` in
                the merged stopping list is used if present.
        """
        if instruction_generator_callback is not None:
            warnings.warn(
                "Passing `instruction_generator_callback` to `generate()` is deprecated. "
                "Please pass it to the constructor instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            instruction_generator_callback = self.instruction_generator_callback

        if respondent_prompt_modifier is not None:
            warnings.warn(
                "Passing `respondent_prompt_modifier` to `generate()` is deprecated. "
                "Please pass it to the constructor instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            respondent_prompt_modifier = self.respondent_prompt_modifier

        if instruction_generator_callback is None:
            error = ValueError("An `instruction_generator_callback` must be provided.")
            self.monitor.log_error("No instruction generator callback set", error)
            raise error
        else:
            self.monitor.log_info(
                "Instruction generator callback set",
                type=instruction_generator_callback.__class__.__name__,
            )

        if instruction_generator_callback.monitor is None:
            instruction_generator_callback.set_monitor(self.monitor)

        if self.correspondent_prompt is None:
            self.monitor.log_info("No correspondent prompt set, initializing...")
            await self.ainitialize(instruction_generator_callback)

        # Handle stopping criteria
        final_stopping_criteria = stopping_criteria or []
        if num_dialogs is not None:
            final_stopping_criteria.append(FixedNumberStoppingCallback(n=num_dialogs))

        # Default if nothing specified
        if not final_stopping_criteria:
            num_dialogs = 5
            final_stopping_criteria.append(FixedNumberStoppingCallback(n=num_dialogs))

        effective_num_requested = num_requested
        if effective_num_requested is None:
            for c in final_stopping_criteria:
                if isinstance(c, FixedNumberStoppingCallback):
                    effective_num_requested = c.n
                    break

        # Delegate to orchestrator
        await self._orchestrator.run(
            generator=self,
            num_requested=effective_num_requested,
            max_turns=max_turns,
            stopping_criteria=final_stopping_criteria,
            instruction_generator_callback=instruction_generator_callback,
            respondent_prompt_modifier=respondent_prompt_modifier,
            storage=self.storage,
            model_provider_name=self.model_provider_name,
            max_concurrency=max_concurrency,
        )

    def _resolve_max_concurrency(self, max_concurrency: int | None) -> int:
        return resolve_generation_max_concurrency(
            self.model_provider_name,
            max_concurrency,
        )

    def to_preference_generator(
        self,
        judge,
        config=None,
        secondary_llm_provider=None,
    ):
        """Create a :class:`~afterimage.preference.PreferenceGenerator` from this generator.

        Args:
            judge: :class:`~afterimage.evaluator.ConversationJudge` for scoring responses.
            config: :class:`~afterimage.preference.PreferenceConfig` (optional).
            secondary_llm_provider: Secondary LLM for model-variation strategy (optional).

        Returns:
            Configured :class:`~afterimage.preference.PreferenceGenerator`.
        """
        from .preference.generator import PreferenceGenerator

        return PreferenceGenerator(
            conversation_generator=self,
            judge=judge,
            config=config,
            secondary_llm_provider=secondary_llm_provider,
        )
