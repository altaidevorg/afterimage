import asyncio
import logging
import random
import time
import traceback
import warnings
from typing import AsyncGenerator, Dict, List, Literal, Optional, Union

from tqdm.asyncio import tqdm

from .base import BaseGenerator, BaseStoppingCallback
from .callbacks import (
    BaseInstructionGeneratorCallback,
    BaseRespondentPromptModifierCallback,
    FixedNumberStoppingCallback,
)
from .common import default_model_name, default_safety_settings
from .evaluator import HybridSyntheticDatasetEvaluator, SimpleSyntheticDatasetEvaluator
from .key_management import SmartKeyPool
from .monitoring import GenerationMonitor
from .prompts import get_correspondent_instruction_generation_prompt
from .providers import ChatSession, LLMFactory
from .storage import BaseStorage, JSONLStorage
from .types import (
    Conversation,
    ConversationEntry,
    ConversationWithContext,
    EvaluatedConversationWithContext,
    GenerationState,
    GradeSchema,
    Role,
)

logger = logging.getLogger(__name__)


class AsyncConversationGenerator(BaseGenerator):
    """Generates conversations between a correspondent (question generator) and a respondent (answer generator) asynchronously."""

    def __init__(
        self,
        respondent_prompt: str,
        api_key: str | SmartKeyPool,
        correspondent_prompt: str | None = None,
        model_name: str | None = None,
        safety_settings: List[Dict[str, str]] | None = None,
        auto_improve: bool = False,
        evaluator_model_name: str | None = None,
        model_provider_name: Literal["gemini", "openai"] = "gemini",
        evaluator_method: Literal["simple", "hybrid"] = "simple",
        storage: Optional[BaseStorage] = None,
        monitor: Optional[GenerationMonitor] = None,
        instruction_generator_callback: BaseInstructionGeneratorCallback | None = None,
        respondent_prompt_modifier: BaseRespondentPromptModifierCallback | None = None,
    ):
        """Initialize the generator with API key(s).

        Args:
            respondent_prompt: System prompt to the respondent, e.g., assistant that you want you fine-tune on this dataset
            api_key: Either a single API key string or a SmartKeyPool instance for LLM use
            correspondent_prompt: System prompt to the correspondent, e.g., model that roleplays a user of the assistant
                that you want to fine-tune on this dataset
            model_name: Model name to use
            safety_settings: Safety settings for the model
            auto_improve: Whether to try to improve low-quality generations
            evaluator_model_name: Model name for the evaluator when auto_improve is True
            evaluator_method: method to be used for evaluation.
            model_provider_name: Provider used for accessing LLMs. `"gemini"` or `"openai"` for Openai-compatible APIs.
            storage: Storage implementation for saving conversations.
                    If `None`, creates JSONLStorage with datetime-based filename.
            monitor: GenerationMonitor instance for tracking generation metrics.
                If  `None`, a default one is created.
            instruction_generator_callback: Callback for instruction generation. Can also be passed to generate() method (deprecated).
            respondent_prompt_modifier: Callback to modify respondent prompts. Can also be passed to generate() method (deprecated).
        """
        self.monitor: GenerationMonitor = (
            monitor or GenerationMonitor()
        )  # ensure it's always created

        self.key_pool: SmartKeyPool = (
            api_key
            if isinstance(api_key, SmartKeyPool)
            else SmartKeyPool.from_single_key(api_key)
        )

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

        self.evaluator = None
        if auto_improve:
            evaluator_model_name = (
                evaluator_model_name
                if evaluator_model_name is not None
                else self.model_name
            )
            if evaluator_method == "simple":
                self.evaluator = SimpleSyntheticDatasetEvaluator(
                    api_key=self.key_pool,
                    model_name=evaluator_model_name,
                    safety_settings=self.safety_settings,
                    monitor=self.monitor,
                )
            elif evaluator_method == "hybrid":
                evaluator_llm = LLMFactory.create(
                    self.model_provider_name, evaluator_model_name, self.key_pool
                )
                self.evaluator = HybridSyntheticDatasetEvaluator(
                    llm=evaluator_llm, monitor=self.monitor
                )

        self.initiators = []
        self.storage = storage or JSONLStorage()

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
                "gemini",
                "gemini-2.5-pro",
                api_key=api_key,
                safety_settings=self.safety_settings,
            )

            response = await model.agenerate_content(prompt=prompt, temperature=0.7)
            prompt = (
                response.text.strip()
                .lstrip("<user_system_prompt>")
                .rstrip("</user_system_prompt>")
                .strip()
            )

            if self.monitor:
                self.monitor.record_metric(
                    "prompt_generation_time",
                    time.time() - start_time,
                    metadata={
                        "prompt_type": "correspondent",
                        "success": True,
                    },
                )

            return prompt
        except Exception as e:
            if self.monitor:
                self.monitor.record_metric(
                    "prompt_generation_time",
                    time.time() - start_time,
                    metadata={
                        "prompt_type": "correspondent",
                        "success": False,
                        "error": str(e),
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
                self.model_provider_name,
                self.model_name,
                api_key=api_key,
                system_instruction=prompt,
                safety_settings=self.safety_settings,
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
    ) -> str:
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

            return answer
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

            question = first_question or await self.ask(
                correspondent, "Ask your first question."
            )
            self.initiators.append(question)
            conversation.append(ConversationEntry(role=Role.USER, content=question))

            for turn in range(turns):
                answer = await self.answer(respondent, question)

                conversation.append(
                    ConversationEntry(role=Role.ASSISTANT, content=answer)
                )

                if (turn + 1) == turns:
                    break
                else:
                    question = await self.ask(correspondent, answer)

                    conversation.append(
                        ConversationEntry(role=Role.USER, content=question)
                    )
                    self.initiators.append(question)

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

                conversation = await self.go(
                    turns=turns,
                    first_question=instruction,
                    check_for_near_duplicates=check_for_near_duplicates,
                    correspondent_prompt=correspondent_prompt,
                    respondent_prompt=current_respondent_prompt,
                )

                conversation_row = ConversationWithContext(
                    conversations=conversation,
                    instruction_context=instruction_context,
                    response_context=response_context,
                    persona=persona,
                    metadata={
                        "context_id": gen_instructions.context_id,
                        "persona_name": persona,
                    },
                )

                evaluation_grade = GradeSchema.NOT_ACCEPTABLE
                while self.evaluator and evaluation_grade in [
                    GradeSchema.NOT_ACCEPTABLE,
                    GradeSchema.BAD,
                    GradeSchema.NEEDS_IMPROVEMENT,
                ]:
                    evaluated_conversation = await asyncio.to_thread(
                        self.evaluator.evaluate_row, conversation_row
                    )

                    if evaluated_conversation.evaluation.overall_grade in [
                        GradeSchema.NOT_ACCEPTABLE,
                        GradeSchema.BAD,
                        GradeSchema.NEEDS_IMPROVEMENT,
                    ]:
                        conversation = await self.go(
                            turns=turns,
                            first_question=instruction,
                            check_for_near_duplicates=check_for_near_duplicates,
                            correspondent_prompt=correspondent_prompt,
                            respondent_prompt=respondent_prompt,
                        )
                    else:
                        evaluation_grade = (
                            evaluated_conversation.evaluation.overall_grade
                        )
                        conversation_row = evaluated_conversation

                yield conversation_row
        else:
            raise ValueError("An `instruction_generator_callback` must be provided.")

    async def generate(
        self,
        num_dialogs: int | None = None,
        max_turns: int = 1,
        max_concurrency: int = 4,
        stopping_criteria: Optional[List[BaseStoppingCallback]] = None,
        instruction_generator_callback: BaseInstructionGeneratorCallback | None = None,
        respondent_prompt_modifier: BaseRespondentPromptModifierCallback | None = None,
    ) -> None:
        """Generates multiple conversation dialogs until stopping criteria is met.

        Args:
            num_dialogs (int, optional): Number of dialogs to generate. Defaults to 5.
            max_turns (int, optional): Maximum number of turns per dialog. Actual number of turns is randomly sampled from 1 .. max_turns.
            stopping_criteria: A list of callbacks to determine when to stop generation. If num_dialogs is specified, FixedNumberStoppingCallback is added to this list.
            instruction_generator_callback (callable, optional): Callback for instruction generation.
                Deprecated: Pass this to the constructor instead. Defaults to None.
            respondent_prompt_modifier (callable, optional): Callback to modify respondent prompts.
                Deprecated: Pass this to the constructor instead. Defaults to None.
            max_concurrency (int, optional): Number of concurrent generations. Defaults to 4.
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

        state = GenerationState(
            num_requested=num_dialogs or 0,
            monitor=self.monitor,
            stop_event=asyncio.Event(),
        )

        semaphore = asyncio.Semaphore(max_concurrency)

        async def save_conversations(conversations: list[ConversationWithContext]):
            if conversations:
                if hasattr(self.storage, "asave_conversations"):
                    await self.storage.asave_conversations(conversations)
                else:
                    await asyncio.to_thread(
                        self.storage.save_conversations, conversations
                    )

        async def worker_task():
            while not state.stop_event.is_set():
                async with semaphore:
                    if state.stop_event.is_set():
                        break

                    try:
                        async for conversation in self.generate_single(
                            max_turns=max_turns,
                            instruction_generator_callback=instruction_generator_callback,
                            respondent_prompt_modifier=respondent_prompt_modifier,
                        ):
                            # Update state and check stopping criteria
                            state.update(conversation)

                            for criteria in final_stopping_criteria:
                                if await criteria.should_stop(state):
                                    self.monitor.log_info(
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
                        if self.monitor:
                            self.monitor.record_metric("error_rate", 1.0)
                        continue

        pbar = tqdm(total=num_dialogs, desc="Generating...", unit="conversation")
        tasks: list[asyncio.Task] = []

        # Create initial set of tasks
        for _ in range(max_concurrency):
            tasks.append(asyncio.create_task(worker_task()))

        last_count = 0
        while not state.stop_event.is_set() or any(not t.done() for t in tasks):
            # Update progress bar
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

        # Wait for any remaining tasks to finish/cancel
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            self.monitor.log_error("Error while trying to finalize generation", error=e)
            self.monitor.record_metric("error_rate", 1.0)
            traceback.print_exc()
        finally:
            self.monitor.log_info("Generation complete")
            self.monitor.visualize_metrics()
