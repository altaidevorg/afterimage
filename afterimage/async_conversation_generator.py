import traceback
import random
import warnings
import asyncio
from typing import AsyncGenerator, Dict, List, Literal, Optional, Union
import time

from tqdm.asyncio import tqdm

from .base import (
    BaseGenerator,
    BaseInstructionGeneratorCallback,
    BaseRespondentPromptModifierCallback,
)
from .common import default_model_name, default_safety_settings
from .evaluator import SimpleSyntheticDatasetEvaluator, HybridSyntheticDatasetEvaluator
from .prompts import get_correspondent_instruction_generation_prompt
from .types import (
    ConversationEntry,
    Conversation,
    ConversationWithContext,
    EvaluatedConversationWithContext,
    GradeSchema,
    Role,
)
from .key_management import SmartKeyPool
from .providers import ChatSession, LLMFactory
from .monitoring import GenerationMonitor
from .storage import BaseStorage, JSONLStorage


class AsyncConversationGenerator(BaseGenerator):
    """Generates conversations between a correspondent (question generator) and a respondent (answer generator) asynchronously."""

    def __init__(
        self,
        respondent_prompt: str,
        api_key: str | SmartKeyPool,
        correspondent_prompt: str | None = None,
        model_name: str | None = None,
        safety_settings: List[Dict[str, str]] | None = None,
        auto_improve: bool = True,
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
            storage: Storage implementation for saving conversations
                    If None, creates JSONLStorage with datetime-based filename
            monitor: GenerationMonitor instance for tracking generation metrics
            instruction_generator_callback: Callback for instruction generation. Can also be passed to generate() method (deprecated).
            respondent_prompt_modifier: Callback to modify respondent prompts. Can also be passed to generate() method (deprecated).
        """
        self.monitor = monitor
        self.key_pool = (
            api_key
            if isinstance(api_key, SmartKeyPool)
            else SmartKeyPool.from_single_key(api_key)
        )

        self.model_provider_name = model_provider_name
        self.model_name = model_name if model_name is not None else default_model_name
        self.safety_settings = (
            safety_settings if safety_settings is not None else default_safety_settings
        )

        # users should pass at least one of correspondent prompt and instruction generator callback.
        # if both are passed, the correspondent prompt will be used as is passed.
        # if neither are passed, raise an error.
        if correspondent_prompt is None and instruction_generator_callback is None:
            raise ValueError(
                "At least one of `correspondent_prompt` or `instruction_generator_callback` should be passed."
            )
        if correspondent_prompt is None:
            warnings.warn(
                "A correspondent prompt will be automatically created because you did not pass one."
            )

        self.respondent_prompt = respondent_prompt
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
                    self.model_provider_name, self.evaluator_model_name, self.key_pool
                )
                self.evaluator = HybridSyntheticDatasetEvaluator(
                    llm=evaluator_llm, monitor=self.monitor
                )

        self.initiators = []
        self.storage = storage or JSONLStorage()

    async def initialize(self, instruction_generator_callback=None):
        """Initializes the generator by creating the correspondent prompt if it doesn't exist."""
        if self.correspondent_prompt is None:
            # Use provided callback if given, otherwise use instance attribute
            callback = (
                instruction_generator_callback or self.instruction_generator_callback
            )
            # Try to use callback first if available
            if callback is not None:
                if hasattr(callback, "acreate_correspondent_prompt"):
                    created_prompt = await callback.acreate_correspondent_prompt(
                        self.respondent_prompt
                    )
                else:
                    created_prompt = await asyncio.to_thread(
                        callback.create_correspondent_prompt, self.respondent_prompt
                    )
                if created_prompt is not None:
                    self.correspondent_prompt = created_prompt
                    self.log_correspondent_prompt(self.correspondent_prompt)
                    return
            # Fallback to generator's method
            self.correspondent_prompt = await self.create_correspondent_prompt(
                self.respondent_prompt
            )
        self.log_correspondent_prompt(self.correspondent_prompt)

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

            if self.monitor:
                self.monitor.track_generation(
                    duration=time.time() - start_time,
                    success=True,
                    metadata={
                        "operation": "question_generation",
                        "answer_length": len(answer)
                        if isinstance(answer, str)
                        else len(answer.content),
                        "question_length": len(question),
                    },
                )

            return question
        except Exception as e:
            if self.monitor:
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

            if self.monitor:
                self.monitor.track_generation(
                    duration=time.time() - start_time,
                    success=True,
                    metadata={
                        "operation": "answer_generation",
                        "question_length": len(question)
                        if isinstance(question, str)
                        else len(question.content),
                        "answer_length": len(answer),
                    },
                )

            return answer
        except Exception as e:
            if self.monitor:
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
        turns: int = 5,
        first_question: str | None = None,
        check_for_near_duplicates: bool = False,
        correspondent_prompt: str | None = None,
        respondent_prompt: str | None = None,
    ) -> List[ConversationEntry]:
        """Simulates a multi-turn conversation between the correspondent and respondent."""
        start_time = time.time()
        total_tokens = 0
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

            if hasattr(correspondent, "token_count"):
                total_tokens += correspondent.token_count
            if hasattr(respondent, "token_count"):
                total_tokens += respondent.token_count

            question = first_question or await self.ask(
                correspondent, "Ask your first question."
            )
            self.initiators.append(question)
            conversation.append(ConversationEntry(role=Role.USER, content=question))

            for turn in range(turns):
                answer = await self.answer(respondent, question)
                if hasattr(respondent, "token_count"):
                    total_tokens += respondent.token_count

                conversation.append(
                    ConversationEntry(role=Role.ASSISTANT, content=answer)
                )

                if (turn + 1) == turns:
                    break
                else:
                    question = await self.ask(correspondent, answer)
                    if hasattr(correspondent, "token_count"):
                        total_tokens += correspondent.token_count

                    conversation.append(
                        ConversationEntry(role=Role.USER, content=question)
                    )
                    self.initiators.append(question)

            if self.monitor:
                self.monitor.track_generation(
                    duration=time.time() - start_time,
                    success=True,
                    tokens=total_tokens,
                    turns=len(conversation) // 2,
                    metadata={
                        "operation": "conversation_generation",
                        "planned_turns": turns,
                        "actual_turns": len(conversation) // 2,
                    },
                )

            return conversation

        except Exception as e:
            if self.monitor:
                self.monitor.track_generation(
                    duration=time.time() - start_time,
                    success=False,
                    error=str(e),
                    tokens=total_tokens,
                    metadata={
                        "operation": "conversation_generation",
                        "error_type": e.__class__.__name__,
                        "completed_turns": len(conversation) // 2,
                    },
                )
            raise

    async def generate_single(
        self,
        i,
        max_turns,
        seed_questions,
        add_examples,
        num_random_examples,
        generation_examples_delay,
        check_for_near_duplicates,
        instruction_generator_callback,
        respondent_prompt_modifier,
    ) -> AsyncGenerator[Union[EvaluatedConversationWithContext, Conversation], None]:
        """Generates conversations for a single session and yields them."""
        correspondent_prompt = self.correspondent_prompt
        respondent_prompt = self.respondent_prompt
        turns = random.randint(1, max_turns)

        if instruction_generator_callback:
            # If correspondent_prompt is None, try to create it using the callback
            if correspondent_prompt is None:
                if hasattr(
                    instruction_generator_callback, "acreate_correspondent_prompt"
                ):
                    created_prompt = await instruction_generator_callback.acreate_correspondent_prompt(
                        respondent_prompt
                    )
                else:
                    created_prompt = await asyncio.to_thread(
                        instruction_generator_callback.create_correspondent_prompt,
                        respondent_prompt,
                    )
                if created_prompt is not None:
                    correspondent_prompt = created_prompt
                else:
                    # Fallback to generator's method if callback doesn't implement it
                    correspondent_prompt = await self.create_correspondent_prompt(
                        respondent_prompt
                    )

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
            first_question = seed_questions[i] if seed_questions else None
            # If correspondent_prompt is None and no callback, create it using generator's method
            if correspondent_prompt is None:
                correspondent_prompt = await self.create_correspondent_prompt(
                    respondent_prompt
                )
            conversation = await self.go(
                turns=turns,
                first_question=first_question,
                check_for_near_duplicates=check_for_near_duplicates,
                correspondent_prompt=correspondent_prompt,
                respondent_prompt=respondent_prompt,
            )
            yield Conversation(conversations=conversation)

    async def generate(
        self,
        num_dialogs: int = 5,
        max_turns: int = 3,
        seed_instructions: List = [],
        add_examples: bool = False,
        num_random_examples: int = 3,
        generation_examples_delay: int = 100,
        check_for_near_duplicates: bool = False,
        instruction_generator_callback: BaseInstructionGeneratorCallback | None = None,
        respondent_prompt_modifier: BaseRespondentPromptModifierCallback | None = None,
        max_concurrency: int = 4,
    ) -> None:
        """Generates multiple conversation dialogs and saves them to a file if specified.

        Args:
            num_dialogs (int, optional): Number of dialogs to generate. Defaults to 5.
            max_turns (int, optional): Maximum number of turns per dialog. Defaults to 3.
            seed_instructions (List, optional): Seed instructions to guide question generation. Defaults to [].
            add_examples (bool, optional): Whether to use seed instructions as examples. Defaults to False.
            num_random_examples (int, optional): Number of random examples to use. Defaults to 3.
            generation_examples_delay (int, optional): Delay before using generated examples. Defaults to 100.
            check_for_near_duplicates (bool, optional): Avoid generating duplicate questions. Defaults to False.
            instruction_generator_callback (callable, optional): Callback for instruction generation.
                Deprecated: Pass this to the constructor instead. Defaults to None.
            respondent_prompt_modifier (callable, optional): Callback to modify respondent prompts.
                Deprecated: Pass this to the constructor instead. Defaults to None.
            max_concurrency (int, optional): Number of concurrent generations. Defaults to 4.
        """
        # Use provided callbacks if given, otherwise use instance attributes
        # Show deprecation warnings if callbacks are provided as arguments
        if instruction_generator_callback is not None:
            warnings.warn(
                "Passing `instruction_generator_callback` to `generate()` is deprecated and may be removed in a future version. "
                "Please pass it to the constructor instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            instruction_generator_callback = self.instruction_generator_callback

        if respondent_prompt_modifier is not None:
            warnings.warn(
                "Passing `respondent_prompt_modifier` to `generate()` is deprecated and may be removed in a future version. "
                "Please pass it to the constructor instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            respondent_prompt_modifier = self.respondent_prompt_modifier

        n_conversations = num_dialogs
        gen_iter = None

        if instruction_generator_callback is not None:
            if add_examples:
                warnings.warn(
                    "You set `add_examples`, but `instruction_generator_callback` will take precedence, and examples will be ignored."
                )

            if seed_instructions:
                warnings.warn(
                    "You set `seed_instructions`, but `instruction_generator_callback` will take precedence, and `seed_instructions`will be ignored."
                )

            if seed_instructions and not add_examples:
                n_conversations = len(seed_instructions)
                gen_iter = iter(range(n_conversations))
                warnings.warn(
                    f"`num_dialogs` is set to {n_conversations} because you set {n_conversations} seed instructions"
                )

        await self.initialize(instruction_generator_callback)

        pbar = tqdm(total=num_dialogs, desc="Generating...", unit="conversation")
        stop = asyncio.Event()
        num_generated = 0
        semaphore = asyncio.Semaphore(max_concurrency)
        tasks: list[asyncio.Task] = []

        async def save_conversations(conversations: list[ConversationWithContext]):
            if conversations:
                if hasattr(self.storage, "asave_conversations"):
                    await self.storage.asave_conversations(conversations)
                else:
                    await asyncio.to_thread(
                        self.storage.save_conversations, conversations
                    )

        async def worker_task():
            nonlocal num_generated
            async with semaphore:
                i = next(gen_iter) if gen_iter else 0
                async for conv in self.generate_single(
                    i,
                    max_turns,
                    seed_instructions,
                    add_examples,
                    num_random_examples,
                    generation_examples_delay,
                    check_for_near_duplicates,
                    instruction_generator_callback,
                    respondent_prompt_modifier,
                ):
                    if stop.is_set():
                        break
                    await save_conversations([conv])
                    num_generated += 1
                    pbar.update(1)

                    if num_generated >= n_conversations:
                        stop.set()
                        break

        # dynamically spawn tasks only when needed
        while not stop.is_set() and num_generated < num_dialogs:
            t = asyncio.create_task(worker_task())
            tasks.append(t)
            # optional tiny sleep to let loop schedule
            await asyncio.sleep(0.001)

        # wait for all spawned tasks to finish/cancel cleanly
        for future in asyncio.as_completed(tasks):
            try:
                await future
                if stop.is_set():
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                    break
            except asyncio.CancelledError:
                # swallow cancellations
                pass
            except Exception as e:
                traceback.print_exc()
                warnings.warn(f"Exception in future: {str(e)}")

        pbar.close()
