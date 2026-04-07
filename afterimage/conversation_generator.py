import traceback
import random
import warnings
from concurrent.futures import (
    CancelledError,
    ThreadPoolExecutor,
    as_completed,
)
from typing import Dict, List, Literal, Optional
import time

from tqdm import tqdm

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


class ConversationGenerator(BaseGenerator):
    """Generates conversations between a correspondent (question generator) and a respondent (answer generator).

    This class simulates multi-turn conversations and supports customization via callbacks for instruction generation
    and prompt modification."""

    def __init__(
        self,
        respondent_prompt: str,
        api_key: str | SmartKeyPool,
        correspondent_prompt: str | None = None,
        model_name: str | None = None,
        safety_settings: List[Dict[str, str]] | None = None,
        auto_improve: bool = True,
        evaluator_model_name: str | None = None,
        model_provider_name: Literal["gemini", "openai", "deepseek"] = "gemini",
        evaluator_method: Literal["simple", "hybrid"] = "simple",
        storage: Optional[BaseStorage] = None,
        monitor: Optional[GenerationMonitor] = None,
        instruction_generator_callback: BaseInstructionGeneratorCallback | None = None,
        respondent_prompt_modifier: BaseRespondentPromptModifierCallback | None = None,
    ):
        """Initialize the generator with API key(s).

        Args:
            respondent_prompt: System prompt to the respondent, e.g., assistant that you want you fine-tune on this dataset
            api_key: Either a single API key string or a SmartKeyPool instance
            correspondent_prompt: System prompt to the correspondent, e.g., model that roleplays a user of the assistant
                that you want to fine-tune on this dataset.
            model_name: Model name to use
            safety_settings: Safety settings for the model
            auto_improve: Whether to try to improve low-quality generations
            evaluator_model_name: Model name for the evaluator when auto_improve is True
            evaluator_method: method to be used for evaluation.
            model_provider_name: Provider used for accessing LLMs. Supported values are `"gemini"`, `"openai"`, and `"deepseek"`.
            storage: Storage implementation for saving conversations
                    If None, creates JSONLStorage with datetime-based filename
            monitor: GenerationMonitor instance for tracking generation metrics
            instruction_generator_callback: Callback for instruction generation. Can also be passed to generate() method (deprecated).
            respondent_prompt_modifier: Callback to modify respondent prompts. Can also be passed to generate() method (deprecated).
        """
        warnings.warn(
            "This synchronous implementation is deprecated and may be removed in the future. Consider using AsyncConversationGenerator class instead."
        )
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

    def create_correspondent_prompt(self, assistant_prompt: str) -> str:
        """Create a correspondent prompt based on the assistant prompt."""
        start_time = time.time()
        try:
            prompt = get_correspondent_instruction_generation_prompt(
                assistant_prompt=assistant_prompt
            )
            api_key = self.key_pool.get_next_key()
            model = LLMFactory.create(
                self.model_provider_name,
                self.model_name,
                api_key=api_key,
                safety_settings=self.safety_settings,
            )

            response = model.generate_content(prompt=prompt, temperature=0.7)

            if self.monitor:
                self.monitor.record_metric(
                    "prompt_generation_time",
                    time.time() - start_time,
                    metadata={
                        "prompt_type": "correspondent",
                        "success": True,
                    },
                )

            return response.text
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
            self.key_pool.report_error(api_key)
            raise

    def create_model(self, prompt: str) -> ChatSession:
        """Creates and initializes a chat model with the given prompt."""
        start_time = time.time()
        try:
            api_key = self.key_pool.get_next_key()
            model = LLMFactory.create(
                self.model_provider_name,
                self.model_name,
                api_key=api_key,
                system_instruction=prompt,
                safety_settings=self.safety_settings,
            )

            chat = model.start_chat()

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
            self.key_pool.report_error(api_key)
            raise

    def ask(self, correspondent: ChatSession, answer: str | ConversationEntry) -> str:
        """Generates a question from the correspondent based on the given answer."""
        start_time = time.time()
        try:
            response = correspondent.send_message(answer)
            question = response.text

            if self.monitor:
                self.monitor.track_generation(
                    duration=time.time() - start_time,
                    success=True,
                    prompt_token_count=response.prompt_token_count,
                    completion_token_count=response.completion_token_count,
                    total_token_count=response.total_token_count,
                    model_name=response.model_name,
                    metadata={
                        "operation": "question_generation",
                        "answer_length": len(answer),
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

    def answer(self, respondent: ChatSession, question: str | ConversationEntry) -> str:
        """Generates an answer from the respondent based on the given question."""
        start_time = time.time()
        try:
            response = respondent.send_message(question)
            answer = response.text

            if self.monitor:
                self.monitor.track_generation(
                    duration=time.time() - start_time,
                    success=True,
                    prompt_token_count=response.prompt_token_count,
                    completion_token_count=response.completion_token_count,
                    total_token_count=response.total_token_count,
                    model_name=response.model_name,
                    metadata={
                        "operation": "answer_generation",
                        "question_length": len(question),
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

    def go(
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
                    correspondent_prompt = self.create_correspondent_prompt(
                        self.respondent_prompt
                    )

            if respondent_prompt is None:
                respondent_prompt = self.respondent_prompt

            correspondent = self.create_model(correspondent_prompt)
            respondent = self.create_model(respondent_prompt)

            # Track token usage if available
            if hasattr(correspondent, "token_count"):
                total_tokens += correspondent.token_count
            if hasattr(respondent, "token_count"):
                total_tokens += respondent.token_count

            question = first_question or self.ask(
                correspondent, "Ask your first question."
            )
            self.initiators.append(question)
            conversation.append(ConversationEntry(role=Role.USER, content=question))

            for turn in range(turns):
                answer = self.answer(respondent, question)
                if hasattr(respondent, "token_count"):
                    total_tokens += respondent.token_count

                conversation.append(
                    ConversationEntry(role=Role.ASSISTANT, content=answer)
                )

                if (turn + 1) == turns:
                    break
                else:
                    question = self.ask(correspondent, answer)
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

    def generate_single(
        self,
        i,
        count,
        max_turns,
        seed_questions,
        add_examples,
        num_random_examples,
        generation_examples_delay,
        check_for_near_duplicates,
        instruction_generator_callback,
        respondent_prompt_modifier,
    ) -> List[EvaluatedConversationWithContext] | List[Conversation]:
        """Generates a single conversation session.

        Args:
            i (int): Index of the current conversation.
            count (int): Total number of conversations to generate.
            max_turns (int): Maximum number of turns per conversation.
            seed_questions (List[str]): Seed questions to guide the conversation.
            add_examples (bool): Whether to add seed questions as examples.
            num_random_examples (int): Number of random examples to add.
            generation_examples_delay (int): Delay before adding generated examples as seeds.
            check_for_near_duplicates (bool): Whether to avoid duplicate questions.
            instruction_generator_callback (callable): Callback to generate instructions.
            respondent_prompt_modifier (callable): Callback to modify respondent prompts.

        Returns:
            List[EvaluatedConversationWithContext]: A list containing the generated conversations in this session and their metadata.
        """
        correspondent_prompt = self.correspondent_prompt
        respondent_prompt = self.respondent_prompt
        turns = random.randint(1, max_turns)
        conversations = []

        if instruction_generator_callback:
            # If correspondent_prompt is None, try to create it using the callback
            if correspondent_prompt is None:
                created_prompt = (
                    instruction_generator_callback.create_correspondent_prompt(
                        respondent_prompt
                    )
                )
                if created_prompt is not None:
                    correspondent_prompt = created_prompt
                else:
                    # Fallback to generator's method if callback doesn't implement it
                    correspondent_prompt = self.create_correspondent_prompt(
                        respondent_prompt
                    )

            gen_instructions = instruction_generator_callback(correspondent_prompt)

            for instruction in gen_instructions.instructions:
                instruction_context = gen_instructions.context
                persona = gen_instructions.persona
                response_context = None
                if respondent_prompt_modifier:
                    modified_respondent_prompt = respondent_prompt_modifier(
                        self.respondent_prompt,
                        context=instruction_context,
                        instruction=instruction,
                    )
                    respondent_prompt = modified_respondent_prompt.prompt
                    response_context = modified_respondent_prompt.context

                conversation = self.go(
                    turns=turns,
                    first_question=instruction,
                    check_for_near_duplicates=check_for_near_duplicates,
                    correspondent_prompt=correspondent_prompt,
                    respondent_prompt=respondent_prompt,
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
                        },
                    )

                conversation_row = build_conversation_row(conversation)

                evaluation_grade = GradeSchema.NOT_ACCEPTABLE
                while self.evaluator and evaluation_grade in [
                    GradeSchema.NOT_ACCEPTABLE,
                    GradeSchema.BAD,
                    GradeSchema.NEEDS_IMPROVEMENT,
                ]:
                    evaluated_conversation = self.evaluator.evaluate_row(
                        conversation_row
                    )

                    if evaluated_conversation.evaluation.overall_grade in [
                        GradeSchema.NOT_ACCEPTABLE,
                        GradeSchema.BAD,
                        GradeSchema.NEEDS_IMPROVEMENT,
                    ]:
                        conversation = self.go(
                            turns=turns,
                            first_question=instruction,
                            check_for_near_duplicates=check_for_near_duplicates,
                            correspondent_prompt=correspondent_prompt,
                            respondent_prompt=respondent_prompt,
                        )
                        conversation_row = build_conversation_row(conversation)
                    else:
                        evaluation_grade = (
                            evaluated_conversation.evaluation.overall_grade
                        )
                        conversation_row = evaluated_conversation

                conversations.append(conversation_row)

            return conversations

        else:
            first_question = seed_questions[i] if seed_questions else None
            # If correspondent_prompt is None and no callback, create it using generator's method
            if correspondent_prompt is None:
                correspondent_prompt = self.create_correspondent_prompt(
                    respondent_prompt
                )

        conversation = self.go(
            turns=turns,
            first_question=first_question,
            check_for_near_duplicates=check_for_near_duplicates,
            correspondent_prompt=correspondent_prompt,
            respondent_prompt=respondent_prompt,
        )

        conversations.append(Conversation(conversations=conversation))

        return conversations

    def generate(
        self,
        num_dialogs: int = 5,
        max_turns: int = 1,
        seed_instructions: List = [],
        add_examples: bool = False,
        num_random_examples: int = 3,
        generation_examples_delay: int = 100,
        check_for_near_duplicates: bool = False,
        instruction_generator_callback: BaseInstructionGeneratorCallback | None = None,
        respondent_prompt_modifier: BaseRespondentPromptModifierCallback | None = None,
        max_workers: int = 4,
    ) -> None:
        """Generates multiple conversation dialogs and saves them to a file if specified.

        Args:
            num_dialogs (int, optional): Number of dialogs to generate. Defaults to 5.
            max_turns (int, optional): Maximum number of turns per dialog. Actual number of turns is randomly sampled from 1 .. max_turns.
            seed_instructions (List, optional): Seed instructions to guide question generation. Defaults to [].
            add_examples (bool, optional): Whether to use seed instructions as examples. Defaults to False.
            num_random_examples (int, optional): Number of random examples to use. Defaults to 3.
            generation_examples_delay (int, optional): Delay before using generated examples. Defaults to 100.
            check_for_near_duplicates (bool, optional): Avoid generating duplicate questions. Defaults to False.
            instruction_generator_callback (callable, optional): Callback for instruction generation.
                Deprecated: Pass this to the constructor instead. Defaults to None.
            respondent_prompt_modifier (callable, optional): Callback to modify respondent prompts.
                Deprecated: Pass this to the constructor instead. Defaults to None.
            max_workers (int, optional): Number of threads for parallel execution. Defaults to 4.
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
                warnings.warn(
                    f"`num_dialogs` is set to {n_conversations} because you set {n_conversations} seed instructions"
                )

        self.initialize(instruction_generator_callback)
        self._configure_persona_sampling(
            instruction_generator_callback,
            num_requested=num_dialogs,
        )

        num_generated = 0
        pbar = tqdm(total=n_conversations, desc="Generating...", unit="conversation")

        def save_conversations(conversations: List[ConversationWithContext]):
            if conversations:
                self.storage.save_conversations(conversations)

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self.generate_single,
                        i,
                        num_dialogs,
                        max_turns,
                        seed_instructions,
                        add_examples,
                        num_random_examples,
                        generation_examples_delay,
                        check_for_near_duplicates,
                        instruction_generator_callback,
                        respondent_prompt_modifier,
                    )
                    for i in range(num_dialogs)
                ]

                for future in as_completed(futures):
                    try:
                        conversations = future.result()
                    except Exception as e:
                        if not isinstance(e, CancelledError):
                            warnings.warn(f"Exception in future: {e}")
                            traceback.print_exc()
                    else:
                        for conversation in conversations:
                            self._record_context_usage(
                                instruction_generator_callback,
                                conversation,
                            )
                        save_conversations(conversations)
                        num_generated += len(conversations)
                        pbar.update(len(conversations))

                        if num_generated >= n_conversations:
                            pbar.close()
                            print("Done! Waiting for graceful shutdown...")
                            for pending_future in futures:
                                pending_future.cancel()

        except KeyboardInterrupt:
            pbar.close()
            warnings.warn("Interrupted! Waiting for graceful shutdown...")

        finally:
            pbar.close()
