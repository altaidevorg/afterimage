import json
import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import google.generativeai as genai
from tqdm import tqdm

from .base import (
    BaseGenerator,
    BaseInstructionGeneratorCallback,
    BaseRespondentPromptModifierCallback,
)
from .common import default_model_name, default_safety_settings
from .prompts import (
    example_correspondent_prompt,
    example_respondent_prompt,
    correspondent_instruction_creation_prompt,
)


class ConversationGenerator(BaseGenerator):
    """Generates conversations between a correspondent (question generator) and a respondent (answer generator).

    This class simulates multi-turn conversations and supports customization via callbacks for instruction generation
    and prompt modification."""

    def __init__(
        self,
        respondent_prompt,
        api_key=None,
        correspondent_prompt=None,
        model_name=None,
        safety_settings=None,
    ):
        """Initializes the ConversationGenerator.

        Args:
            respondent_prompt (str): Template for generating respondent answers.
            api_key (str): API key for the generative AI service.
            correspondent_prompt (str, optional): Template for generating correspondent questions.
                If not provided, it will be generated from the respondent prompt.
            model_name (str, optional): Model name to use. Defaults to a common default model.
            safety_settings (dict, optional): Safety settings for the model. Defaults to a pre-defined configuration.
        """
        assert api_key is not None, "You must provide an API key"

        self.model_name = model_name if model_name is not None else default_model_name
        self.safety_settings = (
            safety_settings if safety_settings is not None else default_safety_settings
        )

        genai.configure(api_key=api_key)

        self.respondent_prompt = respondent_prompt

        self.correspondent_prompt = (
            correspondent_prompt
            if correspondent_prompt is not None
            else self.create_correspondent_prompt(self.respondent_prompt)
        )

        self.initiators = []

    def create_correspondent_prompt(self, assistant_prompt):
        """Creates a correspondent prompt based on the assistant prompt.

        Args:
            assistant_prompt (str): The respondent's prompt used as context.

        Returns:
            str: The generated correspondent prompt.
        """
        prompt = correspondent_instruction_creation_prompt.format(
            example_correspondent_prompt=example_correspondent_prompt,
            example_respondent_prompt=example_respondent_prompt,
            new_assistant_prompt=assistant_prompt,
        )

        model = genai.GenerativeModel(
            self.model_name, safety_settings=self.safety_settings
        )

        correspondent_prompt = model.generate_content(prompt).text

        return correspondent_prompt

    def create_model(self, prompt):
        """Creates and initializes a chat model with the given prompt.

        Args:
            prompt (str): System instruction for the chat model.

        Returns:
            GenerativeChat: A chat model instance ready to process messages.
        """
        model = genai.GenerativeModel(
            self.model_name,
            system_instruction=prompt,
            safety_settings=self.safety_settings,
        )

        return model.start_chat(history=[])

    def ask(self, correspondent, answer):
        """Generates a question from the correspondent based on the given answer.

        Args:
            correspondent (GenerativeChat): The correspondent chat model.
            answer (str): The answer provided by the respondent.

        Returns:
            str: The generated question.
        """
        return correspondent.send_message(answer).text

    def answer(self, respondent, question):
        """Generates an answer from the respondent based on the given question.

        Args:
            respondent (GenerativeChat): The respondent chat model.
            question (str): The question provided by the correspondent.

        Returns:
            str: The generated answer.
        """
        return respondent.send_message(question).text

    def go(
        self,
        turns: int = 5,
        first_question: str | None = None,
        check_for_near_duplicates: bool = False,
        correspondent_prompt: str | None = None,
        respondent_prompt: str | None = None,
    ) -> List[Dict[str, str]]:
        """Simulates a multi-turn conversation between the correspondent and respondent.

        Args:
            turns (int, optional): Number of turns in the conversation. Defaults to 5.
            first_question (str, optional): The first question to start the conversation.
            check_for_near_duplicates (bool, optional): Whether to check for near-duplicate questions.
            correspondent_prompt (str, optional): Prompt template for the correspondent.
            respondent_prompt (str, optional): Prompt template for the respondent.

        Returns:
            List[Dict[str, str]]: A list of conversation messages.
        """
        if correspondent_prompt is None:
            correspondent_prompt = self.correspondent_prompt

        if respondent_prompt is None:
            respondent_prompt = self.respondent_prompt

        correspondent = self.create_model(correspondent_prompt)
        respondent = self.create_model(respondent_prompt)
        conversation = []

        question = first_question or self.ask(correspondent, "Ask your first question.")
        self.initiators.append(question)
        conversation.append({"role": "user", "content": question})

        for turn in range(turns):
            answer = self.answer(respondent, question)
            conversation.append({"role": "assistant", "content": answer})

            if (turn + 1) == turns:
                break

            else:
                question = self.ask(correspondent, answer)
                conversation.append({"role": "user", "content": question})
                self.initiators.append(question)

        return conversation

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
    ) -> List[Dict[str, Any]]:
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
            List[Dict[str, Any]]: A list containing the conversation and its metadata.
        """
        correspondent_prompt = self.correspondent_prompt
        respondent_prompt = self.respondent_prompt
        turns = random.randint(1, max_turns)

        conversations = []

        if instruction_generator_callback:
            gen_instructions = instruction_generator_callback(correspondent_prompt)

            for instruction in gen_instructions.instructions:
                if respondent_prompt_modifier:
                    respondent_prompt = respondent_prompt_modifier(
                        self.respondent_prompt,
                        context=gen_instructions.context,
                        instruction=instruction,
                    )

                conversation = self.go(
                    turns=turns,
                    first_question=instruction,
                    check_for_near_duplicates=check_for_near_duplicates,
                    correspondent_prompt=correspondent_prompt,
                    respondent_prompt=respondent_prompt,
                )

                conversations.append(
                    {
                        "conversations": conversation,
                        "context": gen_instructions["context"],
                    }
                )

            return conversations

        else:
            first_question = seed_questions[i] if seed_questions else None

        conversation = self.go(
            turns=turns,
            first_question=first_question,
            check_for_near_duplicates=check_for_near_duplicates,
            correspondent_prompt=correspondent_prompt,
            respondent_prompt=respondent_prompt,
        )

        conversations.append({"conversations": conversation})

        return conversations

    def generate(
        self,
        num_dialogs: int = 5,
        max_turns: int = 3,
        save_to: str | None = None,
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
            max_turns (int, optional): Maximum number of turns per dialog. Defaults to 3.
            save_to (str, optional): Path to save the generated dialogs in JSONL format. Defaults to None.
            seed_instructions (List, optional): Seed instructions to guide question generation. Defaults to [].
            add_examples (bool, optional): Whether to use seed instructions as examples. Defaults to False.
            num_random_examples (int, optional): Number of random examples to use. Defaults to 3.
            generation_examples_delay (int, optional): Delay before using generated examples. Defaults to 100.
            check_for_near_duplicates (bool, optional): Avoid generating duplicate questions. Defaults to False.
            instruction_generator_callback (callable, optional): Callback for instruction generation. Defaults to None.
            respondent_prompt_modifier (callable, optional): Callback to modify respondent prompts. Defaults to None.
            max_workers (int, optional): Number of threads for parallel execution. Defaults to 4.
        """
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

        num_generated = 0
        pbar = tqdm(n_conversations, desc="Generating conversations...")

        def save_conversations(conversations):
            if save_to and conversations:
                with open(save_to, "a+", encoding="utf8") as f:
                    for conv in conversations:
                        f.write(json.dumps(conv, ensure_ascii=False) + "\n")

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
                        save_conversations(conversations)
                        num_generated += len(conversations)
                        pbar.update(num_generated)
                        if num_generated >= n_conversations:
                            executor.shutdown(wait=False, cancel_futures=True)

                    except Exception as e:
                        warnings.warn(f"Exception in future: {e}")

        except KeyboardInterrupt:
            warnings.warn("Waiting for graceful shutdown")
            executor.shutdown(wait=True, cancel_futures=True)
        finally:
            pbar.close()
