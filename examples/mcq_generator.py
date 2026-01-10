import asyncio
import os
from enum import Enum
from typing import List
from pydantic import BaseModel, Field

from afterimage import (
    AsyncStructuredGenerator,
    PersonaGenerator,
    PersonaInstructionGeneratorCallback,
    InMemoryDocumentProvider,
)

# --- Schema Definitions ---


class DifficultyLevel(str, Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


class MultipleChoiceQuestion(BaseModel):
    question_text: str = Field(
        description="The main text of the question. It should be clear and unambiguous."
    )
    options: List[str] = Field(
        description="A list of 4 possible answers.", min_items=4, max_items=4
    )
    correct_option_index: int = Field(
        description="The index of the correct option in the options list (0-3)."
    )
    explanation: str = Field(
        description="A detailed explanation of why the correct answer is right and why others are wrong."
    )
    difficulty: DifficultyLevel = Field(
        description="The estimated difficulty level of the question."
    )
    topic: str = Field(description="The specific sub-topic this question covers.")


class MCQList(BaseModel):
    questions: List[MultipleChoiceQuestion] = Field(
        ..., description="A list of multiple choice questions."
    )


# --- Educational Context ---

HISTORY_TEXTS = [
    """
    The Industrial Revolution was the transition to new manufacturing processes in Great Britain, continental Europe, and the United States, that occurred during the period from around 1760 to about 1820–1840. This transition included going from hand production methods to machines, new chemical manufacturing and iron production processes, the increasing use of steam power and water power, the development of machine tools and the rise of the mechanized factory system. Output greatly increased, and a result was an unprecedented rise in population and the rate of population growth.
    """,
    """
    The French Revolution was a period of radical political and societal change in France that began with the Estates General of 1789 and ended with the formation of the French Consulate in November 1799. Many of its ideas are considered fundamental principles of liberal democracy, while phrases like liberté, égalité, fraternité reappeared in other revolts, such as the 1917 Russian Revolution, and inspired campaigns for the abolition of slavery and universal suffrage.
    """,
    """
    The meiji restoration, referred to at the time as the Honorable Restoration, was a political event that restored practical imperial rule to Japan in 1868 under Emperor Meiji. Although there were ruling Emperors before the Meiji Restoration, the events restored practical abilities and consolidated the political system under the Emperor of Japan. The goals of the restored government were expressed by the new Emperor in the Charter Oath. The Restoration led to enormous changes in Japan's political and social structure and spanned both the late Edo period (often called the Bakumatsu) and the beginning of the Meiji era.
    """,
]


async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable.")
        return

    # 1. Setup Document Provider
    docs = InMemoryDocumentProvider(HISTORY_TEXTS)

    # 2. Setup Persona Generator
    # We generate "Student" personas who might ask for specific types of questions
    # or "Teacher" personas who set the context?
    # In this flow, the PersonaInstructionGeneratorCallback generates the "Prompt" to the agent.
    # So a persona like "Exam Prep Student" might say "Give me a hard question about this topic".
    persona_gen = PersonaGenerator(api_key=api_key)
    await persona_gen.generate_from_documents(docs)

    # 3. Setup Instruction Generator
    # This generates the "User Request" (e.g. "Create a quiz question about industrialization")
    instruction_callback = PersonaInstructionGeneratorCallback(
        api_key=api_key,
        documents=docs,
        prompt="You are an expert role player.",
        num_random_contexts=1,
    )

    # 4. Setup Structured Generator (The "Teacher/Examiner")
    respondent_prompt = """
    You are an expert educational content creator and exam setter. 
    Your goal is to create high-quality, fair, and challenging multiple-choice questions based *strictly* on the provided text.
    
    - Ensure distractor options are plausible but clearly incorrect.
    - Avoid ambiguous questions.
    - Provide clear and helpful explanations.
    """

    correspondent_prompt = """
    Let's play a game with you.
I am an expert educational content creator and exam setter. I can create high-quality multiple-choice questions based on any text you provide.
You will roleplay as a teacher, student, or course creator who needs a quiz.
Given a context and persona, write instructions for a set of questions to be generated.
You can specify the number of questions or the difficulty.
For example, you could start with: "I need 5 multiple-choice questions based on the following article about the Roman Empire."
I will create the questions for you. You can then ask for changes or follow-ups.
Never break the game and always act as a user who genuinely needs to create a quiz.

    """

    generator = AsyncStructuredGenerator(
        output_schema=MCQList,
        respondent_prompt=respondent_prompt,
        correspondent_prompt=correspondent_prompt,
        api_key=api_key,
        model_name="gemini-2.5-flash",
        instruction_generator_callback=instruction_callback,
    )

    print("Starting generation of synthetic MCQ dataset...")
    # Generate 10 samples
    await generator.generate(num_samples=10)
    print("Generation complete.")
    print("correspondent prompt", generator.correspondent_prompt)


if __name__ == "__main__":
    asyncio.run(main())
