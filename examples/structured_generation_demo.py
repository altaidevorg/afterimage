import asyncio
import os
from enum import Enum
from pydantic import BaseModel, Field
from afterimage import (
    AsyncStructuredGenerator,
    PersonaGenerator,
    PersonaInstructionGeneratorCallback,
    InMemoryDocumentProvider,
)


class SentimentCategory(str, Enum):
    MUST_WATCH = "Must Watch"
    WATCH = "Watch"
    AVOID = "Avoid"
    WASTE_OF_TIME = "Waste of Time"


class Genre(str, Enum):
    ACTION = "Action"
    ANIMATION = "Animation"
    CRIME = "Crime"
    DRAMA = "Drama"
    FANTASY = "Fantasy"
    HORROR = "Horror"
    SCIENCE_FICTION = "Science Fiction"
    THRILLER = "Thriller"
    WAR = "War"
    WESTERN = "Western"


# Define the output schema
class MovieReview(BaseModel):
    movie_title: str = Field(description="The title of the movie")
    genre: Genre = Field(description="The genre of the movie, e.g. Fantasy, Sci-Fi")
    rating: int = Field(description="Rating out of 10")
    summary: str = Field(description="A brief summary of the movie")
    response: str = Field(description="The response to the user's question")
    category: SentimentCategory = Field(
        description="your strong opinion about this movie"
    )


async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable.")
        return

    # Define some context documents (e.g. movie database entries or facts)
    # we will create both personas and instruction based on these.
    docs = InMemoryDocumentProvider(
        [
            "Terminator 2: Judgment Day is a 1991 science fiction action film directed by James Cameron. It stars Arnold Schwarzenegger.",
            "The Shawshank Redemption is a 1994 American prison drama film written and directed by Frank Darabont.",
            "Inception is a 2010 science fiction action film written and directed by Christopher Nolan, who also produced the film with Emma Thomas.",
            "Parasite is a 2019 South Korean black comedy thriller film directed by Bong Joon-ho.",
        ]
    )

    # 2. Setup Persona Generator
    persona_gen = PersonaGenerator(api_key=api_key)

    # Generate personas for the documents
    # This will populate the .personas attribute of each Document in the provider
    await persona_gen.generate_from_documents(docs)

    # Initialize callback
    instruction_callback = PersonaInstructionGeneratorCallback(
        api_key=api_key,
        documents=docs,
        num_random_contexts=1,
        # We can also generate personas if we wanted to run the persona generator first,
        # but here we'll let it use default/random personas if documents don't have them yet.
    )

    # Initialize the generator
    generator = AsyncStructuredGenerator(
        output_schema=MovieReview,
        respondent_prompt="You are an experienced movie critic. You always have a very sharp language and you are not afraid to use it. Generate reviews based on the user's questions.",
        api_key=api_key,
        model_name="gemini-2.5-flash",
        instruction_generator_callback=instruction_callback,
    )

    # Run generation
    print("Starting structured generation...")
    # Generate 10 samples (it samples docs randomly)
    await generator.generate(num_samples=10)
    print("Generation complete. Check the output JSONL file.")


if __name__ == "__main__":
    asyncio.run(main())
