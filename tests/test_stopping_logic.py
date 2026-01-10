import asyncio
import os
from afterimage import (
    AsyncConversationGenerator,
    ContextualInstructionGeneratorCallback,
    InMemoryDocumentProvider,
    FixedNumberStoppingCallback,
    ContextCoverageStoppingCallback,
    PersonaUsageStoppingCallback,
    Document,
    PersonaEntry,
    PersonaInstructionGeneratorCallback,
)


async def test_fixed_number():
    print("\n--- Testing FixedNumberStoppingCallback ---")
    docs = ["Doc 1", "Doc 2", "Doc 3"]
    callback = ContextualInstructionGeneratorCallback(
        api_key=os.getenv("GEMINI_API_KEY"),
        documents=docs,
        n_instructions=5,
    )
    generator = AsyncConversationGenerator(
        respondent_prompt="You are a helper.",
        api_key=os.getenv("GEMINI_API_KEY"),
        instruction_generator_callback=callback,
    )

    # Current behavior: num_dialogs=7
    await generator.generate(num_dialogs=7)
    convs = generator.storage.load_conversations()
    print(
        f"Generated {len(convs)} conversations (expected 7+ due to batching, but should stop at 7 or slightly more depending on worker finish)"
    )
    assert len(convs) >= 7
    print("Success!")


async def test_context_coverage():
    print("\n--- Testing ContextCoverageStoppingCallback ---")
    docs = ["Doc 1", "Doc 2", "Doc 3"]
    provider = InMemoryDocumentProvider(docs)
    instruction_callback = ContextualInstructionGeneratorCallback(
        api_key=os.getenv("GEMINI_API_KEY"),
        documents=provider,
        n_instructions=2,  # Moderate batch per context
    )

    # We want to see each context at least once.
    # Total docs = 3.
    coverage_callback = ContextCoverageStoppingCallback(
        provider=provider, target_visits=1
    )

    generator = AsyncConversationGenerator(
        respondent_prompt="You are a helper.",
        api_key=os.getenv("GEMINI_API_KEY"),
        instruction_generator_callback=instruction_callback,
    )

    # Set num_dialogs as a safety limit, it should stop early due to coverage
    await generator.generate(
        num_dialogs=50,
        stopping_criteria=[coverage_callback],
    )
    convs = generator.storage.load_conversations()

    print(f"Generated {len(convs)} conversations.")
    # Since we have 3 docs and 2 instructions per doc,
    # it might generate ~15-20 conversations.
    assert len(convs) < 50

    # Check if all contexts are present
    context_ids = set()
    for c in convs:
        if c.metadata and "context_id" in c.metadata:
            context_ids.add(c.metadata["context_id"])

    print(f"Unique context IDs seen: {len(context_ids)} (expected 3)")
    assert len(context_ids) == 3
    print("Success!")


async def test_persona_usage():
    print("\n--- Testing PersonaUsageStoppingCallback ---")
    docs = [
        Document(
            text="Doc A", personas=[PersonaEntry(descriptions=["Expert", "Chef"])]
        ),
        Document(
            text="Doc B", personas=[PersonaEntry(descriptions=["Novice", "Student"])]
        ),
    ]
    provider = InMemoryDocumentProvider(docs)

    instruction_callback = PersonaInstructionGeneratorCallback(
        api_key=os.getenv("GEMINI_API_KEY"),
        documents=provider,
        num_random_contexts=1,
    )

    # We want to see 3 unique personas.
    # Since we have Expert, Chef, Novice, Student (4 total), it should find 3 eventually.
    persona_callback = PersonaUsageStoppingCallback(n_personas=3)

    generator = AsyncConversationGenerator(
        respondent_prompt="You are a helper.",
        api_key=os.getenv("GEMINI_API_KEY"),
        instruction_generator_callback=instruction_callback,
    )

    await generator.generate(
        num_dialogs=50,
        stopping_criteria=[persona_callback],
    )
    convs = generator.storage.load_conversations()

    print(f"Generated {len(convs)} conversations.")
    assert len(convs) < 50

    unique_personas = set()
    for c in convs:
        if c.persona:
            unique_personas.add(c.persona)

    print(f"Unique personas seen: {len(unique_personas)} (expected >= 3)")
    assert len(unique_personas) >= 3
    print("Success!")


async def main():
    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set. Skipping tests.")
        return

    await test_fixed_number()
    await test_context_coverage()
    await test_persona_usage()


if __name__ == "__main__":
    asyncio.run(main())
