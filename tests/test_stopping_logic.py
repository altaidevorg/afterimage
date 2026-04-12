"""Integration tests for stopping callbacks (require GEMINI_API_KEY)."""

import pytest

from afterimage import (
    AsyncConversationGenerator,
    ContextualInstructionGeneratorCallback,
    InMemoryDocumentProvider,
    ContextCoverageStoppingCallback,
    PersonaUsageStoppingCallback,
    Document,
    PersonaEntry,
    PersonaInstructionGeneratorCallback,
)


pytestmark = pytest.mark.integration


@pytest.fixture
def api_key(gemini_api_key):
    if not gemini_api_key:
        pytest.skip("GEMINI_API_KEY not set")
    return gemini_api_key


@pytest.mark.asyncio
async def test_fixed_number_stopping_callback(api_key):
    """FixedNumberStoppingCallback stops after num_dialogs."""
    docs = ["Doc 1", "Doc 2", "Doc 3"]
    callback = ContextualInstructionGeneratorCallback(
        api_key=api_key,
        documents=docs,
        n_instructions=5,
    )
    generator = AsyncConversationGenerator(
        respondent_prompt="You are a helper.",
        api_key=api_key,
        instruction_generator_callback=callback,
    )

    await generator.generate(num_dialogs=7)
    convs = generator.storage.load_conversations()

    assert len(convs) >= 7


@pytest.mark.asyncio
async def test_context_coverage_stopping_callback(api_key):
    """ContextCoverageStoppingCallback stops when each context is visited."""
    docs = ["Doc 1", "Doc 2", "Doc 3"]
    provider = InMemoryDocumentProvider(docs)
    instruction_callback = ContextualInstructionGeneratorCallback(
        api_key=api_key,
        documents=provider,
        n_instructions=2,
    )
    coverage_callback = ContextCoverageStoppingCallback(
        provider=provider, target_visits=1
    )
    generator = AsyncConversationGenerator(
        respondent_prompt="You are a helper.",
        api_key=api_key,
        instruction_generator_callback=instruction_callback,
    )

    await generator.generate(
        num_dialogs=50,
        stopping_criteria=[coverage_callback],
    )
    convs = generator.storage.load_conversations()

    assert len(convs) < 50
    context_ids = {
        c.metadata["context_id"]
        for c in convs
        if c.metadata and "context_id" in c.metadata
    }
    assert len(context_ids) == 3


@pytest.mark.asyncio
async def test_persona_usage_stopping_callback(api_key):
    """PersonaUsageStoppingCallback stops after n_personas unique personas."""
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
        api_key=api_key,
        documents=provider,
        num_random_contexts=1,
    )
    persona_callback = PersonaUsageStoppingCallback(n_personas=3)
    generator = AsyncConversationGenerator(
        respondent_prompt="You are a helper.",
        api_key=api_key,
        instruction_generator_callback=instruction_callback,
    )

    await generator.generate(
        num_dialogs=50,
        stopping_criteria=[persona_callback],
    )
    convs = generator.storage.load_conversations()

    assert len(convs) < 50
    unique_personas = {c.persona for c in convs if c.persona}
    assert len(unique_personas) >= 3
