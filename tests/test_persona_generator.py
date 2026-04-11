"""Tests for PersonaGenerator."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from afterimage.monitoring import GenerationMonitor
from afterimage.persona_generator import (
    EXPECTED_PERSONA_COUNT,
    PersonaGenerationContractError,
    PersonaGenerator,
)
from afterimage.providers import InMemoryDocumentProvider
from afterimage.storage import JSONLStorage
from afterimage.types import Document, PersonaEntry


GOOD_PERSONA_RESPONSE_TEXT = "\n".join(
    [
        "Persona 1: A support lead troubleshooting repeated billing errors.",
        "Persona 2: A new customer comparing onboarding steps and setup effort.",
        "Persona 3: A technical admin validating configuration details before rollout.",
        "Persona 4: An operations manager looking for policy edge cases and exceptions.",
        "Persona 5: A frustrated user trying to unblock an urgent workflow quickly.",
    ]
)


def make_response(text: str):
    response = MagicMock()
    response.text = text
    response.prompt_token_count = 10
    response.completion_token_count = 20
    response.total_token_count = 30
    response.model_name = "mock_model"
    return response


class MockPersonaLLM:
    def __init__(self, sync_responses=None, async_responses=None, default_text=None):
        self.default_text = default_text or GOOD_PERSONA_RESPONSE_TEXT
        self.sync_responses = list(sync_responses or [])
        self.async_responses = list(async_responses or [])
        self.sync_calls = 0
        self.async_calls = 0

    def _next_sync_response(self):
        self.sync_calls += 1
        if self.sync_responses:
            return self.sync_responses.pop(0)
        return make_response(self.default_text)

    def _next_async_response(self):
        self.async_calls += 1
        if self.async_responses:
            return self.async_responses.pop(0)
        return make_response(self.default_text)

    def generate_content(self, prompt):
        return self._next_sync_response()

    async def agenerate_content(self, prompt):
        return self._next_async_response()


@pytest.fixture
def persona_generator():
    storage_mock = MagicMock(spec=JSONLStorage)
    storage_mock.asave_documents = AsyncMock()
    monitor_mock = MagicMock(spec=GenerationMonitor)
    return (
        PersonaGenerator(
            api_key="test_key",
            storage=storage_mock,
            monitor=monitor_mock,
        ),
        storage_mock,
        monitor_mock,
    )


@pytest.fixture(autouse=True)
def patch_llm_factory():
    from afterimage import persona_generator as persona_generator_module

    mock_llm = MockPersonaLLM()
    original_create = persona_generator_module.LLMFactory.create
    persona_generator_module.LLMFactory.create = MagicMock(return_value=mock_llm)
    yield mock_llm
    persona_generator_module.LLMFactory.create = original_create


def test_generate_success(persona_generator, patch_llm_factory):
    generator, _, monitor_mock = persona_generator

    personas = generator.generate_from_text("Sample text")

    assert len(personas) == EXPECTED_PERSONA_COUNT
    assert len(set(personas)) == 5
    assert personas[0] == "A support lead troubleshooting repeated billing errors."
    monitor_mock.track_generation.assert_called_once()
    _, kwargs = monitor_mock.track_generation.call_args
    assert kwargs["success"] is True
    assert kwargs["metadata"]["operation"] == "text_to_persona_generation"
    assert kwargs["total_token_count"] == 30


def test_generate_from_text_retries_until_five_unique_personas(
    persona_generator,
    patch_llm_factory,
):
    generator, _, monitor_mock = persona_generator
    patch_llm_factory.sync_responses = [
        make_response(
            "\n".join(
                [
                    "Persona 1:  Duplicate persona   ",
                    "Persona 2: Duplicate persona",
                    "Persona 3: Persona C",
                    "Persona 4: Persona D",
                    "Persona 5: Persona E",
                ]
            )
        ),
        make_response(
            "\n".join(
                [
                    "Persona 1: Persona A",
                    "Persona 2: Persona B",
                    "Persona 3: Persona C",
                    "Persona 4: Persona D",
                    "Persona 5: Persona E",
                    "Persona 6: Persona F",
                ]
            )
        ),
    ]

    personas = generator.generate_from_text("Sample text")

    assert personas == ["Persona A", "Persona B", "Persona C", "Persona D", "Persona E"]
    assert patch_llm_factory.sync_calls == 2
    _, kwargs = monitor_mock.track_generation.call_args
    assert kwargs["success"] is True


def test_generate_from_text_raises_after_three_invalid_attempts(
    persona_generator,
    patch_llm_factory,
):
    generator, _, monitor_mock = persona_generator
    patch_llm_factory.sync_responses = [
        make_response("Persona 1: Persona A"),
        make_response("Persona 1: Persona A\nPersona 2: Persona A"),
        make_response(
            "\n".join(
                [
                    "Persona 1: Persona A",
                    "Persona 2: Persona B",
                    "Persona 3: Persona C",
                    "Persona 4: Persona C",
                ]
            )
        ),
    ]

    with pytest.raises(PersonaGenerationContractError):
        generator.generate_from_text("Sample text")

    assert patch_llm_factory.sync_calls == 3
    _, kwargs = monitor_mock.track_generation.call_args
    assert kwargs["success"] is False


def test_generate_from_persona_tracks_generation_metadata(
    persona_generator,
    patch_llm_factory,
):
    generator, _, monitor_mock = persona_generator

    personas = generator.generate_from_persona("Seed persona", generation=2)

    assert len(personas) == EXPECTED_PERSONA_COUNT
    _, kwargs = monitor_mock.track_generation.call_args
    assert kwargs["metadata"]["operation"] == "persona_to_persona_generation"
    assert kwargs["metadata"]["generation"] == 2


@pytest.mark.parametrize(
    ("target_per_document", "expected_iterations"),
    [
        (1, 0),
        (5, 0),
        (6, 0),
        (20, 1),
        (31, 1),
        (155, 2),
        (156, 2),
        (1000, 3),
        (3905, 4),
        (100000, 6),
    ],
)
def test_resolve_auto_n_iterations_chooses_closest_pool(
    persona_generator,
    target_per_document,
    expected_iterations,
):
    generator, _, _ = persona_generator

    assert (
        generator._resolve_auto_n_iterations(target_per_document) == expected_iterations
    )


@pytest.mark.asyncio
async def test_generate_for_documents_batching(persona_generator):
    generator, storage_mock, _ = persona_generator
    generator.agenerate_from_text = AsyncMock(
        return_value=[
            "Persona A",
            "Persona B",
            "Persona C",
            "Persona D",
            "Persona E",
        ]
    )

    await generator.generate_from_documents(["doc1", "doc2", "doc3"])

    assert generator.agenerate_from_text.call_count == 3
    assert storage_mock.asave_documents.call_count == 1
    args, _ = storage_mock.asave_documents.call_args
    assert len(args[0]) == 3
    assert isinstance(args[0][0], Document)


@pytest.mark.asyncio
async def test_generate_from_documents_builds_fixed_width_persona_tree(
    persona_generator,
):
    generator, _, _ = persona_generator
    doc = Document(id="doc1", text="doc")
    provider = InMemoryDocumentProvider([doc])

    await generator.generate_from_documents(provider, n_iterations=2)

    assert [len(entry.descriptions) for entry in doc.personas] == [5, 25, 125]
    assert [entry.metadata["generation_depth"] for entry in doc.personas] == [0, 1, 2]


@pytest.mark.asyncio
async def test_generate_from_documents_auto_resolves_iterations_from_target_data_count(
    persona_generator,
):
    generator, _, _ = persona_generator
    provider = InMemoryDocumentProvider([Document(id="doc1", text="doc")])

    await generator.generate_from_documents(provider, target_data_count=20)

    doc = provider.get_all()[0]
    assert [len(entry.descriptions) for entry in doc.personas] == [5, 25]
    assert [entry.metadata["generation_depth"] for entry in doc.personas] == [0, 1]


@pytest.mark.asyncio
async def test_generate_from_documents_auto_uses_provider_target_usage_count(
    persona_generator,
):
    generator, _, _ = persona_generator
    provider = InMemoryDocumentProvider(
        [Document(id="doc1", text="doc")],
        target_context_usage_count=1000,
    )

    await generator.generate_from_documents(provider)

    doc = provider.get_all()[0]
    assert [len(entry.descriptions) for entry in doc.personas] == [5, 25, 125, 625]
    assert [entry.metadata["generation_depth"] for entry in doc.personas] == [
        0,
        1,
        2,
        3,
    ]


@pytest.mark.asyncio
async def test_generate_from_documents_target_data_count_does_not_scale_with_num_random_contexts(
    persona_generator,
):
    generator, _, _ = persona_generator
    provider = InMemoryDocumentProvider(
        [
            Document(id="doc1", text="doc1"),
            Document(id="doc2", text="doc2"),
        ]
    )

    await generator.generate_from_documents(
        provider,
        target_data_count=20,
        num_random_contexts=2,
    )

    for doc in provider.get_all():
        assert [len(entry.descriptions) for entry in doc.personas] == [5]
        assert [entry.metadata["generation_depth"] for entry in doc.personas] == [0]


@pytest.mark.asyncio
async def test_generate_from_documents_provider_target_usage_accounts_for_multi_context_rows(
    persona_generator,
):
    generator, _, _ = persona_generator
    provider = InMemoryDocumentProvider(
        [Document(id="doc1", text="doc")],
        target_context_usage_count=20,
    )

    await generator.generate_from_documents(provider, num_random_contexts=2)

    doc = provider.get_all()[0]
    assert [len(entry.descriptions) for entry in doc.personas] == [5]
    assert [entry.metadata["generation_depth"] for entry in doc.personas] == [0]


@pytest.mark.asyncio
async def test_generate_from_documents_auto_scales_by_active_doc_count(
    persona_generator,
):
    generator, _, _ = persona_generator
    provider = InMemoryDocumentProvider(
        [
            Document(id="doc1", text="doc1"),
            Document(id="doc2", text="doc2"),
            Document(id="doc3", text="doc3"),
            Document(id="doc4", text="doc4"),
            Document(id="doc5", text="doc5"),
        ]
    )

    await generator.generate_from_documents(provider, target_data_count=20)

    for doc in provider.get_all():
        assert [len(entry.descriptions) for entry in doc.personas] == [5]
        assert [entry.metadata["generation_depth"] for entry in doc.personas] == [0]


@pytest.mark.asyncio
async def test_generate_from_documents_preserves_existing_personas(persona_generator):
    generator, storage_mock, _ = persona_generator
    doc = Document(
        id="doc1",
        text="doc",
        personas=[
            PersonaEntry(
                descriptions=["Existing persona"],
                metadata={"generation_depth": 99},
            )
        ],
    )
    provider = InMemoryDocumentProvider([doc])

    await generator.generate_from_documents(provider)

    assert doc.personas[0].metadata["generation_depth"] == 99
    assert len(doc.personas) == 2
    storage_mock.asave_documents.assert_called_once()


@pytest.mark.asyncio
async def test_agenerate_from_text_retries_until_fixed_width_contract(
    persona_generator,
    patch_llm_factory,
):
    generator, _, _ = persona_generator
    patch_llm_factory.async_responses = [
        make_response(
            "\n".join(
                [
                    "Persona 1: Persona A",
                    "Persona 2: Persona A",
                    "Persona 3: Persona B",
                    "Persona 4: Persona C",
                ]
            )
        ),
        make_response(GOOD_PERSONA_RESPONSE_TEXT),
    ]

    personas = await generator.agenerate_from_text("Sample text")

    assert len(personas) == EXPECTED_PERSONA_COUNT
    assert patch_llm_factory.async_calls == 2


@pytest.mark.asyncio
async def test_generate_from_documents_tolerates_partial_persona_chain_failures(
    persona_generator,
):
    generator, _, _ = persona_generator
    doc = Document(id="doc1", text="doc")
    provider = InMemoryDocumentProvider([doc])
    generator.agenerate_from_text = AsyncMock(
        return_value=[
            "Persona A",
            "Persona B",
            "Persona C",
            "Persona D",
            "Persona E",
        ]
    )

    async def partial_failure(persona: str, generation: int = 1):
        if persona == "Persona C":
            raise RuntimeError("boom")
        return [f"{persona} child {index}" for index in range(EXPECTED_PERSONA_COUNT)]

    generator.agenerate_from_persona = partial_failure

    await generator.generate_from_documents(provider, n_iterations=1)

    assert [len(entry.descriptions) for entry in doc.personas] == [5, 20]
    assert [entry.metadata["generation_depth"] for entry in doc.personas] == [0, 1]


@pytest.mark.asyncio
async def test_generate_from_documents_raises_when_a_worker_fails(
    persona_generator,
):
    generator, storage_mock, _ = persona_generator
    provider = InMemoryDocumentProvider(
        [
            Document(id="doc1", text="doc1"),
            Document(id="doc2", text="doc2"),
        ]
    )

    async def maybe_fail(text: str):
        if text == "doc2":
            raise PersonaGenerationContractError("bad personas")
        return [
            "Persona A",
            "Persona B",
            "Persona C",
            "Persona D",
            "Persona E",
        ]

    generator.agenerate_from_text = maybe_fail

    with pytest.raises(PersonaGenerationContractError):
        await generator.generate_from_documents(provider)

    assert provider.get_all()[0].personas == []
    assert provider.get_all()[1].personas == []
    storage_mock.asave_documents.assert_not_called()
