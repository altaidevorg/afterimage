"""Tests for instruction generator callbacks."""

from collections import Counter
import random
import pytest
from unittest.mock import MagicMock

from afterimage.callbacks.instruction_generator_callbacks import (
    ContextualInstructionGeneratorCallback,
    PersonaInstructionGeneratorCallback,
    SimpleInstructionGeneratorCallback,
    ToolCallingInstructionGeneratorCallback,
    LLMFactory,
)
from afterimage.common import GeneratedInstructions
from afterimage.providers import InMemoryDocumentProvider
from afterimage.types import Document, PersonaEntry


class MockLLM:
    def generate_structured(self, prompt, schema):
        return MagicMock(
            parsed=MagicMock(instructions=["Test instruction"]),
            prompt_token_count=10,
            completion_token_count=10,
            total_token_count=20,
            finish_reason="stop",
            model_name="mock_model",
        )

    async def agenerate_structured(self, prompt, schema):
        return MagicMock(
            parsed=MagicMock(instructions=["Test instruction"]),
            prompt_token_count=10,
            completion_token_count=10,
            total_token_count=20,
            finish_reason="stop",
            model_name="mock_model",
        )


@pytest.fixture
def mock_llm_factory():
    mock_llm = MockLLM()
    factory = MagicMock()
    factory.create.return_value = mock_llm
    return factory, mock_llm


@pytest.fixture
def documents():
    return [Document(id="doc1", text="Context 1", personas=[])]


def build_persona_document(layer_sizes, doc_id="doc1"):
    return Document(
        id=doc_id,
        text="Context 1",
        personas=[
            PersonaEntry(
                descriptions=[
                    f"Depth {depth} Persona {index}" for index in range(layer_size)
                ],
                metadata={"generation_depth": depth},
            )
            for depth, layer_size in enumerate(layer_sizes)
        ],
    )


def expected_layer_counts(layer_sizes, target):
    remaining = target
    counts = Counter()

    for depth, layer_size in enumerate(layer_sizes):
        if remaining <= 0:
            break
        take = min(layer_size, remaining)
        if take:
            counts[depth] = take
        remaining -= take

    return counts


@pytest.fixture(autouse=True)
def patch_llm_factory(mock_llm_factory):
    factory, _ = mock_llm_factory
    original = LLMFactory.create
    LLMFactory.create = factory.create
    yield
    LLMFactory.create = original


def test_simple_callback_generate():
    callback = SimpleInstructionGeneratorCallback(api_key="test_key")
    result = callback.generate("Ask something interesting.")
    assert result.instructions == ["Test instruction"]
    assert result.context == ""
    assert result.context_id is None
    assert result.context_ids == []
    assert result.persona is None
    assert not hasattr(callback, "provider")


@pytest.mark.asyncio
async def test_simple_callback_agenerate():
    callback = SimpleInstructionGeneratorCallback(api_key="test_key")
    result = await callback.agenerate("Ask something interesting.")
    assert result.instructions == ["Test instruction"]
    assert result.context == ""
    assert result.context_id is None
    assert result.context_ids == []


def test_contextual_callback_generate(documents):
    callback = ContextualInstructionGeneratorCallback(
        api_key="test_key", documents=documents, num_random_contexts=1
    )
    result = callback.generate("Test prompt")
    assert result.instructions == ["Test instruction"]
    assert "Context 1" in result.context
    assert result.context_id == "doc1"
    assert result.context_ids == ["doc1"]


@pytest.mark.asyncio
async def test_contextual_callback_agenerate(documents):
    callback = ContextualInstructionGeneratorCallback(
        api_key="test_key", documents=documents, num_random_contexts=1
    )
    result = await callback.agenerate("Test prompt")
    assert result.instructions == ["Test instruction"]
    assert "Context 1" in result.context
    assert result.context_id == "doc1"
    assert result.context_ids == ["doc1"]


def test_persona_callback_generate(documents):
    callback = PersonaInstructionGeneratorCallback(
        api_key="test_key",
        documents=documents,
        num_random_contexts=1,
    )
    result = callback.generate("Test prompt {persona}")
    assert result.instructions == ["Test instruction"]
    assert result.persona == "A curious user"
    assert result.context_ids == ["doc1"]
    assert result.persona_generation_depth is None


@pytest.mark.asyncio
async def test_persona_callback_agenerate(documents):
    callback = PersonaInstructionGeneratorCallback(
        api_key="test_key", documents=documents, num_random_contexts=1
    )
    result = await callback.agenerate("Test prompt {persona}")
    assert result.instructions == ["Test instruction"]
    assert result.persona == "A curious user"
    assert result.context_ids == ["doc1"]
    assert result.persona_generation_depth is None


def test_persona_callback_supports_legacy_personas_without_generation_depth():
    provider = InMemoryDocumentProvider(
        [
            Document(
                id="doc1",
                text="Context 1",
                personas=[PersonaEntry(descriptions=["Legacy persona"], metadata={})],
            )
        ]
    )
    callback = PersonaInstructionGeneratorCallback(
        api_key="test_key",
        documents=provider,
        num_random_contexts=1,
    )
    callback.configure_persona_sampling(num_requested=1)

    result = callback.generate("Test prompt {persona}")

    assert result.persona == "Legacy persona"
    assert result.persona_generation_depth == 0


def test_contextual_callback_does_not_report_usage_before_generation_succeeds(
    documents,
):
    provider = InMemoryDocumentProvider(documents)
    callback = ContextualInstructionGeneratorCallback(
        api_key="test_key",
        documents=provider,
        num_random_contexts=1,
    )

    callback.generate("Test prompt")

    assert provider._doc_usage_counts["doc1"] == 0
    assert provider._doc_sampling_weights["doc1"] == 1.0


def test_persona_callback_does_not_report_usage_before_generation_succeeds(documents):
    provider = InMemoryDocumentProvider(documents)
    callback = PersonaInstructionGeneratorCallback(
        api_key="test_key",
        documents=provider,
        num_random_contexts=1,
    )

    callback.generate("Test prompt {persona}")

    assert provider._doc_usage_counts["doc1"] == 0
    assert provider._doc_sampling_weights["doc1"] == 1.0


def test_contextual_callback_returns_all_sampled_context_ids():
    provider = InMemoryDocumentProvider(
        [
            Document(id="doc1", text="Context 1", personas=[]),
            Document(id="doc2", text="Context 2", personas=[]),
        ]
    )
    callback = ContextualInstructionGeneratorCallback(
        api_key="test_key",
        documents=provider,
        num_random_contexts=2,
    )

    result = callback.generate("Test prompt")

    assert result.context_id == "doc1"
    assert result.context_ids == ["doc1", "doc2"]


def test_tool_calling_callback_returns_all_sampled_context_ids():
    provider = InMemoryDocumentProvider(
        [
            Document(id="doc1", text="Context 1", personas=[]),
            Document(id="doc2", text="Context 2", personas=[]),
        ]
    )
    callback = ToolCallingInstructionGeneratorCallback(
        api_key="test_key",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup_status",
                    "description": "Look up ticket status",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        documents=provider,
        num_random_contexts=2,
    )

    result = callback.generate("Test prompt")

    assert result.context_id == "doc1"
    assert result.context_ids == ["doc1", "doc2"]


@pytest.mark.asyncio
async def test_tool_calling_callback_agenerate_returns_all_sampled_context_ids():
    provider = InMemoryDocumentProvider(
        [
            Document(id="doc1", text="Context 1", personas=[]),
            Document(id="doc2", text="Context 2", personas=[]),
        ]
    )
    callback = ToolCallingInstructionGeneratorCallback(
        api_key="test_key",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup_status",
                    "description": "Look up ticket status",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        documents=provider,
        num_random_contexts=2,
    )

    result = await callback.agenerate("Test prompt")

    assert result.context_id == "doc1"
    assert result.context_ids == ["doc1", "doc2"]


def test_generated_instructions_context_ids_do_not_share_default_state():
    first = GeneratedInstructions(instructions=["One"], context="Context 1")
    second = GeneratedInstructions(instructions=["Two"], context="Context 2")

    first.context_ids.append("doc1")

    assert second.context_ids == []


def test_persona_callback_undersamples_shallow_layers_first_for_target_20():
    doc = build_persona_document([5, 25, 125, 625, 3125])
    callback = PersonaInstructionGeneratorCallback(
        api_key="test_key",
        documents=InMemoryDocumentProvider([doc]),
        num_random_contexts=1,
    )
    callback.configure_persona_sampling(num_requested=20)

    state = callback._get_persona_selection_state(doc)
    depth_counts = Counter(
        candidate.generation_depth for candidate in state.active_pool
    )

    assert state.mode == "cycle"
    assert len(state.active_pool) == 20
    assert depth_counts == Counter({0: 5, 1: 15})


@pytest.mark.parametrize(
    ("target", "expected_counts"),
    [
        (1, Counter({0: 1})),
        (5, Counter({0: 5})),
        (6, Counter({0: 5, 1: 1})),
        (20, Counter({0: 5, 1: 15})),
        (30, Counter({0: 5, 1: 25})),
        (31, Counter({0: 5, 1: 25, 2: 1})),
        (155, Counter({0: 5, 1: 25, 2: 125})),
        (156, Counter({0: 5, 1: 25, 2: 125, 3: 1})),
        (780, Counter({0: 5, 1: 25, 2: 125, 3: 625})),
        (781, Counter({0: 5, 1: 25, 2: 125, 3: 625, 4: 1})),
        (1000, Counter({0: 5, 1: 25, 2: 125, 3: 625, 4: 220})),
        (3905, Counter({0: 5, 1: 25, 2: 125, 3: 625, 4: 3125})),
    ],
)
def test_persona_callback_target_scenarios_match_expected_layer_counts(
    target,
    expected_counts,
):
    doc = build_persona_document([5, 25, 125, 625, 3125])
    callback = PersonaInstructionGeneratorCallback(
        api_key="test_key",
        documents=InMemoryDocumentProvider([doc]),
        num_random_contexts=1,
    )
    callback.configure_persona_sampling(num_requested=target)

    state = callback._get_persona_selection_state(doc)
    depth_counts = Counter(
        candidate.generation_depth for candidate in state.active_pool
    )

    assert state.mode == "cycle"
    assert len(state.active_pool) == target
    assert depth_counts == expected_counts
    assert depth_counts == expected_layer_counts([5, 25, 125, 625, 3125], target)
    assert state.active_pool == sorted(
        state.active_pool,
        key=lambda candidate: candidate.generation_depth,
    )


def test_persona_callback_uses_full_pool_when_target_matches_total_personas():
    doc = build_persona_document([5, 25, 125, 625, 3125])
    callback = PersonaInstructionGeneratorCallback(
        api_key="test_key",
        documents=InMemoryDocumentProvider([doc]),
        num_random_contexts=1,
    )
    callback.configure_persona_sampling(num_requested=3905)

    state = callback._get_persona_selection_state(doc)
    depth_counts = Counter(
        candidate.generation_depth for candidate in state.active_pool
    )

    assert state.mode == "cycle"
    assert len(state.active_pool) == 3905
    assert depth_counts == Counter({0: 5, 1: 25, 2: 125, 3: 625, 4: 3125})


def test_persona_callback_round_robins_after_pruned_pool_is_exhausted():
    doc = build_persona_document([5, 25])
    callback = PersonaInstructionGeneratorCallback(
        api_key="test_key",
        documents=InMemoryDocumentProvider([doc]),
        num_random_contexts=1,
    )
    callback.configure_persona_sampling(num_requested=6)

    state = callback._get_persona_selection_state(doc)
    first_pass = [state.next_candidate() for _ in range(6)]
    second_pass_first = state.next_candidate()

    assert [candidate.text for candidate in first_pass] == [
        candidate.text for candidate in state.active_pool
    ]
    assert second_pass_first == state.active_pool[0]


def test_persona_callback_round_robins_after_exact_full_pool_is_exhausted():
    doc = build_persona_document([2, 3])
    callback = PersonaInstructionGeneratorCallback(
        api_key="test_key",
        documents=InMemoryDocumentProvider([doc]),
        num_random_contexts=1,
    )
    callback.configure_persona_sampling(num_requested=5)

    state = callback._get_persona_selection_state(doc)
    first_pass = [state.next_candidate() for _ in range(5)]
    second_pass_first = state.next_candidate()

    assert [candidate.text for candidate in first_pass] == [
        candidate.text for candidate in state.active_pool
    ]
    assert second_pass_first == state.active_pool[0]


def test_persona_callback_uses_full_pool_when_requested_target_is_unknown():
    doc = build_persona_document([5, 25, 125])
    callback = PersonaInstructionGeneratorCallback(
        api_key="test_key",
        documents=InMemoryDocumentProvider([doc]),
        num_random_contexts=1,
    )
    callback.configure_persona_sampling(num_requested=None)

    state = callback._get_persona_selection_state(doc)

    assert state.mode == "cycle"
    assert len(state.active_pool) == 155


def test_persona_callback_weighted_oversampling_prefers_shallow_personas():
    doc = build_persona_document([5, 25, 125, 625, 3125])
    callback = PersonaInstructionGeneratorCallback(
        api_key="test_key",
        documents=InMemoryDocumentProvider([doc]),
        num_random_contexts=1,
    )
    callback.configure_persona_sampling(num_requested=3906)

    state = callback._get_persona_selection_state(doc)
    random.seed(0)
    depth_counts = Counter(state.next_candidate().generation_depth for _ in range(5000))

    assert state.mode == "weighted"
    assert depth_counts[0] > depth_counts[4]


def test_persona_callback_weighted_oversampling_normalizes_weights_by_layer_size():
    doc = build_persona_document([5, 25, 125, 625, 3125])
    callback = PersonaInstructionGeneratorCallback(
        api_key="test_key",
        documents=InMemoryDocumentProvider([doc]),
        num_random_contexts=1,
    )
    callback.configure_persona_sampling(num_requested=3906)

    state = callback._get_persona_selection_state(doc)
    weight_totals = Counter()
    for candidate, weight in zip(state.population, state.weights):
        weight_totals[candidate.generation_depth] += weight

    assert state.mode == "weighted"
    assert len(state.population) == 3905
    assert weight_totals[0] == pytest.approx(5.0)
    assert weight_totals[1] == pytest.approx(4.0)
    assert weight_totals[2] == pytest.approx(3.0)
    assert weight_totals[3] == pytest.approx(2.0)
    assert weight_totals[4] == pytest.approx(1.0)
    assert weight_totals[0] > weight_totals[4]


def test_persona_callback_selection_state_does_not_mutate_document_personas():
    doc = build_persona_document([5, 25, 125])
    original_personas = doc.model_copy(deep=True).personas
    callback = PersonaInstructionGeneratorCallback(
        api_key="test_key",
        documents=InMemoryDocumentProvider([doc]),
        num_random_contexts=1,
    )
    callback.configure_persona_sampling(num_requested=20)

    callback._get_persona_selection_state(doc)

    assert doc.personas == original_personas


def test_persona_callback_sets_persona_generation_depth_in_output():
    provider = InMemoryDocumentProvider(
        [
            Document(
                id="doc1",
                text="Context 1",
                personas=[
                    PersonaEntry(
                        descriptions=["Depth 2 Persona"],
                        metadata={"generation_depth": 2},
                    )
                ],
            )
        ]
    )
    callback = PersonaInstructionGeneratorCallback(
        api_key="test_key",
        documents=provider,
        num_random_contexts=1,
    )
    callback.configure_persona_sampling(num_requested=1)

    result = callback.generate("Test prompt {persona}")

    assert result.persona == "Depth 2 Persona"
    assert result.persona_generation_depth == 2


def test_tool_calling_callback_sets_persona_generation_depth_in_output():
    provider = InMemoryDocumentProvider(
        [
            Document(
                id="doc1",
                text="Context 1",
                personas=[
                    PersonaEntry(
                        descriptions=["Depth 3 Persona"],
                        metadata={"generation_depth": 3},
                    )
                ],
            )
        ]
    )
    callback = ToolCallingInstructionGeneratorCallback(
        api_key="test_key",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup_status",
                    "description": "Look up ticket status",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        documents=provider,
        num_random_contexts=1,
    )
    callback.configure_persona_sampling(num_requested=1)

    result = callback.generate("Test prompt")

    assert result.persona == "Depth 3 Persona"
    assert result.persona_generation_depth == 3
