from pydantic import BaseModel
import pytest

from afterimage.async_conversation_generator import AsyncConversationGenerator
from afterimage.callbacks import (
    AndStoppingCallback,
    ContextCoverageStoppingCallback,
    FixedNumberStoppingCallback,
)
from afterimage.callbacks.instruction_generator_callbacks import (
    ContextualInstructionGeneratorCallback,
    PersonaInstructionGeneratorCallback,
)
from afterimage.common import (
    deepseek_default_max_concurrency,
    default_max_concurrency,
    resolve_generation_max_concurrency,
)
from afterimage.metadata_utils import extract_unique_context_ids
from afterimage.types import ConversationWithContext, Document, GenerationState
from afterimage.persona_generator import PersonaGenerator
from afterimage.providers import InMemoryDocumentProvider
from afterimage.structured_generator import AsyncStructuredGenerator


class DummySchema(BaseModel):
    value: str


def make_provider(target_context_usage_count: int | None = None):
    return InMemoryDocumentProvider(
        [
            Document(id="doc1", text="Context 1"),
            Document(id="doc2", text="Context 2"),
        ],
        target_context_usage_count=target_context_usage_count,
    )


def test_report_doc_usage_updates_sampling_weights():
    provider = make_provider(target_context_usage_count=3)

    provider.get_all()
    assert provider._doc_sampling_weights["doc1"] == 3.0
    assert provider._doc_sampling_weights["doc2"] == 3.0

    usage_count = provider.report_doc_usage("doc1")

    assert usage_count == 1
    assert provider._doc_usage_counts["doc1"] == 1
    assert provider._doc_sampling_weights["doc1"] == 2.0
    assert provider._doc_sampling_weights["doc2"] == 3.0


def test_mark_fully_covered_excludes_document_from_sampling():
    provider = make_provider()

    provider.mark_fully_covered("doc1")
    sampled = provider.get_documents(2)

    assert [doc.id for doc in sampled] == ["doc2"]


def test_target_context_usage_count_zeroes_document_weight():
    provider = make_provider(target_context_usage_count=3)

    for _ in range(3):
        provider.report_doc_usage("doc1")
    sampled = provider.get_documents(2)

    assert [doc.id for doc in sampled] == ["doc2"]


def test_default_weights_use_soft_decay():
    provider = make_provider(target_context_usage_count=None)

    provider.get_all()
    assert provider._doc_sampling_weights["doc1"] == 1.0
    assert provider._doc_sampling_weights["doc2"] == 1.0

    provider.report_doc_usage("doc1")

    assert provider._doc_sampling_weights["doc1"] == 0.5
    assert provider._doc_sampling_weights["doc2"] == 1.0


def test_set_target_context_usage_count_recalculates_weights():
    provider = make_provider(target_context_usage_count=None)
    provider.get_all()
    provider.report_doc_usage("doc1")

    provider.set_target_context_usage_count(4)

    assert provider.get_target_context_usage_count() == 4
    assert provider._doc_sampling_weights["doc1"] == 3.0
    assert provider._doc_sampling_weights["doc2"] == 4.0


def test_set_target_context_usage_count_is_preserved_as_explicit_configuration():
    provider = make_provider(target_context_usage_count=None)
    callback = ContextualInstructionGeneratorCallback(
        api_key="test_key",
        documents=provider,
        num_random_contexts=1,
    )
    generator = AsyncConversationGenerator(
        respondent_prompt="You are a helpful assistant.",
        correspondent_prompt="You are a curious user.",
        api_key="test_key",
        instruction_generator_callback=callback,
    )

    provider.set_target_context_usage_count(2)

    generator._configure_context_sampling(
        callback,
        [ContextCoverageStoppingCallback(provider=provider, target_visits=4)],
    )

    assert provider.get_target_context_usage_count() == 2


def test_resolve_generation_max_concurrency_uses_deepseek_default():
    assert (
        resolve_generation_max_concurrency("deepseek", None)
        == deepseek_default_max_concurrency
    )
    assert (
        resolve_generation_max_concurrency("gemini", None)
        == default_max_concurrency
    )
    assert resolve_generation_max_concurrency("deepseek", 3) == 3

    with pytest.raises(ValueError):
        resolve_generation_max_concurrency("deepseek", 0)


def test_generators_use_deepseek_default_max_concurrency():
    conversation_generator = AsyncConversationGenerator(
        respondent_prompt="You are a helpful assistant.",
        correspondent_prompt="You are a curious user.",
        api_key="test_key",
        model_provider_name="deepseek",
    )
    structured_generator = AsyncStructuredGenerator(
        output_schema=DummySchema,
        respondent_prompt="You are a helpful assistant.",
        correspondent_prompt="You are a curious user.",
        api_key="test_key",
        model_provider_name="deepseek",
    )
    persona_generator = PersonaGenerator(
        api_key="test_key",
        model_provider_name="deepseek",
        max_concurrency=None,
    )

    assert (
        conversation_generator._resolve_max_concurrency(None)
        == deepseek_default_max_concurrency
    )
    assert (
        structured_generator._resolve_max_concurrency(None)
        == deepseek_default_max_concurrency
    )
    assert persona_generator.max_concurrency == deepseek_default_max_concurrency


def test_generator_infers_target_usage_count_from_context_coverage_stopping():
    provider = make_provider(target_context_usage_count=None)
    callback = ContextualInstructionGeneratorCallback(
        api_key="test_key",
        documents=provider,
        num_random_contexts=1,
    )
    generator = AsyncConversationGenerator(
        respondent_prompt="You are a helpful assistant.",
        correspondent_prompt="You are a curious user.",
        api_key="test_key",
        instruction_generator_callback=callback,
    )

    generator._configure_context_sampling(
        callback,
        [ContextCoverageStoppingCallback(provider=provider, target_visits=3)],
    )

    assert provider.get_target_context_usage_count() == 3
    assert provider._doc_sampling_weights["doc1"] == 3.0
    assert provider._doc_sampling_weights["doc2"] == 3.0


def test_generator_infers_target_usage_count_from_nested_and_stopping():
    provider = make_provider(target_context_usage_count=None)
    callback = ContextualInstructionGeneratorCallback(
        api_key="test_key",
        documents=provider,
        num_random_contexts=1,
    )
    generator = AsyncConversationGenerator(
        respondent_prompt="You are a helpful assistant.",
        correspondent_prompt="You are a curious user.",
        api_key="test_key",
        instruction_generator_callback=callback,
    )

    generator._configure_context_sampling(
        callback,
        [
            AndStoppingCallback(
                [ContextCoverageStoppingCallback(provider=provider, target_visits=4)]
            )
        ],
    )

    assert provider.get_target_context_usage_count() == 4


def test_generator_does_not_override_explicit_target_usage_count():
    provider = make_provider(target_context_usage_count=2)
    callback = ContextualInstructionGeneratorCallback(
        api_key="test_key",
        documents=provider,
        num_random_contexts=1,
    )
    generator = AsyncConversationGenerator(
        respondent_prompt="You are a helpful assistant.",
        correspondent_prompt="You are a curious user.",
        api_key="test_key",
        instruction_generator_callback=callback,
    )

    generator._configure_context_sampling(
        callback,
        [ContextCoverageStoppingCallback(provider=provider, target_visits=4)],
    )

    assert provider.get_target_context_usage_count() == 2


def test_generator_configures_persona_sampling_from_requested_rows():
    provider = InMemoryDocumentProvider(
        [
            Document(id="doc1", text="Context 1"),
            Document(id="doc2", text="Context 2"),
        ]
    )
    callback = PersonaInstructionGeneratorCallback(
        api_key="test_key",
        documents=provider,
        num_random_contexts=2,
    )
    generator = AsyncConversationGenerator(
        respondent_prompt="You are a helpful assistant.",
        correspondent_prompt="You are a curious user.",
        api_key="test_key",
        instruction_generator_callback=callback,
    )

    generator._configure_persona_sampling(callback, num_requested=7)

    assert callback._persona_target_per_document == 4


def test_generator_configures_persona_sampling_from_provider_target_usage():
    provider = InMemoryDocumentProvider(
        [
            Document(id="doc1", text="Context 1"),
            Document(id="doc2", text="Context 2"),
        ],
        target_context_usage_count=3,
    )
    callback = PersonaInstructionGeneratorCallback(
        api_key="test_key",
        documents=provider,
        num_random_contexts=2,
    )
    generator = AsyncConversationGenerator(
        respondent_prompt="You are a helpful assistant.",
        correspondent_prompt="You are a curious user.",
        api_key="test_key",
        instruction_generator_callback=callback,
    )

    generator._configure_persona_sampling(callback, num_requested=50)

    assert callback._persona_target_per_document == 2


@pytest.mark.parametrize(
    ("doc_count", "num_random_contexts", "num_requested", "expected_target"),
    [
        (1, 1, 1, 1),
        (2, 1, 20, 10),
        (2, 2, 20, 10),
        (4, 1, 1000, 250),
        (4, 2, 1000, 250),
        (5, 1, 100000, 20000),
        (5, 3, 100000, 20000),
    ],
)
def test_persona_sampling_target_inference_matches_requested_dataset_shape(
    doc_count,
    num_random_contexts,
    num_requested,
    expected_target,
):
    provider = InMemoryDocumentProvider(
        [Document(id=f"doc{i}", text=f"Context {i}") for i in range(doc_count)]
    )
    callback = PersonaInstructionGeneratorCallback(
        api_key="test_key",
        documents=provider,
        num_random_contexts=num_random_contexts,
    )

    callback.configure_persona_sampling(num_requested=num_requested)

    assert callback._persona_target_per_document == expected_target


def test_persona_sampling_target_is_unset_when_requested_rows_are_unknown():
    provider = InMemoryDocumentProvider(
        [
            Document(id="doc1", text="Context 1"),
            Document(id="doc2", text="Context 2"),
        ]
    )
    callback = PersonaInstructionGeneratorCallback(
        api_key="test_key",
        documents=provider,
        num_random_contexts=2,
    )

    callback.configure_persona_sampling(num_requested=None)

    assert callback._persona_target_per_document is None


def test_persona_sampling_target_inference_uses_only_active_documents():
    provider = InMemoryDocumentProvider(
        [
            Document(id="doc1", text="Context 1"),
            Document(id="doc2", text="Context 2"),
            Document(id="doc3", text="Context 3"),
        ]
    )
    provider.get_all()
    provider.mark_fully_covered("doc3")
    callback = PersonaInstructionGeneratorCallback(
        api_key="test_key",
        documents=provider,
        num_random_contexts=2,
    )

    callback.configure_persona_sampling(num_requested=12)

    assert callback._persona_target_per_document == 6


def test_generator_configures_persona_sampling_from_fixed_number_stopping_callback():
    provider = InMemoryDocumentProvider(
        [
            Document(id="doc1", text="Context 1"),
            Document(id="doc2", text="Context 2"),
        ]
    )
    callback = PersonaInstructionGeneratorCallback(
        api_key="test_key",
        documents=provider,
        num_random_contexts=2,
    )
    generator = AsyncConversationGenerator(
        respondent_prompt="You are a helpful assistant.",
        correspondent_prompt="You are a curious user.",
        api_key="test_key",
        instruction_generator_callback=callback,
    )

    generator._configure_persona_sampling(
        callback,
        num_requested=None,
        stopping_criteria=[FixedNumberStoppingCallback(n=7)],
    )

    assert callback._persona_target_per_document == 4


def test_record_context_usage_reports_all_context_ids_after_success():
    provider = make_provider(target_context_usage_count=None)
    callback = ContextualInstructionGeneratorCallback(
        api_key="test_key",
        documents=provider,
        num_random_contexts=2,
    )
    generator = AsyncConversationGenerator(
        respondent_prompt="You are a helpful assistant.",
        correspondent_prompt="You are a curious user.",
        api_key="test_key",
        instruction_generator_callback=callback,
    )
    row = ConversationWithContext(
        conversations=[],
        metadata={"context_id": "doc1", "context_ids": ["doc1", "doc2"]},
    )

    provider.get_all()
    generator._record_context_usage(callback, row)

    assert provider._doc_usage_counts["doc1"] == 1
    assert provider._doc_usage_counts["doc2"] == 1
    assert provider._doc_sampling_weights["doc1"] == 0.5
    assert provider._doc_sampling_weights["doc2"] == 0.5


def test_extract_unique_context_ids_preserves_order_and_legacy_fallback():
    metadata = {
        "context_id": "doc3",
        "context_ids": ["doc1", "doc2", "doc1", "", None, "doc2"],
    }

    assert extract_unique_context_ids(metadata) == ["doc1", "doc2", "doc3"]


def test_generation_state_counts_all_context_ids_once():
    row = ConversationWithContext(
        conversations=[],
        metadata={"context_id": "doc1", "context_ids": ["doc1", "doc2", "doc1"]},
    )
    state = GenerationState()

    state.update(row)

    assert state.context_counts["doc1"] == 1
    assert state.context_counts["doc2"] == 1
