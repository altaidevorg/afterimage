"""Tests for SamplingStrategy."""

import pytest

from afterimage.sampling import SamplingStrategy
from afterimage.callbacks import (
    ContextCoverageStoppingCallback,
    FixedNumberStoppingCallback,
    AndStoppingCallback,
)
from afterimage.providers import InMemoryDocumentProvider


class TestIterStoppingCallbacks:
    def test_flat_list(self):
        cb1 = FixedNumberStoppingCallback(n=5)
        cb2 = FixedNumberStoppingCallback(n=10)
        result = list(SamplingStrategy.iter_stopping_callbacks([cb1, cb2]))
        assert result == [cb1, cb2]

    def test_nested_composite(self):
        inner = FixedNumberStoppingCallback(n=5)
        composite = AndStoppingCallback(callbacks=[inner])
        result = list(SamplingStrategy.iter_stopping_callbacks([composite]))
        assert inner in result
        assert composite in result

    def test_empty_list(self):
        assert list(SamplingStrategy.iter_stopping_callbacks([])) == []

    def test_none_input(self):
        assert list(SamplingStrategy.iter_stopping_callbacks(None)) == []


class TestInferTargetContextUsageCount:
    def test_infers_from_matching_provider(self):
        provider = InMemoryDocumentProvider(["doc1", "doc2"])
        cb = ContextCoverageStoppingCallback(provider=provider, target_visits=3)
        strategy = SamplingStrategy()
        result = strategy.infer_target_context_usage_count(provider, [cb])
        assert result == 3

    def test_returns_none_for_unrelated_callbacks(self):
        provider = InMemoryDocumentProvider(["doc1"])
        cb = FixedNumberStoppingCallback(n=10)
        strategy = SamplingStrategy()
        result = strategy.infer_target_context_usage_count(provider, [cb])
        assert result is None

    def test_takes_max_of_multiple_targets(self):
        provider = InMemoryDocumentProvider(["doc1"])
        cb1 = ContextCoverageStoppingCallback(provider=provider, target_visits=2)
        cb2 = ContextCoverageStoppingCallback(provider=provider, target_visits=5)
        strategy = SamplingStrategy()
        result = strategy.infer_target_context_usage_count(provider, [cb1, cb2])
        assert result == 5

    def test_ignores_different_provider(self):
        provider_a = InMemoryDocumentProvider(["doc1"])
        provider_b = InMemoryDocumentProvider(["doc2"])
        cb = ContextCoverageStoppingCallback(provider=provider_a, target_visits=3)
        strategy = SamplingStrategy()
        result = strategy.infer_target_context_usage_count(provider_b, [cb])
        assert result is None


class TestConfigureContextSampling:
    def test_sets_target_on_provider(self):
        provider = InMemoryDocumentProvider(["doc1", "doc2"])
        cb = ContextCoverageStoppingCallback(provider=provider, target_visits=4)

        class FakeCallback:
            def __init__(self, p):
                self.provider = p

        callback = FakeCallback(provider)
        strategy = SamplingStrategy()
        strategy.configure_context_sampling(callback, [cb])
        assert provider.target_context_usage_count == 4

    def test_skips_when_explicit_target_set(self):
        provider = InMemoryDocumentProvider(["doc1"], target_context_usage_count=10)
        cb = ContextCoverageStoppingCallback(provider=provider, target_visits=4)

        class FakeCallback:
            def __init__(self, p):
                self.provider = p

        callback = FakeCallback(provider)
        strategy = SamplingStrategy()
        strategy.configure_context_sampling(callback, [cb])
        # Should not override the explicit target
        assert provider.target_context_usage_count == 10

    def test_skips_when_no_provider(self):
        class FakeCallback:
            provider = None

        strategy = SamplingStrategy()
        # Should not raise
        strategy.configure_context_sampling(FakeCallback(), [])


class TestRecordContextUsage:
    def test_reports_all_context_ids(self):
        provider = InMemoryDocumentProvider(["doc1", "doc2", "doc3"])
        docs = provider.get_all()
        doc_ids = [d.id for d in docs]

        class FakeCallback:
            def __init__(self, p):
                self.provider = p

        class FakeItem:
            def __init__(self, ids):
                self.metadata = {"context_ids": ids}

        callback = FakeCallback(provider)
        item = FakeItem(doc_ids[:2])
        SamplingStrategy.record_context_usage(callback, item)

        assert provider._doc_usage_counts[doc_ids[0]] == 1
        assert provider._doc_usage_counts[doc_ids[1]] == 1
        assert provider._doc_usage_counts[doc_ids[2]] == 0

    def test_skips_when_no_provider(self):
        class FakeCallback:
            provider = None

        class FakeItem:
            metadata = {"context_ids": ["x"]}

        # Should not raise
        SamplingStrategy.record_context_usage(FakeCallback(), FakeItem())

    def test_skips_when_no_metadata(self):
        provider = InMemoryDocumentProvider(["doc1"])

        class FakeCallback:
            def __init__(self, p):
                self.provider = p

        class FakeItem:
            metadata = None

        SamplingStrategy.record_context_usage(FakeCallback(provider), FakeItem())
