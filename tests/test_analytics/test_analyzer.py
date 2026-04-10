"""Tests for DatasetAnalyzer."""

import json
from pathlib import Path

import pytest

from afterimage.analytics.analyzer import (
    DatasetAnalyzer,
    _word_tokenize,
    _make_histogram,
)
from afterimage.analytics.models import DatasetReport


def _make_row(
    user_content="Hello?",
    asst_content="Hi there!",
    persona=None,
    context_id=None,
    depth=None,
    evaluation=None,
    final_score=None,
):
    meta = {}
    if context_id is not None:
        meta["context_id"] = context_id
        meta["context_ids"] = []
    if persona is not None:
        meta["persona_name"] = persona
    if depth is not None:
        meta["persona_generation_depth"] = depth

    row = {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": asst_content},
        ],
        "metadata": meta,
        "instruction_context": "",
        "response_context": None,
    }
    if persona is not None:
        row["persona"] = persona
    if evaluation is not None:
        row["evaluation"] = evaluation
    if final_score is not None:
        row["final_score"] = final_score
    return row


@pytest.fixture
def sample_rows():
    return [
        _make_row(
            "What is Python?",
            "A programming language.",
            persona="Student",
            context_id="doc1",
            depth=0,
        ),
        _make_row(
            "Explain OOP",
            "Object-oriented programming...",
            persona="Student",
            context_id="doc1",
            depth=0,
        ),
        _make_row(
            "How does async work?",
            "Async allows concurrent...",
            persona="Developer",
            context_id="doc2",
            depth=1,
        ),
        _make_row(
            "What is REST?",
            "Representational state transfer.",
            persona="Beginner",
            context_id="doc3",
            depth=0,
        ),
        _make_row(
            "What about GraphQL?",
            "A query language for APIs.",
            persona="Developer",
            context_id="doc2",
            depth=1,
        ),
    ]


@pytest.fixture
def sample_jsonl(tmp_path, sample_rows):
    p = tmp_path / "test.jsonl"
    with open(p, "w") as f:
        for row in sample_rows:
            f.write(json.dumps(row) + "\n")
    return p


class TestWordTokenize:
    def test_basic(self):
        assert _word_tokenize("Hello, world!") == ["hello", "world"]

    def test_empty(self):
        assert _word_tokenize("") == []

    def test_unicode(self):
        tokens = _word_tokenize("café résumé")
        assert "café" in tokens


class TestMakeHistogram:
    def test_empty(self):
        counts, labels = _make_histogram([])
        assert counts == []
        assert labels == []

    def test_single_value(self):
        counts, labels = _make_histogram([5, 5, 5])
        assert counts == [3]
        assert labels == ["5"]

    def test_range(self):
        counts, labels = _make_histogram([1, 2, 3, 4, 5], n_bins=5)
        assert len(counts) == 5
        assert sum(counts) == 5


class TestSummary:
    def test_counts(self, sample_rows):
        report = DatasetAnalyzer(sample_rows).analyze()
        assert report.summary.total_conversations == 5
        assert report.summary.total_turns == 10  # 2 turns each
        assert report.summary.avg_turns_per_conversation == 2.0

    def test_unique_personas(self, sample_rows):
        report = DatasetAnalyzer(sample_rows).analyze()
        assert report.summary.unique_personas == 3  # Student, Developer, Beginner

    def test_unique_contexts(self, sample_rows):
        report = DatasetAnalyzer(sample_rows).analyze()
        assert report.summary.unique_contexts == 3  # doc1, doc2, doc3

    def test_word_counts(self, sample_rows):
        report = DatasetAnalyzer(sample_rows).analyze()
        assert report.summary.total_words > 0
        assert report.summary.avg_words_per_turn > 0

    def test_empty_dataset(self):
        report = DatasetAnalyzer([]).analyze()
        assert report.summary.total_conversations == 0
        assert report.summary.avg_turns_per_conversation == 0


class TestPersonas:
    def test_persona_counts(self, sample_rows):
        report = DatasetAnalyzer(sample_rows).analyze()
        assert report.personas.persona_counts["Student"] == 2
        assert report.personas.persona_counts["Developer"] == 2
        assert report.personas.persona_counts["Beginner"] == 1

    def test_depth_distribution(self, sample_rows):
        report = DatasetAnalyzer(sample_rows).analyze()
        assert report.personas.depth_distribution[0] == 3
        assert report.personas.depth_distribution[1] == 2


class TestCoverage:
    def test_context_counts(self, sample_rows):
        report = DatasetAnalyzer(sample_rows).analyze()
        assert report.coverage.context_counts["doc1"] == 2
        assert report.coverage.context_counts["doc2"] == 2

    def test_once_vs_multiple(self, sample_rows):
        report = DatasetAnalyzer(sample_rows).analyze()
        assert report.coverage.contexts_used_once == 1  # doc3
        assert report.coverage.contexts_used_multiple == 2  # doc1, doc2


class TestQuality:
    def test_no_evaluations(self, sample_rows):
        report = DatasetAnalyzer(sample_rows).analyze()
        assert report.quality.has_evaluations is False

    def test_with_evaluations(self):
        ev = {
            "coherence": {"score": 0.9, "feedback": "good"},
            "factuality": {"score": 0.8, "feedback": "ok"},
            "grounding": {"score": 0.7, "feedback": "ok"},
            "helpfulness": {"score": 0.85, "feedback": "ok"},
            "relevance": {"score": 0.75, "feedback": "ok"},
            "overall_grade": "good",
        }
        rows = [
            _make_row(evaluation=ev, final_score=0.8),
            _make_row(evaluation=ev, final_score=0.85),
        ]
        report = DatasetAnalyzer(rows).analyze()
        assert report.quality.has_evaluations is True
        assert report.quality.grade_counts["good"] == 2
        assert abs(report.quality.avg_scores["coherence"] - 0.9) < 0.01


class TestDiversity:
    def test_basic_diversity(self, sample_rows):
        report = DatasetAnalyzer(sample_rows).analyze()
        assert report.diversity.vocabulary_size > 0
        assert 0 < report.diversity.type_token_ratio <= 1
        assert report.diversity.shannon_entropy > 0

    def test_empty(self):
        report = DatasetAnalyzer([]).analyze()
        assert report.diversity.vocabulary_size == 0


class TestLengths:
    def test_length_stats(self, sample_rows):
        report = DatasetAnalyzer(sample_rows).analyze()
        assert report.lengths.avg_user_length > 0
        assert report.lengths.avg_assistant_length > 0
        assert len(report.lengths.user_lengths) == 5
        assert len(report.lengths.assistant_lengths) == 5


class TestFromJsonl:
    def test_roundtrip(self, sample_jsonl):
        report = DatasetAnalyzer.from_jsonl(sample_jsonl)
        assert isinstance(report, DatasetReport)
        assert report.summary.total_conversations == 5
        assert report.dataset_path == str(sample_jsonl)

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        report = DatasetAnalyzer.from_jsonl(p)
        assert report.summary.total_conversations == 0
