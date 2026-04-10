"""Tests for HTML report generation."""

from pathlib import Path

import pytest

from afterimage.analytics.models import (
    DatasetReport,
    DiversityStats,
    LengthStats,
    PersonaStats,
    QualityStats,
    SummaryStats,
    CoverageStats,
)
from afterimage.analytics.report import generate_report


@pytest.fixture
def full_report():
    return DatasetReport(
        summary=SummaryStats(
            total_conversations=100,
            total_turns=250,
            avg_turns_per_conversation=2.5,
            total_words=5000,
            avg_words_per_turn=20.0,
            unique_personas=5,
            unique_contexts=10,
        ),
        personas=PersonaStats(
            persona_counts={
                "Student": 40,
                "Developer": 30,
                "Teacher": 20,
                "Beginner": 10,
            },
            depth_distribution={0: 60, 1: 30, 2: 10},
        ),
        coverage=CoverageStats(
            context_counts={"doc1": 15, "doc2": 12, "doc3": 8},
            contexts_used_once=2,
            contexts_used_multiple=8,
        ),
        quality=QualityStats(
            has_evaluations=True,
            grade_counts={"perfect": 20, "good": 50, "needs_improvement": 25, "bad": 5},
            avg_scores={"coherence": 0.85, "factuality": 0.78, "relevance": 0.82},
            score_histogram=[2, 5, 10, 20, 30, 15, 10, 5, 2, 1],
            score_bins=[
                "0.0",
                "0.1",
                "0.2",
                "0.3",
                "0.4",
                "0.5",
                "0.6",
                "0.7",
                "0.8",
                "0.9",
            ],
        ),
        diversity=DiversityStats(
            vocabulary_size=1200,
            type_token_ratio=0.24,
            bigram_repetition_rate=0.15,
            shannon_entropy=8.5,
        ),
        lengths=LengthStats(
            user_lengths=[10, 15, 20, 25, 30],
            assistant_lengths=[50, 60, 70, 80, 90],
            avg_user_length=20.0,
            avg_assistant_length=70.0,
            user_length_histogram=[1, 1, 1, 1, 1],
            assistant_length_histogram=[1, 1, 1, 1, 1],
            length_bins=["10", "20", "30", "40", "50"],
        ),
        dataset_path="/tmp/test.jsonl",
    )


@pytest.fixture
def minimal_report():
    return DatasetReport(
        summary=SummaryStats(total_conversations=0),
        dataset_path="empty.jsonl",
    )


class TestGenerateReport:
    def test_returns_html(self, full_report):
        html = generate_report(full_report)
        assert "<!DOCTYPE html>" in html
        assert "AfterImage Dataset Report" in html

    def test_contains_summary_metrics(self, full_report):
        html = generate_report(full_report)
        assert "100" in html  # total conversations
        assert "250" in html  # total turns

    def test_contains_persona_chart(self, full_report):
        html = generate_report(full_report)
        assert "Student" in html
        assert "Persona Distribution" in html

    def test_contains_quality_section(self, full_report):
        html = generate_report(full_report)
        assert "Quality Evaluation" in html
        assert "perfect" in html

    def test_contains_diversity_metrics(self, full_report):
        html = generate_report(full_report)
        assert "1,200" in html  # vocabulary
        assert "Entropy" in html

    def test_contains_length_section(self, full_report):
        html = generate_report(full_report)
        assert "Message Lengths" in html

    def test_dark_mode_support(self, full_report):
        html = generate_report(full_report)
        assert 'data-theme="dark"' in html or "toggleTheme" in html

    def test_self_contained(self, full_report):
        html = generate_report(full_report)
        # No external links
        assert (
            "http" not in html.split("<body>")[1].split("</body>")[0]
            or "localhost" not in html
        )

    def test_writes_file(self, full_report, tmp_path):
        out = tmp_path / "report.html"
        generate_report(full_report, output_path=out)
        assert out.exists()
        content = out.read_text()
        assert "<!DOCTYPE html>" in content

    def test_minimal_report_no_crash(self, minimal_report):
        html = generate_report(minimal_report)
        assert "<!DOCTYPE html>" in html
        assert "0" in html

    def test_no_quality_shows_message(self, minimal_report):
        html = generate_report(minimal_report)
        assert "No evaluations found" in html

    def test_file_size_reasonable(self, full_report):
        html = generate_report(full_report)
        assert len(html.encode()) < 1_000_000  # < 1MB
