"""Tests for preference analytics computation."""

from __future__ import annotations

import pytest

from afterimage.preference.analytics import compute_analytics
from afterimage.preference.types import PreferencePair, ScoredResponse


def _make_pair(chosen_score=0.9, rejected_score=0.2, chosen_label="temperature_0.10"):
    return PreferencePair(
        prompt="Test prompt",
        chosen=ScoredResponse(content="chosen", score=chosen_score, variation_label=chosen_label),
        rejected=ScoredResponse(content="rejected", score=rejected_score, variation_label="temperature_0.90"),
    )


class TestAnalyticsComputed:
    def test_analytics_computed(self):
        """All fields should be populated."""
        pairs = [_make_pair(0.9, 0.2), _make_pair(0.8, 0.3)]
        analytics = compute_analytics(pairs, total_attempted=4)

        assert analytics.total_attempted == 4
        assert analytics.total_valid == 2
        assert analytics.total_discarded == 2
        assert analytics.discard_rate == pytest.approx(0.5)
        assert analytics.avg_chosen_score > 0
        assert analytics.avg_rejected_score > 0
        assert analytics.avg_score_gap > 0

    def test_discard_rate_correct(self):
        """discard_rate should match actual discarded count."""
        pairs = [_make_pair()]  # 1 valid out of 3 attempted
        analytics = compute_analytics(pairs, total_attempted=3)
        assert analytics.discard_rate == pytest.approx(2 / 3)

    def test_strategy_distribution(self):
        """strategy_distribution should track chosen labels."""
        pairs = [
            _make_pair(chosen_label="temperature_0.10"),
            _make_pair(chosen_label="temperature_0.10"),
            _make_pair(chosen_label="prompt_enhanced"),
        ]
        analytics = compute_analytics(pairs, total_attempted=3)
        assert analytics.strategy_distribution["temperature_0.10"] == 2
        assert analytics.strategy_distribution["prompt_enhanced"] == 1

    def test_strategy_warning(self):
        """Warning when one variation dominates (>75% chosen)."""
        pairs = [_make_pair(chosen_label="temperature_0.10")] * 9 + [
            _make_pair(chosen_label="prompt_enhanced")
        ]
        analytics = compute_analytics(pairs, total_attempted=10)
        assert any("temperature_0.10" in w for w in analytics.warnings)

    def test_no_warning_when_balanced(self):
        """No dominance warning when distribution is balanced."""
        pairs = [
            _make_pair(chosen_label="temperature_0.10"),
            _make_pair(chosen_label="prompt_enhanced"),
        ]
        analytics = compute_analytics(pairs, total_attempted=2)
        dominance_warnings = [w for w in analytics.warnings if "dominates" in w]
        assert len(dominance_warnings) == 0

    def test_high_discard_rate_warning(self):
        """Warning when discard rate > 40%."""
        pairs = [_make_pair()] * 5
        analytics = compute_analytics(pairs, total_attempted=10)  # 50% discard
        assert any("discard" in w.lower() for w in analytics.warnings)

    def test_analytics_with_no_valid_pairs(self):
        """Handles 100% discard rate gracefully."""
        analytics = compute_analytics([], total_attempted=5)
        assert analytics.total_valid == 0
        assert analytics.total_discarded == 5
        assert analytics.discard_rate == pytest.approx(1.0)
        assert len(analytics.warnings) >= 1

    def test_analytics_zero_attempted(self):
        """Handles zero attempted without division error."""
        analytics = compute_analytics([], total_attempted=0)
        assert analytics.total_attempted == 0
        assert analytics.discard_rate == 0.0

    def test_avg_score_gap(self):
        """avg_score_gap should be mean of (chosen - rejected) per pair."""
        pairs = [
            _make_pair(chosen_score=0.9, rejected_score=0.1),  # gap 0.8
            _make_pair(chosen_score=0.7, rejected_score=0.5),  # gap 0.2
        ]
        analytics = compute_analytics(pairs, total_attempted=2)
        assert analytics.avg_score_gap == pytest.approx(0.5)
