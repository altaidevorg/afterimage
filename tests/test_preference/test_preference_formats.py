"""Tests for preference output formats."""

from __future__ import annotations

import json

import pytest

from afterimage.preference.formats import format_preference_pairs
from afterimage.preference.types import PreferencePair, ScoredResponse


def _make_pair(
    prompt="What is Python?",
    chosen_content="Python is a high-level programming language...",
    rejected_content="A snake.",
    chosen_score=0.9,
    rejected_score=0.2,
    chosen_label="temperature_0.10",
    rejected_label="temperature_0.90",
    shared_prefix=None,
    system_prompt="You are helpful.",
):
    system_msg = {"role": "system", "content": system_prompt}
    chosen_messages = [
        system_msg,
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": chosen_content},
    ]
    rejected_messages = [
        system_msg,
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": rejected_content},
    ]
    return PreferencePair(
        prompt=prompt,
        chosen=ScoredResponse(
            content=chosen_content,
            score=chosen_score,
            variation_label=chosen_label,
            messages=chosen_messages,
        ),
        rejected=ScoredResponse(
            content=rejected_content,
            score=rejected_score,
            variation_label=rejected_label,
            messages=rejected_messages,
        ),
        shared_prefix=shared_prefix,
        metadata={
            "all_scores": [
                {
                    "content": chosen_content[:50],
                    "score": chosen_score,
                    "label": chosen_label,
                },
                {
                    "content": rejected_content[:50],
                    "score": rejected_score,
                    "label": rejected_label,
                },
            ]
        },
    )


class TestDPOFormat:
    def test_dpo_format(self):
        """Standard DPO: prompt, chosen, rejected keys."""
        pair = _make_pair()
        rows = format_preference_pairs([pair], fmt="dpo")
        assert len(rows) == 1
        row = rows[0]
        assert set(row.keys()) == {"prompt", "chosen", "rejected"}
        assert row["prompt"] == pair.prompt
        assert row["chosen"] == pair.chosen.content
        assert row["rejected"] == pair.rejected.content

    def test_dpo_valid_json(self):
        """DPO rows should be JSON-serialisable."""
        pair = _make_pair()
        rows = format_preference_pairs([pair], fmt="dpo")
        json_str = json.dumps(rows[0])
        loaded = json.loads(json_str)
        assert loaded["prompt"] == pair.prompt


class TestChatDPOFormat:
    def test_chat_dpo_format(self):
        """chat_dpo: chosen and rejected are message lists."""
        pair = _make_pair()
        rows = format_preference_pairs([pair], fmt="chat_dpo")
        assert len(rows) == 1
        row = rows[0]
        assert "prompt" in row
        assert "chosen" in row
        assert "rejected" in row
        assert isinstance(row["chosen"], list)
        assert isinstance(row["rejected"], list)

    def test_chat_dpo_identical_except_last(self):
        """chosen and rejected should be identical except for last assistant message."""
        pair = _make_pair()
        rows = format_preference_pairs([pair], fmt="chat_dpo")
        row = rows[0]
        chosen = row["chosen"]
        rejected = row["rejected"]
        # All but last message should be the same
        assert chosen[:-1] == rejected[:-1]
        # Last messages differ in content
        assert chosen[-1]["content"] != rejected[-1]["content"]
        assert chosen[-1]["role"] == "assistant"
        assert rejected[-1]["role"] == "assistant"

    def test_system_prompt_in_chat_dpo(self):
        """System prompt should be present and identical in chosen and rejected."""
        pair = _make_pair(system_prompt="You are a coding assistant.")
        rows = format_preference_pairs([pair], fmt="chat_dpo")
        row = rows[0]
        chosen_sys = row["chosen"][0]
        rejected_sys = row["rejected"][0]
        assert chosen_sys["role"] == "system"
        assert rejected_sys["role"] == "system"
        assert chosen_sys == rejected_sys


class TestUltraFeedbackFormat:
    def test_ultrafeedback_format(self):
        """ultrafeedback: includes all responses with scores."""
        pair = _make_pair()
        rows = format_preference_pairs([pair], fmt="ultrafeedback")
        row = rows[0]
        assert "instruction" in row
        assert "completions" in row
        assert "chosen" in row
        assert "rejected" in row
        assert isinstance(row["completions"], list)
        assert len(row["completions"]) >= 2
        # Each completion has response + score + label
        for comp in row["completions"]:
            assert "response" in comp
            assert "score" in comp

    def test_ultrafeedback_scores_present(self):
        """Chosen and rejected should have scores."""
        pair = _make_pair(chosen_score=0.9, rejected_score=0.2)
        rows = format_preference_pairs([pair], fmt="ultrafeedback")
        row = rows[0]
        assert row["chosen"]["score"] == pytest.approx(0.9)
        assert row["rejected"]["score"] == pytest.approx(0.2)


class TestAnthropicHHFormat:
    def test_anthropic_hh_format(self):
        """anthropic_hh: Human:/Assistant: prefix format."""
        pair = _make_pair()
        rows = format_preference_pairs([pair], fmt="anthropic_hh")
        row = rows[0]
        assert "chosen" in row
        assert "rejected" in row
        assert "Human:" in row["chosen"]
        assert "Assistant:" in row["chosen"]
        assert "Human:" in row["rejected"]
        assert "Assistant:" in row["rejected"]

    def test_anthropic_hh_content(self):
        """Response text should appear in the formatted string."""
        pair = _make_pair(
            prompt="What is AI?",
            chosen_content="AI stands for artificial intelligence.",
            rejected_content="Dunno.",
        )
        rows = format_preference_pairs([pair], fmt="anthropic_hh")
        row = rows[0]
        assert "What is AI?" in row["chosen"]
        assert "AI stands for artificial intelligence." in row["chosen"]
        assert "Dunno." in row["rejected"]


class TestORPOFormat:
    def test_orpo_format(self):
        """orpo: DPO plus scores."""
        pair = _make_pair(chosen_score=0.85, rejected_score=0.3)
        rows = format_preference_pairs([pair], fmt="orpo")
        row = rows[0]
        assert "prompt" in row
        assert "chosen" in row
        assert "rejected" in row
        assert "chosen_score" in row
        assert "rejected_score" in row
        assert row["chosen_score"] == pytest.approx(0.85)
        assert row["rejected_score"] == pytest.approx(0.3)


class TestMultiTurnDPO:
    def test_multiturn_dpo(self):
        """Multi-turn: shared_prefix in anthropic_hh should appear."""
        pair = _make_pair(
            shared_prefix=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        )
        rows = format_preference_pairs([pair], fmt="anthropic_hh")
        row = rows[0]
        # prefix should appear before the final question
        assert "Hello" in row["chosen"]
        assert "Hi there!" in row["chosen"]


class TestSpecialCharacters:
    def test_special_characters(self):
        """Quotes, newlines, unicode should survive all formats."""
        prompt = 'He said "hello"\nAnd then: über cool 🎉'
        chosen = 'The answer is "yes"\nWith unicode: café'
        rejected = "Simple answer."

        pair = _make_pair(
            prompt=prompt, chosen_content=chosen, rejected_content=rejected
        )

        for fmt in ("dpo", "chat_dpo", "ultrafeedback", "anthropic_hh", "orpo"):
            rows = format_preference_pairs([pair], fmt=fmt)
            # Must be JSON-serialisable
            json_str = json.dumps(rows[0], ensure_ascii=False)
            loaded = json.loads(json_str)
            assert loaded is not None


class TestUnknownFormat:
    def test_unknown_format_raises(self):
        pair = _make_pair()
        with pytest.raises(ValueError, match="Unknown preference format"):
            format_preference_pairs([pair], fmt="nonexistent")
