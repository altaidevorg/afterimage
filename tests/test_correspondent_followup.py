"""Tests for correspondent follow-up shaping (multi-turn role clarity)."""

from afterimage.conversation_generator import format_correspondent_followup_user_message


def test_followup_wraps_assistant_text():
    body = "Here is the answer about AI adoption."
    out = format_correspondent_followup_user_message(body)
    assert body in out
    assert "Assistant's last message:" in out
    assert "Your next message as the user" in out
    assert "same human user" in out
    assert "not in English" in out


def test_followup_strips_whitespace():
    out = format_correspondent_followup_user_message("  trimmed  \n")
    assert "---\ntrimmed\n---" in out


def test_followup_empty_assistant():
    out = format_correspondent_followup_user_message("")
    assert "Assistant's last message:" in out
    assert "---" in out
