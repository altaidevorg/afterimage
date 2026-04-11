"""Tests for Alpaca exporter."""

import pytest

from afterimage.integrations.alpaca import AlpacaExporter


@pytest.fixture
def exporter():
    return AlpacaExporter()


@pytest.fixture
def split_exporter():
    return AlpacaExporter(split_turns=True)


class TestAlpaca:
    def test_single_turn(self, exporter):
        row = {
            "conversations": [
                {"role": "user", "content": "What is X?"},
                {"role": "assistant", "content": "X is Y."},
            ],
            "instruction_context": "",
        }
        result = exporter.convert_conversation(row)
        assert len(result) == 1
        assert result[0]["instruction"] == "What is X?"
        assert result[0]["output"] == "X is Y."
        assert result[0]["input"] == ""

    def test_multi_turn_truncated(self, exporter):
        row = {
            "conversations": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ],
            "instruction_context": "",
        }
        result = exporter.convert_conversation(row)
        assert len(result) == 1
        assert result[0]["instruction"] == "Q1"
        assert result[0]["output"] == "A1"

    def test_multi_turn_split(self, split_exporter):
        row = {
            "conversations": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ],
            "instruction_context": "",
        }
        result = split_exporter.convert_conversation(row)
        assert len(result) == 2
        assert result[0]["instruction"] == "Q1"
        assert result[0]["input"] == ""
        assert result[1]["instruction"] == "Q2"
        assert "User: Q1" in result[1]["input"]
        assert "Assistant: A1" in result[1]["input"]

    def test_context_in_input(self, exporter):
        row = {
            "conversations": [
                {"role": "user", "content": "Summarize"},
                {"role": "assistant", "content": "Summary here."},
            ],
            "instruction_context": "Some important context.",
        }
        result = exporter.convert_conversation(row)
        assert result[0]["input"] == "Some important context."

    def test_empty_input_not_null(self, exporter):
        row = {
            "conversations": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ],
        }
        result = exporter.convert_conversation(row)
        assert result[0]["input"] is not None
        assert isinstance(result[0]["input"], str)

    def test_no_assistant_skipped(self, exporter):
        row = {
            "conversations": [{"role": "user", "content": "Hello?"}],
        }
        assert exporter.convert_conversation(row) == []

    def test_validate_output(self, exporter):
        good = {"instruction": "Q", "input": "", "output": "A"}
        assert exporter.validate_output(good) == []
        bad = {"instruction": "Q", "input": None, "output": "A"}
        assert len(exporter.validate_output(bad)) > 0
