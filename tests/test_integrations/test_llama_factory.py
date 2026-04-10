"""Tests for LLaMA-Factory exporter."""

import pytest

from afterimage.integrations.llama_factory import LLaMAFactoryExporter


@pytest.fixture
def exporter():
    return LLaMAFactoryExporter()


class TestLLaMAFactory:
    def test_single_turn(self, exporter):
        row = {
            "conversations": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
            ],
            "instruction_context": "",
        }
        result = exporter.convert_conversation(row)
        assert len(result) == 1
        assert result[0]["instruction"] == "Q1"
        assert result[0]["output"] == "A1"
        assert result[0]["history"] == []
        assert result[0]["input"] == ""

    def test_multi_turn(self, exporter):
        row = {
            "conversations": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
                {"role": "user", "content": "Q3"},
                {"role": "assistant", "content": "A3"},
            ],
            "instruction_context": "",
        }
        result = exporter.convert_conversation(row)
        r = result[0]
        assert r["instruction"] == "Q3"
        assert r["output"] == "A3"
        assert r["history"] == [["Q1", "A1"], ["Q2", "A2"]]

    def test_system_field(self, exporter):
        row = {
            "conversations": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ],
            "instruction_context": "",
        }
        result = exporter.convert_conversation(row, system_prompt="Be concise.")
        assert result[0]["system"] == "Be concise."

    def test_context_in_input(self, exporter):
        row = {
            "conversations": [
                {"role": "user", "content": "Summarize"},
                {"role": "assistant", "content": "Summary."},
            ],
            "instruction_context": "Important context.",
        }
        result = exporter.convert_conversation(row)
        assert result[0]["input"] == "Important context."

    def test_empty_conversations(self, exporter):
        assert exporter.convert_conversation({"conversations": []}) == []

    def test_no_pairs(self, exporter):
        row = {"conversations": [{"role": "user", "content": "Hello?"}]}
        assert exporter.convert_conversation(row) == []
