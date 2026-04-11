"""Tests for OpenAI fine-tuning exporter."""

import pytest

from afterimage.integrations.openai_finetune import OpenAIFineTuneExporter


@pytest.fixture
def exporter():
    return OpenAIFineTuneExporter()


class TestOpenAI:
    def test_basic(self, exporter):
        row = {
            "conversations": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ]
        }
        result = exporter.convert_conversation(row)
        assert len(result) == 1
        msgs = result[0]["messages"]
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_requires_assistant(self, exporter):
        row = {"conversations": [{"role": "user", "content": "Hello?"}]}
        assert exporter.convert_conversation(row) == []

    def test_system_prompt(self, exporter):
        row = {
            "conversations": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ]
        }
        result = exporter.convert_conversation(row, system_prompt="System.")
        msgs = result[0]["messages"]
        assert msgs[0] == {"role": "system", "content": "System."}

    def test_validate_no_assistant(self, exporter):
        bad = {"messages": [{"role": "user", "content": "Q"}]}
        warnings = exporter.validate_output(bad)
        assert any("assistant" in w for w in warnings)
