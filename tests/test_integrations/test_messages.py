"""Tests for HuggingFace Messages exporter."""

import pytest

from afterimage.integrations.messages import MessagesExporter


@pytest.fixture
def exporter():
    return MessagesExporter()


class TestMessages:
    def test_basic(self, exporter):
        row = {
            "conversations": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        }
        result = exporter.convert_conversation(row)
        assert len(result) == 1
        msgs = result[0]["messages"]
        assert msgs[0] == {"role": "user", "content": "Hi"}
        assert msgs[1] == {"role": "assistant", "content": "Hello!"}

    def test_system_prompt(self, exporter):
        row = {
            "conversations": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ]
        }
        result = exporter.convert_conversation(row, system_prompt="Be helpful.")
        msgs = result[0]["messages"]
        assert msgs[0] == {"role": "system", "content": "Be helpful."}
        assert msgs[1]["role"] == "user"

    def test_validates_has_assistant(self, exporter):
        row = {
            "conversations": [{"role": "user", "content": "Hello?"}]
        }
        result = exporter.convert_conversation(row)
        assert result == []

    def test_multi_turn(self, exporter):
        row = {
            "conversations": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ]
        }
        result = exporter.convert_conversation(row)
        msgs = result[0]["messages"]
        assert len(msgs) == 4

    def test_validate_output(self, exporter):
        good = {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
        assert exporter.validate_output(good) == []
        no_asst = {"messages": [{"role": "user", "content": "Q"}]}
        assert len(exporter.validate_output(no_asst)) > 0
