"""Tests for Oumi exporter."""

import pytest

from afterimage.integrations.oumi import OumiExporter


@pytest.fixture
def exporter():
    return OumiExporter()


class TestOumi:
    def test_matches_messages_format(self, exporter):
        row = {
            "conversations": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ]
        }
        result = exporter.convert_conversation(row)
        assert len(result) == 1
        assert "messages" in result[0]
        msgs = result[0]["messages"]
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_system_prompt_included(self, exporter):
        row = {
            "conversations": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ]
        }
        result = exporter.convert_conversation(row, system_prompt="Be brief.")
        msgs = result[0]["messages"]
        assert msgs[0] == {"role": "system", "content": "Be brief."}

    def test_empty_conversations(self, exporter):
        assert exporter.convert_conversation({"conversations": []}) == []
