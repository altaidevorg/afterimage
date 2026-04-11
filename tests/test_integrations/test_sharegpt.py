"""Tests for ShareGPT exporter."""


import pytest

from afterimage.integrations.sharegpt import ShareGPTExporter


@pytest.fixture
def exporter():
    return ShareGPTExporter()


class TestShareGPT:
    def test_single_turn(self, exporter):
        row = {
            "conversations": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        result = exporter.convert_conversation(row)
        assert len(result) == 1
        convs = result[0]["conversations"]
        assert convs[0] == {"from": "human", "value": "Hello"}
        assert convs[1] == {"from": "gpt", "value": "Hi there!"}

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
        convs = result[0]["conversations"]
        assert len(convs) == 4
        roles = [c["from"] for c in convs]
        assert roles == ["human", "gpt", "human", "gpt"]

    def test_system_prompt(self, exporter):
        row = {
            "conversations": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        }
        result = exporter.convert_conversation(row, system_prompt="You are helpful.")
        convs = result[0]["conversations"]
        assert convs[0] == {"from": "system", "value": "You are helpful."}
        assert convs[1]["from"] == "human"

    def test_consecutive_same_role_merged(self, exporter):
        row = {
            "conversations": [
                {"role": "user", "content": "Part 1"},
                {"role": "user", "content": "Part 2"},
                {"role": "assistant", "content": "Combined answer."},
            ]
        }
        result = exporter.convert_conversation(row)
        convs = result[0]["conversations"]
        assert len(convs) == 2
        assert convs[0]["value"] == "Part 1\nPart 2"

    def test_output_schema(self, exporter):
        row = {
            "conversations": [
                {"role": "user", "content": "X"},
                {"role": "assistant", "content": "Y"},
            ]
        }
        result = exporter.convert_conversation(row)
        assert "conversations" in result[0]
        for entry in result[0]["conversations"]:
            assert "from" in entry
            assert "value" in entry

    def test_empty_conversations(self, exporter):
        assert exporter.convert_conversation({"conversations": []}) == []

    def test_validate_output(self, exporter):
        good = {"conversations": [{"from": "human", "value": "hi"}]}
        assert exporter.validate_output(good) == []
        bad = {"conversations": [{"missing": True}]}
        assert len(exporter.validate_output(bad)) > 0
