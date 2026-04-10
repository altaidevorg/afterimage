"""Tests for BaseExporter streaming file export."""

import json

import pytest

from afterimage.integrations.base import BaseExporter, ExportResult


class EchoExporter(BaseExporter):
    """Minimal exporter for testing base class."""

    format_name = "echo"

    def convert_conversation(self, conversation, *, system_prompt=None):
        return [{"echo": conversation.get("conversations", [])}]


class SkipExporter(BaseExporter):
    """Exporter that skips everything."""

    format_name = "skip"

    def convert_conversation(self, conversation, *, system_prompt=None):
        return []


class ErrorExporter(BaseExporter):
    """Exporter that raises on every row."""

    format_name = "error"

    def convert_conversation(self, conversation, *, system_prompt=None):
        raise ValueError("intentional error")


class TestBaseExporter:
    def test_export_file_streams(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        rows = [
            {"conversations": [{"role": "user", "content": f"Q{i}"}]} for i in range(5)
        ]
        with open(inp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        result = EchoExporter().export_file(inp, out)
        assert result.total_input == 5
        assert result.total_output == 5
        assert result.skipped == 0

    def test_export_file_skips_empty(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        inp.write_text(
            json.dumps({"conversations": [{"role": "user", "content": "Q"}]}) + "\n"
        )

        result = SkipExporter().export_file(inp, out)
        assert result.total_input == 1
        assert result.total_output == 0
        assert result.skipped == 1

    def test_export_file_handles_errors(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        inp.write_text(json.dumps({"conversations": []}) + "\n")

        result = ErrorExporter().export_file(inp, out)
        assert result.total_input == 1
        assert result.skipped == 1
        assert any("intentional error" in w for w in result.warnings)

    def test_export_file_handles_bad_json(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        inp.write_text("not json\n" + json.dumps({"conversations": []}) + "\n")

        result = EchoExporter().export_file(inp, out)
        assert result.total_input == 2
        assert result.skipped == 1
        assert any("invalid JSON" in w for w in result.warnings)

    def test_export_creates_output_dir(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        inp.write_text(json.dumps({"conversations": []}) + "\n")
        out = tmp_path / "deep" / "nested" / "out.jsonl"

        EchoExporter().export_file(inp, out)
        assert out.exists()

    def test_export_result_fields(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        inp.write_text("")

        result = EchoExporter().export_file(inp, out)
        assert isinstance(result, ExportResult)
        assert result.format_name == "echo"
        assert result.input_path == str(inp)
        assert result.output_path == str(out)
