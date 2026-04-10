"""Tests for the AfterImage CLI."""

import json
import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from afterimage.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def basic_config(tmp_path):
    """Write a basic config file and return its path."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
        generation:
          num_dialogs: 5
          max_turns: 1
        model:
          provider: gemini
          model_name: gemini-2.0-flash
          api_key_env: GEMINI_API_KEY
        respondent:
          system_prompt: "You are a helpful assistant."
        output:
          path: {output}
    """.format(output=str(tmp_path / "out.jsonl"))
        ),
        encoding="utf-8",
    )
    return cfg


@pytest.fixture
def local_config(tmp_path):
    """Write a local model config file."""
    cfg = tmp_path / "local.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
        model:
          provider: local
          base_url: http://localhost:9999/v1
          model_name: test-model
        respondent:
          system_prompt: "You are helpful."
        output:
          path: {output}
    """.format(output=str(tmp_path / "out.jsonl"))
        ),
        encoding="utf-8",
    )
    return cfg


class TestGenerateDryRun:
    def test_dry_run_prints_plan(self, runner, basic_config):
        result = runner.invoke(main, ["generate", "-c", str(basic_config), "--dry-run"])
        assert result.exit_code == 0
        assert "Generation Plan" in result.output
        assert "gemini" in result.output
        assert "5" in result.output  # num_dialogs

    def test_dry_run_local_config(self, runner, local_config):
        result = runner.invoke(main, ["generate", "-c", str(local_config), "--dry-run"])
        assert result.exit_code == 0
        assert "local" in result.output
        assert "localhost" in result.output


class TestValidate:
    def test_validate_missing_api_key(self, runner, basic_config, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        result = runner.invoke(main, ["validate", "-c", str(basic_config)])
        assert result.exit_code == 1
        assert "FAIL" in result.output
        assert "API key" in result.output

    def test_validate_local_no_server(self, runner, local_config):
        result = runner.invoke(main, ["validate", "-c", str(local_config)])
        # Should fail on connectivity but pass config syntax and API key
        assert "OK" in result.output
        assert "Config syntax" in result.output


class TestExport:
    @pytest.fixture
    def sample_jsonl(self, tmp_path):
        """Create a sample AfterImage JSONL file."""
        data = [
            {
                "conversations": [
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "A programming language."},
                ],
                "metadata": {},
            },
            {
                "conversations": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4."},
                ],
                "metadata": {},
            },
        ]
        p = tmp_path / "input.jsonl"
        with open(p, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")
        return p

    def test_export_sharegpt(self, runner, sample_jsonl, tmp_path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        result = runner.invoke(
            main,
            ["export", "-i", str(sample_jsonl), "-f", "sharegpt", "-o", str(out_dir)],
        )
        assert result.exit_code == 0
        out = out_dir / "input_sharegpt.jsonl"
        rows = [json.loads(line) for line in out.read_text().strip().split("\n")]
        assert len(rows) == 2
        assert rows[0]["conversations"][0]["from"] == "human"
        assert rows[0]["conversations"][1]["from"] == "gpt"

    def test_export_alpaca(self, runner, sample_jsonl, tmp_path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        result = runner.invoke(
            main,
            ["export", "-i", str(sample_jsonl), "-f", "alpaca", "-o", str(out_dir)],
        )
        assert result.exit_code == 0
        out = out_dir / "input_alpaca.jsonl"
        rows = [json.loads(line) for line in out.read_text().strip().split("\n")]
        assert len(rows) == 2
        assert rows[0]["instruction"] == "What is Python?"
        assert rows[0]["output"] == "A programming language."
        assert rows[0]["input"] == ""

    def test_export_messages(self, runner, sample_jsonl, tmp_path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        result = runner.invoke(
            main,
            ["export", "-i", str(sample_jsonl), "-f", "messages", "-o", str(out_dir)],
        )
        assert result.exit_code == 0
        out = out_dir / "input_messages.jsonl"
        rows = [json.loads(line) for line in out.read_text().strip().split("\n")]
        assert len(rows) == 2
        assert rows[0]["messages"][0]["role"] == "user"
        assert rows[0]["messages"][1]["role"] == "assistant"

    def test_export_default_output_path(self, runner, sample_jsonl):
        result = runner.invoke(
            main, ["export", "-i", str(sample_jsonl), "-f", "sharegpt"]
        )
        assert result.exit_code == 0
        expected = sample_jsonl.with_name("input_sharegpt.jsonl")
        assert expected.exists()


class TestInvalidConfig:
    def test_nonexistent_config(self, runner):
        result = runner.invoke(main, ["generate", "-c", "/tmp/no_such_file.yaml"])
        assert result.exit_code != 0

    def test_invalid_yaml_gives_error(self, runner, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("respondent:\n  system_prompt: [invalid", encoding="utf-8")
        result = runner.invoke(main, ["generate", "-c", str(bad), "--dry-run"])
        assert result.exit_code != 0

    def test_missing_required_field(self, runner, tmp_path):
        cfg = tmp_path / "missing.yaml"
        cfg.write_text("model:\n  provider: gemini\n", encoding="utf-8")
        result = runner.invoke(main, ["generate", "-c", str(cfg), "--dry-run"])
        assert result.exit_code != 0
