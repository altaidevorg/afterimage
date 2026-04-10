"""Tests for the 'afterimage analyze' CLI command and auto_analyze hook."""

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
def sample_jsonl(tmp_path):
    rows = [
        {
            "conversations": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "A programming language."},
            ],
            "metadata": {"context_id": "doc1", "persona_name": "Student"},
            "persona": "Student",
        },
        {
            "conversations": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4."},
            ],
            "metadata": {"context_id": "doc2", "persona_name": "Learner"},
            "persona": "Learner",
        },
    ]
    p = tmp_path / "input.jsonl"
    with open(p, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return p


class TestAnalyzeCommand:
    def test_basic_analyze(self, runner, sample_jsonl, tmp_path):
        out = tmp_path / "report.html"
        result = runner.invoke(main, ["analyze", "-i", str(sample_jsonl), "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        assert "Report saved" in result.output

    def test_default_output_path(self, runner, sample_jsonl):
        result = runner.invoke(main, ["analyze", "-i", str(sample_jsonl)])
        assert result.exit_code == 0
        expected = sample_jsonl.with_suffix(".html")
        assert expected.exists()

    def test_report_is_valid_html(self, runner, sample_jsonl, tmp_path):
        out = tmp_path / "report.html"
        runner.invoke(main, ["analyze", "-i", str(sample_jsonl), "-o", str(out)])
        content = out.read_text()
        assert "<!DOCTYPE html>" in content
        assert "AfterImage Dataset Report" in content

    def test_nonexistent_input(self, runner, tmp_path):
        result = runner.invoke(main, ["analyze", "-i", str(tmp_path / "nope.jsonl")])
        assert result.exit_code != 0


class TestAutoAnalyzeConfig:
    def test_config_parses_auto_analyze(self, tmp_path):
        from afterimage.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(textwrap.dedent("""
            respondent:
              system_prompt: "test"
            analytics:
              auto_analyze: true
        """), encoding="utf-8")
        cfg = load_config(cfg_file)
        assert cfg.analytics.auto_analyze is True

    def test_config_defaults_false(self, tmp_path):
        from afterimage.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(textwrap.dedent("""
            respondent:
              system_prompt: "test"
        """), encoding="utf-8")
        cfg = load_config(cfg_file)
        assert cfg.analytics.auto_analyze is False

    def test_config_custom_output_path(self, tmp_path):
        from afterimage.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(textwrap.dedent("""
            respondent:
              system_prompt: "test"
            analytics:
              auto_analyze: true
              output_path: /tmp/my_report.html
        """), encoding="utf-8")
        cfg = load_config(cfg_file)
        assert cfg.analytics.output_path == "/tmp/my_report.html"
