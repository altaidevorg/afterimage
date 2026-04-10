"""Tests for the preference CLI command."""

from __future__ import annotations

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
def preference_config(tmp_path):
    """Write a minimal preference config and return its path."""
    cfg = tmp_path / "pref_config.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            model:
              provider: openai
              model_name: gpt-4o-mini
              api_key_env: OPENAI_API_KEY
            respondent:
              system_prompt: "You are a helpful assistant."
            preference:
              num_pairs: 2
              num_responses: 2
              strategy: temperature
              min_score_gap: 0.05
              output_format: dpo
              output_path: {output}
            """.format(output=str(tmp_path / "preferences.jsonl"))
        ),
        encoding="utf-8",
    )
    return cfg


class TestPreferenceCommand:
    def test_preference_command_exists(self, runner):
        """afterimage preference --help should exit 0."""
        result = runner.invoke(main, ["preference", "--help"])
        assert result.exit_code == 0
        assert "preference" in result.output.lower() or "DPO" in result.output

    def test_preference_dry_run(self, runner, preference_config):
        """--dry-run prints plan without generating."""
        result = runner.invoke(
            main, ["preference", "-c", str(preference_config), "--dry-run"]
        )
        assert result.exit_code == 0
        assert "Plan" in result.output or "plan" in result.output.lower()
        assert "temperature" in result.output
        assert "dpo" in result.output.lower() or "DPO" in result.output

    def test_preference_dry_run_shows_num_pairs(self, runner, preference_config):
        """Dry run should display the number of pairs."""
        result = runner.invoke(
            main, ["preference", "-c", str(preference_config), "--dry-run"]
        )
        assert result.exit_code == 0
        assert "2" in result.output  # num_pairs=2

    def test_preference_format_flag(self, runner, preference_config):
        """--format flag should override config format in dry run."""
        result = runner.invoke(
            main,
            ["preference", "-c", str(preference_config), "--dry-run", "--format", "orpo"],
        )
        assert result.exit_code == 0
        assert "orpo" in result.output.lower()

    def test_preference_output_flag(self, runner, preference_config, tmp_path):
        """--output flag should override output path in dry run."""
        override_path = str(tmp_path / "override.jsonl")
        result = runner.invoke(
            main,
            [
                "preference",
                "-c",
                str(preference_config),
                "--dry-run",
                "-o",
                override_path,
            ],
        )
        assert result.exit_code == 0
        assert "override.jsonl" in result.output

    def test_preference_save_log_flag(self, runner, preference_config):
        """--save-log flag should appear in dry run plan."""
        result = runner.invoke(
            main,
            ["preference", "-c", str(preference_config), "--dry-run", "--save-log"],
        )
        assert result.exit_code == 0
        assert "yes" in result.output.lower() or "log" in result.output.lower()

    def test_preference_missing_config(self, runner):
        """Missing config file should return exit code 2."""
        result = runner.invoke(main, ["preference", "-c", "/nonexistent/config.yaml"])
        assert result.exit_code != 0
