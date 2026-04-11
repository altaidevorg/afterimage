"""Tests for the enhanced export CLI command."""

import json

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
            "metadata": {},
        },
        {
            "conversations": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4."},
            ],
            "metadata": {},
        },
        {
            "conversations": [
                {"role": "user", "content": "Explain REST."},
                {"role": "assistant", "content": "REST is an architecture."},
            ],
            "metadata": {},
        },
    ]
    p = tmp_path / "dataset.jsonl"
    with open(p, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return p


class TestListFormats:
    def test_list_formats_flag(self, runner, sample_jsonl):
        result = runner.invoke(main, ["export", "--list-formats"])
        assert result.exit_code == 0
        assert "sharegpt" in result.output
        assert "alpaca" in result.output
        assert "messages" in result.output
        assert "dpo" in result.output

    def test_list_formats_without_input(self, runner):
        """--list-formats should work without -i."""
        result = runner.invoke(main, ["export", "--list-formats"])
        assert result.exit_code == 0


class TestSingleFormat:
    def test_single_format(self, runner, sample_jsonl, tmp_path):
        result = runner.invoke(
            main,
            [
                "export",
                "-i",
                str(sample_jsonl),
                "-f",
                "sharegpt",
                "-o",
                str(tmp_path / "exports"),
            ],
        )
        assert result.exit_code == 0
        out_file = tmp_path / "exports" / "dataset_sharegpt.jsonl"
        assert out_file.exists()
        rows = [json.loads(l) for l in out_file.read_text().strip().split("\n")]
        assert len(rows) == 3
        assert rows[0]["conversations"][0]["from"] == "human"


class TestMultipleFormats:
    def test_two_formats(self, runner, sample_jsonl, tmp_path):
        result = runner.invoke(
            main,
            [
                "export",
                "-i",
                str(sample_jsonl),
                "-f",
                "sharegpt",
                "-f",
                "alpaca",
                "-o",
                str(tmp_path / "exports"),
            ],
        )
        assert result.exit_code == 0
        assert (tmp_path / "exports" / "dataset_sharegpt.jsonl").exists()
        assert (tmp_path / "exports" / "dataset_alpaca.jsonl").exists()


class TestAllFlag:
    def test_all_flag(self, runner, sample_jsonl, tmp_path):
        result = runner.invoke(
            main,
            [
                "export",
                "-i",
                str(sample_jsonl),
                "--all",
                "-o",
                str(tmp_path / "exports"),
            ],
        )
        assert result.exit_code == 0
        exports_dir = tmp_path / "exports"
        # Should have at least sharegpt, alpaca, messages, etc.
        assert (exports_dir / "dataset_sharegpt.jsonl").exists()
        assert (exports_dir / "dataset_alpaca.jsonl").exists()
        assert (exports_dir / "dataset_messages.jsonl").exists()


class TestSplit:
    def test_split(self, runner, sample_jsonl, tmp_path):
        result = runner.invoke(
            main,
            [
                "export",
                "-i",
                str(sample_jsonl),
                "-f",
                "sharegpt",
                "-o",
                str(tmp_path / "exports"),
                "--split",
                "0.34",
            ],
        )
        assert result.exit_code == 0
        train = tmp_path / "exports" / "dataset_sharegpt_train.jsonl"
        val = tmp_path / "exports" / "dataset_sharegpt_val.jsonl"
        assert train.exists()
        assert val.exists()
        n_train = len(train.read_text().strip().split("\n"))
        n_val = len(val.read_text().strip().split("\n"))
        assert n_train + n_val == 3

    def test_deterministic_split(self, runner, sample_jsonl, tmp_path):
        for run_dir in ["run1", "run2"]:
            runner.invoke(
                main,
                [
                    "export",
                    "-i",
                    str(sample_jsonl),
                    "-f",
                    "messages",
                    "-o",
                    str(tmp_path / run_dir),
                    "--split",
                    "0.34",
                    "--seed",
                    "123",
                ],
            )
        train1 = (tmp_path / "run1" / "dataset_messages_train.jsonl").read_text()
        train2 = (tmp_path / "run2" / "dataset_messages_train.jsonl").read_text()
        assert train1 == train2


class TestErrors:
    def test_missing_input(self, runner):
        result = runner.invoke(main, ["export", "-f", "sharegpt"])
        assert result.exit_code != 0

    def test_invalid_format(self, runner, sample_jsonl):
        result = runner.invoke(
            main,
            [
                "export",
                "-i",
                str(sample_jsonl),
                "-f",
                "nonexistent",
            ],
        )
        assert result.exit_code != 0
        assert "Unknown format" in result.output

    def test_no_format_specified(self, runner, sample_jsonl):
        result = runner.invoke(main, ["export", "-i", str(sample_jsonl)])
        assert result.exit_code != 0
        assert "Specify" in result.output

    def test_output_dir(self, runner, sample_jsonl, tmp_path):
        out = tmp_path / "custom_dir"
        result = runner.invoke(
            main,
            [
                "export",
                "-i",
                str(sample_jsonl),
                "-f",
                "sharegpt",
                "-o",
                str(out),
            ],
        )
        assert result.exit_code == 0
        assert (out / "dataset_sharegpt.jsonl").exists()

    def test_default_output_dir(self, runner, sample_jsonl):
        result = runner.invoke(
            main,
            [
                "export",
                "-i",
                str(sample_jsonl),
                "-f",
                "sharegpt",
            ],
        )
        assert result.exit_code == 0
        expected = sample_jsonl.parent / "dataset_sharegpt.jsonl"
        assert expected.exists()
