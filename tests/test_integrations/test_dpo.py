"""Tests for DPO exporter."""

import json

import pytest

from afterimage.integrations.dpo import DPOExporter


@pytest.fixture
def exporter():
    return DPOExporter()


def _make_scored_row(prompt, response, score):
    return {
        "conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "metadata": {},
        "final_score": score,
    }


class TestDPO:
    def test_with_scores(self, exporter, tmp_path):
        rows = [
            _make_scored_row("What is X?", "X is great.", 0.9),
            _make_scored_row("What is X?", "X is bad.", 0.4),
        ]
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        with open(inp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        result = exporter.export_file(inp, out)
        assert result.total_output == 1

        pair = json.loads(out.read_text().strip())
        assert pair["prompt"] == "What is X?"
        assert pair["chosen"] == "X is great."
        assert pair["rejected"] == "X is bad."

    def test_without_scores(self, exporter, tmp_path):
        rows = [
            {
                "conversations": [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "A"},
                ],
                "metadata": {},
            },
        ]
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        with open(inp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        result = exporter.export_file(inp, out)
        assert result.total_output == 0
        assert any("quality scores" in w for w in result.warnings)

    def test_score_gap_threshold(self, exporter, tmp_path):
        rows = [
            _make_scored_row("Q", "A1", 0.8),
            _make_scored_row("Q", "A2", 0.75),  # gap < 0.2
        ]
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        with open(inp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        result = exporter.export_file(inp, out)
        assert result.total_output == 0

    def test_no_matching_pairs(self, exporter, tmp_path):
        rows = [
            _make_scored_row("Q1", "A1", 0.9),
            _make_scored_row("Q2", "A2", 0.4),  # different prompts
        ]
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        with open(inp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        result = exporter.export_file(inp, out)
        assert result.total_output == 0
        assert any("No instruction pairs" in w for w in result.warnings)

    def test_system_prompt(self, exporter, tmp_path):
        rows = [
            _make_scored_row("Q", "Good", 0.95),
            _make_scored_row("Q", "Bad", 0.3),
        ]
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        with open(inp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        exporter.export_file(inp, out, system_prompt="Be helpful.")
        pair = json.loads(out.read_text().strip())
        assert pair["prompt"].startswith("Be helpful.")
