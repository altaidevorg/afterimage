"""Test that export is streaming and doesn't load entire dataset into memory."""

import json
import tracemalloc

import pytest

from afterimage.integrations.sharegpt import ShareGPTExporter


class TestStreaming:
    def test_streaming_memory(self, tmp_path):
        """Export 10k rows and verify peak memory stays reasonable."""
        inp = tmp_path / "large.jsonl"
        out = tmp_path / "large_sharegpt.jsonl"

        # Write 10,000 rows
        with open(inp, "w") as f:
            for i in range(10_000):
                row = {
                    "conversations": [
                        {"role": "user", "content": f"Question {i} " + "x" * 200},
                        {"role": "assistant", "content": f"Answer {i} " + "y" * 500},
                    ],
                    "metadata": {},
                }
                f.write(json.dumps(row) + "\n")

        exporter = ShareGPTExporter()

        tracemalloc.start()
        result = exporter.export_file(inp, out)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert result.total_input == 10_000
        assert result.total_output == 10_000

        # Peak memory should stay well under 100MB for streaming
        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 100, f"Peak memory {peak_mb:.1f}MB exceeds 100MB limit"

    def test_output_valid_json(self, tmp_path):
        """Every line of exported file must be valid JSON."""
        inp = tmp_path / "test.jsonl"
        out = tmp_path / "test_sharegpt.jsonl"

        with open(inp, "w") as f:
            for i in range(100):
                row = {
                    "conversations": [
                        {"role": "user", "content": f"Q{i}"},
                        {"role": "assistant", "content": f"A{i}"},
                    ],
                    "metadata": {},
                }
                f.write(json.dumps(row) + "\n")

        ShareGPTExporter().export_file(inp, out)

        with open(out) as f:
            for i, line in enumerate(f):
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON on line {i+1}")
