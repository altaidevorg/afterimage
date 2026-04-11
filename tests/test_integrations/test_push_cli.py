"""Tests for the push CLI command."""

import json
from unittest.mock import MagicMock, patch

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
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ],
            "metadata": {},
        },
    ]
    p = tmp_path / "data.jsonl"
    with open(p, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return p


class TestPush:
    @patch("afterimage.cli.HfApi")
    def test_push_workflow(self, mock_hf_api_cls, runner, sample_jsonl):
        """Mock HfApi and verify push invokes Hub upload APIs."""
        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api
        result = runner.invoke(
            main,
            [
                "push",
                "-i",
                str(sample_jsonl),
                "--repo",
                "user/test",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_hf_api_cls.assert_called_once_with()
        mock_api.create_repo.assert_called_once()
        assert mock_api.upload_file.call_count >= 3
