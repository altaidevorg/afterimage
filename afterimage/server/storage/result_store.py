"""Manages generated dataset files on disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ResultStore:
    """Saves and retrieves generated dataset files."""

    def __init__(self, results_dir: str | Path = "./results"):
        self._dir = Path(results_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        d = self._dir / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def conversations_path(self, job_id: str, fmt: str = "jsonl") -> Path:
        return self.job_dir(job_id) / f"conversations.{fmt}"

    def save_conversations_json(
        self,
        job_id: str,
        conversations: list[dict[str, Any]],
        system_prompt_parts: list[str] | None = None,
    ) -> Path:
        path = self.conversations_path(job_id, "json")
        data: dict[str, Any] = {"conversations": conversations}
        if system_prompt_parts is not None:
            data["system_prompt_parts"] = system_prompt_parts
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    def get_result_path(self, job_id: str, fmt: str = "jsonl") -> Path | None:
        """Return path only if the file exists."""
        p = self.conversations_path(job_id, fmt)
        return p if p.exists() else None

    def delete_job_files(self, job_id: str) -> None:
        import shutil

        d = self._dir / job_id
        if d.exists():
            shutil.rmtree(d)
