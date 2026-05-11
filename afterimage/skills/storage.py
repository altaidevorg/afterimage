"""Directory-backed skill storage."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from ..types import Document
from .types import SkillProbe, SkillProbeResult, SkillSelectionResult, SkillVersion


def context_hash(text: str | None) -> str:
    normalized = (text or "").strip().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def _append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class DirectorySkillStore:
    """Persist context-specific skills in a directory tree."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.jsonl"

    def context_dir(self, context_id: str) -> Path:
        return self.root / context_id

    def register_context(self, document: Document, context_id: str | None = None) -> str:
        resolved_id = context_id or document.id
        record = {
            "context_id": resolved_id,
            "document_id": document.id,
            "context_hash": context_hash(document.text),
            "metadata": document.metadata,
        }
        _append_jsonl(self.manifest_path, record)
        self.context_dir(resolved_id).mkdir(parents=True, exist_ok=True)
        return resolved_id

    def find_context_id_by_text(self, text: str | None) -> str | None:
        h = context_hash(text)
        for row in reversed(_load_jsonl(self.manifest_path)):
            if row.get("context_hash") == h:
                return row.get("context_id")
        return None

    def save_probes(self, context_id: str, probes: list[SkillProbe]) -> None:
        for probe in probes:
            _append_jsonl(self.context_dir(context_id) / "probes.jsonl", probe.model_dump())

    def save_probe_results(
        self, context_id: str, results: list[SkillProbeResult]
    ) -> None:
        for result in results:
            _append_jsonl(
                self.context_dir(context_id) / "results.jsonl",
                result.model_dump(),
            )

    def save_version(self, version: SkillVersion) -> Path:
        skill_dir = self.context_dir(version.context_id)
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path = skill_dir / f"skill-iter-{version.iteration}.md"
        frontmatter = (
            "---\n"
            f"name: {version.name}\n"
            f"description: {version.description}\n"
            f"context_id: {version.context_id}\n"
            f"version_id: {version.id}\n"
            f"iteration: {version.iteration}\n"
            "---\n\n"
        )
        skill_path.write_text(frontmatter + version.content.strip() + "\n", encoding="utf-8")
        _append_jsonl(skill_dir / "versions.jsonl", version.model_dump())
        return skill_path

    def load_versions(self, context_id: str) -> list[SkillVersion]:
        return [
            SkillVersion.model_validate(row)
            for row in _load_jsonl(self.context_dir(context_id) / "versions.jsonl")
        ]

    def load_results(self, context_id: str) -> list[SkillProbeResult]:
        return [
            SkillProbeResult.model_validate(row)
            for row in _load_jsonl(self.context_dir(context_id) / "results.jsonl")
        ]

    def write_selection(self, selection: SkillSelectionResult) -> Path:
        skill_dir = self.context_dir(selection.context_id)
        selection_path = skill_dir / "selection.json"
        selection_path.write_text(
            json.dumps(selection.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        selected = None
        for version in self.load_versions(selection.context_id):
            if version.id == selection.selected_version_id:
                selected = version
                break
        if selected is not None:
            source = skill_dir / f"skill-iter-{selected.iteration}.md"
            if source.exists():
                shutil.copyfile(source, skill_dir / "SKILL.md")
        return selection_path

    def load_selected(
        self,
        context_id: str | None = None,
        *,
        context_text: str | None = None,
    ) -> SkillVersion | None:
        resolved_id = context_id or self.find_context_id_by_text(context_text)
        if not resolved_id:
            return None
        selection_path = self.context_dir(resolved_id) / "selection.json"
        if selection_path.exists():
            selection = SkillSelectionResult.model_validate_json(
                selection_path.read_text(encoding="utf-8")
            )
            for version in self.load_versions(resolved_id):
                if version.id == selection.selected_version_id:
                    return version
        versions = self.load_versions(resolved_id)
        return versions[-1] if versions else None
