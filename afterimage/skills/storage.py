"""Directory-backed skill storage."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from threading import RLock
from typing import Any

from ..types import Document
from .types import (
    SkillProbe,
    SkillProbeResult,
    SkillSelectionResult,
    SkillSide,
    SkillVersion,
)


def context_hash(text: str | None) -> str:
    normalized = (text or "").strip().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def _append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _append_jsonl_many(path: Path, items: list[dict[str, Any]]) -> None:
    if not items:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for item in items:
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


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class DirectorySkillStore:
    """Persist context-specific skills in a directory tree."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._manifest_loaded = False
        self._context_id_by_hash: dict[str, str] = {}
        self._context_hash_by_id: dict[str, str] = {}
        self._versions_cache: dict[tuple[str, str], list[SkillVersion]] = {}
        self._selected_cache_by_context_id: dict[str, SkillVersion | None] = {}
        self._selected_cache_by_hash: dict[str, SkillVersion | None] = {}

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.jsonl"

    def context_dir(self, context_id: str) -> Path:
        return self.root / context_id

    def register_context(
        self, document: Document, context_id: str | None = None
    ) -> str:
        resolved_id = context_id or document.id
        resolved_hash = context_hash(document.text)
        record = {
            "context_id": resolved_id,
            "document_id": document.id,
            "context_hash": resolved_hash,
            "metadata": document.metadata,
        }
        _append_jsonl(self.manifest_path, record)
        with self._lock:
            self._context_id_by_hash[resolved_hash] = resolved_id
            self._context_hash_by_id[resolved_id] = resolved_hash
        self.context_dir(resolved_id).mkdir(parents=True, exist_ok=True)
        return resolved_id

    def find_context_id_by_text(self, text: str | None) -> str | None:
        lookup_hash = context_hash(text)
        with self._lock:
            cached = self._context_id_by_hash.get(lookup_hash)
            if cached is not None:
                return cached
        self._ensure_manifest_index()
        with self._lock:
            return self._context_id_by_hash.get(lookup_hash)

    def save_probes(self, context_id: str, probes: list[SkillProbe]) -> None:
        _append_jsonl_many(
            self.context_dir(context_id) / "probes.jsonl",
            [probe.model_dump() for probe in probes],
        )

    def save_probe_results(
        self, context_id: str, results: list[SkillProbeResult]
    ) -> None:
        _append_jsonl_many(
            self.context_dir(context_id) / "results.jsonl",
            [result.model_dump() for result in results],
        )

    def save_version(self, version: SkillVersion) -> Path:
        side = version.side
        skill_dir = self.context_dir(version.context_id)
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path = skill_dir / self._skill_filename(version.iteration, side)
        frontmatter = (
            "---\n"
            f"name: {version.name}\n"
            f"description: {version.description}\n"
            f"context_id: {version.context_id}\n"
            f"version_id: {version.id}\n"
            f"iteration: {version.iteration}\n"
            f"side: {version.side}\n"
            "---\n\n"
        )
        skill_path.write_text(
            frontmatter + version.content.strip() + "\n", encoding="utf-8"
        )
        _append_jsonl(
            self._versions_path(version.context_id, side), version.model_dump()
        )
        with self._lock:
            cache_key = (version.context_id, side)
            cached = self._versions_cache.get(cache_key)
            if cached is not None:
                cached.append(version)
            if side == "reasoner":
                self._selected_cache_by_context_id.pop(version.context_id, None)
                context_h = self._context_hash_by_id.get(version.context_id)
                if context_h:
                    self._selected_cache_by_hash.pop(context_h, None)
        return skill_path

    def load_versions(
        self, context_id: str, side: SkillSide = "reasoner"
    ) -> list[SkillVersion]:
        cache_key = (context_id, side)
        with self._lock:
            cached = self._versions_cache.get(cache_key)
            if cached is not None:
                return list(cached)
        loaded = [
            SkillVersion.model_validate(row)
            for row in _load_jsonl(self._versions_path(context_id, side))
        ]
        with self._lock:
            self._versions_cache[cache_key] = loaded
        return list(loaded)

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
        for version in self.load_versions(selection.context_id, side="reasoner"):
            if version.id == selection.selected_version_id:
                selected = version
                break
        if selected is not None:
            source = skill_dir / self._skill_filename(selected.iteration, selected.side)
            if source.exists():
                shutil.copyfile(source, skill_dir / "SKILL.md")
        with self._lock:
            self._selected_cache_by_context_id[selection.context_id] = selected
            context_h = self._context_hash_by_id.get(selection.context_id)
            if context_h:
                self._selected_cache_by_hash[context_h] = selected
        return selection_path

    def load_selected(
        self,
        context_id: str | None = None,
        *,
        context_text: str | None = None,
    ) -> SkillVersion | None:
        lookup_hash = context_hash(context_text) if context_text is not None else None
        with self._lock:
            if (
                context_id is not None
                and context_id in self._selected_cache_by_context_id
            ):
                return self._selected_cache_by_context_id[context_id]
            if lookup_hash and lookup_hash in self._selected_cache_by_hash:
                return self._selected_cache_by_hash[lookup_hash]

        resolved_id = context_id or self.find_context_id_by_text(context_text)
        if not resolved_id:
            return None
        selection_path = self.context_dir(resolved_id) / "selection.json"
        selected: SkillVersion | None = None
        if selection_path.exists():
            selection = SkillSelectionResult.model_validate_json(
                selection_path.read_text(encoding="utf-8")
            )
            for version in self.load_versions(resolved_id, side="reasoner"):
                if version.id == selection.selected_version_id:
                    selected = version
                    break
        if selected is None:
            versions = self.load_versions(resolved_id, side="reasoner")
            selected = versions[-1] if versions else None

        with self._lock:
            self._selected_cache_by_context_id[resolved_id] = selected
            context_h = self._context_hash_by_id.get(resolved_id) or lookup_hash
            if context_h:
                self._selected_cache_by_hash[context_h] = selected
        return selected

    def _versions_path(self, context_id: str, side: SkillSide) -> Path:
        filename = "versions.jsonl" if side == "reasoner" else f"{side}_versions.jsonl"
        return self.context_dir(context_id) / filename

    @staticmethod
    def _skill_filename(iteration: int, side: SkillSide) -> str:
        if side == "reasoner":
            return f"skill-iter-{iteration}.md"
        return f"{side}-skill-iter-{iteration}.md"

    def _ensure_manifest_index(self) -> None:
        with self._lock:
            if self._manifest_loaded:
                return

        context_id_by_hash: dict[str, str] = {}
        context_hash_by_id: dict[str, str] = {}
        for row in _iter_jsonl(self.manifest_path):
            context_h = row.get("context_hash")
            context_id = row.get("context_id")
            if isinstance(context_h, str) and isinstance(context_id, str):
                context_id_by_hash[context_h] = context_id
                context_hash_by_id[context_id] = context_h

        with self._lock:
            if not self._manifest_loaded:
                self._context_id_by_hash.update(context_id_by_hash)
                self._context_hash_by_id.update(context_hash_by_id)
            self._manifest_loaded = True
