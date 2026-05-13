# document_providers.py
from __future__ import annotations

import glob
import json
import math
import random
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Iterable, List, Optional, Protocol, runtime_checkable

from ..types import Document

logger = logging.getLogger(__name__)


@runtime_checkable
class DocumentProvider(Protocol):
    """
    Unified DocumentProvider protocol.

    Minimal required method for implementations:
        - _load_documents() -> list[Document]

    Public helpers (provided by protocol defaults below):
        - get_documents(n: int) -> list[Document]
        - get_all() -> list[Document]
        - sample(n: int) -> list[Document]
        - report_doc_usage(document_id: str) -> int
        - set_target_context_usage_count(target_context_usage_count: int | None)
        - mark_fully_covered(document_id: str)
        - clear_cache()
        - __len__(), __iter__(), __getitem__(i)
    """

    @abstractmethod
    def _load_documents(self) -> list[Document]:
        """Load (and return) all documents. Implementations may cache internally."""
        ...

    # --- default helpers (implementations can override) ---
    def _ensure_usage_tracking_state(self, docs: list[Document]) -> None:
        """Initialize and refresh usage/weight bookkeeping for loaded documents."""
        if not hasattr(self, "_doc_usage_counts"):
            self._doc_usage_counts = {}
        if not hasattr(self, "_doc_sampling_weights"):
            self._doc_sampling_weights = {}
        if not hasattr(self, "_fully_covered_doc_ids"):
            self._fully_covered_doc_ids = set()
        if not hasattr(self, "target_context_usage_count"):
            self.target_context_usage_count = None
        if not hasattr(self, "_target_context_usage_count_explicit"):
            self._target_context_usage_count_explicit = False

        doc_ids = {doc.id for doc in docs}
        self._doc_usage_counts = {
            doc_id: self._doc_usage_counts.get(doc_id, 0) for doc_id in doc_ids
        }
        self._doc_sampling_weights = {
            doc_id: self._doc_sampling_weights.get(doc_id, 1.0) for doc_id in doc_ids
        }
        self._fully_covered_doc_ids = {
            doc_id for doc_id in self._fully_covered_doc_ids if doc_id in doc_ids
        }
        self._recalculate_sampling_weights(docs)

    def _get_sampling_weight(self, document_id: str) -> float:
        if document_id in self._fully_covered_doc_ids:
            return 0.0

        usage_count = self._doc_usage_counts.get(document_id, 0)
        target_usage_count = self.target_context_usage_count
        if target_usage_count is not None:
            if target_usage_count <= 0:
                return 0.0
            return float(max(target_usage_count - usage_count, 0))

        return 1.0 / (usage_count + 1)

    def _recalculate_sampling_weights(self, docs: list[Document]) -> None:
        for doc in docs:
            self._doc_sampling_weights[doc.id] = self._get_sampling_weight(doc.id)

    def _weighted_sample_without_replacement(
        self, docs: list[Document], k: int
    ) -> list[Document]:
        remaining_docs = list(docs)
        sampled_docs: list[Document] = []

        while remaining_docs and len(sampled_docs) < k:
            weights = [
                self._doc_sampling_weights.get(doc.id, 0.0) for doc in remaining_docs
            ]
            if not any(weight > 0 for weight in weights):
                break

            selected_index = random.choices(
                range(len(remaining_docs)),
                weights=weights,
                k=1,
            )[0]
            sampled_docs.append(remaining_docs.pop(selected_index))

        return sampled_docs

    def get_all(self) -> list[Document]:
        """Return all documents (loads once if implementation caches)."""
        docs = self._load_documents()
        self._ensure_usage_tracking_state(docs)
        return docs

    def get_documents(self, n: int) -> list[Document]:
        """Return up to n random documents. If n is math.inf, return all documents."""
        if n is None:
            n = math.inf
        if n == math.inf:
            return self.get_all()
        docs = self.get_all()
        k = min(int(n), len(docs))
        if k == 0:
            return []

        active_docs = [
            doc for doc in docs if self._doc_sampling_weights.get(doc.id, 0.0) > 0
        ]
        if not active_docs:
            return []
        if k >= len(active_docs):
            return list(active_docs)

        return self._weighted_sample_without_replacement(active_docs, k)

    def sample(self, n: int) -> list[Document]:
        """Alias for get_documents."""
        return self.get_documents(n)

    def report_doc_usage(self, document_id: str) -> int:
        """Record usage for a document and refresh sampling weights."""
        docs = self.get_all()
        if document_id not in self._doc_usage_counts:
            raise KeyError(f"Unknown document id: {document_id}")

        self._doc_usage_counts[document_id] += 1
        self._recalculate_sampling_weights(docs)
        return self._doc_usage_counts[document_id]

    def set_target_context_usage_count(
        self,
        target_context_usage_count: int | None,
    ) -> None:
        """Update the target usage count used by weight calculation."""
        self.target_context_usage_count = target_context_usage_count
        self._target_context_usage_count_explicit = (
            target_context_usage_count is not None
        )
        docs = self.get_all()
        self._recalculate_sampling_weights(docs)

    def get_target_context_usage_count(self) -> int | None:
        """Return the current target usage count, if configured."""
        self.get_all()
        return self.target_context_usage_count

    def mark_fully_covered(self, document_id: str) -> None:
        """Exclude a document from future weighted sampling."""
        self.get_all()
        if document_id not in self._doc_sampling_weights:
            raise KeyError(f"Unknown document id: {document_id}")

        self._fully_covered_doc_ids.add(document_id)
        self._doc_sampling_weights[document_id] = 0.0

    def clear_cache(self) -> None:
        """Optional: implementations can override to clear internal caches."""
        # protocol default: no-op
        return None

    def __len__(self) -> int:
        """Length if supported (may force load)."""
        return len(self.get_all())

    def __iter__(self) -> Iterable[Document]:
        return iter(self.get_all())

    def __getitem__(self, index: int) -> Document:
        """Index access (forces load)."""
        return self.get_all()[index]


# ---------- Concrete implementations ----------


class InMemoryDocumentProvider(DocumentProvider):
    """Simple provider backed by a list of strings."""

    def __init__(
        self,
        texts: list[str | Document],
        target_context_usage_count: int | None = None,
    ):
        if not isinstance(texts, list) or not (
            all(isinstance(d, str) for d in texts)
            or all(isinstance(d, Document) for d in texts)
        ):
            raise TypeError("texts must be a list[str|Document] but got: " + str(texts))
        self.target_context_usage_count = target_context_usage_count
        self._target_context_usage_count_explicit = (
            target_context_usage_count is not None
        )
        self._documents = (
            [Document(text=text) for text in texts]
            if len(texts) > 0 and isinstance(texts[0], str)
            else texts
        )

    def _load_documents(self) -> list[Document]:
        return self._documents

    def clear_cache(self) -> None:
        # nothing to clear for in-memory
        return None


class FileSystemDocumentProvider(DocumentProvider):
    """Load text files matched by a glob pattern."""

    def __init__(
        self,
        path_pattern: str,
        encoding: str = "utf-8",
        recursive: bool = False,
        min_length: int = 1,
        cache: bool = True,
        target_context_usage_count: int | None = None,
    ):
        self.pattern = path_pattern
        self.encoding = encoding
        self.recursive = recursive
        self.min_length = min_length
        self.target_context_usage_count = target_context_usage_count
        self._target_context_usage_count_explicit = (
            target_context_usage_count is not None
        )
        self._cache_enabled = bool(cache)
        self._cache: Optional[list[Document]] = None

    def _find_files(self) -> List[str]:
        return glob.glob(self.pattern, recursive=self.recursive)

    def _load_documents(self) -> list[Document]:
        if self._cache_enabled and self._cache is not None:
            return self._cache

        files = self._find_files()
        if not files:
            raise FileNotFoundError(f"No files matching pattern: {self.pattern}")

        docs: list[Document] = []
        for path in files:
            try:
                with open(path, "r", encoding=self.encoding) as f:
                    text = f.read().strip()
                    if len(text) >= self.min_length:
                        docs.append(Document(id=str(Path(path).resolve()), text=text))
            except Exception as exc:
                logger.warning("Failed to read %s: %s", path, exc)
                continue

        if not docs:
            raise ValueError(
                f"No documents found after filtering for pattern: {self.pattern}"
            )

        if self._cache_enabled:
            self._cache = docs
        return docs

    def clear_cache(self) -> None:
        self._cache = None


class DirectoryDocumentProvider(DocumentProvider):
    """Search a directory for several filename patterns (txt/md/jsonl etc)."""

    def __init__(
        self,
        directory: str | Path,
        file_patterns: Optional[List[str]] = None,
        encoding: str = "utf-8",
        recursive: bool = True,
        min_length: int = 1,
        cache: bool = True,
        target_context_usage_count: int | None = None,
    ):
        self.directory = Path(directory)
        self.patterns = file_patterns or ["*.txt", "*.md"]
        self.encoding = encoding
        self.recursive = recursive
        self.min_length = min_length
        self.target_context_usage_count = target_context_usage_count
        self._target_context_usage_count_explicit = (
            target_context_usage_count is not None
        )
        self._cache_enabled = bool(cache)
        self._cache: Optional[list[Document]] = None

    def _find_files(self) -> List[Path]:
        patterns = self.patterns
        files: List[Path] = []
        for p in patterns:
            glob_pat = f"**/{p}" if self.recursive else p
            files.extend(self.directory.glob(glob_pat))
        return files

    def _load_documents(self) -> list[Document]:
        if self._cache_enabled and self._cache is not None:
            return self._cache

        files = self._find_files()
        if not files:
            raise FileNotFoundError(
                f"No files found in {self.directory} for {self.patterns}"
            )

        docs: list[Document] = []
        for path in files:
            if not path.is_file():
                continue
            try:
                with path.open("r", encoding=self.encoding) as f:
                    text = f.read().strip()
                    if len(text) >= self.min_length:
                        docs.append(Document(id=str(path.resolve()), text=text))
            except Exception as exc:
                logger.debug("skip %s: %s", path, exc)
                continue

        if not docs:
            raise ValueError("No valid documents found in directory after filtering")

        if self._cache_enabled:
            self._cache = docs
        return docs

    def clear_cache(self) -> None:
        self._cache = None


class JSONLDocumentProvider(DocumentProvider):
    """Load text fields from one or more JSONL files.

    content_key selects which key from each JSON object to use.
    """

    def __init__(
        self,
        path_pattern: str,
        content_key: str = "text",
        encoding: str = "utf-8",
        recursive: bool = False,
        cache: bool = True,
        max_docs: Optional[int] = None,
        target_context_usage_count: int | None = None,
        preserve_ids: bool = False,
        include_metadata: bool = False,
    ):
        self.pattern = path_pattern
        self.content_key = content_key
        self.encoding = encoding
        self.recursive = recursive
        self.target_context_usage_count = target_context_usage_count
        self._target_context_usage_count_explicit = (
            target_context_usage_count is not None
        )
        self._cache_enabled = bool(cache)
        self._cache: Optional[list[Document]] = None
        self._max_docs = max_docs
        self.preserve_ids = preserve_ids
        self.include_metadata = include_metadata

    def _find_files(self) -> List[str]:
        return glob.glob(self.pattern, recursive=self.recursive)

    def _load_documents(self) -> list[Document]:
        if self._cache_enabled and self._cache is not None:
            return self._cache

        files = self._find_files()
        if not files:
            raise FileNotFoundError(f"No JSONL files matching: {self.pattern}")

        docs: list[Document] = []
        for fp in files:
            try:
                with open(fp, "r", encoding=self.encoding) as f:
                    for line_number, line in enumerate(f, start=1):
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            logger.debug("invalid json line in %s - skipping", fp)
                            continue
                        if isinstance(obj, dict) and self.content_key in obj:
                            val = obj[self.content_key]
                            if isinstance(val, str) and val.strip():
                                fallback_id = f"{Path(fp).resolve()}:{line_number}"
                                metadata_obj = obj.get("metadata")
                                metadata = (
                                    metadata_obj
                                    if isinstance(metadata_obj, dict)
                                    else {}
                                )
                                doc_id = fallback_id
                                if self.preserve_ids:
                                    doc_id = (
                                        obj.get("id")
                                        or metadata.get("context_id")
                                        or fallback_id
                                    )
                                if self.include_metadata:
                                    metadata = dict(metadata)
                                    for key, value in obj.items():
                                        if key not in {
                                            self.content_key,
                                            "id",
                                            "metadata",
                                        }:
                                            metadata.setdefault(key, value)
                                else:
                                    metadata = {}
                                docs.append(
                                    Document(
                                        id=str(doc_id),
                                        text=val.strip(),
                                        metadata=metadata,
                                    )
                                )
                                if self._max_docs and len(docs) >= self._max_docs:
                                    break
            except Exception as exc:
                logger.warning("Failed to read %s: %s", fp, exc)
                continue
            if self._max_docs and len(docs) >= self._max_docs:
                break

        if not docs:
            raise ValueError("No documents extracted from JSONL files")

        if self._cache_enabled:
            self._cache = docs
        return docs

    def clear_cache(self) -> None:
        self._cache = None


# Qdrant optional provider
try:
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.http.models import Filter, ScoredPoint  # type: ignore

    class QdrantDocumentProvider(DocumentProvider):
        """Load text payloads from a Qdrant collection via scroll.

        Note: requires qdrant-client package.
        """

        def __init__(
            self,
            client: QdrantClient,
            collection_name: str,
            content_key: str = "text",
            batch_size: int = 500,
            scroll_filter: Optional[Filter] = None,
            with_payload_keys: Optional[List[str]] = None,
            cache: bool = True,
            max_docs: Optional[int] = None,
            target_context_usage_count: int | None = None,
        ):
            self.client = client
            self.collection_name = collection_name
            self.content_key = content_key
            self.batch_size = batch_size
            self.scroll_filter = scroll_filter
            self.target_context_usage_count = target_context_usage_count
            self._target_context_usage_count_explicit = (
                target_context_usage_count is not None
            )
            self.with_payload_keys = (
                [content_key] if with_payload_keys is None else with_payload_keys
            )
            self._cache_enabled = bool(cache)
            self._cache: Optional[list[Document]] = None
            self._max_docs = max_docs

        def _scroll_once(self, offset: Optional[int] = None) -> List[ScoredPoint]:
            # using client.scroll - returns (points, next_page) depending on client version
            resp = self.client.scroll(
                collection_name=self.collection_name,
                offset=offset,
                limit=self.batch_size,
                scroll_filter=self.scroll_filter,
                with_payload=True,
                with_vectors=False,
            )
            # qdrant client may return a tuple; safe-guard:
            if isinstance(resp, tuple) or isinstance(resp, list):
                points = resp[0]
            else:
                points = resp
            return points

        def _load_documents(self) -> list[Document]:
            if self._cache_enabled and self._cache is not None:
                return self._cache

            docs: list[Document] = []
            offset = None
            while True:
                points = self._scroll_once(offset)
                if not points:
                    break
                for p in points:
                    if getattr(p, "payload", None) and self.content_key in p.payload:
                        val = p.payload[self.content_key]
                        if isinstance(val, str) and val.strip():
                            docs.append(Document(id=str(p.id), text=val.strip()))
                            if self._max_docs and len(docs) >= self._max_docs:
                                break
                if self._max_docs and len(docs) >= self._max_docs:
                    break
                offset = points[-1].id if points else None
                if len(points) < self.batch_size:
                    break

            if not docs:
                raise ValueError(
                    f"No documents found in Qdrant collection {self.collection_name}"
                )

            if self._cache_enabled:
                self._cache = docs
            return docs

        def clear_cache(self) -> None:
            self._cache = None

except Exception:
    # Qdrant not installed - define a lightweight placeholder for type-checkers/usage errors.
    QdrantDocumentProvider = None  # type: ignore


# ---------- small usage / test snippet ----------
if __name__ == "__main__":
    # quick smoke test
    mem = InMemoryDocumentProvider(["a", "b", "c", "d"])
    assert len(mem) == 4
    assert len(mem.get_documents(2)) == 2
    assert len(mem.get_documents(math.inf)) == 4

    # JSONL/FileSystem/Directory providers: create small files and test manually as needed.
    print("document_providers module loaded - smoke tests passed.")
