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
        - clear_cache()
        - __len__(), __iter__(), __getitem__(i)
    """

    @abstractmethod
    def _load_documents(self) -> list[Document]:
        """Load (and return) all documents. Implementations may cache internally."""
        ...

    # --- default helpers (implementations can override) ---
    def get_all(self) -> list[Document]:
        """Return all documents (loads once if implementation caches)."""
        docs = self._load_documents()
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
        if k == len(docs):
            return list(docs)
        # sample without replacement
        return random.sample(docs, k)

    def sample(self, n: int) -> list[Document]:
        """Alias for get_documents."""
        return self.get_documents(n)

    def clear_cache(self) -> None:
        """Optional: implementations can override to clear internal caches."""
        # protocol default: no-op
        return None

    def __len__(self) -> int:
        """Length if supported (may force load)."""
        return len(self.get_all())

    def __iter__(self) -> Iterable[str]:
        return iter(self.get_all())

    def __getitem__(self, index: int) -> str:
        """Index access (forces load)."""
        return self.get_all()[index]


# ---------- Concrete implementations ----------


class InMemoryDocumentProvider(DocumentProvider):
    """Simple provider backed by a list of strings."""

    def __init__(self, texts: list[str]):
        if not isinstance(texts, list) or not all(
            isinstance(d, str) for d in texts
        ):
            raise TypeError("texts must be a List[str]")
        self._documents = [Document(text=text) for text in texts]

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
    ):
        self.pattern = path_pattern
        self.encoding = encoding
        self.recursive = recursive
        self.min_length = min_length
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
                        docs.append(Document(text=text))
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
    ):
        self.directory = Path(directory)
        self.patterns = file_patterns or ["*.txt", "*.md"]
        self.encoding = encoding
        self.recursive = recursive
        self.min_length = min_length
        self._cache_enabled = bool(cache)
        self._cache: Optional[list[str]] = None

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
                        docs.append(Document(text=text))
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
    ):
        self.pattern = path_pattern
        self.content_key = content_key
        self.encoding = encoding
        self.recursive = recursive
        self._cache_enabled = bool(cache)
        self._cache: Optional[list[Document]] = None
        self._max_docs = max_docs

    def _find_files(self) -> List[str]:
        return glob.glob(self.pattern, recursive=self.recursive)

    def _load_documents(self) -> List[str]:
        if self._cache_enabled and self._cache is not None:
            return self._cache

        files = self._find_files()
        if not files:
            raise FileNotFoundError(f"No JSONL files matching: {self.pattern}")

        docs: list[Document] = []
        for fp in files:
            try:
                with open(fp, "r", encoding=self.encoding) as f:
                    for line in f:
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
                                docs.append(Document(text=val.strip()))
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
        ):
            self.client = client
            self.collection_name = collection_name
            self.content_key = content_key
            self.batch_size = batch_size
            self.scroll_filter = scroll_filter
            self.with_payload_keys = (
                [content_key] if with_payload_keys is None else with_payload_keys
            )
            self._cache_enabled = bool(cache)
            self._cache: Optional[List[str]] = None
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
                            docs.append(Document(text=val.strip()))
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
