import glob
import json
from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Protocol, runtime_checkable
import random
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, ScoredPoint


@runtime_checkable
class DocumentProvider(Protocol):
    """Protocol defining the interface for document providers."""

    @abstractmethod
    def get_documents(self, n_samples: int) -> List[str]:
        """Get n random documents for context.

        Args:
            n_samples: Number of documents to retrieve

        Returns:
            List[str]: List of document contents
        """
        pass


class InMemoryDocumentProvider(DocumentProvider):
    """Simple in-memory document provider using a list of documents."""

    def __init__(self, documents: List[str]):
        """Initialize with a list of documents.

        Args:
            documents: List of document contents
        """
        assert (
            isinstance(documents, List)
            and len(documents) >= 1
            and isinstance(documents[0], str)
        ), "`documents` must be a list of strings"
        self.documents = documents

    def get_documents(self, n_samples: int) -> List[str]:
        """Get random documents from the in-memory list."""
        return random.sample(self.documents, min(n_samples, len(self.documents)))


class FileSystemDocumentProvider(DocumentProvider):
    """Provides documents from the filesystem using glob patterns."""

    def __init__(
        self,
        path_pattern: str,
        encoding: str = "utf-8",
        recursive: bool = False,
    ):
        """Initialize the filesystem document provider.

        Args:
            path_pattern: Glob pattern for finding files (e.g., "data/*.txt")
            encoding: File encoding to use
            recursive: Whether to search directories recursively
        """
        self.pattern = path_pattern
        self.encoding = encoding
        self.recursive = recursive
        self._cached_files = None

    def _get_file_paths(self) -> List[str]:
        """Get list of matching file paths."""
        if self._cached_files is None:
            self._cached_files = glob.glob(self.pattern, recursive=self.recursive)
        return self._cached_files

    def get_documents(self, n_samples: int) -> List[str]:
        """Get random documents from matching files."""
        files = self._get_file_paths()
        assert len(files) > 0, f"No files found matching pattern: {self.pattern}"

        selected_files = random.sample(files, min(n_samples, len(files)))
        documents = []

        for file_path in selected_files:
            with open(file_path, "r", encoding=self.encoding) as f:
                documents.append(f.read().strip())

        return documents


class JSONLDocumentProvider(DocumentProvider):
    """Provides documents from JSONL files with configurable key extraction."""

    def __init__(
        self,
        path_pattern: str,
        content_key: str = "text",
        encoding: str = "utf-8",
        recursive: bool = False,
        cache_size: Optional[int] = None,
    ):
        """Initialize the JSONL document provider.

        Args:
            path_pattern: Glob pattern for finding JSONL files
            content_key: Key to extract text content from each JSON object
            encoding: File encoding to use
            recursive: Whether to search directories recursively
            cache_size: Maximum number of documents to cache in memory (None for all)
        """
        self.pattern = path_pattern
        self.content_key = content_key
        self.encoding = encoding
        self.recursive = recursive
        self.cache_size = cache_size
        self._document_cache = None

    def _load_documents(self) -> List[str]:
        """Load and cache documents from JSONL files."""
        if self._document_cache is not None:
            return self._document_cache

        documents = []
        files = glob.glob(self.pattern, recursive=self.recursive)
        assert len(files) > 0, f"No JSONL files found matching pattern: {self.pattern}"

        for file_path in files:
            with open(file_path, "r", encoding=self.encoding) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if isinstance(data, dict) and self.content_key in data:
                            documents.append(data[self.content_key])
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON lines

                    # Check cache limit
                    if self.cache_size and len(documents) >= self.cache_size:
                        break

        self._document_cache = documents
        return documents

    def get_documents(self, n_samples: int) -> List[str]:
        """Get random documents from JSONL files."""
        documents = self._load_documents()
        return random.sample(documents, min(n_samples, len(documents)))


class DirectoryDocumentProvider(DocumentProvider):
    """Provides documents from a directory with support for multiple file types."""

    def __init__(
        self,
        directory: str | Path,
        file_patterns: List[str] = ["*.txt", "*.md"],
        encoding: str = "utf-8",
        recursive: bool = True,
        min_length: int = 10,
    ):
        """Initialize the directory document provider.

        Args:
            directory: Base directory to search
            file_patterns: List of glob patterns for file types
            encoding: File encoding to use
            recursive: Whether to search directories recursively
            min_length: Minimum document length to include
        """
        self.directory = Path(directory)
        self.patterns = file_patterns
        self.encoding = encoding
        self.recursive = recursive
        self.min_length = min_length
        self._cached_files = None

    def _get_matching_files(self) -> List[Path]:
        """Get list of all matching files in directory."""
        if self._cached_files is not None:
            return self._cached_files

        files = []
        for pattern in self.patterns:
            glob_pattern = "**/" + pattern if self.recursive else pattern
            files.extend(self.directory.glob(glob_pattern))

        self._cached_files = files
        return files

    def get_documents(self, n_samples: int) -> List[str]:
        """Get random documents from matching files."""
        files = self._get_matching_files()
        assert len(files) > 0, f"No matching files found in: {self.directory}"

        documents = []
        shuffled_files = random.sample(files, len(files))

        for file_path in shuffled_files:
            try:
                with open(file_path, "r", encoding=self.encoding) as f:
                    content = f.read().strip()
                    if len(content) >= self.min_length:
                        documents.append(content)
                        if len(documents) >= n_samples:
                            break
            except Exception:
                continue  # Skip problematic files

        return documents


class QdrantDocumentProvider(DocumentProvider):
    """Provides documents from Qdrant collection using scroll API."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        content_key: str = "text",
        batch_size: int = 100,
        filter: Optional[Filter] = None,
        with_payload_key: Optional[List[str]] = None,
        cache_size: Optional[int] = None,
    ):
        """Initialize the Qdrant document provider.

        Args:
            client: Initialized QdrantClient
            collection_name: Name of the collection to scroll
            content_key: Key in the payload containing the text content
            batch_size: Number of documents to fetch per scroll request
            filter: Optional filter conditions for document selection
            with_payload_key: Optional list of specific payload keys to retrieve
                            (optimization for large payloads)
            cache_size: Maximum number of documents to cache (None for all)
        """
        self.client = client
        self.collection_name = collection_name
        self.content_key = content_key
        self.batch_size = batch_size
        self.filter = filter
        self.with_payload_key = (
            [content_key] if with_payload_key is None else with_payload_key
        )
        self.cache_size = cache_size
        self._document_cache = None
        self._total_docs = None

    def _scroll_documents(self) -> List[str]:
        """Scroll through documents in the collection."""
        documents = []
        offset = None

        while True:
            # Scroll request with optional filter and payload selection
            scroll_response = self.client.scroll(
                collection_name=self.collection_name,
                offset=offset,
                limit=self.batch_size,
                scroll_filter=self.filter,
                with_payload=True,
                with_vectors=False,
            )

            # Extract documents from response
            points: List[ScoredPoint] = scroll_response[0]
            if not points:
                break

            # Process each point
            for point in points:
                if (
                    point.payload
                    and self.content_key in point.payload
                    and isinstance(point.payload[self.content_key], str)
                ):
                    documents.append(point.payload[self.content_key])

                    # Check cache limit
                    if self.cache_size and len(documents) >= self.cache_size:
                        return documents

            # Update offset for next batch
            offset = points[-1].id

            # If we got fewer points than batch_size, we're done
            if len(points) < self.batch_size:
                break

        return documents

    def _load_documents(self) -> List[str]:
        """Load and cache documents from Qdrant."""
        if self._document_cache is not None:
            return self._document_cache

        print("Filling document cash from Qdrant...")
        documents = self._scroll_documents()
        assert len(documents) > 0, (
            f"No documents found in collection {self.collection_name} "
            f"with content key: {self.content_key}"
        )

        self._document_cache = documents
        self._total_docs = len(documents)
        return documents

    def get_documents(self, n_samples: int) -> List[str]:
        """Get random documents from Qdrant collection.

        Args:
            n_samples: Number of documents to retrieve

        Returns:
            List[str]: List of randomly sampled documents
        """
        documents = self._load_documents()
        return random.sample(documents, min(n_samples, len(documents)))

    def clear_cache(self):
        """Clear the document cache to force reloading."""
        self._document_cache = None
        self._total_docs = None

    @property
    def total_documents(self) -> Optional[int]:
        """Get the total number of cached documents."""
        return self._total_docs
