from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Protocol, Dict, Any
import json
from filelock import FileLock
from datetime import datetime
import asyncio

from .types import ConversationWithContext, EvaluatedConversationWithContext


class DatasetStorage(Protocol):
    """Protocol defining the interface for dataset storage implementations."""

    @abstractmethod
    def save_conversations(
        self,
        conversations: List[ConversationWithContext | EvaluatedConversationWithContext],
    ) -> None:
        """Save conversations to storage.

        Args:
            conversations: List of conversations to save
        """
        pass

    @abstractmethod
    async def asave_conversations(
        self,
        conversations: List[ConversationWithContext | EvaluatedConversationWithContext],
    ) -> None:
        """Save conversations to storage asynchronously.

        Args:
            conversations: List of conversations to save
        """
        pass

    @abstractmethod
    def load_conversations(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[ConversationWithContext]:
        """Load conversations from storage.

        Args:
            limit: Maximum number of conversations to load
            offset: Number of conversations to skip

        Returns:
            List of conversations
        """
        pass


class JSONLStorage(DatasetStorage):
    """Stores conversations in JSONL format with thread-safe file access."""

    @staticmethod
    def _get_default_path() -> Path:
        """Generate default path using current datetime.

        Returns:
            Path: Default path like 'conversations_YYYYMMDD_HHMMSS.jsonl'
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(f"conversations_{timestamp}.jsonl")

    def __init__(
        self,
        path: Optional[str | Path] = None,
        encoding: str = "utf-8",
        lock_timeout: int = 30,
    ):
        """Initialize JSONL storage.

        Args:
            path: Path to JSONL file. If None, uses datetime-based filename
            encoding: File encoding to use
            lock_timeout: Maximum seconds to wait for file lock
        """
        self.path = Path(path) if path is not None else self._get_default_path()
        self.encoding = encoding
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.lock_timeout = lock_timeout
        self._async_lock = asyncio.Lock()

    def save_conversations(
        self,
        conversations: List[ConversationWithContext | EvaluatedConversationWithContext],
    ) -> None:
        """Save conversations to JSONL file with thread-safe access.

        Args:
            conversations: List of conversations to save
        """
        with FileLock(self.lock_path, timeout=self.lock_timeout):
            mode = "a" if self.path.exists() else "w"
            with open(self.path, mode, encoding=self.encoding) as f:
                for conv in conversations:
                    f.write(json.dumps(conv.model_dump(), ensure_ascii=False) + "\n")

    async def asave_conversations(
        self,
        conversations: List[ConversationWithContext | EvaluatedConversationWithContext],
    ) -> None:
        """Save conversations to JSONL file asynchronously."""
        def _save():
            with FileLock(self.lock_path, timeout=self.lock_timeout):
                mode = "a" if self.path.exists() else "w"
                with open(self.path, mode, encoding=self.encoding) as f:
                    for conv in conversations:
                        f.write(json.dumps(conv.model_dump(), ensure_ascii=False) + "\n")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _save)

    def load_conversations(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[ConversationWithContext]:
        """Load conversations from JSONL file.

        Args:
            limit: Maximum number of conversations to load
            offset: Number of conversations to skip

        Returns:
            List of conversations
        """
        if not self.path.exists():
            return []

        with FileLock(self.lock_path, timeout=self.lock_timeout):
            conversations = []
            current_idx = 0

            with open(self.path, "r", encoding=self.encoding) as f:
                for line in f:
                    if offset and current_idx < offset:
                        current_idx += 1
                        continue

                    conv_data = json.loads(line.strip())
                    conversations.append(ConversationWithContext(**conv_data))

                    current_idx += 1
                    if limit and len(conversations) >= limit:
                        break

            return conversations


class SQLStorage(DatasetStorage):
    """Stores conversations using SQLAlchemy with support for multiple backends."""

    def __init__(
        self,
        url: str,  # e.g., "sqlite:///data.db", "postgresql://user:pass@localhost/db"
        table_name: str = "conversations",
        metadata_fields: Optional[List[str]] = None,
        batch_size: int = 100,
    ):
        """Initialize SQL storage.

        Args:
            url: Database URL (supports SQLite, PostgreSQL, MySQL, etc.)
            table_name: Name of the table to store conversations
            metadata_fields: List of metadata fields to index
            batch_size: Number of records to insert in one batch

        Raises:
            ImportError: If sqlalchemy is not installed
        """
        try:
            from sqlalchemy import (  # type: ignore
                create_engine,
                MetaData,
                Table,
                Column,
                Integer,
                String,
                JSON,
                DateTime,
            )
            from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
        except ImportError:
            raise ImportError(
                "SQL storage requires 'sqlalchemy' package. "
                "Install it with: pip install 'sqlalchemy>=2.0'"
            )

        self.engine = create_engine(url)
        self.async_engine = create_async_engine(url)
        self.metadata = MetaData()
        self.batch_size = batch_size
        self.async_session_maker = async_sessionmaker(self.async_engine)

        # Define table
        self.table = Table(
            table_name,
            self.metadata,
            Column("id", Integer, primary_key=True),
            Column("conversations", JSON),
            Column("context", String),
            Column("metadata", JSON),
            Column("timestamp", DateTime),
            Column("evaluation", JSON, nullable=True),
        )

        # Create table if it doesn't exist
        self.metadata.create_all(self.engine)

        # Create indexes for metadata fields
        if metadata_fields:
            for field in metadata_fields:
                idx_name = f"idx_{table_name}_{field}"
                self.engine.execute(
                    f"CREATE INDEX IF NOT EXISTS {idx_name} "
                    f"ON {table_name} ((metadata->'{field}'))"
                )

    def save_conversations(
        self,
        conversations: List[ConversationWithContext | EvaluatedConversationWithContext],
    ) -> None:
        """Save conversations to database.

        Args:
            conversations: List of conversations to save
        """
        records = []
        for conv in conversations:
            data = conv.model_dump()
            record = {
                "conversations": data["conversations"],
                "context": data["context"],
                "metadata": data.get("metadata", {}),
                "evaluation": data.get("evaluation"),
                "timestamp": datetime.now(),
            }
            records.append(record)

        # Insert in batches
        with self.engine.begin() as conn:
            for i in range(0, len(records), self.batch_size):
                batch = records[i : i + self.batch_size]
                conn.execute(self.table.insert(), batch)

    async def asave_conversations(
        self,
        conversations: List[ConversationWithContext | EvaluatedConversationWithContext],
    ) -> None:
        """Save conversations to database asynchronously."""
        records = []
        for conv in conversations:
            data = conv.model_dump()
            record = {
                "conversations": data["conversations"],
                "context": data.get("context"),
                "metadata": data.get("metadata", {}),
                "evaluation": data.get("evaluation"),
                "timestamp": datetime.now(),
            }
            records.append(record)

        async with self.async_session_maker() as session:
            async with session.begin():
                for i in range(0, len(records), self.batch_size):
                    batch = records[i : i + self.batch_size]
                    await session.execute(self.table.insert(), batch)

    def load_conversations(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[List[tuple]] = None,
    ) -> List[ConversationWithContext]:
        """Load conversations from database with filtering and sorting.

        Args:
            limit: Maximum number of conversations to load
            offset: Number of conversations to skip
            filters: Dict of field-value pairs for filtering
            order_by: List of (field, direction) tuples for sorting

        Returns:
            List of conversations
        """
        query = self.table.select()

        if filters:
            for field, value in filters.items():
                if field.startswith("metadata."):
                    # Handle metadata field filtering
                    _, key = field.split(".", 1)
                    query = query.where(self.table.c.metadata[key] == value)
                else:
                    # Handle regular field filtering
                    query = query.where(getattr(self.table.c, field) == value)

        if order_by:
            for field, direction in order_by:
                col = getattr(self.table.c, field)
                query = query.order_by(col.desc() if direction == -1 else col)
        else:
            # Default sort by timestamp descending
            query = query.order_by(self.table.c.timestamp.desc())

        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)

        with self.engine.connect() as conn:
            result = conn.execute(query)
            return [
                ConversationWithContext(
                    conversations=row.conversations,
                    context=row.context,
                    metadata=row.metadata,
                    evaluation=row.evaluation,
                )
                for row in result
            ]
