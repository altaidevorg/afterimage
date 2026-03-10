from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Protocol, Dict, Any
import json
from filelock import FileLock
from datetime import datetime
import asyncio

from .types import (
    ConversationWithContext,
    EvaluatedConversationWithContext,
    Document,
)
from pydantic import BaseModel


class BaseStorage(Protocol):
    """Protocol defining the interface for storage implementations."""

    @abstractmethod
    def save_conversations(
        self,
        conversations: List[
            EvaluatedConversationWithContext | ConversationWithContext | BaseModel
        ],
    ) -> None:
        pass

    @abstractmethod
    async def asave_conversations(
        self,
        conversations: List[
            ConversationWithContext | EvaluatedConversationWithContext | BaseModel
        ],
    ) -> None:
        pass

    @abstractmethod
    def load_conversations(
        self,
        limit: int | None = None,
        offset: int | None = None,
    ) -> List[ConversationWithContext]:
        pass

    @abstractmethod
    def load_documents(
        self,
        limit: int | None = None,
        offset: int | None = None,
    ) -> List[Document]:
        pass

    @abstractmethod
    def save_documents(self, documents: List[Document]) -> None:
        pass

    @abstractmethod
    async def asave_documents(self, documents: List[Document]) -> None:
        pass


class JSONLStorage(BaseStorage):
    """Stores conversations and documents in JSONL format."""

    def __init__(
        self,
        conversations_path: Optional[str | Path] = None,
        documents_path: Optional[str | Path] = None,
        encoding: str = "utf-8",
        lock_timeout: int = 30,
    ):
        self.encoding = encoding
        self.lock_timeout = lock_timeout

        self.conversations_path = (
            Path(conversations_path)
            if conversations_path
            else self._get_default_path("conversations")
        )
        self.conversations_lock_path = self.conversations_path.with_suffix(
            self.conversations_path.suffix + ".lock"
        )

        self.documents_path = (
            Path(documents_path)
            if documents_path
            else self._get_default_path("documents")
        )
        self.documents_lock_path = self.documents_path.with_suffix(
            self.documents_path.suffix + ".lock"
        )

    @staticmethod
    def _get_default_path(prefix: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(f"{prefix}_{timestamp}.jsonl")

    def save_conversations(
        self,
        conversations: List[
            ConversationWithContext | EvaluatedConversationWithContext | BaseModel
        ],
    ) -> None:
        with FileLock(self.conversations_lock_path, timeout=self.lock_timeout):
            mode = "a" if self.conversations_path.exists() else "w"
            with open(self.conversations_path, mode, encoding=self.encoding) as f:
                for conv in conversations:
                    f.write(json.dumps(conv.model_dump(), ensure_ascii=False) + "\n")

    async def asave_conversations(
        self,
        conversations: List[
            ConversationWithContext | EvaluatedConversationWithContext | BaseModel
        ],
    ) -> None:
        def _save():
            self.save_conversations(conversations)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _save)

    def load_conversations(
        self,
        limit: int | None = None,
        offset: int | None = None,
    ) -> List[EvaluatedConversationWithContext]:
        """Load conversations from JSONL file.

        Args:
            limit: Maximum number of conversations to load
            offset: Number of conversations to skip

        Returns:
            List of conversations
        """
        if not self.conversations_path.exists():
            return []

        with FileLock(self.conversations_lock_path, timeout=self.lock_timeout):
            conversations = []
            current_idx = 0

            with open(self.conversations_path, "r", encoding=self.encoding) as f:
                for line in f:
                    if offset and current_idx < offset:
                        current_idx += 1
                        continue

                    conv_data = json.loads(line.strip())
                    conversations.append(EvaluatedConversationWithContext(**conv_data))

                    current_idx += 1
                    if limit and len(conversations) >= limit:
                        break

            return conversations

    def load_documents(
        self,
        limit: int | None = None,
        offset: int | None = None,
    ) -> List[Document]:
        """Load documents from JSONL file.

        Args:
            limit: Maximum number of documents to load
            offset: Number of documents to skip

        Returns:
            List of documents
        """
        if not self.documents_path.exists():
            return []

        with FileLock(self.documents_lock_path, timeout=self.lock_timeout):
            documents = []
            current_idx = 0

            with open(self.documents_path, "r", encoding=self.encoding) as f:
                for line in f:
                    if offset and current_idx < offset:
                        current_idx += 1
                        continue

                    doc_data = json.loads(line.strip())
                    documents.append(Document(**doc_data))

                    current_idx += 1
                    if limit and len(documents) >= limit:
                        break

            return documents

    def save_documents(self, documents: List[Document]) -> None:
        with FileLock(self.documents_lock_path, timeout=self.lock_timeout):
            mode = "a" if self.documents_path.exists() else "w"
            with open(self.documents_path, mode, encoding=self.encoding) as f:
                for entry in documents:
                    f.write(entry.model_dump_json() + "\n")

    async def asave_documents(self, documents: List[Document]) -> None:
        def _save():
            self.save_documents(documents)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _save)


class SQLStorage(BaseStorage):
    """Stores conversations and documents using SQLAlchemy."""

    def __init__(
        self,
        url: str,
        conversations_table_name: str = "conversations",
        documents_table_name: str = "documents",
        metadata_fields: Optional[List[str]] = None,
        batch_size: int = 100,
    ):
        try:
            from sqlalchemy import (
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
            raise ImportError("SQL storage requires 'sqlalchemy'.")

        self.engine = create_engine(url)
        self.async_engine = create_async_engine(url)
        self.metadata = MetaData()
        self.batch_size = batch_size
        self.async_session_maker = async_sessionmaker(self.async_engine)

        self.conversations_table = Table(
            conversations_table_name,
            self.metadata,
            Column("id", Integer, primary_key=True),
            Column("conversations", JSON),
            Column("instruction_context", String, nullable=True),
            Column("response_context", String, nullable=True),
            Column("metadata", JSON, nullable=True),
            Column("timestamp", DateTime),
            Column("evaluation", JSON, nullable=True),
        )

        self.documents_table = Table(
            documents_table_name,
            self.metadata,
            Column("_id", Integer, primary_key=True),
            Column("id", String),
            Column("text", String),
            Column("personas", JSON, nullable=True),
            Column("metadata", JSON, nullable=True),
        )

        # Create table if it doesn't exist
        self.metadata.create_all(self.engine)

        # Create indexes for metadata fields
        if metadata_fields:
            for field in metadata_fields:
                idx_name = f"idx_{conversations_table_name}_{field}"
                self.engine.execute(
                    f"CREATE INDEX IF NOT EXISTS {idx_name} "
                    f"ON {conversations_table_name} ((metadata->'{field}'))"
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
            # If it's a generic BaseModel, wrapping it in a structure compatible with the table
            # or we might need a separate table/method for generic types.
            # GUIDANCE: For now, we assume if it's NOT a Conversation object, we try to fit it or fail if table doesn't match.
            # But the requirement was mainly for JSONL.
            # Let's simple check if keys exist.
            if "conversations" in data:
                record = {
                    "conversations": data["conversations"],
                    "instruction_context": data.get("instruction_context"),
                    "response_context": data.get("response_context"),
                    "metadata": data.get("metadata", {}),
                    "evaluation": data.get("evaluation"),
                    "timestamp": datetime.now(),
                }
            else:
                # Fallback for generic models: store the whole model in 'metadata' or 'conversations' column?
                # SQLStorage relies on specific schema. Storing generic models in 'conversations' column as JSON.
                record = {
                    "conversations": data,  # Storing the whole object in the JSON column
                    "timestamp": datetime.now(),
                    "metadata": {},
                    "instruction_context": None,
                    "response_context": None,
                }
            records.append(record)

        # Insert in batches
        with self.engine.begin() as conn:
            for i in range(0, len(records), self.batch_size):
                batch = records[i : i + self.batch_size]
                conn.execute(self.conversations_table.insert(), batch)

    async def asave_conversations(
        self,
        conversations: List[
            ConversationWithContext | EvaluatedConversationWithContext | BaseModel
        ],
    ) -> None:
        """Save conversations to database asynchronously."""
        records = []
        for conv in conversations:
            data = conv.model_dump()
            if "conversations" in data:
                record = {
                    "conversations": data["conversations"],
                    "instruction_context": data.get("instruction_context"),
                    "response_context": data.get("response_context"),
                    "metadata": data.get("metadata", {}),
                    "evaluation": data.get("evaluation"),
                    "timestamp": datetime.now(),
                }
            else:
                record = {
                    "conversations": data,
                    "timestamp": datetime.now(),
                    "metadata": {},
                    "instruction_context": None,
                    "response_context": None,
                }
            records.append(record)

        async with self.async_session_maker() as session:
            async with session.begin():
                for i in range(0, len(records), self.batch_size):
                    batch = records[i : i + self.batch_size]
                    await session.execute(self.conversations_table.insert(), batch)

    def load_conversations(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[List[tuple]] = None,
    ) -> List[EvaluatedConversationWithContext]:
        """Load conversations from database with filtering and sorting.

        Args:
            limit: Maximum number of conversations to load
            offset: Number of conversations to skip
            filters: Dict of field-value pairs for filtering
            order_by: List of (field, direction) tuples for sorting

        Returns:
            List of conversations
        """
        query = self.conversations_table.select()

        if filters:
            for field, value in filters.items():
                if field.startswith("metadata."):
                    # Handle metadata field filtering
                    _, key = field.split(".", 1)
                    query = query.where(
                        self.conversations_table.c.metadata[key] == value
                    )
                else:
                    # Handle regular field filtering
                    query = query.where(
                        getattr(self.conversations_table.c, field) == value
                    )

        if order_by:
            for field, direction in order_by:
                col = getattr(self.conversations_table.c, field)
                query = query.order_by(col.desc() if direction == -1 else col)
        else:
            # Default sort by timestamp descending
            query = query.order_by(self.conversations_table.c.timestamp.desc())

        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)

        with self.engine.connect() as conn:
            result = conn.execute(query)
            return [
                EvaluatedConversationWithContext(
                    conversations=row.conversations,
                    instruction_context=row.instruction_context,
                    response_context=row.response_context,
                    metadata=row.metadata,
                    evaluation=row.evaluation,
                )
                for row in result
            ]

    def load_documents(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[List[tuple]] = None,
    ) -> List[Document]:
        """Load documents from database with filtering and sorting.

        Args:
            limit: Maximum number of documents to load
            offset: Number of documents to skip
            filters: Dict of field-value pairs for filtering
            order_by: List of (field, direction) tuples for sorting

        Returns:
            List of documents
        """
        query = self.documents_table.select()

        if filters:
            for field, value in filters.items():
                if field.startswith("metadata."):
                    # Handle metadata field filtering
                    _, key = field.split(".", 1)
                    query = query.where(self.documents_table.c.metadata[key] == value)
                else:
                    # Handle regular field filtering
                    query = query.where(getattr(self.documents_table.c, field) == value)

        if order_by:
            for field, direction in order_by:
                col = getattr(self.documents_table.c, field)
                query = query.order_by(col.desc() if direction == -1 else col)

        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)

        with self.engine.connect() as conn:
            result = conn.execute(query)
            return [
                Document(
                    id=row.id,
                    text=row.text,
                    personas=row.personas,
                    metadata=row.metadata,
                )
                for row in result
            ]

    def save_documents(self, documents: List[Document]) -> None:
        records = [document.model_dump(mode="json") for document in documents]
        with self.engine.begin() as conn:
            for i in range(0, len(records), self.batch_size):
                batch = records[i : i + self.batch_size]
                conn.execute(self.documents_table.insert(), batch)

    async def asave_documents(self, documents: List[Document]) -> None:
        records = [document.model_dump(mode="json") for document in documents]
        async with self.async_session_maker() as session:
            async with session.begin():
                for i in range(0, len(records), self.batch_size):
                    batch = records[i : i + self.batch_size]
                    await session.execute(self.documents_table.insert(), batch)
