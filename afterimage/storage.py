from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Protocol, Dict, Any
import json
from filelock import FileLock
from datetime import datetime
import asyncio
import dataclasses

from .types import (
    ConversationWithContext,
    EvaluatedConversationWithContext,
    PersonaEntry,
)


class BaseStorage(Protocol):
    """Protocol defining the interface for storage implementations."""

    @abstractmethod
    def save_conversations(
        self,
        conversations: List[EvaluatedConversationWithContext | EvaluatedConversationWithContext],
    ) -> None:
        pass

    @abstractmethod
    async def asave_conversations(
        self,
        conversations: List[ConversationWithContext | EvaluatedConversationWithContext],
    ) -> None:
        pass

    @abstractmethod
    def load_conversations(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[ConversationWithContext]:
        pass

    @abstractmethod
    def save_personas(self, personas: List[PersonaEntry]) -> None:
        pass

    @abstractmethod
    async def asave_personas(self, personas: List[PersonaEntry]) -> None:
        pass


class JSONLStorage(BaseStorage):
    """Stores conversations and personas in JSONL format."""

    def __init__(
        self,
        conversations_path: Optional[str | Path] = None,
        personas_path: Optional[str | Path] = None,
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

        self.personas_path = (
            Path(personas_path) if personas_path else self._get_default_path("personas")
        )
        self.personas_lock_path = self.personas_path.with_suffix(
            self.personas_path.suffix + ".lock"
        )

    @staticmethod
    def _get_default_path(prefix: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(f"{prefix}_{timestamp}.jsonl")

    def save_conversations(
        self,
        conversations: List[ConversationWithContext | EvaluatedConversationWithContext],
    ) -> None:
        with FileLock(self.conversations_lock_path, timeout=self.lock_timeout):
            mode = "a" if self.conversations_path.exists() else "w"
            with open(self.conversations_path, mode, encoding=self.encoding) as f:
                for conv in conversations:
                    f.write(json.dumps(conv.model_dump(), ensure_ascii=False) + "\n")

    async def asave_conversations(
        self,
        conversations: List[ConversationWithContext | EvaluatedConversationWithContext],
    ) -> None:
        def _save():
            self.save_conversations(conversations)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _save)

    def load_conversations(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
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

    def save_personas(self, personas: List[PersonaEntry]) -> None:
        with FileLock(self.personas_lock_path, timeout=self.lock_timeout):
            mode = "a" if self.personas_path.exists() else "w"
            with open(self.personas_path, mode, encoding=self.encoding) as f:
                for entry in personas:
                    f.write(json.dumps(dataclasses.asdict(entry), default=str) + "\n")

    async def asave_personas(self, personas: List[PersonaEntry]) -> None:
        def _save():
            self.save_personas(personas)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _save)


class SQLStorage(BaseStorage):
    """Stores conversations and personas using SQLAlchemy."""

    def __init__(
        self,
        url: str,
        conversations_table_name: str = "conversations",
        personas_table_name: str = "personas",
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

        self.personas_table = Table(
            personas_table_name,
            self.metadata,
            Column("id", Integer, primary_key=True),
            Column("source_document", String),
            Column("personas", JSON),
            Column("timestamp", DateTime),
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
            record = {
                "conversations": data["conversations"],
                "instruction_context": data["instruction_context"],
                "response_context": data["response_context"],
                "metadata": data.get("metadata", {}),
                "evaluation": data.get("evaluation"),
                "timestamp": datetime.now(),
            }
            records.append(record)

        # Insert in batches
        with self.engine.begin() as conn:
            for i in range(0, len(records), self.batch_size):
                batch = records[i : i + self.batch_size]
                conn.execute(self.conversations_table.insert(), batch)

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
                "instruction_context": data.get("instruction_context"),
                "response_context": data.get("response_context"),
                "metadata": data.get("metadata", {}),
                "evaluation": data.get("evaluation"),
                "timestamp": datetime.now(),
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

    def save_personas(self, personas: List[PersonaEntry]) -> None:
        records = [dataclasses.asdict(p) for p in personas]
        with self.engine.begin() as conn:
            for i in range(0, len(records), self.batch_size):
                batch = records[i : i + self.batch_size]
                conn.execute(self.personas_table.insert(), batch)

    async def asave_personas(self, personas: List[PersonaEntry]) -> None:
        records = [dataclasses.asdict(p) for p in personas]
        async with self.async_session_maker() as session:
            async with session.begin():
                for i in range(0, len(records), self.batch_size):
                    batch = records[i : i + self.batch_size]
                    await session.execute(self.personas_table.insert(), batch)
