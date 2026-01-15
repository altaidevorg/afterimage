"""
Custom storage implementation for the demo UI.
Captures items in-memory for live updates and writes to a temporary file for download.
"""

import tempfile
from typing import List

from pydantic import BaseModel

from afterimage.storage import BaseStorage, JSONLStorage
from afterimage.types import (
    ConversationWithContext,
    Document,
    EvaluatedConversationWithContext,
    StructuredGenerationRow,
)


class CaptureStorage(BaseStorage):
    """Storage that captures items in-memory for UI and writes to a temporary file."""

    def __init__(self):
        self.captured_items: list[StructuredGenerationRow] = []
        self.temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".jsonl", mode="w+", encoding="utf-8"
        )
        self.jsonl_storage = JSONLStorage(conversations_path=self.temp_file.name)
        self.temp_file.close()

    async def save_conversations(
        self,
        conversations: List[
            EvaluatedConversationWithContext | ConversationWithContext | BaseModel
        ],
    ) -> None:
        self.captured_items.extend(conversations)
        self.jsonl_storage.save_conversations(conversations)

    async def asave_conversations(
        self,
        conversations: List[
            ConversationWithContext | EvaluatedConversationWithContext | BaseModel
        ],
    ) -> None:
        self.captured_items.extend(conversations)
        await self.jsonl_storage.asave_conversations(conversations)

    def load_conversations(
        self, limit: int | None = None, offset: int | None = None
    ) -> List[ConversationWithContext]:
        return []

    def save_documents(self, documents: List[Document]) -> None:
        pass

    async def asave_documents(self, documents: List[Document]) -> None:
        pass

    def get_download_path(self) -> str:
        return self.jsonl_storage.conversations_path.absolute().as_posix()

    def clear(self) -> None:
        """Clear captured items."""
        self.captured_items.clear()
