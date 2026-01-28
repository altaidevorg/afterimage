"""
Custom storage implementation for the demo UI.
Captures items in-memory for live updates and writes to a temporary file for download.
"""

import json
import os
from datetime import datetime
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
    """Storage that captures items in-memory for UI and writes to datasets folder."""

    def __init__(
        self, 
        dataset_name: str | None = None, 
        tools_used: List[str] | None = None,
        category: str | None = None
    ):
        from .config import get_datasets_dir
        
        self.captured_items: list[StructuredGenerationRow] = []
        self.tools_used = tools_used or []
        self.category = category or "Uncategorized"
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = dataset_name or f"dataset_{timestamp}"
        self.dataset_name = name
        filename = f"{name}.jsonl"
        
        # Save to datasets folder
        datasets_dir = get_datasets_dir()
        self.filepath = f"{datasets_dir}/{filename}"
        self.metadata_path = f"{datasets_dir}/{name}.meta.json"
        
        self.jsonl_storage = JSONLStorage(conversations_path=self.filepath)

    def save_metadata(self):
        """Save metadata JSON alongside the dataset."""
        # Count tool usage from captured items
        tool_distribution = {}
        for item in self.captured_items:
            tool_calls = None
            
            # Handle dict structure (common case)
            if isinstance(item, dict):
                output = item.get('output', {})
                if isinstance(output, dict):
                    tool_calls = output.get('tool_calls', [])
            # Handle Pydantic model with output attribute
            elif hasattr(item, 'output'):
                output = item.output
                if hasattr(output, 'tool_calls'):
                    tool_calls = output.tool_calls
                elif isinstance(output, dict):
                    tool_calls = output.get('tool_calls', [])
            # Handle direct tool_calls attribute
            elif hasattr(item, 'tool_calls'):
                tool_calls = item.tool_calls
            
            # Extract tool names from tool_calls
            if tool_calls:
                for tc in tool_calls:
                    tool_name = None
                    if isinstance(tc, dict):
                        func = tc.get('function', {})
                        tool_name = func.get('name') if isinstance(func, dict) else None
                    elif hasattr(tc, 'function'):
                        func = tc.function
                        tool_name = func.get('name') if isinstance(func, dict) else getattr(func, 'name', None)
                    
                    if tool_name:
                        tool_distribution[tool_name] = tool_distribution.get(tool_name, 0) + 1
        
        metadata = {
            "name": self.dataset_name,
            "category": self.category,
            "created_at": datetime.now().isoformat(),
            "total_samples": len(self.captured_items),
            "tools_used": self.tools_used,
            "tool_distribution": tool_distribution,
            "file_path": self.filepath,
        }
        
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    async def save_conversations(
        self,
        conversations: List[
            EvaluatedConversationWithContext | ConversationWithContext | BaseModel
        ],
    ) -> None:
        self.captured_items.extend(conversations)
        self.jsonl_storage.save_conversations(conversations)
        self.save_metadata()

    async def asave_conversations(
        self,
        conversations: List[
            ConversationWithContext | EvaluatedConversationWithContext | BaseModel
        ],
    ) -> None:
        self.captured_items.extend(conversations)
        await self.jsonl_storage.asave_conversations(conversations)
        self.save_metadata()

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
