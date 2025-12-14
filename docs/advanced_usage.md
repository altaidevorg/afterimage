# Advanced Configuration

As you move from prototyping to production, you'll need more robustness. This guide covers how to handle API rate limits and store data in production-grade databases.

## Smart Key Management

LLM APIs often have strict rate limits (RPM/TPM). Using a single key can bottleneck your generation speed. Afterimage provides `SmartKeyPool` to rotate through multiple keys automatically.

### `SmartKeyPool`

This manager rotates keys, respects rate limits, and automatically cools down keys that hit errors.

```python
from afterimage import AsyncConversationGenerator, SmartKeyPool

# 1. Configure the Pool
key_pool = SmartKeyPool(
    api_keys=["key_A...", "key_B...", "key_C..."],
    hourly_limit=1000,    # Max 1000 requests per key per hour
    cooldown_period=600   # Wait 10 mins if a key errors out
)

# 2. Use it in Generator
generator = AsyncConversationGenerator(
    ...,
    api_key=key_pool  # Pass the pool instead of a string
)
```

The generator will now automatically cycle through available keys, maximizing your aggregate throughput.

## Storage Backends

By default, Afterimage saves to local JSONL files. For larger datasets or better querying, you should use a SQL database.

### `SQLStorage`

Supports SQLite, PostgreSQL, MySQL, and any other database supported by SQLAlchemy.

```python
from afterimage import AsyncConversationGenerator
from afterimage.storage import SQLStorage

# 1. Initialize Storage
# Example: PostgreSQL
storage = SQLStorage(
    url="postgresql://user:pass@localhost:5432/mydb",
    conversations_table_name="synthetic_dataset"
)

# 2. Use in Generator
generator = AsyncConversationGenerator(
    ...,
    storage=storage
)
```

When using `SQLStorage`, generated conversations are inserted as rows. The table schema is automatically handled by Afterimage (using SQLAlchemy ORM).

### `JSONLStorage` (Advanced)

If you stick with files, you can check where they are being saved.

```python
from afterimage.storage import JSONLStorage

storage = JSONLStorage(
    conversations_path="output/my_dataset.jsonl",
    documents_path="output/my_docs.jsonl"
)
```

## Custom Document Providers

If your data lives in a custom API or a specific format, you can write your own `DocumentProvider`. You just need to implement the protocol that yields `Document` objects.

```python
from typing import AsyncIterator
from afterimage.types import Document

class MyAPIDocumentProvider:
    def __init__(self, api_endpoint):
        self.api_endpoint = api_endpoint

    async def get_documents(self, batch_size: int = 10) -> AsyncIterator[Document]:
        # Fetch from your API and yield Document objects
        items = await self._fetch_from_api() 
        for item in items:
            yield Document(content=item['text'], metadata=item['meta'])
```

---
[Previous: Monitoring & Observability](monitoring.md) | [Next: Architecture & Design](architecture.md)
