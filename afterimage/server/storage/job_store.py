"""SQLite-backed job store using aiosqlite."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite

from ..models import GenerationRequest, JobProgress, JobResult, JobStatus

if TYPE_CHECKING:
    pass


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus
    request: GenerationRequest
    created_at: datetime
    updated_at: datetime
    progress: JobProgress = field(default_factory=JobProgress)
    result: JobResult | None = None
    error: str | None = None
    # In-memory cancel event — not persisted
    _cancel_event: asyncio.Event = field(
        default_factory=asyncio.Event, repr=False, compare=False
    )


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    request_json TEXT NOT NULL,
    progress_json TEXT NOT NULL,
    result_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


class JobStore:
    """Persists job metadata to SQLite; also keeps an in-memory cache for speed."""

    def __init__(self, db_path: str | Path = "jobs.db"):
        self._db_path = str(db_path)
        self._cache: dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()
        self._initialized = False

    async def _ensure_init(self) -> None:
        if self._initialized:
            return
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(_CREATE_TABLE)
            await db.commit()
        self._initialized = True

    async def save(self, record: JobRecord) -> None:
        await self._ensure_init()
        async with self._lock:
            self._cache[record.job_id] = record
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    """INSERT OR REPLACE INTO jobs
                       (job_id, status, request_json, progress_json, result_json,
                        error, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record.job_id,
                        record.status,
                        record.request.model_dump_json(),
                        record.progress.model_dump_json(),
                        record.result.model_dump_json() if record.result else None,
                        record.error,
                        record.created_at.isoformat(),
                        record.updated_at.isoformat(),
                    ),
                )
                await db.commit()

    async def get(self, job_id: str) -> JobRecord | None:
        await self._ensure_init()
        if job_id in self._cache:
            return self._cache[job_id]
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ) as cur:
                row = await cur.fetchone()
        if row is None:
            return None
        record = _row_to_record(row)
        self._cache[job_id] = record
        return record

    async def list_jobs(
        self, page: int = 1, per_page: int = 20
    ) -> tuple[list[JobRecord], int]:
        await self._ensure_init()
        offset = (page - 1) * per_page
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM jobs") as cur:
                total = (await cur.fetchone())[0]
            async with db.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (per_page, offset),
            ) as cur:
                rows = await cur.fetchall()
        records = [_row_to_record(r) for r in rows]
        for r in records:
            if r.job_id not in self._cache:
                self._cache[r.job_id] = r
        return records, total

    async def update_progress(self, job_id: str, progress: JobProgress) -> None:
        """Fast in-memory update + async DB write (non-blocking)."""
        record = await self.get(job_id)
        if record:
            record.progress = progress
            record.updated_at = datetime.utcnow()
            asyncio.create_task(
                self._persist_progress(job_id, progress, record.updated_at)
            )

    async def _persist_progress(
        self, job_id: str, progress: JobProgress, updated_at: datetime
    ) -> None:
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE jobs SET progress_json = ?, updated_at = ? WHERE job_id = ?",
                (progress.model_dump_json(), updated_at.isoformat(), job_id),
            )
            await db.commit()

    async def update_status(
        self,
        job_id: str,
        status: JobStatus,
        result: JobResult | None = None,
        error: str | None = None,
    ) -> None:
        record = await self.get(job_id)
        if record:
            record.status = status
            record.result = result
            record.error = error
            record.updated_at = datetime.utcnow()
            await self.save(record)


def _row_to_record(row: tuple) -> JobRecord:
    (
        job_id,
        status,
        request_json,
        progress_json,
        result_json,
        error,
        created_at,
        updated_at,
    ) = row
    return JobRecord(
        job_id=job_id,
        status=status,
        request=GenerationRequest.model_validate_json(request_json),
        progress=JobProgress.model_validate_json(progress_json),
        result=JobResult.model_validate_json(result_json) if result_json else None,
        error=error,
        created_at=datetime.fromisoformat(created_at),
        updated_at=datetime.fromisoformat(updated_at),
    )
