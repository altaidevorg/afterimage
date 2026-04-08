"""Job lifecycle management — submit, track, cancel, list."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime

from ..models import GenerationRequest, JobProgress, JobResult, JobStatus
from ..storage.job_store import JobRecord, JobStore
from ..storage.result_store import ResultStore
from .generation_service import GenerationResult, GenerationService


class JobManager:
    """Manages background generation jobs with concurrency control."""

    def __init__(
        self,
        job_store: JobStore,
        result_store: ResultStore,
        max_concurrent_jobs: int = 3,
    ):
        self._job_store = job_store
        self._result_store = result_store
        self._generation_service = GenerationService(result_store)
        self._semaphore = asyncio.Semaphore(max_concurrent_jobs)
        # Map job_id -> asyncio.Task for active jobs
        self._tasks: dict[str, asyncio.Task] = {}
        # SSE subscriber queues: job_id -> list of asyncio.Queue
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit(self, request: GenerationRequest) -> JobRecord:
        now = datetime.utcnow()
        record = JobRecord(
            job_id=str(uuid.uuid4()),
            status="queued",
            request=request,
            created_at=now,
            updated_at=now,
        )
        await self._job_store.save(record)
        task = asyncio.create_task(self._run_job(record))
        self._tasks[record.job_id] = task
        return record

    async def get(self, job_id: str) -> JobRecord | None:
        return await self._job_store.get(job_id)

    async def list_jobs(self, page: int = 1, per_page: int = 20):
        return await self._job_store.list_jobs(page=page, per_page=per_page)

    async def cancel(self, job_id: str) -> bool:
        record = await self._job_store.get(job_id)
        if record is None or record.status in ("completed", "failed", "cancelled"):
            return False
        record._cancel_event.set()
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
        await self._job_store.update_status(job_id, "cancelled")
        await self._broadcast(job_id, {"event": "cancelled", "job_id": job_id})
        return True

    def subscribe(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(job_id, []).append(q)
        return q

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        subs = self._subscribers.get(job_id, [])
        if queue in subs:
            subs.remove(queue)

    async def active_job_count(self) -> int:
        records, _ = await self._job_store.list_jobs(page=1, per_page=1000)
        return sum(1 for r in records if r.status in ("queued", "running"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_job(self, record: JobRecord) -> None:
        async with self._semaphore:
            await self._update_status(record, "running")

            async def _progress_cb(progress: JobProgress) -> None:
                await self._job_store.update_progress(record.job_id, progress)
                await self._broadcast(
                    record.job_id,
                    {
                        "event": "progress",
                        "job_id": record.job_id,
                        **progress.model_dump(),
                    },
                )

            try:
                result: GenerationResult = await self._generation_service.run(
                    job_id=record.job_id,
                    request=record.request,
                    progress_callback=_progress_cb,
                    cancel_event=record._cancel_event,
                )
                job_result = JobResult(
                    num_conversations=result.num_conversations,
                    download_url=f"/api/v1/jobs/{record.job_id}/result",
                    output_format=result.output_format,
                )
                await self._job_store.update_status(
                    record.job_id, "completed", result=job_result
                )
                await self._broadcast(
                    record.job_id,
                    {
                        "event": "complete",
                        "job_id": record.job_id,
                        "download_url": job_result.download_url,
                        "num_conversations": result.num_conversations,
                    },
                )
            except asyncio.CancelledError:
                await self._job_store.update_status(record.job_id, "cancelled")
                await self._broadcast(
                    record.job_id, {"event": "cancelled", "job_id": record.job_id}
                )
            except Exception as exc:
                await self._job_store.update_status(
                    record.job_id, "failed", error=str(exc)
                )
                await self._broadcast(
                    record.job_id,
                    {"event": "error", "job_id": record.job_id, "error": str(exc)},
                )
            finally:
                self._tasks.pop(record.job_id, None)

    async def _update_status(self, record: JobRecord, status: JobStatus) -> None:
        record.status = status
        record.updated_at = datetime.utcnow()
        await self._job_store.save(record)

    async def _broadcast(self, job_id: str, payload: dict) -> None:
        for q in list(self._subscribers.get(job_id, [])):
            await q.put(payload)
