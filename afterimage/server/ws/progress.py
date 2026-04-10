"""Server-Sent Events (SSE) progress streaming."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, AsyncGenerator

if TYPE_CHECKING:
    from ..services.job_manager import JobManager


async def sse_progress_generator(
    job_manager: "JobManager",
    job_id: str,
    heartbeat_interval: float = 15.0,
) -> AsyncGenerator[dict, None]:
    """Async generator that yields SSE-compatible dicts until the job finishes.

    Yields dicts of the form::

        {"event": "progress", "data": "<json string>"}

    which ``sse-starlette`` converts to valid SSE text/event-stream responses.
    """
    record = await job_manager.get(job_id)
    if record is None:
        yield {"event": "error", "data": json.dumps({"error": "job not found"})}
        return

    # If already terminal, stream the final state immediately
    if record.status == "completed" and record.result:
        yield {
            "event": "complete",
            "data": json.dumps(
                {
                    "job_id": job_id,
                    "download_url": record.result.download_url,
                    "num_conversations": record.result.num_conversations,
                }
            ),
        }
        return
    if record.status in ("failed", "cancelled"):
        yield {
            "event": record.status,
            "data": json.dumps({"job_id": job_id, "error": record.error}),
        }
        return

    queue = job_manager.subscribe(job_id)
    try:
        while True:
            try:
                payload = await asyncio.wait_for(
                    queue.get(), timeout=heartbeat_interval
                )
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                yield {"event": "heartbeat", "data": "{}"}
                continue

            event = payload.pop("event", "progress")
            yield {"event": event, "data": json.dumps(payload)}

            if event in ("complete", "error", "cancelled"):
                break
    finally:
        job_manager.unsubscribe(job_id, queue)
