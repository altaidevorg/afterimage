"""Job management endpoints: status, list, cancel, result download, SSE stream."""

from __future__ import annotations


from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from ..dependencies import get_job_manager, get_result_store, verify_api_key
from ..models import (
    JobListResponse,
    JobStatusResponse,
    JobSummary,
)
from ..services.job_manager import JobManager
from ..storage.result_store import ResultStore
from ..ws.progress import sse_progress_generator

router = APIRouter(
    prefix="/api/v1/jobs",
    tags=["jobs"],
    dependencies=[Depends(verify_api_key)],
)


# ---------------------------------------------------------------------------
# GET /api/v1/jobs — list
# ---------------------------------------------------------------------------


@router.get("", response_model=JobListResponse)
async def list_jobs(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    job_manager: JobManager = Depends(get_job_manager),
):
    records, total = await job_manager.list_jobs(page=page, per_page=per_page)
    summaries = [
        JobSummary(
            job_id=r.job_id,
            status=r.status,
            num_dialogs=r.request.num_dialogs,
            model_name=r.request.model_name,
            model_provider_name=r.request.model_provider_name,
            created_at=r.created_at,
            updated_at=r.updated_at,
        )
        for r in records
    ]
    return JobListResponse(jobs=summaries, total=total, page=page, per_page=per_page)


# ---------------------------------------------------------------------------
# GET /api/v1/jobs/{job_id} — status
# ---------------------------------------------------------------------------


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
):
    record = await job_manager.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobStatusResponse(
        job_id=record.job_id,
        status=record.status,
        progress=record.progress,
        result=record.result,
        error=record.error,
        created_at=record.created_at,
        updated_at=record.updated_at,
        model_name=record.request.model_name,
        model_provider_name=record.request.model_provider_name,
        num_dialogs=record.request.num_dialogs,
        output_format=record.request.output_format,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/jobs/{job_id}/stream — SSE
# ---------------------------------------------------------------------------


@router.get("/{job_id}/stream")
async def stream_job(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
):
    record = await job_manager.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return EventSourceResponse(
        sse_progress_generator(job_manager, job_id),
        media_type="text/event-stream",
    )


# ---------------------------------------------------------------------------
# GET /api/v1/jobs/{job_id}/result — download
# ---------------------------------------------------------------------------


@router.get("/{job_id}/result")
async def download_result(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
    result_store: ResultStore = Depends(get_result_store),
):
    record = await job_manager.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if record.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job is {record.status}. Results only available for completed jobs.",
        )

    fmt = record.request.output_format
    path = result_store.get_result_path(job_id, fmt)

    # Fallback: try the other format
    if path is None:
        other = "json" if fmt == "jsonl" else "jsonl"
        path = result_store.get_result_path(job_id, other)
        if path is not None:
            fmt = other

    if path is None:
        raise HTTPException(status_code=404, detail="Result file not found on disk.")

    media_type = "application/json" if fmt == "json" else "application/x-ndjson"
    return FileResponse(
        path=str(path),
        media_type=media_type,
        filename=f"dataset_{job_id}.{fmt}",
    )


# ---------------------------------------------------------------------------
# DELETE /api/v1/jobs/{job_id} — cancel
# ---------------------------------------------------------------------------


@router.delete("/{job_id}", status_code=204)
async def cancel_job(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
):
    cancelled = await job_manager.cancel(job_id)
    if not cancelled:
        record = await job_manager.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        raise HTTPException(
            status_code=409,
            detail=f"Cannot cancel job in status '{record.status}'.",
        )
