from fastapi import APIRouter, Depends

from ..dependencies import get_job_manager
from ..models import HealthResponse
from ..services.job_manager import JobManager

import afterimage

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(job_manager: JobManager = Depends(get_job_manager)):
    active = await job_manager.active_job_count()
    return HealthResponse(status="ok", version=afterimage.__version__, active_jobs=active)


@router.get("/ready")
async def ready():
    return {"ready": True}
