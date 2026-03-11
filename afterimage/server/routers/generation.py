"""POST /api/v1/generate — start a generation job."""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from ..config import ServerConfig, get_config
from ..dependencies import get_job_manager, verify_api_key
from ..models import GenerationJobResponse, GenerationRequest
from ..services.job_manager import JobManager

router = APIRouter(
    prefix="/api/v1",
    tags=["generation"],
    dependencies=[Depends(verify_api_key)],
)


@router.post("/generate", response_model=GenerationJobResponse, status_code=202)
async def start_generation(
    request: GenerationRequest,
    job_manager: JobManager = Depends(get_job_manager),
    config: ServerConfig = Depends(get_config),
):
    if request.num_dialogs > config.max_dialogs_per_request:
        raise HTTPException(
            status_code=422,
            detail=f"num_dialogs exceeds server limit of {config.max_dialogs_per_request}.",
        )
    if not any([request.document_text, request.document_url, request.document_chunks]):
        raise HTTPException(
            status_code=422,
            detail="Supply one of: document_text, document_url, document_chunks.",
        )

    job = await job_manager.submit(request)
    base = f"/api/v1/jobs/{job.job_id}"
    return GenerationJobResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        links={
            "self": base,
            "status": f"{base}/status",
            "stream": f"{base}/stream",
            "result": f"{base}/result",
            "cancel": base,
        },
    )


@router.post("/generate/upload", response_model=GenerationJobResponse, status_code=202)
async def start_generation_with_upload(
    file: UploadFile = File(...),
    num_dialogs: int = Form(10),
    max_turns: int = Form(1),
    max_concurrency: int = Form(4),
    auto_generate_prompts: bool = Form(True),
    use_personas: bool = Form(True),
    model_name: str = Form("gemini-2.0-flash"),
    model_provider_name: str = Form("gemini"),
    output_format: str = Form("jsonl"),
    chunk_size: int = Form(5000),
    job_manager: JobManager = Depends(get_job_manager),
    config: ServerConfig = Depends(get_config),
):
    """Accept a plain-text file upload and start a generation job."""
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=422, detail="File must be UTF-8 encoded text.")

    request = GenerationRequest(
        document_text=text,
        num_dialogs=num_dialogs,
        max_turns=max_turns,
        max_concurrency=max_concurrency,
        auto_generate_prompts=auto_generate_prompts,
        use_personas=use_personas,
        model_name=model_name,
        model_provider_name=model_provider_name,
        output_format=output_format,
        chunk_size=chunk_size,
    )

    if request.num_dialogs > config.max_dialogs_per_request:
        raise HTTPException(
            status_code=422,
            detail=f"num_dialogs exceeds server limit of {config.max_dialogs_per_request}.",
        )

    job = await job_manager.submit(request)
    base = f"/api/v1/jobs/{job.job_id}"
    return GenerationJobResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        links={
            "self": base,
            "status": f"{base}/status",
            "stream": f"{base}/stream",
            "result": f"{base}/result",
            "cancel": base,
        },
    )
