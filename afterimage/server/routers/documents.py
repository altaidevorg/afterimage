"""Document analysis endpoint — preview auto-generated prompts before a full job."""

from fastapi import APIRouter, Depends, HTTPException

from ..config import ServerConfig, get_config
from ..dependencies import verify_api_key
from ..models import AnalyzeDocumentRequest, AnalyzeDocumentResponse
from ..services.prompt_analyzer import PromptAnalyzer

router = APIRouter(
    prefix="/api/v1",
    tags=["documents"],
    dependencies=[Depends(verify_api_key)],
)


@router.post("/analyze-document", response_model=AnalyzeDocumentResponse)
async def analyze_document(
    body: AnalyzeDocumentRequest,
    config: ServerConfig = Depends(get_config),
):
    api_key = config.get_api_key()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="No LLM API key configured. Set AFTERIMAGE_GEMINI_API_KEY or similar.",
        )
    analyzer = PromptAnalyzer(api_key=api_key, model_name=config.default_model)
    return await analyzer.analyze(
        body.document_text, excerpt_length=body.excerpt_length
    )
