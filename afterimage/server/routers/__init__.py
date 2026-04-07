from .documents import router as documents_router
from .generation import router as generation_router
from .health import router as health_router
from .jobs import router as jobs_router

__all__ = ["documents_router", "generation_router", "health_router", "jobs_router"]
