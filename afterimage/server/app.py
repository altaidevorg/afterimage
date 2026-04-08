"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import afterimage

from .config import ServerConfig
from .routers import documents_router, generation_router, health_router, jobs_router
from .services.job_manager import JobManager
from .storage.job_store import JobStore
from .storage.result_store import ResultStore


def create_app(config: ServerConfig | None = None) -> FastAPI:
    if config is None:
        from .config import get_config
        config = get_config()

    job_store = JobStore(db_path=config.job_db_path)
    result_store = ResultStore(results_dir=config.results_dir)
    job_manager = JobManager(
        job_store=job_store,
        result_store=result_store,
        max_concurrent_jobs=config.max_concurrent_jobs,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.config = config
        app.state.job_store = job_store
        app.state.result_store = result_store
        app.state.job_manager = job_manager
        yield

    app = FastAPI(
        title="AfterImage Server",
        description=(
            "Production-grade REST API for AfterImage synthetic dataset generation. "
            "Supports on-demand generation, SSE progress streaming, and job management."
        ),
        version=afterimage.__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(generation_router)
    app.include_router(jobs_router)
    app.include_router(documents_router)

    return app
