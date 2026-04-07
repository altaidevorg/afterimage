"""FastAPI dependency providers shared across routers."""

from __future__ import annotations

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from .config import ServerConfig, get_config  # ServerConfig used by get_config_dep

_api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def verify_api_key(
    authorization: str | None = Security(_api_key_header),
) -> None:
    config = get_config()
    if config.api_key and authorization != f"Bearer {config.api_key}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def get_job_manager(request: Request):
    return request.app.state.job_manager


def get_result_store(request: Request):
    return request.app.state.result_store


def get_config_dep(request: Request) -> ServerConfig:
    return request.app.state.config
