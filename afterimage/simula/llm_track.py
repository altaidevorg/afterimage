"""Wrap ``LLMProvider.agenerate_structured`` with optional :class:`~afterimage.monitoring.GenerationMonitor` tracking."""

from __future__ import annotations

import time
from typing import Any, TypeVar

from pydantic import BaseModel

from ..monitoring import GenerationMonitor
from ..providers.llm_providers import LLMProvider

T = TypeVar("T", bound=BaseModel)


def _merge_metadata(
    operation: str,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {"component": "opensimula", "operation": operation}
    if extra:
        meta.update(extra)
    return meta


async def agenerate_structured_tracked(
    monitor: GenerationMonitor | None,
    llm: LLMProvider,
    *,
    operation: str,
    prompt: str,
    schema: type[T],
    temperature: float = 0.7,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Call ``llm.agenerate_structured`` and emit ``track_generation`` when ``monitor`` is set."""
    start = time.perf_counter()
    meta = _merge_metadata(operation, metadata)
    try:
        resp = await llm.agenerate_structured(
            prompt=prompt,
            schema=schema,
            temperature=temperature,
            **kwargs,
        )
        if monitor is not None:
            dur = time.perf_counter() - start
            monitor.track_generation(
                dur,
                True,
                prompt_token_count=getattr(resp, "prompt_token_count", None),
                completion_token_count=getattr(resp, "completion_token_count", None),
                total_token_count=getattr(resp, "total_token_count", None),
                finish_reason=getattr(resp, "finish_reason", None),
                model_name=getattr(resp, "model_name", None),
                metadata=meta,
            )
        return resp
    except Exception as e:
        if monitor is not None:
            dur = time.perf_counter() - start
            monitor.track_generation(
                dur,
                False,
                error=str(e),
                metadata={**meta, "error_type": type(e).__name__},
            )
        raise
