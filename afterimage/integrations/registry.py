"""Exporter registry with decorator-based registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseExporter

EXPORTERS: dict[str, type[BaseExporter]] = {}


def register(name: str):
    """Class decorator: ``@register("sharegpt")``."""

    def wrapper(cls):
        EXPORTERS[name] = cls
        return cls

    return wrapper


def get_exporter(name: str) -> BaseExporter:
    """Instantiate the exporter registered under *name*."""
    if name not in EXPORTERS:
        available = ", ".join(sorted(EXPORTERS.keys()))
        raise ValueError(f"Unknown format '{name}'. Available: {available}")
    return EXPORTERS[name]()


def list_formats() -> list[dict]:
    """Return metadata about all registered exporters."""
    return [
        {
            "name": name,
            "description": cls.description,
            "multi_turn": cls.supports_multi_turn,
            "system_prompt": cls.supports_system_prompt,
            "tool_calls": cls.supports_tool_calls,
            "used_by": cls.used_by,
        }
        for name, cls in sorted(EXPORTERS.items())
    ]
