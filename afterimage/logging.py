"""Central logging utilities for AfterImage."""

from __future__ import annotations

import logging

_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "httpcore.connection",
    "httpcore.http11",
    "google_genai",
    "google_genai._api_client",
    "google_genai.client",
    "google_genai.models",
    "google.auth",
    "google.auth.transport",
)


def silence_noisy_third_party_loggers(level: int = logging.ERROR) -> None:
    """Turn down chatty HTTP and google-genai log lines (warnings/info)."""
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(level)
