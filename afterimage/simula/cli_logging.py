"""Logging helpers for CLI / examples (quiet third-party noise)."""

from __future__ import annotations

import logging

_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "httpcore.connection",
    "httpcore.http11",
    "google_genai",
    "google_genai.models",
    "google.auth",
    "google.auth.transport",
)


def silence_noisy_third_party_loggers(level: int = logging.WARNING) -> None:
    """Turn down chatty HTTP and google-genai log lines during OpenSimula runs."""
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(level)


def configure_example_console(
    *,
    simula_level: int = logging.WARNING,
    root_level: int = logging.WARNING,
) -> None:
    """One-line setup for example scripts: quiet root, optional simula detail, no httpx spam.

    Use ``simula_level=logging.INFO`` when you want ``afterimage.simula`` DEBUG/INFO
    without tqdm (e.g. ``show_progress=False`` on ``build_taxonomy``).
    """
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=root_level,
            format="%(levelname)s %(name)s: %(message)s",
        )
    else:
        root.setLevel(root_level)
        for handler in root.handlers:
            handler.setLevel(root_level)
    logging.getLogger("afterimage.simula").setLevel(simula_level)
    silence_noisy_third_party_loggers()
