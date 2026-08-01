"""Logging helpers for CLI / examples (quiet third-party noise)."""

from __future__ import annotations

import logging

from ..logging import silence_noisy_third_party_loggers


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
