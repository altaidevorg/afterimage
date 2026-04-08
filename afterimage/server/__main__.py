"""
Run with:
    python -m afterimage.server
    python -m afterimage.server --port 8080 --workers 2
"""

from __future__ import annotations

import argparse


def _parse_args():
    parser = argparse.ArgumentParser(description="AfterImage Server")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--no-reload", action="store_true")
    return parser.parse_args()


def main():
    import uvicorn

    from .config import get_config

    args = _parse_args()
    config = get_config()

    # CLI overrides env-based config
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.workers:
        config.workers = args.workers
    if args.results_dir:
        config.results_dir = args.results_dir

    from .app import create_app

    app = create_app(config)
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        workers=config.workers,
    )


if __name__ == "__main__":
    main()
