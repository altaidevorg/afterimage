"""AfterImage Server — FastAPI-based dataset generation API."""

from .app import create_app
from .config import ServerConfig, get_config

__all__ = ["create_app", "get_config", "ServerConfig"]


def main() -> None:
    """Entry point for the ``afterimage-server`` CLI command."""
    import uvicorn

    config = get_config()
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, workers=config.workers)
