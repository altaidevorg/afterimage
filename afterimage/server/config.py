from __future__ import annotations

import os

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerConfig(BaseSettings):
    """Server configuration loaded from environment variables (prefix: AFTERIMAGE_).

    API keys are also read from the bare names (GEMINI_API_KEY, DEEPSEEK_API_KEY,
    OPENAI_API_KEY) as a fallback, so the existing .env file works without changes.
    """

    model_config = SettingsConfigDict(env_prefix="AFTERIMAGE_", env_file=".env", extra="ignore")

    # API keys — populated by the prefixed vars or bare fallbacks (see validator below)
    gemini_api_key: str | None = None
    deepseek_api_key: str | None = None
    openai_api_key: str | None = None

    @model_validator(mode="after")
    def _fill_bare_key_fallbacks(self) -> "ServerConfig":
        """If a prefixed key is absent, fall back to the bare env/dotenv name.

        pydantic-settings loads .env into its own namespace but doesn't inject bare
        keys into os.environ, so we read the .env file directly for the fallback.
        """
        bare: dict[str, str] = {}
        env_file = os.path.join(os.getcwd(), ".env")
        if os.path.isfile(env_file):
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        bare[k.strip()] = v.strip().strip('"').strip("'")

        def _get(env_key: str) -> str | None:
            return os.environ.get(env_key) or bare.get(env_key)

        if not self.gemini_api_key:
            self.gemini_api_key = _get("GEMINI_API_KEY")
        if not self.deepseek_api_key:
            self.deepseek_api_key = _get("DEEPSEEK_API_KEY")
        if not self.openai_api_key:
            self.openai_api_key = _get("OPENAI_API_KEY")
        return self

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # Generation defaults
    default_model: str = "gemini-2.0-flash"
    default_provider: str = "gemini"
    max_concurrent_jobs: int = 3
    max_dialogs_per_request: int = 1000

    # Storage
    results_dir: str = "./results"
    job_db_path: str = "jobs.db"

    # Security
    api_key: str | None = None
    cors_origins: list[str] = ["*"]

    def get_api_key(self, provider: str | None = None) -> str | None:
        """Return the API key for the given provider (or best available)."""
        provider = provider or self.default_provider
        if provider == "gemini":
            return self.gemini_api_key
        if provider == "deepseek":
            return self.deepseek_api_key
        if provider == "openai":
            return self.openai_api_key
        return self.gemini_api_key or self.openai_api_key or self.deepseek_api_key


_config: ServerConfig | None = None


def get_config() -> ServerConfig:
    global _config
    if _config is None:
        _config = ServerConfig()
    return _config
