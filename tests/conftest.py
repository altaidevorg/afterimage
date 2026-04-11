"""Shared pytest fixtures and configuration."""

import os

import pytest


@pytest.fixture
def gemini_api_key():
    """Return GEMINI_API_KEY from env, or None if not set."""
    return os.environ.get("GEMINI_API_KEY")
