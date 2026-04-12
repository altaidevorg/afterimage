"""Shared pytest fixtures and configuration."""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _afterimage_jsonl_under_pytest_tmp(tmp_path_factory):
    """Keep default JSONLStorage paths out of the repo root (see AFTERIMAGE_JSONL_DIR)."""
    d = tmp_path_factory.mktemp("afterimage_jsonl")
    previous = os.environ.get("AFTERIMAGE_JSONL_DIR")
    os.environ["AFTERIMAGE_JSONL_DIR"] = str(d)
    yield
    if previous is None:
        os.environ.pop("AFTERIMAGE_JSONL_DIR", None)
    else:
        os.environ["AFTERIMAGE_JSONL_DIR"] = previous


@pytest.fixture
def gemini_api_key():
    """Return GEMINI_API_KEY from env, or None if not set."""
    return os.environ.get("GEMINI_API_KEY")
