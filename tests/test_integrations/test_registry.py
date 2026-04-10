"""Tests for exporter registry."""

import pytest

from afterimage.integrations import EXPORTERS, get_exporter, list_formats


EXPECTED_FORMATS = ["alpaca", "dpo", "llama_factory", "messages", "oumi", "openai", "raw", "sharegpt"]


class TestRegistry:
    def test_all_registered(self):
        for name in EXPECTED_FORMATS:
            assert name in EXPORTERS, f"'{name}' not registered"

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown format"):
            get_exporter("nonexistent_format")

    def test_unknown_format_lists_available(self):
        with pytest.raises(ValueError, match="alpaca"):
            get_exporter("wrong")

    def test_get_exporter_returns_instance(self):
        e = get_exporter("sharegpt")
        assert e.format_name == "sharegpt"

    def test_list_formats_returns_all(self):
        fmts = list_formats()
        names = [f["name"] for f in fmts]
        for expected in EXPECTED_FORMATS:
            assert expected in names

    def test_list_formats_metadata(self):
        fmts = list_formats()
        for f in fmts:
            assert "name" in f
            assert "description" in f
            assert "multi_turn" in f
            assert "system_prompt" in f
            assert "used_by" in f
