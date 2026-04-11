"""Tests for YAML config loading and validation."""

import os
import textwrap
from pathlib import Path

import pytest

from afterimage.config import AfterImageConfig, load_config, resolve_api_key


@pytest.fixture
def tmp_config(tmp_path):
    """Helper to write a YAML config and return its path."""

    def _write(content: str, name: str = "config.yaml") -> Path:
        p = tmp_path / name
        p.write_text(textwrap.dedent(content), encoding="utf-8")
        return p

    return _write


class TestLoadConfig:
    def test_minimal_config(self, tmp_config):
        cfg_path = tmp_config("""
            respondent:
              system_prompt: "You are helpful."
        """)
        cfg = load_config(cfg_path)
        assert cfg.respondent.system_prompt == "You are helpful."
        assert cfg.generation.num_dialogs == 10
        assert cfg.model.provider == "gemini"

    def test_full_config(self, tmp_config):
        cfg_path = tmp_config("""
            generation:
              num_dialogs: 50
              max_turns: 3
              max_concurrency: 5
            model:
              provider: openai
              model_name: gpt-4o
              api_key_env: OPENAI_API_KEY
            respondent:
              system_prompt: "You are an expert."
            documents:
              provider: directory
              path: ./docs
            context:
              enabled: true
              num_random_contexts: 3
              n_instructions: 5
            personas:
              enabled: true
            quality:
              auto_improve: true
            output:
              path: ./out/dataset.jsonl
              storage: jsonl
        """)
        cfg = load_config(cfg_path)
        assert cfg.generation.num_dialogs == 50
        assert cfg.generation.max_turns == 3
        assert cfg.model.provider == "openai"
        assert cfg.personas.enabled is True
        assert cfg.quality.auto_improve is True

    def test_system_prompt_file(self, tmp_config, tmp_path):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("You are a pirate.", encoding="utf-8")
        cfg_path = tmp_config(f"""
            respondent:
              system_prompt_file: prompt.txt
        """)
        cfg = load_config(cfg_path)
        assert cfg.respondent.system_prompt == "You are a pirate."

    def test_system_prompt_file_not_found(self, tmp_config):
        cfg_path = tmp_config("""
            respondent:
              system_prompt_file: nonexistent.txt
        """)
        with pytest.raises(FileNotFoundError, match="System prompt file not found"):
            load_config(cfg_path)

    def test_missing_respondent(self, tmp_config):
        cfg_path = tmp_config("""
            model:
              provider: gemini
        """)
        with pytest.raises(Exception):
            load_config(cfg_path)

    def test_both_prompts_fails(self, tmp_config, tmp_path):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("test", encoding="utf-8")
        cfg_path = tmp_config("""
            respondent:
              system_prompt: "inline"
              system_prompt_file: prompt.txt
        """)
        with pytest.raises(ValueError, match="only one of"):
            load_config(cfg_path)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

    def test_invalid_yaml(self, tmp_config):
        cfg_path = tmp_config("not: [valid: yaml: {{")
        with pytest.raises(Exception):
            load_config(cfg_path)

    def test_documents_path_resolved_relative_to_config(self, tmp_config, tmp_path):
        docs_dir = tmp_path / "my_docs"
        docs_dir.mkdir()
        cfg_path = tmp_config("""
            respondent:
              system_prompt: "test"
            documents:
              provider: directory
              path: ./my_docs
        """)
        cfg = load_config(cfg_path)
        assert cfg.documents.path == str(docs_dir.resolve())

    def test_local_provider_config(self, tmp_config):
        cfg_path = tmp_config("""
            model:
              provider: local
              base_url: http://localhost:8000/v1
              model_name: my-model
            respondent:
              system_prompt: "test"
        """)
        cfg = load_config(cfg_path)
        assert cfg.model.provider == "local"
        assert cfg.model.base_url == "http://localhost:8000/v1"


class TestGenerationStopping:
    def test_budget_only_requires_null_num_dialogs_or_extra_rules(self, tmp_config):
        cfg_path = tmp_config("""
            generation:
              num_dialogs: null
              stopping:
                - type: budget
                  max_total_tokens: 1000
            respondent:
              system_prompt: "You are helpful."
        """)
        cfg = load_config(cfg_path)
        assert cfg.generation.num_dialogs is None
        assert len(cfg.generation.stopping) == 1

    def test_budget_requires_a_limit(self, tmp_config):
        cfg_path = tmp_config("""
            generation:
              num_dialogs: null
              stopping:
                - type: budget
            respondent:
              system_prompt: "You are helpful."
        """)
        with pytest.raises(Exception, match="at least one"):
            load_config(cfg_path)

    def test_context_coverage_requires_documents(self, tmp_config):
        cfg_path = tmp_config("""
            generation:
              num_dialogs: null
              stopping:
                - type: context_coverage
                  target_visits: 1
            respondent:
              system_prompt: "You are helpful."
        """)
        with pytest.raises(Exception, match="documents"):
            load_config(cfg_path)

    def test_personas_require_documents(self, tmp_config):
        cfg_path = tmp_config("""
            respondent:
              system_prompt: "You are helpful."
            personas:
              enabled: true
        """)
        with pytest.raises(Exception, match="personas.enabled requires"):
            load_config(cfg_path)

    def test_documents_require_context_enabled(self, tmp_config, tmp_path):
        (tmp_path / "docs").mkdir()
        cfg_path = tmp_config("""
            respondent:
              system_prompt: "x"
            documents:
              provider: directory
              path: ./docs
            context:
              enabled: false
        """)
        with pytest.raises(Exception, match="context.enabled"):
            load_config(cfg_path)


class TestBuildConversationRun:
    def test_simple_instruction_and_stopping(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        cfg = AfterImageConfig(respondent={"system_prompt": "You are helpful."})
        from afterimage.callbacks import SimpleInstructionGeneratorCallback
        from afterimage.config_to_generator import build_conversation_run

        run = build_conversation_run(cfg)
        assert isinstance(
            run.generator.instruction_generator_callback,
            SimpleInstructionGeneratorCallback,
        )
        assert run.num_requested == 10
        assert len(run.stopping_criteria) >= 1

    def test_build_stopping_nested_all(self, tmp_config, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        cfg_path = tmp_config("""
            generation:
              num_dialogs: 5
              stopping:
                - type: all
                  conditions:
                    - type: fixed
                      n: 3
                    - type: budget
                      max_total_tokens: 999999999
            respondent:
              system_prompt: "hi"
        """)
        cfg = load_config(cfg_path)
        from afterimage.config_to_generator import build_stopping_criteria

        stopping, num_req = build_stopping_criteria(cfg, None)
        assert num_req == 5


class TestResolveApiKey:
    def test_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY", "sk-123")
        cfg = AfterImageConfig(
            respondent={"system_prompt": "test"},
            model={"provider": "gemini", "api_key_env": "TEST_KEY"},
        )
        assert resolve_api_key(cfg) == "sk-123"

    def test_missing_env_raises(self, monkeypatch):
        monkeypatch.delenv("MISSING_KEY", raising=False)
        cfg = AfterImageConfig(
            respondent={"system_prompt": "test"},
            model={"provider": "gemini", "api_key_env": "MISSING_KEY"},
        )
        with pytest.raises(ValueError, match="export MISSING_KEY"):
            resolve_api_key(cfg)

    def test_local_returns_none_without_key(self):
        cfg = AfterImageConfig(
            respondent={"system_prompt": "test"},
            model={"provider": "local", "base_url": "http://localhost:8000/v1"},
        )
        assert resolve_api_key(cfg) is None

    def test_default_env_var_inferred(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
        cfg = AfterImageConfig(
            respondent={"system_prompt": "test"},
            model={"provider": "gemini"},
        )
        assert resolve_api_key(cfg) == "gem-key"
