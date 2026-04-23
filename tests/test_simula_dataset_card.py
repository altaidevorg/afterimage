"""Hub dataset README (auto card) for OpenSimula checkpoints."""

from __future__ import annotations

from afterimage.simula.checkpoint import (
    AFTERIMAGE_REPO_URL,
    OpenSimulaManifest,
    OpenSimulaRunConfig,
    _default_dataset_readme,
    _hub_dataset_tags,
)


def _minimal_manifest() -> OpenSimulaManifest:
    return OpenSimulaManifest(
        created_at="2026-01-01T00:00:00Z",
        instruction_y_sha256="a" * 64,
        taxonomy_bundle_sha256="b" * 64,
    )


def test_hub_dataset_tags_base_only() -> None:
    assert _hub_dataset_tags(None) == ["afterimage", "simula", "opensimula"]


def test_hub_dataset_tags_mcq_and_batch() -> None:
    run = OpenSimulaRunConfig(num_choices=4, num_samples=8)
    tags = _hub_dataset_tags(run)
    assert "mcq" in tags
    assert "multiple-choice" in tags
    assert "batch-generation" in tags


def test_hub_dataset_tags_single_qa_batch() -> None:
    run = OpenSimulaRunConfig(num_samples=3)
    tags = _hub_dataset_tags(run)
    assert "single-qa" in tags
    assert "question-answering" in tags
    assert "mcq" not in tags


def test_default_dataset_readme_links_and_frontmatter() -> None:
    txt = _default_dataset_readme(
        manifest=_minimal_manifest(),
        run=OpenSimulaRunConfig(name="unit", num_samples=2),
        repo_id="org/test-dataset",
    )
    assert "---" in txt
    assert "tags:" in txt
    assert "afterimage" in txt
    assert "opensimula" in txt
    assert AFTERIMAGE_REPO_URL in txt
    assert "openreview.net" in txt
    assert "research.google/blog" in txt
    assert "org/test-dataset" in txt
    assert "**unit**" in txt
