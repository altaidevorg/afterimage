"""OpenSimula checkpoint manifest + save/load + optional Hub (skipped without token)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from afterimage.simula.checkpoint import (
    Checkpointer,
    OpenSimulaManifest,
    load_checkpoint,
    pull_checkpoint_from_hub,
    save_checkpoint,
)
from afterimage.simula.types import (
    FactorTaxonomy,
    SamplingStrategySpec,
    SimulaFactor,
    StrategyMixRule,
    TaxonomyBundle,
    TaxonomyNode,
)


def _minimal_bundle() -> TaxonomyBundle:
    f = SimulaFactor(name="size", description="object size")
    root = TaxonomyNode(
        factor_id=f.id,
        parent_id=None,
        depth=0,
        label="root",
        description="root",
    )
    tree = FactorTaxonomy(
        factor_id=f.id,
        root_id=root.id,
        nodes={root.id: root},
    )
    return TaxonomyBundle(
        instruction_y="Answer briefly.",
        target_depth_D=2,
        proposal_N=2,
        factors=[f],
        taxonomies=[tree],
    )


def test_checkpointer_save_methods_roundtrip(tmp_path: Path) -> None:
    bundle = _minimal_bundle()
    strat = SamplingStrategySpec(
        strategies=[StrategyMixRule(name="s1", weight=1.0, factor_ids=[bundle.factors[0].id])],
    )
    with Checkpointer(tmp_path) as cp:
        bundle.save(cp)
        strat.save(cp)
        cp.write_run_config({"k": 1})
    assert cp.manifest is not None
    ckpt = load_checkpoint(tmp_path)
    assert ckpt.bundle.instruction_y == bundle.instruction_y
    assert ckpt.sampling_strategy is not None
    assert ckpt.run_config == {"k": 1}


def test_taxonomy_bundle_save_type_error() -> None:
    bundle = _minimal_bundle()
    with pytest.raises(TypeError, match="Checkpointer"):
        bundle.save(object())  # type: ignore[arg-type]


def test_checkpointer_requires_context_manager(tmp_path: Path) -> None:
    cp = Checkpointer(tmp_path)
    with pytest.raises(RuntimeError, match="context manager"):
        cp.write_taxonomy_bundle(_minimal_bundle())


def test_push_to_hub_requires_manifest(tmp_path: Path) -> None:
    cp = Checkpointer(tmp_path)
    with pytest.raises(FileNotFoundError, match="No OpenSimula manifest"):
        cp.push_to_hub("dummy/repo")


def test_save_load_roundtrip(tmp_path: Path) -> None:
    bundle = _minimal_bundle()
    strat = SamplingStrategySpec(
        strategies=[StrategyMixRule(name="s1", weight=1.0, factor_ids=[bundle.factors[0].id])],
    )
    run_cfg = {"model": "gemini-2.5-flash", "max_factors": 8}

    manifest = save_checkpoint(
        tmp_path,
        bundle=bundle,
        sampling_strategy=strat,
        run_config=run_cfg,
    )
    assert isinstance(manifest, OpenSimulaManifest)
    assert manifest.producer == "afterimage"
    assert manifest.format == "opensimula"
    assert manifest.format_version == "1.0"
    assert manifest.sampling_strategy_file == "sampling_strategy.json"
    assert manifest.run_config_file == "run_config.json"

    ckpt = load_checkpoint(tmp_path)
    assert ckpt.bundle.instruction_y == bundle.instruction_y
    assert ckpt.sampling_strategy is not None
    assert ckpt.sampling_strategy.strategies[0].name == "s1"
    assert ckpt.run_config == run_cfg


def test_save_without_optional_clears_stale_files(tmp_path: Path) -> None:
    bundle = _minimal_bundle()
    save_checkpoint(
        tmp_path,
        bundle=bundle,
        sampling_strategy=SamplingStrategySpec(
            strategies=[
                StrategyMixRule(name="s1", weight=1.0, factor_ids=[bundle.factors[0].id]),
            ],
        ),
        run_config={"x": 1},
    )
    save_checkpoint(tmp_path, bundle=bundle)
    ckpt = load_checkpoint(tmp_path)
    assert ckpt.sampling_strategy is None
    assert ckpt.run_config is None


def test_unsupported_manifest_version(tmp_path: Path) -> None:
    odir = tmp_path / "opensimula"
    odir.mkdir(parents=True)
    bad = {
        "producer": "afterimage",
        "format": "opensimula",
        "format_version": "99.0",
        "created_at": "2026-01-01T00:00:00Z",
        "instruction_y_sha256": "a" * 64,
        "taxonomy_bundle_sha256": "b" * 64,
    }
    (odir / "manifest.json").write_text(__import__("json").dumps(bad), encoding="utf-8")
    (odir / "taxonomy_bundle.json").write_text(
        _minimal_bundle().model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported OpenSimula manifest"):
        load_checkpoint(tmp_path, verify_digests=False)


@pytest.mark.integration
def test_hub_push_pull_roundtrip(tmp_path: Path) -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        pytest.skip("Set HF_TOKEN to run Hub push/pull integration test")

    repo_id = os.environ.get("AFTERIMAGE_SIMULA_HF_TEST_REPO")
    if not repo_id:
        pytest.skip("Set AFTERIMAGE_SIMULA_HF_TEST_REPO to a writable dataset repo id")

    bundle = _minimal_bundle()
    local_a = tmp_path / "a"
    local_b = tmp_path / "b"
    save_checkpoint(local_a, bundle=bundle)
    url = Checkpointer(local_a).push_to_hub(
        repo_id,
        token=token,
        commit_message="test simula checkpoint",
    )
    assert "huggingface.co" in url
    pull_checkpoint_from_hub(repo_id, local_b, token=token)
    ckpt = load_checkpoint(local_b)
    assert ckpt.bundle.instruction_y == bundle.instruction_y
