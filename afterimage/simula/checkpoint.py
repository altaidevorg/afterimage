"""OpenSimula checkpoint layout: versioned manifest + JSON artifacts under ``opensimula/``.

Use :class:`Checkpointer` as a context manager and call ``bundle.save(cp)``,
``spec.save(cp)``, and optionally :meth:`Checkpointer.write_run_config`, then
:meth:`Checkpointer.push_to_hub` once ``manifest.json`` exists—or call
:func:`save_checkpoint` / :func:`push_checkpoint_to_hub` for shorthand.

On-disk layout (``format_version`` ``1.0``)::

    <checkpoint_root>/
      opensimula/
        manifest.json          # producer, format, format_version, digests, file names
        taxonomy_bundle.json
        sampling_strategy.json # optional
        run_config.json        # optional JSON object (e.g. caps, model id)

``huggingface-hub`` is used for optional push/pull of the ``opensimula/`` subtree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from afterimage.simula.types import (
    SamplingStrategySpec,
    TaxonomyBundle,
    sha256_text,
    validate_factor_taxonomy,
)


OPENSIMULA_SUBDIR = "opensimula"
MANIFEST_FILENAME = "manifest.json"
TAXONOMY_BUNDLE_FILENAME = "taxonomy_bundle.json"
SAMPLING_STRATEGY_FILENAME = "sampling_strategy.json"
RUN_CONFIG_FILENAME = "run_config.json"

SUPPORTED_MANIFEST_FORMAT_VERSIONS = frozenset({"1.0"})


def _package_version(dist_name: str = "afterimage") -> str | None:
    try:
        from importlib.metadata import version

        return version(dist_name)
    except Exception:
        return None


def _sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def opensimula_dir(checkpoint_root: Path) -> Path:
    return Path(checkpoint_root) / OPENSIMULA_SUBDIR


class OpenSimulaManifest(BaseModel):
    """Versioned checkpoint manifest (portable across tools that understand ``format``)."""

    producer: Literal["afterimage"] = "afterimage"
    format: Literal["opensimula"] = "opensimula"
    format_version: str = Field(
        "1.0",
        description="Contract version for manifest fields and sibling file layout.",
    )

    created_at: str = Field(
        ...,
        description="ISO 8601 timestamp in UTC when the checkpoint was written.",
    )
    afterimage_version: str | None = Field(
        None,
        description="Installed afterimage distribution version, if available.",
    )

    instruction_y_sha256: str = Field(
        ...,
        description="SHA256 hex of UTF-8 bytes of instruction_y (cross-check with bundle).",
    )
    taxonomy_bundle_sha256: str = Field(
        ...,
        description="SHA256 hex of on-disk taxonomy_bundle.json bytes.",
    )
    sampling_strategy_sha256: str | None = Field(
        None,
        description="SHA256 hex of sampling_strategy.json if present.",
    )

    taxonomy_bundle_file: str = TAXONOMY_BUNDLE_FILENAME
    sampling_strategy_file: str | None = None
    run_config_file: str | None = None


@dataclass(frozen=True)
class SimulaCheckpoint:
    """Loaded checkpoint: manifest + parsed models + optional extras."""

    manifest: OpenSimulaManifest
    bundle: TaxonomyBundle
    sampling_strategy: SamplingStrategySpec | None
    run_config: dict[str, Any] | None
    root: Path


def _validate_bundle_trees(bundle: TaxonomyBundle) -> None:
    for t in bundle.taxonomies:
        validate_factor_taxonomy(t)


class Checkpointer:
    """Collect OpenSimula artifacts under ``<root>/opensimula/`` and write ``manifest.json`` on exit.

    Typical usage::

        with Checkpointer("./run") as cp:
            bundle.save(cp)
            spec.save(cp)
            cp.write_run_config({"model": "..."})
        url = cp.push_to_hub("org/dataset-repo")

    Call :meth:`write_taxonomy_bundle` (or ``bundle.save(cp)``) at least once before the
    context exits. Optional files are removed on enter when ``clear_stale_optional`` is
    true so omitted ``spec.save`` / ``write_run_config`` do not leave stale JSON.
    """

    def __init__(
        self,
        checkpoint_root: Path | str,
        *,
        validate_taxonomies: bool = True,
        clear_stale_optional: bool = True,
    ) -> None:
        self.root = Path(checkpoint_root)
        self.validate_taxonomies = validate_taxonomies
        self.clear_stale_optional = clear_stale_optional
        self._odir = opensimula_dir(self.root)
        self._bundle_written = False
        self._instruction_y_digest: str | None = None
        self._bundle_digest: str | None = None
        self._strat_digest: str | None = None
        self._strat_file: str | None = None
        self._run_file: str | None = None
        self.manifest: OpenSimulaManifest | None = None
        self._entered = False

    @property
    def opensimula_dir(self) -> Path:
        return self._odir

    def __enter__(self) -> Checkpointer:
        self._entered = True
        self._bundle_written = False
        self._instruction_y_digest = None
        self._bundle_digest = None
        self._strat_digest = None
        self._strat_file = None
        self._run_file = None
        self.manifest = None
        self._odir.mkdir(parents=True, exist_ok=True)
        if self.clear_stale_optional:
            stale_strat = self._odir / SAMPLING_STRATEGY_FILENAME
            stale_run = self._odir / RUN_CONFIG_FILENAME
            if stale_strat.is_file():
                stale_strat.unlink()
            if stale_run.is_file():
                stale_run.unlink()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]:
        self._entered = False
        if exc_type is None and self._bundle_written:
            self._write_manifest()
        return False

    def _require_context(self) -> None:
        if not self._entered:
            raise RuntimeError("Use Checkpointer as a context manager: with Checkpointer(path) as cp: ...")

    def write_taxonomy_bundle(self, bundle: TaxonomyBundle) -> None:
        """Write ``taxonomy_bundle.json`` and record digests for the manifest."""
        self._require_context()
        if self.validate_taxonomies:
            _validate_bundle_trees(bundle)

        bundle_path = self._odir / TAXONOMY_BUNDLE_FILENAME
        bundle_path.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")

        inst_digest = sha256_text(bundle.instruction_y)
        roundtrip = TaxonomyBundle.model_validate_json(
            bundle_path.read_text(encoding="utf-8"),
        )
        if sha256_text(roundtrip.instruction_y) != inst_digest:
            raise RuntimeError("taxonomy_bundle.json round-trip instruction_y digest mismatch")

        self._bundle_written = True
        self._instruction_y_digest = inst_digest
        self._bundle_digest = _sha256_file(bundle_path)

    def write_sampling_strategy(self, spec: SamplingStrategySpec) -> None:
        """Write ``sampling_strategy.json`` (call after :meth:`write_taxonomy_bundle`)."""
        self._require_context()
        if not self._bundle_written:
            raise RuntimeError("write_taxonomy_bundle (or bundle.save) before write_sampling_strategy")

        strat_path = self._odir / SAMPLING_STRATEGY_FILENAME
        strat_path.write_text(spec.model_dump_json(indent=2) + "\n", encoding="utf-8")
        self._strat_digest = _sha256_file(strat_path)
        self._strat_file = SAMPLING_STRATEGY_FILENAME

    def write_run_config(self, data: dict[str, Any]) -> None:
        """Write ``run_config.json`` (call after :meth:`write_taxonomy_bundle`)."""
        self._require_context()
        if not self._bundle_written:
            raise RuntimeError("write_taxonomy_bundle (or bundle.save) before write_run_config")

        run_path = self._odir / RUN_CONFIG_FILENAME
        run_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._run_file = RUN_CONFIG_FILENAME

    def finalize(self) -> OpenSimulaManifest:
        """Write ``manifest.json`` immediately (usually you rely on context exit instead)."""
        self._require_context()
        self._write_manifest()
        assert self.manifest is not None
        return self.manifest

    def _write_manifest(self) -> None:
        if not self._bundle_written or self._instruction_y_digest is None or self._bundle_digest is None:
            raise RuntimeError("Cannot finalize: taxonomy bundle was not written")

        manifest = OpenSimulaManifest(
            created_at=datetime.now(UTC)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            afterimage_version=_package_version(),
            instruction_y_sha256=self._instruction_y_digest,
            taxonomy_bundle_sha256=self._bundle_digest,
            sampling_strategy_sha256=self._strat_digest,
            sampling_strategy_file=self._strat_file,
            run_config_file=self._run_file,
        )
        (self._odir / MANIFEST_FILENAME).write_text(
            manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        self.manifest = manifest

    def push_to_hub(
        self,
        repo_id: str,
        *,
        repo_type: Literal["dataset", "model", "space"] = "dataset",
        token: str | None = None,
        commit_message: str | None = None,
        private: bool = False,
        path_in_repo: str = OPENSIMULA_SUBDIR,
    ) -> str:
        """Upload ``<root>/opensimula/`` to the Hugging Face Hub (creates the repo if missing).

        Requires ``manifest.json`` on disk—for example after the ``with`` block exits or
        after :meth:`finalize`.
        """
        from huggingface_hub import HfApi, create_repo

        manifest_path = self._odir / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"No OpenSimula manifest at {manifest_path}. "
                "Finish saving (exit `with Checkpointer(...)` or call finalize()) before push_to_hub.",
            )

        api = HfApi(token=token)
        create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True, token=token)
        api.upload_folder(
            folder_path=str(self._odir),
            path_in_repo=path_in_repo.strip("/") or OPENSIMULA_SUBDIR,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message or "Upload OpenSimula checkpoint (opensimula/)",
            token=token,
        )
        host = "https://huggingface.co"
        if repo_type == "dataset":
            return f"{host}/datasets/{repo_id}"
        if repo_type == "space":
            return f"{host}/spaces/{repo_id}"
        return f"{host}/{repo_id}"


def save_checkpoint(
    checkpoint_root: Path | str,
    *,
    bundle: TaxonomyBundle,
    sampling_strategy: SamplingStrategySpec | None = None,
    run_config: dict[str, Any] | None = None,
    validate_taxonomies: bool = True,
) -> OpenSimulaManifest:
    """Write ``opensimula/`` under ``checkpoint_root`` and return the manifest.

    Equivalent to using :class:`Checkpointer` with ``bundle.save`` / ``spec.save`` /
    :meth:`Checkpointer.write_run_config`.
    """
    cp = Checkpointer(checkpoint_root, validate_taxonomies=validate_taxonomies)
    with cp:
        bundle.save(cp)
        if sampling_strategy is not None:
            sampling_strategy.save(cp)
        if run_config is not None:
            cp.write_run_config(run_config)
    assert cp.manifest is not None
    return cp.manifest


def load_checkpoint(
    checkpoint_root: Path | str,
    *,
    verify_digests: bool = True,
    validate_taxonomies: bool = True,
) -> SimulaCheckpoint:
    """Load ``opensimula/`` from ``checkpoint_root``."""
    root = Path(checkpoint_root)
    odir = opensimula_dir(root)
    manifest_path = odir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing manifest: {manifest_path} (expected OpenSimula checkpoint layout)",
        )

    manifest = OpenSimulaManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8"),
    )
    if manifest.format_version not in SUPPORTED_MANIFEST_FORMAT_VERSIONS:
        raise ValueError(
            f"Unsupported OpenSimula manifest format_version={manifest.format_version!r}; "
            f"supported: {sorted(SUPPORTED_MANIFEST_FORMAT_VERSIONS)}",
        )
    if manifest.producer != "afterimage" or manifest.format != "opensimula":
        raise ValueError(
            f"Unexpected manifest producer/format: {manifest.producer!r} / {manifest.format!r}",
        )

    bundle_path = odir / manifest.taxonomy_bundle_file
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Missing taxonomy bundle file: {bundle_path}")
    if verify_digests:
        got = _sha256_file(bundle_path)
        if got != manifest.taxonomy_bundle_sha256:
            raise ValueError(
                f"taxonomy_bundle.json digest mismatch: expected {manifest.taxonomy_bundle_sha256}, got {got}",
            )

    bundle = TaxonomyBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))
    if verify_digests and sha256_text(bundle.instruction_y) != manifest.instruction_y_sha256:
        raise ValueError("instruction_y digest does not match manifest")

    if validate_taxonomies:
        _validate_bundle_trees(bundle)

    strategy: SamplingStrategySpec | None = None
    strat_name = manifest.sampling_strategy_file or SAMPLING_STRATEGY_FILENAME
    strat_candidate = odir / strat_name
    if strat_candidate.is_file():
        if verify_digests and manifest.sampling_strategy_sha256:
            sg = _sha256_file(strat_candidate)
            if sg != manifest.sampling_strategy_sha256:
                raise ValueError(
                    f"sampling strategy digest mismatch: expected {manifest.sampling_strategy_sha256}, got {sg}",
                )
        strategy = SamplingStrategySpec.model_validate_json(
            strat_candidate.read_text(encoding="utf-8"),
        )

    run_cfg: dict[str, Any] | None = None
    if manifest.run_config_file:
        rp = odir / manifest.run_config_file
        if rp.is_file():
            run_cfg = json.loads(rp.read_text(encoding="utf-8"))

    return SimulaCheckpoint(
        manifest=manifest,
        bundle=bundle,
        sampling_strategy=strategy,
        run_config=run_cfg,
        root=root,
    )


def push_checkpoint_to_hub(
    checkpoint_root: Path | str,
    repo_id: str,
    *,
    repo_type: Literal["dataset", "model", "space"] = "dataset",
    token: str | None = None,
    commit_message: str | None = None,
    private: bool = False,
    path_in_repo: str = OPENSIMULA_SUBDIR,
) -> str:
    """Upload local ``opensimula/`` to the Hub under ``path_in_repo`` (default ``opensimula``).

    Same as ``Checkpointer(checkpoint_root).push_to_hub(...)``. Returns the canonical repo URL.
    """
    return Checkpointer(checkpoint_root).push_to_hub(
        repo_id,
        repo_type=repo_type,
        token=token,
        commit_message=commit_message,
        private=private,
        path_in_repo=path_in_repo,
    )


def pull_checkpoint_from_hub(
    repo_id: str,
    checkpoint_root: Path | str,
    *,
    repo_type: Literal["dataset", "model", "space"] = "dataset",
    revision: str | None = None,
    token: str | None = None,
    path_in_repo: str = OPENSIMULA_SUBDIR,
) -> Path:
    """Download ``path_in_repo/**`` from the Hub into ``checkpoint_root`` (merging with ``snapshot_download``).

    Returns ``opensimula_dir(checkpoint_root)``.
    """
    from huggingface_hub import snapshot_download

    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    prefix = path_in_repo.strip("/")
    allow = [f"{prefix}/**"] if prefix else None
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(root),
        allow_patterns=allow,
        token=token,
    )
    odir = opensimula_dir(root)
    if not (odir / MANIFEST_FILENAME).is_file():
        raise FileNotFoundError(
            f"After Hub download, expected manifest at {odir / MANIFEST_FILENAME}. "
            f"Check repo layout and path_in_repo (got {path_in_repo!r}).",
        )
    return odir
