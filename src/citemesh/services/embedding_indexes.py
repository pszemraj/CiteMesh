"""Distribute portable corpus configurations through Hugging Face Storage Buckets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from citemesh.data.embedding_cache.portable import PortableManifest


class IndexSelectionError(ValueError):
    """Indicate a missing or unknown user-selected publication configuration."""


@dataclass(frozen=True)
class IndexCatalog:
    """Map independently published configuration names to their manifests."""

    configs: dict[str, str]
    default_config: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> IndexCatalog:
        """Read the supported catalog and reject ambiguous configurations."""
        if value.get("format_version") != 1 or not isinstance(
            value.get("configs"), dict
        ):
            raise ValueError("Unsupported or malformed index catalog")
        configs = value["configs"]
        for name, manifest in configs.items():
            if (
                not isinstance(name, str)
                or not name
                or name in {".", ".."}
                or "/" in name
                or "\\" in name
            ):
                raise ValueError("Invalid index configuration name")
            parts = PurePosixPath(manifest).parts if isinstance(manifest, str) else ()
            if (
                len(parts) != 4
                or parts[:2] != ("variants", name)
                or parts[2] in {".", ".."}
                or parts[3] != "index.json"
                or "\\" in manifest
                or str(PurePosixPath(manifest)) != manifest
            ):
                raise ValueError(f"Invalid manifest path for configuration {name!r}")
        default = value.get("default_config")
        if default is not None and default not in configs:
            raise ValueError("The catalog default is not a published configuration")
        return cls(dict(configs), default)

    def select(self, config_name: str | None) -> str:
        """Select an explicit configuration or the declared default."""
        selected = config_name or self.default_config
        if selected is None:
            raise IndexSelectionError(
                "Specify --config; the catalog has no default configuration"
            )
        if selected not in self.configs:
            raise IndexSelectionError(f"Unknown index configuration {selected!r}")
        return selected

    def to_dict(self) -> dict[str, Any]:
        """Return the catalog's portable JSON representation."""
        return {
            "format_version": 1,
            "default_config": self.default_config,
            "configs": self.configs,
        }


@dataclass(frozen=True)
class ResolvedPublication:
    """Describe one completed snapshot in a bucket."""

    bucket_id: str
    config_name: str
    snapshot_id: str
    manifest_path: Path
    manifest: PortableManifest
    manifest_bucket_path: str
    cache_dir: Path | None = None

    @property
    def origin(self) -> dict[str, str]:
        """Return the publication identity recorded by native import."""
        return {
            "bucket_id": self.bucket_id,
            "config_name": self.config_name,
            "snapshot_id": self.snapshot_id,
        }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _manifest_bucket_path(config_name: str, snapshot_id: str) -> str:
    path = f"variants/{config_name}/{snapshot_id}/index.json"
    IndexCatalog.from_dict({"format_version": 1, "configs": {config_name: path}})
    return path


def _cache_root(bucket_id: str, cache_dir: Path | None) -> Path:
    if cache_dir is None:
        from huggingface_hub.constants import HF_HUB_CACHE

        cache_dir = Path(HF_HUB_CACHE)
    return Path(cache_dir) / "citemesh-indexes" / bucket_id


def _download_files(bucket_id: str, files: list[tuple[str, Path]]) -> None:
    """Keep incomplete transfers separate from reusable completed files."""
    from huggingface_hub import HfApi

    if not files:
        return
    partials = [
        (remote, local.with_name(local.name + ".incomplete")) for remote, local in files
    ]
    for _, path in partials:
        path.parent.mkdir(parents=True, exist_ok=True)
    HfApi().download_bucket_files(
        bucket_id, files=partials, raise_on_missing_files=True
    )
    for (_, partial), (_, target) in zip(partials, files):
        partial.replace(target)


def _catalog(bucket_id: str) -> IndexCatalog:
    # The catalog is mutable; fetch it afresh rather than reusing its disk copy.
    with TemporaryDirectory(prefix="citemesh-catalog-") as directory:
        path = Path(directory) / "catalog.json"
        _download_files(bucket_id, [("catalog.json", path)])
        return IndexCatalog.from_dict(_load_json(path))


def resolve_publication(
    bucket_id: str,
    config_name: str | None = None,
    snapshot_id: str | None = None,
    cache_dir: Path | None = None,
) -> ResolvedPublication:
    """Select the latest snapshot, or reuse an explicitly named cached snapshot."""
    from citemesh.data.embedding_cache.portable import load_manifest

    if snapshot_id is not None:
        if config_name is None:
            raise IndexSelectionError("Specify --config with --snapshot")
        remote_path = _manifest_bucket_path(config_name, snapshot_id)
        selected = config_name
    else:
        catalog = _catalog(bucket_id)
        selected = catalog.select(config_name)
        remote_path = catalog.configs[selected]
        snapshot_id = PurePosixPath(remote_path).parts[2]
    manifest_path = _cache_root(bucket_id, cache_dir) / remote_path
    if not manifest_path.is_file():
        _download_files(bucket_id, [(remote_path, manifest_path)])
    manifest = load_manifest(manifest_path)
    if manifest.config_name != selected or manifest.snapshot_id != snapshot_id:
        raise ValueError("Selected manifest disagrees with its bucket snapshot path")
    return ResolvedPublication(
        bucket_id,
        selected,
        snapshot_id,
        manifest_path,
        manifest,
        remote_path,
        cache_dir,
    )


def download_publication(publication: ResolvedPublication) -> Path:
    """Download only missing declared artifacts from the selected snapshot."""
    directory = PurePosixPath(publication.manifest_bucket_path).parent
    root = publication.manifest_path.parent
    files = [
        (str(directory / name), root / name)
        for name in publication.manifest.artifact_paths
        if not (root / name).is_file()
    ]
    _download_files(publication.bucket_id, files)
    return root


def preflight_publication_destination(
    publication: ResolvedPublication, cache_dir: Path | None = None
) -> str:
    """Refuse incompatible native destinations before payload transfers."""
    from citemesh.data.embedding_cache.lifecycle import preflight_import_destination
    from citemesh.data.embedding_cache.portable import manifest_import_identity

    manifest = publication.manifest
    return preflight_import_destination(
        cache_dir=cache_dir,
        namespace_serialization=manifest.namespace_serialization,
        contract=manifest.contract,
        source=manifest.source,
        identity=manifest_import_identity(manifest),
        namespace_selectors=manifest.namespace_selectors,
        origin=publication.origin,
    )


def _readme(catalog: IndexCatalog, manifests: dict[str, PortableManifest]) -> str:
    lines = [
        "# CiteMesh corpus indexes",
        "",
        "Portable retrieval-document embeddings with normalized paper metadata. "
        "No model weights are included. Query and graph encoding still use the selected model.",
        "",
    ]
    for name, manifest in manifests.items():
        contract, source = manifest.contract, manifest.source
        lines.extend(
            [
                f"## {name}",
                "",
                f"{manifest.row_count:,} active papers; {contract.dimension} dimensions; {contract.stored_dtype} storage.",
                f"Snapshot: `{manifest.snapshot_id}`.",
                f"Source: [{source.repo_id}](https://huggingface.co/datasets/{source.repo_id}), "
                f"revision `{source.revision}`, split `{source.split}`. See the source dataset for attribution and licensing.",
                f"Model: [{contract.model_id}](https://huggingface.co/{contract.model_id}), revision `{contract.model_revision}`.",
                f"Document formatter: `{contract.formatter_identity}`; normalization: `{contract.normalization}`; "
                f"quantization: `{contract.quantization}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Install a configuration",
            "",
            "```bash",
            "citemesh cache import --bucket <this-bucket> --config <configuration>",
            "citemesh cache scan",
            "```",
            "",
            f"Default configuration: `{catalog.default_config}`. "
            "Each configuration retains its current snapshot; older snapshot directories are removed after publication.",
            "",
            'Use the installed namespace with `citemesh search --mode local --cache-namespace <namespace> "your query"`. '
            "Run `citemesh cache refresh --namespace <namespace>` to reconcile newer source data locally.",
            "",
        ]
    )
    return "\n".join(lines)


def _prune_snapshots(
    api: Any, bucket_id: str, config_name: str, snapshot_id: str
) -> None:
    prefix = f"variants/{config_name}/"
    current = f"{prefix}{snapshot_id}/"
    obsolete = [
        item.path
        for item in api.list_bucket_tree(bucket_id, prefix=prefix, recursive=True)
        if item.path.startswith(prefix)
        and not item.path.startswith(current)
        and getattr(item, "type", "file") == "file"
    ]
    if obsolete:
        api.batch_bucket_files(bucket_id, delete=obsolete)


def publish_snapshot(
    bucket_id: str, snapshot_dir: Path, *, cache_dir: Path | None = None
) -> str:
    """Upload payloads, then their manifest, and finally update the catalog."""
    from huggingface_hub import HfApi

    from citemesh.data.embedding_cache.portable import load_manifest, validate_snapshot

    manifest = validate_snapshot(snapshot_dir)
    selected_path = _manifest_bucket_path(manifest.config_name, manifest.snapshot_id)
    api = HfApi()
    if not api.bucket_info(bucket_id).private:
        raise ValueError("Publishing is authorized only to private buckets")
    bucket_paths = {
        item.path
        for item in api.list_bucket_tree(bucket_id, recursive=True)
        if item.type == "file"
    }
    if "catalog.json" in bucket_paths:
        previous = _catalog(bucket_id)
    elif any(
        path.startswith("variants/")
        and not path.startswith(f"variants/{manifest.config_name}/")
        for path in bucket_paths
    ):
        raise ValueError("Missing catalog in an established index bucket")
    else:
        previous = IndexCatalog({})
    manifests = {}
    for name, path in previous.configs.items():
        local = _cache_root(bucket_id, cache_dir) / path
        if not local.is_file():
            _download_files(bucket_id, [(path, local)])
        old = load_manifest(local)
        if old.config_name != name or old.snapshot_id != PurePosixPath(path).parts[2]:
            raise ValueError("Published manifest disagrees with its catalog snapshot")
        if name == manifest.config_name:
            if old.contract != manifest.contract:
                raise ValueError(
                    "A changed embedding contract requires a new configuration name"
                )
            if old.calibration_ranges is not None:
                import numpy as np

                old_ranges = local.parent / old.calibration_ranges
                if not old_ranges.is_file():
                    _download_files(
                        bucket_id,
                        [
                            (
                                str(
                                    PurePosixPath(path).parent / old.calibration_ranges
                                ),
                                old_ranges,
                            )
                        ],
                    )
                if not np.array_equal(
                    np.load(old_ranges, allow_pickle=False),
                    np.load(
                        snapshot_dir / manifest.calibration_ranges, allow_pickle=False
                    ),
                ):
                    raise ValueError(
                        "Changed calibration ranges require a new configuration name"
                    )
        manifests[name] = old
    if selected_path in bucket_paths:
        remote = _cache_root(bucket_id, cache_dir) / selected_path
        if not remote.is_file():
            _download_files(bucket_id, [(selected_path, remote)])
        if load_manifest(remote) != manifest:
            raise ValueError("A different snapshot must use a new snapshot ID")
    configs = {**previous.configs, manifest.config_name: selected_path}
    catalog = IndexCatalog(configs, previous.default_config or manifest.config_name)
    manifests[manifest.config_name] = manifest
    directory = PurePosixPath(selected_path).parent
    payloads = [
        (snapshot_dir / name, str(directory / name)) for name in manifest.artifact_paths
    ]
    if selected_path not in bucket_paths:
        if payloads:
            api.batch_bucket_files(bucket_id, add=payloads)
        api.batch_bucket_files(
            bucket_id, add=[(snapshot_dir / "index.json", selected_path)]
        )
    if previous.configs.get(manifest.config_name) != selected_path:
        api.batch_bucket_files(
            bucket_id,
            add=[(json.dumps(catalog.to_dict(), indent=2).encode(), "catalog.json")],
        )
    _prune_snapshots(api, bucket_id, manifest.config_name, manifest.snapshot_id)
    api.batch_bucket_files(
        bucket_id, add=[(_readme(catalog, manifests).encode(), "README.md")]
    )
    return manifest.snapshot_id
