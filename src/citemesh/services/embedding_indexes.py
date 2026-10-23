"""Resolve, download, and atomically publish portable corpus configurations."""

from __future__ import annotations

import json
from collections.abc import Mapping
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
            if manifest != f"variants/{name}/index.json":
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
    """Describe one configuration at an immutable publication revision."""

    repo_id: str
    config_name: str
    revision: str
    manifest_path: Path
    manifest: PortableManifest
    manifest_repo_path: str
    cache_dir: Path | None = None

    @property
    def origin(self) -> dict[str, str]:
        """Return the publication identity recorded by native import."""
        return {
            "repo_id": self.repo_id,
            "config_name": self.config_name,
            "revision": self.revision,
        }


def resolve_dataset_revision(repo_id: str, requested_revision: str = "main") -> str:
    """Resolve a dataset branch or tag once to its immutable commit."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo_id=repo_id, revision=requested_revision)
    if not info.sha:
        raise RuntimeError(f"Cannot resolve dataset revision for {repo_id}")
    return info.sha


def download_declared_files(
    repo_id: str,
    resolved_revision: str,
    repo_paths: list[str],
    cache_dir: Path | None = None,
) -> dict[str, Path]:
    """Download exact filenames at one resolved revision using the Hub cache."""
    from huggingface_hub import hf_hub_download

    return {
        name: Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=resolved_revision,
                filename=name,
                cache_dir=cache_dir,
            )
        )
        for name in repo_paths
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _artifact_repo_paths(
    manifest: PortableManifest, manifest_repo_path: str
) -> list[str]:
    directory = PurePosixPath(manifest_repo_path).parent
    paths = list(manifest.artifact_paths)
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate portable artifact paths")
    for path in paths:
        relative = PurePosixPath(path)
        if (
            not path
            or relative.is_absolute()
            or any(part in {".", ".."} for part in path.split("/"))
            or "\\" in path
        ):
            raise ValueError(
                f"Artifact path must stay inside its configuration: {path!r}"
            )
    return [str(directory / path) for path in paths]


def resolve_publication(
    repo_id: str,
    config_name: str | None = None,
    revision: str = "main",
    cache_dir: Path | None = None,
) -> ResolvedPublication:
    """Resolve the catalog and selected manifest before downloading payloads."""
    from citemesh.data.embedding_cache.portable import load_manifest

    resolved = resolve_dataset_revision(repo_id, revision)
    files = download_declared_files(repo_id, resolved, ["catalog.json"], cache_dir)
    catalog = IndexCatalog.from_dict(_load_json(files["catalog.json"]))
    selected = catalog.select(config_name)
    manifest_repo_path = catalog.configs[selected]
    files = download_declared_files(repo_id, resolved, [manifest_repo_path], cache_dir)
    manifest_path = files[manifest_repo_path]
    manifest = load_manifest(manifest_path)
    if manifest.config_name != selected:
        raise ValueError("Selected manifest disagrees with the catalog configuration")
    _artifact_repo_paths(manifest, manifest_repo_path)
    return ResolvedPublication(
        repo_id,
        selected,
        resolved,
        manifest_path,
        manifest,
        manifest_repo_path,
        cache_dir,
    )


def download_publication(publication: ResolvedPublication) -> Path:
    """Download every declared selected artifact, failing on missing files."""
    paths = _artifact_repo_paths(publication.manifest, publication.manifest_repo_path)
    download_declared_files(
        publication.repo_id, publication.revision, paths, publication.cache_dir
    )
    return publication.manifest_path.parent


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
        origin={"snapshot_id": manifest.snapshot_id, **publication.origin},
    )


def _readme(catalog: IndexCatalog, manifests: Mapping[str, PortableManifest]) -> str:
    lines = ["---", "configs:"]
    for name, manifest_path in catalog.configs.items():
        manifest = manifests[name]
        lines.append(f"  - config_name: {json.dumps(name)}")
        if name == catalog.default_config:
            lines.append("    default: true")
        lines.extend(["    data_files:", "      - split: train", "        path:"])
        base = PurePosixPath(manifest_path).parent
        lines.extend(
            f"          - {json.dumps(str(base / shard.path))}"
            for shard in manifest.shards
        )
    lines.extend(
        [
            "---",
            "",
            "# CiteMesh corpus indexes",
            "",
            "Portable retrieval-document embeddings with normalized paper metadata. "
            "No model weights are included. Query and graph encoding still use the selected model.",
            "",
        ]
    )
    for name, manifest in manifests.items():
        contract, source = manifest.contract, manifest.source
        lines.extend(
            [
                f"## {name}",
                "",
                f"{manifest.row_count:,} active papers; {contract.dimension} dimensions; {contract.stored_dtype} storage.",
                f"Source: [{source.repo_id}](https://huggingface.co/datasets/{source.repo_id}), "
                f"revision `{source.revision}`, split `{source.split}`. See the source dataset for corpus attribution and licensing.",
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
            "citemesh cache import --repo <this-repository> --config <configuration> --revision <publication-commit>",
            "citemesh cache scan",
            "```",
            "",
            'Use the installed namespace with `citemesh search --mode local --cache-namespace <namespace> "your query"`. '
            "Run `citemesh cache refresh --namespace <namespace>` to reconcile newer source data locally.",
            "",
        ]
    )
    return "\n".join(lines)


def publish_complete_snapshot(
    repo_id: str,
    files: Mapping[str, Path],
    obsolete_paths: list[str],
    parent_revision: str,
) -> str:
    """Expose one complete publication with an optimistic parent constraint."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    api = HfApi()
    if not api.dataset_info(repo_id=repo_id, revision=parent_revision).private:
        raise ValueError(
            "Publishing is authorized only to private dataset repositories"
        )
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=path)
        for name, path in files.items()
    ]
    operations.extend(
        CommitOperationDelete(path_in_repo=name)
        for name in obsolete_paths
        if name not in files
    )
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        revision="main",
        parent_commit=parent_revision,
        operations=operations,
        commit_message="Publish complete CiteMesh corpus snapshot",
    )
    return commit.oid


def publish_snapshot(
    repo_id: str, snapshot_dir: Path, *, cache_dir: Path | None = None
) -> str:
    """Publish a locally completed export, preserving other configurations."""
    from huggingface_hub import HfApi

    from citemesh.data.embedding_cache.portable import load_manifest, validate_snapshot

    manifest = validate_snapshot(snapshot_dir)
    IndexCatalog.from_dict(
        {
            "format_version": 1,
            "configs": {
                manifest.config_name: f"variants/{manifest.config_name}/index.json"
            },
        }
    )
    selected_path = f"variants/{manifest.config_name}/index.json"
    artifact_paths = _artifact_repo_paths(manifest, selected_path)
    files = {selected_path: snapshot_dir / "index.json"}
    files.update(
        zip(artifact_paths, (snapshot_dir / name for name in manifest.artifact_paths))
    )
    for path in files.values():
        if not path.is_file():
            raise ValueError(f"Missing portable artifact: {path}")

    info = HfApi().dataset_info(repo_id=repo_id, revision="main")
    if not info.private:
        raise ValueError(
            "Publishing is authorized only to private dataset repositories"
        )
    parent = info.sha
    if not parent:
        raise ValueError(
            "Create a private dataset repository with an initial README branch before publishing"
        )
    repo_paths = {sibling.rfilename for sibling in info.siblings or []}
    if "catalog.json" in repo_paths:
        path = download_declared_files(repo_id, parent, ["catalog.json"], cache_dir)[
            "catalog.json"
        ]
        previous = IndexCatalog.from_dict(_load_json(path))
    elif "README.md" in repo_paths and repo_paths <= {
        "README.md",
        ".gitattributes",
        ".gitignore",
    }:
        previous = IndexCatalog({})
    else:
        raise ValueError(
            "Missing catalog in an established index repository; initialize only an empty README repository"
        )

    manifests = {}
    obsolete = []
    for name, path in previous.configs.items():
        local = download_declared_files(repo_id, parent, [path], cache_dir)[path]
        old = load_manifest(local)
        if old.config_name != name:
            raise ValueError(
                "Published manifest disagrees with its catalog configuration"
            )
        if name == manifest.config_name:
            if old.contract != manifest.contract:
                raise ValueError(
                    "A changed embedding contract requires a new configuration name"
                )
            if old.calibration_ranges is not None:
                import numpy as np

                old_path = str(PurePosixPath(path).parent / old.calibration_ranges)
                downloaded = download_declared_files(
                    repo_id, parent, [old_path], cache_dir
                )[old_path]
                if not np.array_equal(
                    np.load(downloaded, allow_pickle=False),
                    np.load(
                        snapshot_dir / manifest.calibration_ranges, allow_pickle=False
                    ),
                ):
                    raise ValueError(
                        "Changed calibration ranges require a new configuration name"
                    )
            if old.snapshot_id == manifest.snapshot_id:
                return parent
            obsolete = [path, *_artifact_repo_paths(old, path)]
        else:
            manifests[name] = old
    configs = dict(previous.configs)
    configs[manifest.config_name] = selected_path
    catalog = IndexCatalog.from_dict(
        {
            "format_version": 1,
            "configs": configs,
            "default_config": previous.default_config or manifest.config_name,
        }
    )
    manifests[manifest.config_name] = manifest
    with TemporaryDirectory(prefix="citemesh-publication-") as directory:
        root = Path(directory)
        catalog_path, readme_path = root / "catalog.json", root / "README.md"
        catalog_path.write_text(
            json.dumps(catalog.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        readme_path.write_text(_readme(catalog, manifests), encoding="utf-8")
        files.update({"catalog.json": catalog_path, "README.md": readme_path})
        try:
            return publish_complete_snapshot(repo_id, files, obsolete, parent)
        except Exception as error:
            # A final response can be lost after the commit has become visible.
            try:
                current = resolve_publication(
                    repo_id, manifest.config_name, cache_dir=cache_dir
                )
            except Exception:
                raise error
            if current.manifest.snapshot_id == manifest.snapshot_id:
                return current.revision
            raise
