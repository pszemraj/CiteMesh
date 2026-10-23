"""Pinned configuration downloads and atomic private Hub publication."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from citemesh.data.embedding_contract import serialize_embedding_namespace
from citemesh.services import embedding_indexes as indexes


def manifest(config="one", snapshot="snapshot-one", *, shard=None, source="source-one"):
    selectors = {
        "model_identity": "unsloth/embeddinggemma-300m",
        "requested_revision": "main",
        "artifact_identity": "artifact",
        "profile_schema": "v1",
        "representation": "retrieval-document-v1",
        "dimension": 2,
        "stored_dtype": "float32",
        "calibration_sample_size": None,
        "formatter_identity": "formatter",
    }
    return {
        "format_version": 1,
        "config_name": config,
        "snapshot_id": snapshot,
        "created_at": "2026-10-23T00:00:00Z",
        "producer_version": "0.1.0",
        "contract": {
            "model_id": "unsloth/embeddinggemma-300m",
            "model_revision": "model-pin",
            "model_artifact_identity": "artifact",
            "profile_key": "embeddinggemma",
            "profile_schema": "v1",
            "representation": "retrieval-document-v1",
            "formatter_identity": "formatter",
            "normalization": "l2-v1",
            "dimension": 2,
            "stored_dtype": "float32",
            "quantization": None,
            "calibration_sample_size": None,
        },
        "namespace_serialization": serialize_embedding_namespace(**selectors),
        "namespace_selectors": selectors,
        "source": {
            "repo_id": "source/arxiv",
            "config_name": None,
            "split": "train",
            "revision": source,
            "selection": "all",
            "limit": None,
        },
        "row_count": 1 if shard else 0,
        "source_records_examined": 1 if shard else 0,
        "row_order": "native-relative",
        "shards": [{"path": shard, "first_row": 0, "row_count": 1}] if shard else [],
        "calibration_ranges": None,
        "producer_provenance": {},
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize("name", [".", ".."])
def test_catalog_configuration_must_name_one_variant_directory(name):
    with pytest.raises(ValueError, match="configuration name"):
        indexes.IndexCatalog.from_dict(
            {"format_version": 1, "configs": {name: f"variants/{name}/index.json"}}
        )


@dataclass
class Addition:
    path_in_repo: str
    path_or_fileobj: Path


@dataclass
class Deletion:
    path_in_repo: str


@pytest.fixture
def hub(tmp_path, monkeypatch):
    class Hub:
        private = True
        head = "parent"
        fail = None

        def __init__(self):
            self.root = tmp_path / "hub"
            self.downloads = []
            self.resolutions = []
            self.commits = []
            self.tree("parent").mkdir(parents=True)

        def tree(self, revision):
            return self.root / revision

        def dataset_info(self, *, repo_id, revision):
            self.resolutions.append(revision)
            resolved = self.head if revision == "main" else revision
            return SimpleNamespace(
                sha=resolved,
                private=self.private,
                siblings=[
                    SimpleNamespace(
                        rfilename=str(path.relative_to(self.tree(resolved)))
                    )
                    for path in self.tree(resolved).rglob("*")
                    if path.is_file()
                ],
            )

        def download(self, *, filename, revision, **kwargs):
            self.downloads.append((filename, revision))
            path = self.tree(revision) / filename
            if not path.is_file():
                raise FileNotFoundError(filename)
            return str(path)

        def create_commit(self, **kwargs):
            self.commits.append(kwargs)
            if self.fail == "preupload":
                raise ConnectionError("preupload failed")
            if self.fail == "conflict" or kwargs["parent_commit"] != self.head:
                raise ValueError("parent conflict")
            shutil.copytree(self.tree(self.head), self.tree("published"))
            for operation in kwargs["operations"]:
                target = self.tree("published") / operation.path_in_repo
                if isinstance(operation, Addition):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(operation.path_or_fileobj.read_bytes())
                else:
                    target.unlink()
            self.head = "published"
            if self.fail == "response":
                raise ConnectionError("final response lost")
            return SimpleNamespace(oid="published")

    fixture = Hub()
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=lambda: fixture,
            hf_hub_download=fixture.download,
            CommitOperationAdd=Addition,
            CommitOperationDelete=Deletion,
        ),
    )
    return fixture


def catalog(configs, default="one"):
    return {
        "format_version": 1,
        "default_config": default,
        "configs": {name: f"variants/{name}/index.json" for name in configs},
    }


def install(hub, *, selected_shard=None):
    write_json(hub.tree("parent") / "catalog.json", catalog(["one", "two"]))
    write_json(
        hub.tree("parent") / "variants/one/index.json", manifest(shard=selected_shard)
    )
    write_json(
        hub.tree("parent") / "variants/two/index.json",
        manifest("two", "snapshot-two", source="independent-source"),
    )


def local_export(tmp_path):
    path = tmp_path / "export"
    write_json(path / "index.json", manifest(snapshot="new-snapshot"))
    return path


def test_resolve_then_download_only_selected_declared_files(hub):
    install(hub, selected_shard="data/part.parquet")
    shard = hub.tree("parent") / "variants/one/data/part.parquet"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"payload")
    publication = indexes.resolve_publication("owner/index")
    assert publication.origin == {
        "repo_id": "owner/index",
        "config_name": "one",
        "revision": "parent",
    }
    assert hub.downloads == [
        ("catalog.json", "parent"),
        ("variants/one/index.json", "parent"),
    ]
    assert indexes.download_publication(publication) == shard.parent.parent
    assert hub.resolutions == ["main"]
    assert hub.downloads[-1] == ("variants/one/data/part.parquet", "parent")
    assert not any("two/" in name for name, _ in hub.downloads)


def test_missing_declared_shard_fails_download(hub):
    install(hub, selected_shard="data/missing.parquet")
    publication = indexes.resolve_publication("owner/index", "one")
    with pytest.raises(FileNotFoundError, match="missing.parquet"):
        indexes.download_publication(publication)


@pytest.mark.parametrize(
    "path", ["../escape.parquet", "/absolute.parquet", "data/../escape.parquet"]
)
def test_reject_out_of_variant_artifacts_before_payload_transfer(hub, path):
    install(hub, selected_shard=path)
    with pytest.raises(ValueError):
        indexes.resolve_publication("owner/index", "one")
    assert len(hub.downloads) == 2


def test_configuration_selection_requires_declared_default(hub):
    write_json(hub.tree("parent") / "catalog.json", catalog(["one"], default=None))
    with pytest.raises(ValueError, match="--config"):
        indexes.resolve_publication("owner/index")
    with pytest.raises(ValueError, match="Unknown"):
        indexes.resolve_publication("owner/index", "missing")


def test_publish_one_commit_preserves_other_configuration_and_removes_obsolete(
    hub, tmp_path
):
    install(hub, selected_shard="data/obsolete.parquet")
    obsolete = hub.tree("parent") / "variants/one/data/obsolete.parquet"
    obsolete.parent.mkdir(parents=True)
    obsolete.write_bytes(b"old")
    before = (hub.tree("parent") / "variants/two/index.json").read_bytes()
    revision = indexes.publish_snapshot("owner/index", local_export(tmp_path))
    assert revision == "published"
    assert len(hub.commits) == 1
    commit = hub.commits[0]
    assert commit["parent_commit"] == "parent"
    assert commit["revision"] == "main"
    assert commit["repo_type"] == "dataset"
    assert (hub.tree(revision) / "variants/two/index.json").read_bytes() == before
    assert not (hub.tree(revision) / "variants/one/data/obsolete.parquet").exists()
    result = json.loads((hub.tree(revision) / "catalog.json").read_text())
    assert result == catalog(["one", "two"])
    readme = (hub.tree(revision) / "README.md").read_text()
    assert "independent-source" in readme
    assert "obsolete.parquet" not in readme


@pytest.mark.parametrize("failure", ["preupload", "conflict"])
def test_publish_failure_preserves_prior_publication(hub, tmp_path, failure):
    install(hub)
    hub.fail = failure
    with pytest.raises((ConnectionError, ValueError)):
        indexes.publish_snapshot("owner/index", local_export(tmp_path))
    assert hub.head == "parent"
    assert len(hub.commits) == 1


def test_lost_commit_response_detects_selected_snapshot_without_second_commit(
    hub, tmp_path
):
    install(hub)
    hub.fail = "response"
    assert (
        indexes.publish_snapshot("owner/index", local_export(tmp_path)) == "published"
    )
    assert len(hub.commits) == 1
    assert indexes.publish_snapshot("owner/index", tmp_path / "export") == "published"
    assert len(hub.commits) == 1


def test_first_publication_requires_readme_branch_and_private_repository(hub, tmp_path):
    export = local_export(tmp_path)
    with pytest.raises(ValueError, match="Missing catalog"):
        indexes.publish_snapshot("owner/index", export)
    (hub.tree("parent") / "README.md").write_text("Initial dataset", encoding="utf-8")
    hub.private = False
    with pytest.raises(ValueError, match="private"):
        indexes.publish_snapshot("owner/index", export)
    assert not hub.commits
    hub.private = True
    assert indexes.publish_snapshot("owner/index", export) == "published"


def test_missing_catalog_does_not_discard_existing_variants(hub, tmp_path):
    write_json(hub.tree("parent") / "variants/two/index.json", manifest("two"))
    (hub.tree("parent") / "README.md").write_text("Existing dataset", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing catalog"):
        indexes.publish_snapshot("owner/index", local_export(tmp_path))
    assert not hub.commits


def test_preflight_happens_with_manifest_without_payload_downloads(hub, tmp_path):
    install(hub, selected_shard="data/not-yet-downloaded.parquet")
    publication = indexes.resolve_publication("owner/index")
    destination = tmp_path / "native"
    assert (
        indexes.preflight_publication_destination(publication, destination) == "create"
    )
    assert not list(destination.glob("metadata_*.db"))
    assert not list(destination.glob("embeddings_*.h5"))
    assert hub.downloads == [
        ("catalog.json", "parent"),
        ("variants/one/index.json", "parent"),
    ]


def test_changed_contract_requires_a_distinct_configuration(hub, tmp_path):
    install(hub)
    export = local_export(tmp_path)
    value = json.loads((export / "index.json").read_text())
    value["contract"]["model_revision"] = "different-pin"
    write_json(export / "index.json", value)
    with pytest.raises(ValueError, match="new configuration"):
        indexes.publish_snapshot("owner/index", export)
    assert not hub.commits


def test_missing_local_payload_prevents_any_hub_operations(hub, tmp_path):
    export = tmp_path / "export"
    write_json(export / "index.json", manifest(shard="data/missing.parquet"))
    with pytest.raises(FileNotFoundError):
        indexes.publish_snapshot("owner/index", export)
    assert not hub.resolutions
    assert not hub.commits


@pytest.fixture
def publisher():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts/publish_arxiv_index.py"
    spec = importlib.util.spec_from_file_location("citemesh_manual_publisher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("adopt", [False, True])
@pytest.mark.parametrize("revision", [None, "source-pin"])
def test_manual_publisher_reuses_refresh_export_and_commit(
    publisher, monkeypatch, tmp_path, adopt, revision
):
    from citemesh.data.embedding_cache import portable
    from citemesh.strategies import embedding
    from citemesh.strategies.embedding import refresh

    calls = []

    class Builder:
        embedding_cache = object()

        def __init__(self, **options):
            calls.append(("builder", options))

        def refresh_corpus(self, revision):
            calls.append(("refresh", revision))

    monkeypatch.setattr(embedding, "EmbeddingGraphBuilder", Builder)
    monkeypatch.setattr(
        refresh,
        "adopt_legacy_cache",
        lambda builder, namespace, revision: calls.append(
            ("adopt", namespace, revision)
        ),
    )
    monkeypatch.setattr(
        portable,
        "export_snapshot",
        lambda cache, output, config_name: calls.append(
            ("export", output, config_name)
        ),
    )
    monkeypatch.setattr(
        indexes,
        "publish_snapshot",
        lambda repo, output: calls.append(("publish", repo, output)) or "commit-pin",
    )
    output = tmp_path / "export"
    assert (
        publisher.publish_arxiv_index(
            "owner/index",
            output,
            config_name="one",
            model_revision="model-pin",
            dimension=256,
            storage_precision="float32",
            adopt_namespace="0123456789ab" if adopt else None,
            **({"dataset_revision": revision} if revision is not None else {}),
        )
        == "commit-pin"
    )
    assert calls[0][1]["truncate_dim"] == 256
    assert calls[0][1]["model_revision"] == "model-pin"
    assert calls[1] == (
        ("adopt", "0123456789ab", revision) if adopt else ("refresh", revision)
    )
    assert calls[2:] == [("export", output, "one"), ("publish", "owner/index", output)]


def test_manual_publisher_failure_returns_exit_code_without_traceback(
    publisher, monkeypatch, capsys, tmp_path
):
    def fail(*args, **kwargs):
        raise ValueError("repository must be private")

    monkeypatch.setattr(publisher, "publish_arxiv_index", fail)
    assert publisher.main(["owner/index", str(tmp_path / "out")]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert captured.err == "error: repository must be private\n"


def test_same_configuration_cannot_recalibrate_stored_int8_ranges(hub, tmp_path):
    import numpy as np

    old = manifest()
    old["contract"].update(
        stored_dtype="int8",
        quantization="affine-int8-v1",
        calibration_sample_size=2,
    )
    old["namespace_selectors"].update(stored_dtype="int8", calibration_sample_size=2)
    old["namespace_serialization"] = serialize_embedding_namespace(
        **old["namespace_selectors"]
    )
    old["calibration_ranges"] = "calibration_ranges.npy"
    root = hub.tree("parent")
    write_json(root / "catalog.json", catalog(["one"]))
    write_json(root / "variants/one/index.json", old)
    np.save(
        root / "variants/one/calibration_ranges.npy",
        np.array([[-1, -1], [1, 1]], dtype=np.float32),
    )
    export = tmp_path / "int8-export"
    write_json(export / "index.json", {**old, "snapshot_id": "new-snapshot"})
    np.save(
        export / "calibration_ranges.npy",
        np.array([[-2, -2], [2, 2]], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="calibration ranges"):
        indexes.publish_snapshot("owner/index", export)
    assert not hub.commits
