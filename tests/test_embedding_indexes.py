"""Bucket snapshot selection, bounded retention, and private publication."""

from __future__ import annotations

import json
import sys
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
            {
                "format_version": 1,
                "configs": {name: f"variants/{name}/snapshot/index.json"},
            }
        )


@pytest.fixture
def hub(tmp_path, monkeypatch):
    class Hub:
        private = True
        fail_path = None
        download_failure = False
        fail_delete = False

        def __init__(self):
            self.root = tmp_path / "bucket"
            self.root.mkdir()
            self.downloads = []
            self.uploads = []
            self.info_calls = 0

        def bucket_info(self, bucket_id):
            self.info_calls += 1
            return SimpleNamespace(private=self.private)

        def list_bucket_tree(self, bucket_id, prefix=None, recursive=None):
            return [
                SimpleNamespace(
                    path=str(path.relative_to(self.root)),
                    type="file" if path.is_file() else "directory",
                )
                for path in self.root.rglob("*")
                if (
                    prefix is None
                    or str(path.relative_to(self.root)).startswith(prefix)
                )
            ]

        def download_bucket_files(self, bucket_id, files, raise_on_missing_files):
            assert raise_on_missing_files
            self.downloads.extend(remote for remote, _ in files)
            for remote, local in files:
                source = self.root / remote
                if not source.is_file():
                    raise FileNotFoundError(remote)
                Path(local).write_bytes(source.read_bytes())
                if self.download_failure:
                    raise ConnectionError("interrupted download")

        def batch_bucket_files(self, bucket_id, *, add=None, delete=None):
            self.uploads.append(
                {"add": [remote for _, remote in add or []], "delete": delete or []}
            )
            for source, remote in add or []:
                if self.fail_path and remote.endswith(self.fail_path):
                    raise ConnectionError("upload failed")
                target = self.root / remote
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(
                    source if isinstance(source, bytes) else Path(source).read_bytes()
                )
            if delete and self.fail_delete:
                raise ConnectionError("delete failed")
            for remote in delete or []:
                (self.root / remote).unlink()

    fixture = Hub()
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: fixture)
    )
    monkeypatch.setattr(
        indexes,
        "_cache_root",
        lambda bucket, cache: tmp_path / "download-cache" / bucket,
    )
    return fixture


def catalog(configs=("one", "two"), default="one"):
    return {
        "format_version": 1,
        "default_config": default,
        "configs": {
            name: f"variants/{name}/snapshot-{name}/index.json" for name in configs
        },
    }


def install(hub, *, selected_shard=None):
    write_json(hub.root / "catalog.json", catalog())
    write_json(
        hub.root / "variants/one/snapshot-one/index.json",
        manifest(shard=selected_shard),
    )
    write_json(
        hub.root / "variants/two/snapshot-two/index.json",
        manifest("two", "snapshot-two", source="independent-source"),
    )


def local_export(tmp_path, *, with_payload=False):
    path = tmp_path / "export"
    write_json(
        path / "index.json",
        manifest(
            snapshot="new-snapshot", shard="data/part.parquet" if with_payload else None
        ),
    )
    if with_payload:
        import pyarrow as pa
        import pyarrow.parquet as pq

        from citemesh.data.embedding_cache.portable import (
            load_manifest,
            portable_schema,
        )

        schema = portable_schema(load_manifest(path).contract)
        row = {
            "row_number": 0,
            "paper_id": "paper",
            "text_hash": "hash",
            "source_ordinal": 0,
            "embedding": [0.5, 0.5],
        }
        (path / "data").mkdir()
        pq.write_table(
            pa.Table.from_pylist([row], schema=schema), path / "data/part.parquet"
        )
    return path


def test_resolve_then_download_only_selected_declared_files(hub):
    install(hub, selected_shard="data/part.parquet")
    shard = hub.root / "variants/one/snapshot-one/data/part.parquet"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"payload")
    publication = indexes.resolve_publication("owner/index")
    assert publication.origin == {
        "bucket_id": "owner/index",
        "config_name": "one",
        "snapshot_id": "snapshot-one",
    }
    assert hub.downloads == ["catalog.json", "variants/one/snapshot-one/index.json"]
    root = indexes.download_publication(publication)
    assert (root / "data/part.parquet").read_bytes() == b"payload"
    assert hub.downloads[-1] == "variants/one/snapshot-one/data/part.parquet"
    assert not any("two/" in name for name in hub.downloads)
    assert indexes.download_publication(publication) == root
    assert len(hub.downloads) == 3


def test_missing_declared_shard_fails_download_without_completed_local_file(hub):
    install(hub, selected_shard="data/missing.parquet")
    publication = indexes.resolve_publication("owner/index", "one")
    with pytest.raises(FileNotFoundError, match="missing.parquet"):
        indexes.download_publication(publication)
    assert not (publication.manifest_path.parent / "data/missing.parquet").exists()


@pytest.mark.parametrize(
    "path", ["../escape.parquet", "/absolute.parquet", "data/../escape.parquet"]
)
def test_reject_out_of_variant_artifacts_before_payload_transfer(hub, path):
    install(hub, selected_shard=path)
    with pytest.raises(ValueError):
        indexes.resolve_publication("owner/index", "one")
    assert len(hub.downloads) == 2


def test_configuration_selection_requires_declared_default(hub):
    write_json(hub.root / "catalog.json", catalog(("one",), default=None))
    with pytest.raises(ValueError, match="--config"):
        indexes.resolve_publication("owner/index")
    with pytest.raises(ValueError, match="Unknown"):
        indexes.resolve_publication("owner/index", "missing")
    with pytest.raises(ValueError, match="--config"):
        indexes.resolve_publication("owner/index", snapshot_id="snapshot-one")


def test_publish_preserves_other_config_and_prunes_selected_config_history(
    hub, tmp_path
):
    install(hub, selected_shard="data/obsolete.parquet")
    obsolete = hub.root / "variants/one/snapshot-one/data/obsolete.parquet"
    obsolete.parent.mkdir(parents=True)
    obsolete.write_bytes(b"old")
    write_json(hub.root / "variants/one/unfinished/index.json", {"incomplete": True})
    before = (hub.root / "variants/two/snapshot-two/index.json").read_bytes()
    assert (
        indexes.publish_snapshot("owner/index", local_export(tmp_path))
        == "new-snapshot"
    )
    assert (hub.root / "variants/two/snapshot-two/index.json").read_bytes() == before
    assert not obsolete.exists()
    assert not (hub.root / "variants/one/unfinished/index.json").exists()
    result = json.loads((hub.root / "catalog.json").read_text())
    assert result["configs"] == {
        "one": "variants/one/new-snapshot/index.json",
        "two": "variants/two/snapshot-two/index.json",
    }
    assert hub.uploads[0]["add"] == ["variants/one/new-snapshot/index.json"]
    assert hub.uploads[1]["add"] == ["catalog.json"]
    readme = (hub.root / "README.md").read_text()
    assert "independent-source" in readme
    assert not readme.startswith("---")
    assert "--bucket" in readme


@pytest.mark.parametrize("failure", ["index.json", "catalog.json"])
def test_publish_failure_preserves_prior_catalog(hub, tmp_path, failure):
    install(hub)
    before = (hub.root / "catalog.json").read_bytes()
    hub.fail_path = failure
    with pytest.raises(ConnectionError):
        indexes.publish_snapshot("owner/index", local_export(tmp_path))
    assert (hub.root / "catalog.json").read_bytes() == before
    assert (hub.root / "variants/one/snapshot-one/index.json").is_file()
    hub.fail_path = None
    assert (
        indexes.publish_snapshot("owner/index", tmp_path / "export") == "new-snapshot"
    )


def test_empty_private_bucket_accepts_first_publication(hub, tmp_path):
    export = local_export(tmp_path)
    hub.private = False
    with pytest.raises(ValueError, match="private"):
        indexes.publish_snapshot("owner/index", export)
    assert not hub.uploads
    hub.private = True
    assert indexes.publish_snapshot("owner/index", export) == "new-snapshot"


def test_missing_catalog_does_not_discard_other_existing_variants(hub, tmp_path):
    write_json(hub.root / "variants/two/snapshot-two/index.json", manifest("two"))
    with pytest.raises(ValueError, match="Missing catalog"):
        indexes.publish_snapshot("owner/index", local_export(tmp_path))
    assert not hub.uploads


def test_preflight_happens_before_payload_downloads(hub, tmp_path):
    install(hub, selected_shard="data/not-yet-downloaded.parquet")
    publication = indexes.resolve_publication("owner/index")
    destination = tmp_path / "native"
    assert (
        indexes.preflight_publication_destination(publication, destination) == "create"
    )
    assert not list(destination.glob("metadata_*.db"))
    assert not list(destination.glob("embeddings_*.h5"))
    assert hub.downloads == ["catalog.json", "variants/one/snapshot-one/index.json"]


def test_changed_contract_requires_distinct_configuration(hub, tmp_path):
    install(hub)
    export = local_export(tmp_path)
    value = json.loads((export / "index.json").read_text())
    value["contract"]["model_revision"] = "different-pin"
    write_json(export / "index.json", value)
    with pytest.raises(ValueError, match="new configuration"):
        indexes.publish_snapshot("owner/index", export)
    assert not hub.uploads


def test_missing_local_payload_prevents_hub_operations(hub, tmp_path):
    export = tmp_path / "export"
    write_json(export / "index.json", manifest(shard="data/missing.parquet"))
    with pytest.raises(FileNotFoundError):
        indexes.publish_snapshot("owner/index", export)
    assert hub.info_calls == 0
    assert not hub.uploads


def test_explicit_cached_snapshot_still_imports_after_remote_retirement(hub, tmp_path):
    install(hub, selected_shard="data/part.parquet")
    remote = hub.root / "variants/one/snapshot-one/data/part.parquet"
    remote.parent.mkdir(parents=True)
    remote.write_bytes(b"payload")
    old = indexes.resolve_publication("owner/index", "one")
    indexes.download_publication(old)
    indexes.publish_snapshot("owner/index", local_export(tmp_path))
    assert not remote.exists()
    hub.downloads.clear()
    cached = indexes.resolve_publication("owner/index", "one", "snapshot-one")
    assert (
        indexes.download_publication(cached) / "data/part.parquet"
    ).read_bytes() == b"payload"
    assert not hub.downloads
    assert indexes.resolve_publication("owner/index").snapshot_id == "new-snapshot"


def test_uncached_retired_snapshot_fails_without_native_state(hub):
    install(hub)
    with pytest.raises(FileNotFoundError, match="retired"):
        indexes.resolve_publication("owner/index", "one", "retired")


def test_failed_download_is_not_reused_as_completed(hub):
    install(hub)
    hub.download_failure = True
    with pytest.raises(ConnectionError):
        indexes.resolve_publication("owner/index", "one", "snapshot-one")
    hub.download_failure = False
    publication = indexes.resolve_publication("owner/index", "one", "snapshot-one")
    assert publication.manifest_path.is_file()
    assert hub.downloads.count("variants/one/snapshot-one/index.json") == 2


def test_repeat_publication_only_prunes_leftovers(hub, tmp_path):
    install(hub)
    export = local_export(tmp_path)
    indexes.publish_snapshot("owner/index", export)
    hub.uploads.clear()
    assert indexes.publish_snapshot("owner/index", export) == "new-snapshot"
    assert hub.uploads == [{"add": ["README.md"], "delete": []}]


def test_snapshot_id_cannot_overwrite_different_manifest(hub, tmp_path):
    install(hub)
    export = local_export(tmp_path)
    indexes.publish_snapshot("owner/index", export)
    value = json.loads((export / "index.json").read_text())
    value["created_at"] = "different-time"
    write_json(export / "index.json", value)
    with pytest.raises(ValueError, match="new snapshot ID"):
        indexes.publish_snapshot("owner/index", export)


def test_same_configuration_cannot_recalibrate_stored_int8_ranges(hub, tmp_path):
    import numpy as np

    old = manifest()
    old["contract"].update(
        stored_dtype="int8", quantization="affine-int8-v1", calibration_sample_size=2
    )
    old["namespace_selectors"].update(stored_dtype="int8", calibration_sample_size=2)
    old["namespace_serialization"] = serialize_embedding_namespace(
        **old["namespace_selectors"]
    )
    old["calibration_ranges"] = "calibration_ranges.npy"
    write_json(hub.root / "catalog.json", catalog(("one",)))
    write_json(hub.root / "variants/one/snapshot-one/index.json", old)
    np.save(
        hub.root / "variants/one/snapshot-one/calibration_ranges.npy",
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
    assert not hub.uploads


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
def test_manual_publisher_reuses_refresh_export_and_publication(
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
        lambda repo, output: calls.append(("publish", repo, output)) or "snapshot-pin",
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
        == "snapshot-pin"
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
        raise ValueError("bucket must be private")

    monkeypatch.setattr(publisher, "publish_arxiv_index", fail)
    assert publisher.main(["owner/index", str(tmp_path / "out")]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert captured.err == "error: bucket must be private\n"


@pytest.mark.parametrize("failure", ["part.parquet", "index.json", "catalog.json"])
def test_first_publication_upload_failure_can_resume_same_export(
    hub, tmp_path, failure
):
    export = local_export(tmp_path, with_payload=True)
    hub.fail_path = failure
    with pytest.raises(ConnectionError, match="upload failed"):
        indexes.publish_snapshot("owner/index", export)
    assert not (hub.root / "catalog.json").exists()
    hub.fail_path = None
    hub.uploads.clear()
    assert indexes.publish_snapshot("owner/index", export) == "new-snapshot"
    assert indexes.resolve_publication("owner/index").snapshot_id == "new-snapshot"
    writes = [path for upload in hub.uploads for path in upload["add"]]
    if failure != "catalog.json":
        assert (
            writes.index("variants/one/new-snapshot/data/part.parquet")
            < writes.index("variants/one/new-snapshot/index.json")
            < writes.index("catalog.json")
        )
    else:
        assert writes == ["catalog.json", "README.md"]


def test_failed_pruning_can_retry_without_reuploading_payload(hub, tmp_path):
    install(hub)
    export = local_export(tmp_path, with_payload=True)
    hub.fail_delete = True
    with pytest.raises(ConnectionError, match="delete failed"):
        indexes.publish_snapshot("owner/index", export)
    assert indexes.resolve_publication("owner/index").snapshot_id == "new-snapshot"
    assert (hub.root / "variants/one/snapshot-one/index.json").is_file()
    hub.fail_delete = False
    hub.uploads.clear()
    assert indexes.publish_snapshot("owner/index", export) == "new-snapshot"
    assert not (hub.root / "variants/one/snapshot-one/index.json").exists()
    assert not any(
        ".parquet" in path for upload in hub.uploads for path in upload["add"]
    )


def test_pending_import_reports_bucket_snapshot_resume_command(hub, tmp_path):
    from citemesh.data.embedding_cache.lifecycle import import_destination_session
    from citemesh.data.embedding_cache.portable import manifest_import_identity

    install(hub)
    first = indexes.resolve_publication("owner/index")
    settings = dict(
        cache_dir=tmp_path / "native",
        namespace_serialization=first.manifest.namespace_serialization,
        contract=first.manifest.contract,
        source=first.manifest.source,
        identity=manifest_import_identity(first.manifest),
        namespace_selectors=first.manifest.namespace_selectors,
        origin=first.origin,
    )
    with import_destination_session(**settings):
        pass
    indexes.publish_snapshot("owner/index", local_export(tmp_path))
    current = indexes.resolve_publication("owner/index")
    with pytest.raises(ValueError, match="Conflicting incomplete import") as error:
        indexes.preflight_publication_destination(current, settings["cache_dir"])
    assert "--bucket owner/index --config one --snapshot snapshot-one" in str(
        error.value
    )
    assert (
        indexes.preflight_publication_destination(first, settings["cache_dir"])
        == "resume_same_import"
    )
