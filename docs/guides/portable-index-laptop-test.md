# Test a portable index on a laptop

This checks that a laptop can install the published corpus without encoding it,
retrieve from it, and build a graph for a new paper. Run the same
checks on either or both machines.

## Choose the input

Use a completed full-coverage portable index with at least eight active papers
and a [supported embedding model](../reference/embedding-runtime.md#model-selection-and-fallback).
Get either a local export directory or a readable Storage Bucket from its
publisher. Check its manifest for the model, dimension, stored dtype, source
revision, and coverage; allow space for both the portable files and native cache,
plus model weights.

Import needs no GPU or model weights. Search and graph building use the index's
pinned encoder; their first run may download it. No upstream corpus download or
corpus encoding should occur.

## 1. Install from a checkout

Run from a checkout containing portable-index support. To make a fresh checkout:

```bash
git clone https://github.com/pszemraj/CiteMesh.git citemesh-index-test
cd citemesh-index-test
```

Use your usual Python environment, preferably Python 3.12 or 3.13. Install the
appropriate PyTorch wheel first using the [official installation selector](https://pytorch.org/get-started/locally/):
macOS for an Apple Silicon MacBook Air, or Windows with the appropriate CUDA
build for an NVIDIA laptop. CiteMesh requires torch 2.13+ on macOS. Then install:

```bash
python -m pip install -e ".[recommended]"
citemesh cache import --help
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'MPS:', torch.backends.mps.is_available())"
```

| Machine | Device to use below | Check |
| --- | --- | --- |
| Apple Silicon MacBook Air | `mps` | MPS prints `True` |
| Windows with NVIDIA GPU | `cuda` | CUDA prints `True` |
| Intel MacBook Air or CPU-only test | `cpu` | Accelerator availability is not required |

If the intended accelerator is unavailable, resolve the installation/device
issue before calling that accelerator test a pass. CPU can be tested separately.
For a non-NVIDIA Windows GPU, use CPU; CiteMesh has no DirectML backend.

For a private bucket, follow the [bucket authentication instructions](caching.md#portable-corpus-indexes).
A local export needs no bucket credentials.

## 2. Import into a fresh cache

A fresh root avoids an occupied legacy namespace blocking the import. Set this
once in the terminal used for every command below. Existing caches stay intact.

macOS:

```bash
export CITEMESH_CACHE_DIR="$HOME/.cache/citemesh-index-test"
```

Windows PowerShell:

```powershell
$env:CITEMESH_CACHE_DIR = Join-Path $env:LOCALAPPDATA "CiteMesh-index-test"
```

These remaining commands work in either shell; keep each command on one line:

```bash
citemesh cache import --bucket OWNER/INDEX --config CONFIG --log-file index-import.log
citemesh cache scan
```

Replace `OWNER/INDEX` and `CONFIG` with the publication's values. This selects
the current snapshot; add `--snapshot SNAPSHOT_ID` to select a retained snapshot.
For a local export, replace the import command with
`citemesh cache import --path ./arxiv-index --log-file index-import.log`.

**Pass:** import reports a namespace; scan shows ready, certified membership and
the manifest's active row count, dimension, dtype, source revision, and origin.
Import performs data transfer and, for int8, local binary-index reconstruction,
with no model loading or calibration. Record the namespace and snapshot identity,
and replace `NAMESPACE` in every example below. Keep the local snapshot or
completed download available for repeat tests; remote snapshots may be pruned.
See [Portable corpus indexes](caching.md#portable-corpus-indexes) for resume and
retention behavior.

## 3. Search the installed corpus

The examples use `mps`. On Windows substitute `cuda`; for CPU substitute `cpu`.
The first search can download the pinned model. Keep compilation at its default.

```bash
citemesh search "long-context language models and efficient attention" --mode local --cache-namespace NAMESPACE --device mps --all-corpus --limit 5 --verbose --log-file index-search.log
citemesh search "long-context language models and efficient attention" --mode local --cache-namespace NAMESPACE --device mps --corpus-size 8 --limit 5
```

**Pass:** both return scored papers, use the requested device and selected
namespace, and never begin corpus hydration or source refresh. The second query
searches only the eight newest members. Repeat the first command to check warm
reuse. Different devices may produce slightly different scores/orderings.

## 4. Build from an unseen paper's title and abstract

Choose a real new arXiv paper absent from the snapshot, or an unpublished draft
whose contents were never indexed. A new identifier assigned to an old indexed
paper does not establish unseen content. For arXiv use its canonical,
versionless `arxiv:<id>` identifier. For a draft use a unique `draft:<name>` ID.

Create the output directory, then save its real metadata as
`out/unseen-paper.json`, replacing every placeholder:

```bash
python -c "from pathlib import Path; Path('out').mkdir(exist_ok=True)"
```

```json
{
  "paper_id": "arxiv:YOUR_REAL_ID",
  "title": "The actual paper title",
  "abstract": "The complete actual abstract",
  "year": 2026
}
```

Save the following as `out/test-unseen-paper.py`, then run
`python out/test-unseen-paper.py`. Set `DEVICE` for the machine. This uses supplied
metadata and skips optional citation-count enrichment, so a brand-new paper need
not already be discoverable through Semantic Scholar. It exercises the actual
pinned encoder, imported corpus retrieval, and separate graph embedding cache.

```python
import json
from collections.abc import Sequence
from pathlib import Path

from citemesh import EmbeddingGraphBuilder, Paper
from citemesh.data.embedding_cache.lifecycle import (
    open_existing_cache,
    select_existing_cache,
)
from citemesh.services import SemanticScholarClient
from citemesh.visualization import GraphExporter

DEVICE = "mps"  # cuda on NVIDIA Windows; cpu for a CPU test
NAMESPACE = "REPLACE_WITH_IMPORTED_NAMESPACE"


class CorpusOnlyClient(SemanticScholarClient):
    """Skip optional external citation counts during local-index acceptance."""

    def get_papers(
        self,
        paper_ids: Sequence[str],
        *,
        raise_on_unavailable: bool = False,
    ) -> dict[str, Paper]:
        """Skip external paper lookups during local-index acceptance."""
        return {}


seed = Paper(**json.loads(Path("out/unseen-paper.json").read_text(encoding="utf-8")))
selection = select_existing_cache(NAMESPACE)
cache = open_existing_cache(selection)
assert not cache.get_paper_metadata_batch([seed.paper_id]), "Choose an unseen seed"
state, count = cache.corpus_state(), cache.embedding_count()
print(
    "namespace", selection.namespace_id, "contract", selection.contract, "state", state
)

for cap in (None, 8):
    builder = EmbeddingGraphBuilder(
        cache_selection=selection,
        device=DEVICE,
        client=CorpusOnlyClient(),
        corpus_size=cap,
        max_papers=6,
        min_semantic_similarity=0,
    )
    graph, seed_id = builder.build_graph(seed.paper_id, seed_paper=seed)
    assert seed_id == seed.paper_id
    assert graph.number_of_nodes() == 6 and graph.number_of_edges() > 0
    assert all(
        node == seed_id or attributes["paper"].is_local_corpus
        for node, attributes in graph.nodes(data=True)
    )
    assert cache.embedding_count() == count
    assert cache.corpus_state() == state
    assert not cache.get_paper_metadata_batch([seed.paper_id])
    assert builder.graph_embedding_cache.db_path != cache.db_path
    assert builder.graph_embedding_cache.storage_precision == "float32"
    runtime = graph.graph["embedding_runtime"]
    assert runtime["device"] == DEVICE
    assert runtime["resolved_model_revision"] == selection.contract.model_revision
    label = "full" if cap is None else "newest8"
    exporter = GraphExporter(graph, seed_id, theme_name="dark")
    exporter.to_json(Path(f"out/unseen-{label}.json"))
    exporter.to_dashboard_html(Path(f"out/unseen-{label}.dashboard.html"))
    print(label, "PASS", runtime)
```

**Pass:** both runs print `PASS`. Each dashboard contains the actual unseen seed
and five papers from the installed corpus. The full and newest-eight views leave
the retrieval corpus and source provenance unchanged; graph vectors occupy a
separate float32 namespace. Repeating the script should reuse graph vectors.
Open both dashboard files and confirm titles, abstracts, and navigation work.

The Python selection API is subject to
the project's [pre-1.0 API policy](python-api.md).

## 5. Check normal identifier-based CLI use

Once Semantic Scholar resolves that paper with the correct title and abstract,
replace `arxiv:YOUR_REAL_ID` and the device in this command:

```bash
citemesh build arxiv:YOUR_REAL_ID --strategy embedding --cache-namespace NAMESPACE --device mps --all-corpus --max-papers 6 --min-semantic-similarity 0 --export dashboard --export json --output out/unseen-cli --verbose --log-file unseen-cli.log
citemesh cache scan
```

**Pass:** the output contains the real paper as seed and corpus neighbors. A
`query:` seed containing only the unresolved identifier is not a pass for this
paper-metadata test. If the provider does not know the paper yet, use step 4;
that is a metadata-acquisition limitation rather than an index failure.
Do not refresh/rebuild the corpus just to query a new seed.

## Record each laptop's result

Record the checkout commit, namespace, snapshot identity, source revision and
coverage, model contract, OS, RAM, GPU, Python/torch versions, requested and
actual device, import time, first/warm search time, and pass/fail for full/capped
unseen-paper builds. Keep `index-import.log`, `index-search.log`, `unseen-cli.log`, the printed
runtime metadata, and the generated JSON/dashboard outputs. Note optional
provider failures separately from index/model failures.

Compare semantic relevance across machines rather than requiring bit-identical
CPU/MPS/CUDA results. Across query/build checks, no corpus-wide encoding,
calibration, source refresh, or retrieval-row growth should occur.
