# data_vectorize

Shared package for vector database operations in LiteratureAnnotator.
Consumed by `data_generation` and `label_api`; run standalone to populate
the vector DB from scratch.

## What it does

| Capability | Entry point | Description |
|---|---|---|
| **Retrieval** | `VectorDb.query()` | Embed a query with OpenAI and return top-k matching chunks from Pinecone |
| **Ingestion** | `Ingester.run()` | Load JSON articles from a MinIO bucket, chunk, embed, and upsert to Pinecone |
| **Chunking** | `Chunker` | Overlap-window text splitter; exposed for callers that only need chunking |

## Package layout

```
data_vectorize/
├── __init__.py     # public API: VectorDb, Ingester, Chunker
├── vector_db.py    # Pinecone wrapper (upsert + query with internal embedding)
├── ingester.py     # MinIO → chunk → embed → upsert pipeline
├── chnker.py       # overlap-window text splitter
└── data_label.py   # CLI entry point (python data_label.py)
```

## Usage

### Retrieval — called from `data_generation` or `label_api`

```python
from data_vectorize import VectorDb

vdb = VectorDb()
matches = vdb.query(
    namespace="article_upload_test_2",
    query_text="cancer biomarkers in ADBM patients",
    top_k=5,
)
for m in matches:
    print(f"{m.score:.3f}  {m.metadata['title']}  chunk {m.metadata['chunk']}")
    print(m.metadata["text"])
```

### Ingestion — populate the vector DB from MinIO

```python
from data_vectorize import Ingester

Ingester().run()
```

Or as a CLI script from the workspace root:

```bash
uv run python data_vectorize/data_label.py
```

### Single-article ingest

```python
from data_vectorize import Ingester

ingester = Ingester()
ingester.ingest_article({
    "12345678": {
        "Title": "My Article",
        "Abstract": "...",
        "Introduction": ["..."],
    }
})
```

## Configuration

All credentials are read from `.env.yaml` at the workspace root via
`config.app_config.load_app_config()`.  The relevant sections are:

```yaml
pinecone:
  api_key: "..."
  index_name: "adbm"
  namespace: "article_upload_test_2"

embeddings:
  api_key: "..."           # OpenAI key
  model: "text-embedding-ada-002"
  dimensions: 1536

minio:
  url: "localhost:9000"
  access_key: "..."
  secret_key: "..."
  raw_articles_bucket: "raw-pubmed-articles"
  secure: false
```

Any value can be overridden per-environment with `override.env.yaml`.

## How chunking works

`Chunker` walks the article's sections in order.  Once the running
text buffer exceeds 150 words, it flushes the buffer through
`overlap_window_chunker`, which emits two chunks per adjacent section
pair — one with a trailing overlap from the first, one with a leading
overlap from the second.  This ensures each chunk has context from its
neighbour without duplicating large blocks of text.
