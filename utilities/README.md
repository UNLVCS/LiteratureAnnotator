# Utilities — Queue, Vector DB, and Seeding Helpers

Shared runtime utilities for the LiteratureAnnotator paper processing pipelines:
Redis-backed queues, Pinecone vector DB access, queue status monitoring, and
queue seeding scripts.

## Setup

Dependencies are declared in `utilities/pyproject.toml` and managed by [uv](https://docs.astral.sh/uv/).
Run this once from the **workspace root** to install them into the shared `.venv`:

```bash
uv sync --package literature-annotator-utilities
```

No need to activate the venv — prefix commands with `uv run` and it handles that automatically.

## Quick Start

```bash
# Seed the RAG labeling queue from a file of paper IDs
uv run python -m utilities.seed_queue

# Seed the human labeling queue
uv run python -m utilities.seed_human_queue

# Check queue depths across all pipelines
uv run python -m utilities.queue_status

# Check status and print the actual paper IDs in each queue
uv run python -m utilities.queue_status --ids

# Move stuck in-flight papers back to pending
uv run python -m utilities.queue_status --fix
```

## Configuration

All settings come from `.env.yaml` in the workspace root via `load_app_config()`.

### First-time setup

```bash
# from the workspace root
cp .env.yaml.example .env.yaml
# edit .env.yaml and fill in your values
```

### Redis — required for all queue operations

Fill in the `redis:` block of `.env.yaml`:

```yaml
redis:
  url: "redis://localhost:6379/0"

  # Queue and set names — defaults work for a single environment.
  # Override if running multiple environments on one Redis instance.
  paper_queue:            "q:papers:v1"
  paper_processing:       "q:papers:processing:v1"
  paper_dedup_set:        "s:papers:enqueued:v1"
  completed_papers_queue: "q:papers:completed:v1"
  generated_set:          "s:papers:generated:v1"

  ann_queue:              "q:annotations:completed:v1"
  ann_flush_threshold:    1000
  ann_persist_path:       "data_labeling/annotations.jsonl"
  ann_flush_on_exit:      true
  ann_install_signal_handlers: true

  human_paper_queue:      "q:papers:human:v1"
  human_processing_queue: "q:papers:human:processing:v1"
  human_dedup_set:        "s:papers:human:enqueued:v1"
```

### Pinecone — required for `vector_db.py`

```yaml
pinecone:
  api_key:    ""
  index_name: "adbm"
  namespace:  "article_upload_test_2"

embeddings:
  dimensions: 1536
```

### Seed file paths — used by the seeding scripts

```yaml
seed:
  queue_file:         "utilities/test_papers.txt"   # RAG queue input
  human_papers_file:  "utilities/human_papers.txt"  # human queue input
```

## Scripts

### `seed_queue` — seed the RAG labeling queue

Reads paper IDs (one per line) from `seed.queue_file` in `.env.yaml` and
enqueues each into the RAG labeling pipeline. Duplicates are silently skipped
via the Redis dedup set.

```bash
uv run python -m utilities.seed_queue
```

Input file format (`utilities/test_papers.txt`):
```
17299597
34567890
...
```

### `seed_human_queue` — seed the human labeling queue

Same as above but targets the human labeling pipeline. Reads from
`seed.human_papers_file` in `.env.yaml`.

```bash
uv run python -m utilities.seed_human_queue
```

### `queue_status` — queue status dashboard

Prints live queue depths for both the RAG and human labeling pipelines,
annotation buffer size, and Redis connection info.

```bash
uv run python -m utilities.queue_status           # summary
uv run python -m utilities.queue_status --ids     # include paper IDs in each queue
uv run python -m utilities.queue_status --fix     # move stuck in-flight papers → pending
```

Example output:

```
========================================================
  LiteratureAnnotator — Queue Status
========================================================

── RAG Labeling Pipeline ──────────────────────────────
  Pending          42  [████████░░░░░░░░░░░░]
  In-flight         3  [█░░░░░░░░░░░░░░░░░░░]
  Dedup set        45  (papers ever enqueued & not yet acked)
  Generated set   120  (completed successfully)
  Completed log     0  (legacy completed-papers list)

── Human Labeling Pipeline ────────────────────────────
  Pending          10  [██████████░░░░░░░░░░]
  In-flight         1  [█░░░░░░░░░░░░░░░░░░░]
  Dedup set        11

── Annotations ────────────────────────────────────────
  Buffered          0  (pending flush to disk)

── Redis connection ───────────────────────────────────
  URL           redis://localhost:6379/0
  Version       7.2.4
  Uptime        2d
```

## Modules

### `queue_helpers.py` — `PaperQueue`

The core queue abstraction. Can be used directly or via the module-level
convenience functions:

```python
# Module-level functions (load config from .env.yaml automatically)
from utilities.queue_helpers import (
    enqueue_paper_id,
    claim_next_paper,
    ack_paper,
    requeue_inflight,
    push_completed_paper,
    push_completed_annotation,
    flush_annotations_to_persistent,
)

# Or instantiate explicitly with your own AppConfig
from utilities.queue_helpers import PaperQueue
from config.app_config import load_app_config

queue = PaperQueue.from_app_config(load_app_config())
queue.register_shutdown_hooks()   # flush annotations on SIGINT/SIGTERM/exit
paper_id = queue.claim_next_paper()
# ... process paper ...
queue.ack_paper(paper_id)
```

**Paper lifecycle:**

```
enqueue_paper_id()
      ↓
  paper_queue  (pending)
      ↓  claim_next_paper()
  paper_processing  (in-flight)
      ↓  ack_paper()        or  requeue_inflight()  (on failure)
  [removed]                      paper_queue  (retried)
```

### `vector_db.py` — `VectorDb`

Pinecone wrapper used by upstream RAG scripts. Loads index config from
`.env.yaml` automatically when no explicit args are passed:

```python
from utilities.vector_db import VectorDb

db = VectorDb()                          # reads pinecone + embeddings from .env.yaml
db = VectorDb(api_key="...", index_name="my-index", embedding_dimensions=1536)
```

### `criteria.py` — inclusion criteria prompts

Shared LLM prompt strings for the six paper-inclusion criteria used by both the
RAG labeling and human labeling pipelines. Import directly — no config needed:

```python
from utilities.criteria import CRITERIA_PROMPTS, CRITERION_NAMES

for name, prompt in zip(CRITERION_NAMES, CRITERIA_PROMPTS):
    print(name, prompt)
```
