# Label API - FastAPI + Label Studio bridge

`label_api` runs the annotation bridge service for both labeling workflows:

- **RAG Annotation Project**: humans validate model output (`CORRECT` / `INCORRECT` / `ABSTAIN`)
- **Human Labeling Project**: humans label directly from retrieved chunks (`SATISFIED` / `NOT_SATISFIED` / `ABSTAIN`)

It reads pending paper IDs from Redis, imports Label Studio tasks, receives webhook events for completed annotations, and persists results to MinIO.

## Setup

Dependencies are managed by `uv` from the repository root:

```bash
uv sync --package literature-annotator-label-api
```

Run with:

```bash
uv run uvicorn label_api.app:app --host 0.0.0.0 --port 8000 --reload
```

## Pipeline overview

```text
Redis queues
  ├─ q:papers:v1            (RAG verification)
  └─ q:papers:human:v1      (direct human labeling)
        │
        ▼
label_api FastAPI service
  ├─ creates / reuses Label Studio projects
  ├─ imports tasks from queued paper IDs
  ├─ receives /webhook annotation events
  └─ writes completed annotations to MinIO
        │
        ▼
MinIO buckets
  ├─ minio.annotations_bucket
  └─ minio.human_annotations_bucket
```

## Key modules

| File | Purpose |
|---|---|
| `app.py` | FastAPI app entrypoint and startup wiring |
| `routes.py` | Central route registration (`/health`, `/webhook`) |
| `route_handlers/` | Per-route handler implementations |
| `services/` | Startup jobs, queue polling, importers, payload builders, MinIO persistence |
| `label_studio_base.py`, `lstudio_interfacer_sdk.py`, `human_labeller_sdk.py` | Label Studio SDK wrappers and shared base |

## Queue semantics

Both RAG and human imports use the **claim/ack/requeue** pattern only:

- claim: `claim_next_paper()` / `claim_next_paper_human()`
- success: `ack_paper(...)` / `ack_paper_human(...)`
- failure: `requeue_inflight(...)` / `requeue_inflight_human(...)`

This avoids item loss compared to direct pop-based queue consumption.

## Configuration

`label_api` uses the same root `.env.yaml` as the rest of the repo (`config.load_app_config`).
The most important sections are:

```yaml
label_studio:
  label_studio_url: "http://localhost:8080"
  label_studio_api_key: "..."
  webhook_host: "http://host.docker.internal:8000"

redis:
  paper_queue: "q:papers:v1"
  human_paper_queue: "q:papers:human:v1"

pinecone:
  namespace: "article_upload_test_2"

minio:
  url: "localhost:9000"
  access_key: "..."
  secret_key: "..."
  annotations_bucket: "annotations"
  human_annotations_bucket: "human-annotations"
  bucket_name: "classified-articles"
```

## Typical workflow

1. Seed queues:
   - `uv run python -m utilities.seed_queue`
   - `uv run python -m utilities.seed_human_queue`
2. Start Label Studio and the API service.
3. API startup creates webhooks and imports initial tasks.
4. Annotators complete tasks in Label Studio.
5. Webhooks trigger save-to-MinIO for each completed annotation.

## Docker dev stack

`docker-compose.yml` in this directory brings up Postgres, Redis, Label Studio, FastAPI, and MinIO:

```bash
docker compose -f label_api/docker-compose.yml up --build
```