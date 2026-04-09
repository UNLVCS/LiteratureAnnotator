# data_generation

Generates labeled training data by evaluating research papers against inclusion/exclusion
criteria using LLM providers. Papers are pulled from a Redis queue, relevant chunks are
retrieved from Pinecone, and each criterion is evaluated by one or more LLMs. Results are
written to MinIO.

## Pipeline overview

```
Redis queue
    │
    ▼
claim paper ID
    │
    ▼
Pinecone (via data_vectorize)
  vector search per criterion
    │
    ▼
LLM providers (via llm_providers)
  one inference call per criterion per provider
    │
    ▼
response_standardizer
  strip markdown, extract JSON
    │
    ▼
MinIO  ──  results/{provider}/{paper_id}.json
    │
    ▼
Redis  ──  ack (success) or requeue (failure)
```

## Files

| File | Purpose |
|---|---|
| `labeler.py` | Sequential labeler — one paper at a time, class-based. Good for debugging. |
| `labeler_mp.py` | Multiprocessing labeler — one worker process per LLM provider, runs in parallel. Use this in production. |
| `query_db.py` | Utility module: sets up a LangChain `PineconeVectorStore` backed by `data_vectorize.VectorDb`. Import this when you need a LangChain retriever elsewhere in the package. |
| `response_standardizer.py` | Cleans LLM responses (strips ` ```json ` fences, extracts embedded JSON) and parses them into dicts. |

## Usage

### Run the labeler (multiprocessing — production)

```bash
uv run python data_generation/labeler_mp.py
```

Processes every paper currently in the Redis queue, one worker process per configured LLM
provider. Results land in the MinIO bucket defined by `minio.synthetic_data_bucket` in
`.env.yaml`.

### Run the labeler (sequential — debugging)

```bash
uv run python data_generation/labeler.py
```

To populate the vector DB first, use `data_vectorize` directly:

```bash
uv run python data_vectorize/data_label.py
```

## Retrieval strategy

For each paper × criterion pair the labeler:

1. Issues a vector similarity search against Pinecone filtered to `{"doc": paper_id}` — so
   only chunks belonging to that paper are considered.
2. Returns the top-`k` (default 5) chunks most semantically similar to the criterion
   prompt.
3. Concatenates those chunks into a context block and passes them to the LLM.

Because retrieval is per-criterion (not per-paper), each LLM call sees the most relevant
excerpt for the question it is answering, rather than the full text. The vector store can
be swapped for a full-text fallback by passing the entire paper text as a single chunk
during ingestion.

## Configuration

All values come from `.env.yaml` at the workspace root.

```yaml
pinecone:
  api_key: "..."
  index_name: "adbm"
  namespace: "article_upload_test_2"

embeddings:
  api_key: "..."         # OpenAI key used for vector search
  model: "text-embedding-ada-002"
  dimensions: 1536

minio:
  url: "localhost:9000"
  access_key: "..."
  secret_key: "..."
  synthetic_data_bucket: "rag-labeled-data"   # labeling output
  raw_articles_bucket: "raw-pubmed-articles"  # ingestion input

redis:
  url: "redis://localhost:6379/0"
  paper_queue: "q:papers:v1"

llm_providers:
  meta-llama/Llama-3.1-8B-Instruct:
    type: vllm_native
    base_url: "http://localhost:8000/v1"
  gpt-4o-mini:
    type: openai
    api_key: "..."
```

## Output format

Each result is a JSON file at `{provider}/{paper_id}.json` in the output MinIO bucket:

```json
{
  "paper_id": "38291045",
  "provider": "gpt-4o-mini",
  "title": "Blood-based biomarkers for Alzheimer's disease...",
  "criteria_results": [
    {
      "criterion": "criterion_1",
      "response": {"criterion_1": {"satisfied": true, "reason": "..."}},
      "chunks_used": 5,
      "error": null
    }
  ],
  "errors": []
}
```

## Dependencies

| Package | Role |
|---|---|
| `data_vectorize` | `VectorDb` — Pinecone client and index handle |
| `utilities` | Redis queue helpers (`queue_helpers`) and criteria prompts (`criteria`) |
| `llm_providers` | Provider abstraction (`BaseLLMProvider`, `Query`) |
| `config` | `.env.yaml` config models |
| `langchain-pinecone` | `PineconeVectorStore` for metadata-filtered retrieval |
| `langchain-openai` | `OpenAIEmbeddings` for LangChain retriever embedding |
