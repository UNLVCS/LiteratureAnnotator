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

## LLM providers

All providers implement the same `BaseLLMProvider` interface from `llm_providers`.
The labelers call `provider.call_api_batch(queries)` to send all 6 criterion queries
for a paper in one shot. The batching mechanism differs per provider:

| Provider | Type key | `call_api_batch` mechanism |
|---|---|---|
| `OpenAIProvider` | `openai` | LangChain `llm.batch()` — parallel threads, results in order. Retries with exponential backoff on rate-limit / 429 errors. |
| `AnthropicProvider` | `anthropic` | LangChain `llm.batch()` — parallel threads. Retries with exponential backoff. Automatically drops `top_p` when `temperature` is set (Claude 4+ constraint). |
| `VLLMProvider` | `vllm` | LangChain `llm.batch()` — parallel HTTP requests to a running vLLM server. |
| `VLLMNativeProvider` | `vllm_native` | `llm.generate(prompts)` — **true GPU batch**: all prompts submitted in one call so vLLM can maximally fill its KV-cache. Best throughput for local models. |
| `OllamaProvider` | `ollama` | `ThreadPoolExecutor` — up to 6 concurrent HTTP requests to the local Ollama server. Only registered if `check_server_status()` returns True at startup. |
| `HuggingFaceProvider` | `huggingface` | LangChain `llm.batch()` — batched inference via HuggingFace pipeline. |

The base class default for providers that don't override `call_api_batch` is a
sequential loop over `call_api` — so any future provider is safe without
implementing batching explicitly.

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
  output_prefix: ""                           # optional path prefix within the output bucket

redis:
  url: "redis://localhost:6379/0"
  paper_queue: "q:papers:v1"

llm_providers:
  openai:
    api_key: "sk-..."
    models:
      - model: "gpt-4o-mini"
  anthropic:
    api_key: "sk-ant-..."
    models:
      - model: "claude-sonnet-4-6"
  vllm_native:
    models:
      - model: "meta-llama/Llama-3.1-8B-Instruct"
        tensor_parallel_size: 1
        gpu_memory_utilization: 0.90
  ollama:
    models:
      - model: "llama3.1"
        base_url: "http://localhost:11434"
```

## Output format

Each result is a JSON file at `{output_prefix}/{provider}/{paper_id}.json` in
`minio.synthetic_data_bucket`:

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
| `llm_providers` | Provider abstraction (`BaseLLMProvider`, `Query`, all provider classes) |
| `config` | `.env.yaml` config models; `instantiate_provider` used by worker processes |
| `langchain-pinecone` | `PineconeVectorStore` for metadata-filtered retrieval |
| `langchain-openai` | `OpenAIEmbeddings` for LangChain retriever embedding |
