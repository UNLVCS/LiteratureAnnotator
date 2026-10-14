# data_generation

Generates labeled training data by evaluating research papers against inclusion/exclusion
criteria using LLM providers. Papers are pulled from a Redis queue, their **full text** is
loaded from the raw BioC JSON in MinIO (via [`papers`](../papers/README.md)), and each
criterion is evaluated by one or more LLMs. Models cite verbatim evidence quotes, which are
resolved back to the source text. Results are written to MinIO.

> The earlier RAG path (Pinecone retrieval of the top-5 chunks per criterion) was retired:
> retrieval was always scoped to a single paper, so it could only hide evidence, and it
> cannot establish absence-type criteria (C1 original research, C5 no animal models).
> The last RAG version is tagged `rag-final`.

## Pipeline overview

```
Redis queue
    │
    ▼
claim paper ID
    │
    ▼
MinIO raw-pubmed-articles/{prefix}/{pmid}.json
  parse BioC (papers.parse_bioc): keep offsets, drop References
    │
    ▼
render whole paper with "## <section>" headers
    │
    ▼
LLM providers (via llm_providers)
  one call per criterion per provider, whole paper each time
    │
    ▼
response_standardizer → JSON;  papers.resolve_evidence → quote offsets
    │
    ▼
MinIO  ──  {output_prefix}/{provider}/{paper_id}.json
    │
    ▼
Redis  ──  ack (success / unlabelable paper) or requeue (provider failure)
```

## Files

| File | Purpose |
|---|---|
| `labeling.py` | Shared core: `label_paper()` builds the per-criterion queries over the full paper, parses answers, resolves evidence. |
| `labeler.py` | Sequential labeler — one paper at a time, writes a local JSON file. Good for debugging. |
| `labeler_mp.py` | Multiprocessing labeler — one worker process per LLM provider, writes to MinIO. Use this in production. |
| `response_standardizer.py` | Cleans LLM responses (strips ` ```json ` fences, extracts embedded JSON) and parses them into dicts. |
| `tests/` | `pytest` tests for `labeling.py` using a fake provider (no network). |

## Usage

The raw articles must already be in MinIO (`data_download`), and the paper IDs queued
(`utilities.seed_queue`).

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

### Tests

```bash
uv run pytest data_generation/tests papers/tests
```

## Paper context and evidence

- **Whole paper, per criterion.** Each criterion is a separate call over the full rendered
  paper, so one criterion's reasoning cannot leak into another's. The paper comes first in
  the prompt and the task last, which keeps the shared prefix cacheable across the 6 calls.
- **What the model sees.** Passages in `papers.exclude_section_types` (References, author
  contributions, competing interests, funding) are removed. Section labels come from the
  paper's own headings. Abstract-only papers (PubMed fallback) are labeled from title +
  abstract; `paper.is_full_text` records which case applied.
- **Length.** The whole paper is sent unless `papers.max_chars` is set; truncation is
  recorded in `paper.truncated`. Typical full texts are 40k–90k characters (~10k–25k tokens).
- **Evidence.** Models return `evidence: [{section, quote}]`. Each quote is matched against
  the source passages (whitespace, case, typographic quotes and dashes normalized; `...`
  splits a quote into fragments that must appear in order). Matched quotes get BioC
  `start`/`end` offsets; unmatched quotes are kept with `matched: false`, which marks a
  hallucinated or paraphrased citation.
- **Failures.** A provider failure puts the paper in `errors` and requeues it. A paper that
  is missing from MinIO or empty after parsing is acked and skipped (retrying cannot help).
  A malformed answer is saved with an `error` on that criterion only.

## LLM providers

All providers implement the same `BaseLLMProvider` interface from `llm_providers`.
The labelers call `provider.call_api_batch(queries)` to send all 6 criterion queries
(each carrying the whole paper) in one shot. The batching mechanism differs per provider:

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
minio:
  url: "localhost:9000"
  access_key: "..."
  secret_key: "..."
  synthetic_data_bucket: "synthetic-data"      # labeling output
  raw_articles_bucket: "raw-pubmed-articles"   # raw BioC input
  output_prefix: ""                            # optional path prefix within the output bucket

papers:
  # raw_prefix defaults to bioc_download.object_prefix
  exclude_section_types: ["REF", "AUTH_CONT", "COMP_INT", "ACK_FUND"]
  max_chars: null                              # null = whole paper

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
        max_model_len: 32768                   # must fit the whole paper
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
  "schema_version": "whole-paper-v2",
  "paper_id": "41646005",
  "provider": "gpt-4o-mini",
  "title": "Current potential biomarkers for Alzheimer's disease...",
  "labeled_at": "2026-10-13T03:00:00+00:00",
  "paper": {
    "source": "PMC", "source_id": "PMC12885010", "is_full_text": true,
    "rendered_chars": 81919, "truncated": false,
    "passages_included": 74, "passages_total": 74
  },
  "criteria_results": [
    {
      "criterion": "criterion_1",
      "prompt": "Criterion 1 – Original Research ...",
      "response": {"criterion_1": {"satisfied": false, "reason": "...", "evidence": [...]}},
      "raw_response": "...",
      "cleaned_response": "...",
      "error": null,
      "evidence": [
        {
          "quote": "This review systematically analysed the potential biomarkers",
          "cited_section": "Abstract",
          "matched": true, "start": 483, "end": 543,
          "section": "Abstract", "section_agrees": true
        }
      ],
      "finish_reason": "end_turn",
      "usage": {"input_tokens": 312, "output_tokens": 140, "cache_read_input_tokens": 11873, "...": "..."}
    }
  ],
  "errors": []
}
```

`satisfied` is `true` or `false`, or `"not_applicable"` for the conditional criteria
(3: sample size, 6: blood as biomarker) when the condition does not hold, e.g. sample
size for a mouse study. Variants such as `"N/A"` are normalized; any other value is
recorded as a per-criterion `error`. `whole-paper-v1` records predate this and only
have booleans.

Results written before the retirement (RAG era, no `schema_version`) have `chunks_used`
and `full_context` instead of `paper` and `evidence`.

## Dependencies

| Package | Role |
|---|---|
| `papers` | Raw BioC loading from MinIO, rendering, quote resolution |
| `utilities` | Redis queue helpers (`queue_helpers`) and criteria prompts (`criteria`) |
| `llm_providers` | Provider abstraction (`BaseLLMProvider`, `Query`, all provider classes) |
| `config` | `.env.yaml` config models; `instantiate_provider` used by worker processes |
