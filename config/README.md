# `config` — Application Configuration

A unified, YAML-based configuration system for LiteratureAnnotator, built on [Pydantic v2](https://docs.pydantic.dev/latest/).

---

## Overview

All runtime settings live in a single file, `.env.yaml`, at the project root. The config package validates that file into a typed `AppConfig` object, making every setting accessible with IDE autocompletion and full type safety.

```
project root/
├── .env.yaml                      ← your local config (gitignored)
├── .env.yaml.example              ← template — copy and fill in secrets
├── override.env.yaml              ← optional: deep-merged on top of .env.yaml
└── config/
    ├── __init__.py
    ├── app_config.py              ← AppConfig + sub-configs + load_app_config()
    ├── base.py                    ← generic YAML/JSON loaders
    └── llm_providers_config.py    ← LLM provider & model config models
```

---

## Quick Start

```bash
# 1. Copy the example file and fill in your secrets
cp .env.yaml.example .env.yaml
```

```python
from config import load_app_config

config = load_app_config()

# Access nested settings
print(config.redis.url)
print(config.minio.access_key)
print(config.pinecone.index_name)

# Get instantiated LLM provider objects
providers = config.get_providers_dict()  # Dict[str, BaseLLMProvider]
```

The config is **cached** after the first call — subsequent `load_app_config()` calls return the same instance with no file I/O.

---

## `AppConfig`

`AppConfig` is the root Pydantic model. It composes all sub-configs as nested fields.

```python
class AppConfig(LLMProvidersDictMixin, BaseModel):
    redis:         RedisConfig
    minio:         MinioConfig
    pinecone:      PineconeConfig
    label_studio:  LabelStudioConfig
    embeddings:    EmbeddingsConfig
    seed:          SeedConfig
    llm_providers: Dict[str, LLMProviderConfig]
```

---

## Sub-Configs

### `RedisConfig`

Job queues, deduplication sets, and annotation buffering.

| Field | Default | Description |
|---|---|---|
| `url` | `redis://localhost:6379/0` | Redis connection URL |
| `paper_queue` | `q:papers:v1` | Main paper pipeline queue |
| `paper_processing` | `q:papers:processing:v1` | In-flight papers (claim/ack pattern) |
| `paper_dedup_set` | `s:papers:enqueued:v1` | SET used to prevent duplicate enqueues |
| `completed_papers_queue` | `q:papers:completed:v1` | Bookkeeping list for finished paper IDs |
| `generated_set` | `s:papers:generated:v1` | SET recorded after RAG batch generation |
| `ann_queue` | `q:annotations:completed:v1` | Buffer for completed LS annotations |
| `ann_flush_threshold` | `1000` | Drain to disk when queue exceeds this length |
| `ann_persist_path` | `data_labeling/annotations.jsonl` | Append-only JSONL flush target |
| `ann_flush_on_exit` | `true` | Register `atexit` handler to flush on exit |
| `ann_install_signal_handlers` | `true` | Flush on `SIGINT`/`SIGTERM` then exit |
| `human_paper_queue` | `q:papers:human:v1` | Human-labeling pipeline queue |
| `human_processing_queue` | `q:papers:human:processing:v1` | In-flight items in human pipeline |
| `human_dedup_set` | `s:papers:human:enqueued:v1` | Dedup SET for human pipeline |

---

### `MinioConfig`

S3-compatible object storage for raw articles, classification results, and exported annotations.

| Field | Default | Description |
|---|---|---|
| `url` | `localhost:9000` | Host:port (no scheme) |
| `access_key` | `""` | S3 access key (`MINIO_ROOT_USER` or created user) |
| `secret_key` | `""` | S3 secret key |
| `bucket_name` | `v1-criteria-classified-articles` | RAG classification results bucket |
| `raw_articles_bucket` | `raw-pubmed-articles` | Source articles — single source of truth used by downloader, ingester, and queue seeder |
| `secure` | `false` | Enable HTTPS / TLS verification |
| `human_annotations_bucket` | `human-annotations` | Completed human LS annotations |
| `annotations_bucket` | `completed-annotations` | Completed RAG-annotation tasks |

---

### `PineconeConfig`

Pinecone vector index for RAG retrieval.

| Field | Default | Description |
|---|---|---|
| `api_key` | `""` | Pinecone console API key |
| `index_name` | `adbm` | Index name (must match dimension in Pinecone) |
| `namespace` | `article_upload_test_2` | Namespace for retriever queries |

---

### `LabelStudioConfig`

Label Studio annotation UI connection settings.

| Field | Default | Description |
|---|---|---|
| `url` | `""` | Base URL of Label Studio, e.g. `http://localhost:8080` |
| `api_key` | `""` | Personal or project access token |
| `webhook_host` | `http://localhost:8000` | Base URL of *this* FastAPI server as reachable from LS |

The class also exposes `label_studio_url` and `label_studio_api_key` as property aliases for compatibility with the `LabellerSDK` protocol.

---

### `EmbeddingsConfig`

OpenAI-compatible text embeddings used by RAG scripts and human-import chunk retrieval.

| Field | Default | Description |
|---|---|---|
| `api_key` | `""` | OpenAI API key (or compatible provider) |
| `model` | `text-embedding-ada-002` | Embedding model ID |
| `dimensions` | `1536` | Vector dimension — must match Pinecone index |

---

### `SeedConfig`

Default input files for the queue-seeding utilities.

| Field | Default | Description |
|---|---|---|
| `queue_file` | `utilities/test_papers.txt` | One paper ID per line → main RAG queue |
| `human_papers_file` | `utilities/human_papers.txt` | One paper ID per line → human queue |

---

## LLM Provider Configuration

LLM backends are configured under the `llm_providers` key. Each entry maps a **provider type** to a `LLMProviderConfig`.

### Supported Providers

| Provider key | Class instantiated |
|---|---|
| `openai` | `OpenAIProvider` |
| `anthropic` | `AnthropicProvider` |
| `huggingface` | `HuggingFaceProvider` |
| `vllm` | `VLLMProvider` (OpenAI-compatible) |
| `vllm_native` | `VLLMNativeProvider` |
| `ollama` | `OllamaProvider` (health-checked on startup) |

### `LLMProviderConfig`

```yaml
llm_providers:
  openai:
    api_key: "sk-..."        # provider-level key; used unless model overrides
    models:
      - model: gpt-4o
        temperature: 0.1
        skip: false
```

### `LLMModelConfig` fields

| Field | Default | Description |
|---|---|---|
| `model` | _(required)_ | Model ID string; also used as MinIO prefix |
| `temperature` | `0.7` | Sampling temperature |
| `skip` | `false` | If `true`, this model is excluded from `get_providers_dict()` |
| `api_key` | `None` | Per-model key override (rare) |
| `max_tokens` | `None` | Max completion tokens |
| `top_p` | `1.0` | Nucleus sampling probability |
| `frequency_penalty` | `0.0` | Frequency penalty |
| `presence_penalty` | `0.0` | Presence penalty |
| `base_url` | `None` | Base URL for Ollama / vLLM / compatible endpoints |
| `timeout` | `None` | HTTP timeout (seconds) |
| `node_name` | `None` | Optional cluster node label |
| `node_port` | `None` | Optional cluster node port |
| `think` | `None` | Enable chain-of-thought/thinking mode if supported |
| `options` | `None` | Arbitrary extra options passed to the provider |

Extra YAML keys are allowed (`extra="allow"`) for forward compatibility.

### Getting Instantiated Providers

```python
config = load_app_config()
providers = config.get_providers_dict()  # Dict[model_id, BaseLLMProvider]

llm = providers["gpt-4o"]
response = llm.complete("Summarise this paper...")
```

Models with `skip: true` or a missing `api_key` (for cloud providers) are silently excluded. Ollama models are excluded if the local server is not reachable.

---

## Override / Docker Workflow

If `override.env.yaml` exists next to `.env.yaml`, its keys are **deep-merged** on top of the base config. This is useful for Docker, where you can mount environment-specific overrides without touching the base file:

```yaml
# override.env.yaml (example)
redis:
  url: "redis://redis-container:6379/0"
minio:
  url: "minio-container:9000"
  secure: false
```

See `.env.docker-override.yaml.example` for a full Docker override template.

---

## Loading Config Programmatically

```python
from config import load_app_config, get_app_config, AppConfig
from config import load_config_from_yaml_file, load_config_from_json_file

# Standard usage — reads .env.yaml, caches result
config: AppConfig = load_app_config()

# Force a fresh reload (e.g. in tests)
config = load_app_config(reload=True)

# Load from a custom path
config = load_app_config(config_path="/path/to/custom.yaml")

# Load any Pydantic model from a YAML or JSON file
from config import AppConfig
config = load_config_from_yaml_file(AppConfig, "path/to/file.yaml")
```

---

## Error Handling

- **`FileNotFoundError`** — raised by `load_app_config()` if `.env.yaml` is missing. Copy `.env.yaml.example` and fill in your values.
- **`ConfigValidationError`** — alias for `pydantic.ValidationError`. Raised when the YAML contains a value that fails type validation.

---

## Dependencies

Defined in `config/pyproject.toml`:

| Package | Minimum version |
|---|---|
| `pydantic` | `>=2.0.0` |
| `pyyaml` | `>=6.0` |
| `typing-extensions` | `>=4.0.0` |
