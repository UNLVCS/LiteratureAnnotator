# LiteratureAnnotator

LiteratureAnnotator is a pipeline for building labeled training data from biomedical
literature. It downloads papers from PubMed, uses LLMs to evaluate them against a set of
inclusion/exclusion criteria (RAG-style, grounded in retrieved chunks), routes the results
through human review in Label Studio, and produces a labeled dataset for downstream model
training — with Snorkel-based weak supervision available for combining noisy label sources.

At a high level, papers flow through the system like this:

```
PubMed (BioC API)
      │
      ▼
 data_download  ──────────►  MinIO (raw articles)
                                    │
                                    ▼
                              data_vectorize  ──────────►  Pinecone (chunked + embedded)
                                    │
                                    ▼
                    Redis queue ◄──────────► data_generation (LLM labeling via llm_providers)
                                    │
                                    ▼
                              label_api (FastAPI + Label Studio) ◄── human review
                                    │
                                    ▼
                              MinIO (labeled results)
                                    │
                                    ▼
                              weak_supervision / data_analysis
```

`config` (shared settings) and `llm_providers` (a unified LLM abstraction) are used
throughout the pipeline. Secrets (API keys, DB credentials) are managed by Vault; see
[vault/README.md](vault/README.md).

## Components

| Component | Purpose |
|---|---|
| [`config`](config/README.md) | Shared, YAML-based Pydantic config system used by every other package. |
| [`llm_providers`](llm_providers) | Unified abstraction over Anthropic, OpenAI, HuggingFace, Ollama, and vLLM. |
| [`data_download`](data_download/README.md) | Downloads PubMed articles via the NCBI BioC API into MinIO. |
| [`data_vectorize`](data_vectorize/README.md) | Chunks and embeds articles, and handles Pinecone retrieval/ingestion. |
| [`data_generation`](data_generation/README.md) | RAG labeling pipeline: evaluates papers against criteria using LLMs. |
| [`label_api`](label_api/README.md) | FastAPI service bridging the pipeline to Label Studio for human review. |
| [`utilities`](utilities/README.md) | Redis queue helpers, queue seeding, and status monitoring scripts. |
| [`data_analysis`](data_analysis) | Sample generation and semantic entropy analysis over labeling results. |
| `weak_supervision` | Snorkel-based weak supervision experiments for combining noisy labels. |
| [`vault`](vault/README.md) | Vault runbook — where secrets live and how services obtain them. |
| `ops` | Operational scripts (e.g. Vault bootstrap/maintenance). |

## Getting started

This repo is a [uv](https://docs.astral.sh/uv/) workspace: one shared virtual environment
at the root, with each component as a workspace member. See [USAGE.md](USAGE.md) for the
full guide to installing dependencies, running scripts, and adding new packages.

Quick start:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv
uv sync --all-packages                            # install every component's deps
cp .env.yaml.example .env.yaml                    # fill in secrets/config
uv run uvicorn label_api.app:app --reload         # e.g. launch the label API
```

Each component directory has its own README with setup and usage details specific to that
piece of the pipeline.
