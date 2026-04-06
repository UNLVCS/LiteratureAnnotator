# LiteratureAnnotator

A uv workspace containing multiple sub-packages for downloading, labeling, and analyzing biomedical literature.

## Workspace layout

```
LiteratureAnnotator/          ← workspace root (virtual, package = false)
├── config/                   ← installable: shared Pydantic config models
├── llm_providers/            ← installable: unified LLM provider abstraction
├── data_download/            ← scripts: PubMed BioC downloader
├── data_generation/          ← scripts: RAG-based labeling pipeline
├── data_analysis/            ← scripts: sample generation & entropy analysis
├── label_api/                ← scripts: FastAPI + Label Studio webhook server
├── utilities/                ← scripts: queue and vector DB helpers
└── weak_supervision/         ← scripts: Snorkel weak-supervision experiments
```

**Installable members** (`config`, `llm_providers`) have a build backend and are imported by other code.  
**Script members** (everything else) have `package = false` — their code is run directly, not imported as a package.

## Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

All commands run from the **workspace root** (`LiteratureAnnotator/`). uv manages a single shared `.venv` here.

### Install deps for all workspace members

```bash
uv sync --all-packages
```

This installs the dependencies of every member into one `.venv`. Use this when you need to run scripts from multiple sub-packages in the same session.

### Install only the core installable packages

```bash
uv sync
```

This installs just `config` and `llm_providers` (the two installable members the root depends on). Sufficient for tasks that only import from those two packages.

### Install deps for a single script package

```bash
uv sync --package literature-annotator-data-generation
```

Use any of the member package names (see `[tool.uv.workspace]` in `pyproject.toml`).

### Include the SLURM / vLLM group

vLLM is kept out of default deps because it is large and GPU-only. Add it with:

```bash
uv sync --all-packages --group slurm
```

Or combine with a single package:

```bash
uv sync --package literature-annotator-data-analysis --group slurm
```

## Running scripts

Prefix any script invocation with `uv run` so it uses the workspace `.venv` automatically — no need to activate it manually.

```bash
# Example: run a script inside data_generation/
uv run python data_generation/some_script.py

# Example: run a module
uv run python -m data_download.pubmed_downloader

# Example: launch the label API server
uv run uvicorn label_api.main:app --reload
```

`uv run` always resolves against the workspace `.venv`, regardless of which directory you invoke it from.

## Adding a new dependency

Add to a specific member's `pyproject.toml`:

```bash
uv add <package> --package literature-annotator-data-generation
```

Add to the root (for workspace-wide deps):

```bash
uv add <package>
```

After adding, `uv sync` (or `uv sync --all-packages`) updates the lockfile and `.venv`.

## Dependency groups

| Group   | Install flag          | Contents                      |
|---------|-----------------------|-------------------------------|
| slurm   | `--group slurm`       | `vllm`, `pyyaml` for SLURM jobs |

## Optional extras

Some members expose optional dependency sets:

| Member             | Extra            | Contents                                        |
|--------------------|------------------|-------------------------------------------------|
| `llm_providers`    | `local-models`   | `torch`, `transformers`, `accelerate`, `vllm`  |
| `llm_providers`    | `runtime`        | `minio`, `redis`                                |
| `data_generation`  | `providers`      | `langchain-anthropic`, `langchain-huggingface`  |
| `data_analysis`    | `providers`      | `langchain-anthropic`, `langchain-huggingface`, `vllm` |

Install an extra:

```bash
uv sync --package llm-providers --extra local-models
```
