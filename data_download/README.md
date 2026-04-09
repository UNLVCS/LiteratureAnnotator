# Data Download — PubMed BioC Article Downloader

Download PubMed articles via the NCBI BioC RESTful API and store them in MinIO.

## Setup

Dependencies are declared in `data_download/pyproject.toml` and managed by [uv](https://docs.astral.sh/uv/).
Run this once from the **workspace root** to install them into the shared `.venv`:

```bash
uv sync --package literature-annotator-data-download
```

No need to activate the venv — prefix commands with `uv run` and it handles that automatically.

## Quick Start

```bash
# 1. Discover MeSH terms and qualifiers for your topic
uv run python -m data_download.mesh_lookup "Alzheimer Disease" --subheadings

# 2. Preview how many papers different term combinations match
uv run python -m data_download.mesh_lookup --preview \
    "alzheimer disease/blood" "biomarkers/blood"

# 3. Download articles to MinIO
uv run python -m data_download.bioc_downloader \
    --terms "alzheimer disease/blood,biomarkers/blood" --max-results 100
```

## Step-by-Step Guide

### Step 1: Discover MeSH Terms

MeSH (Medical Subject Headings) is NCBI's controlled vocabulary for indexing
PubMed articles. Every PubMed article is tagged with MeSH descriptors that
describe its topics.

Search for descriptors by keyword:

```bash
uv run python -m data_download.mesh_lookup "Alzheimer Disease" -v
```

This returns matching descriptors with their UIDs, scope notes, and synonyms:

```
  D000544  Alzheimer Disease
           A degenerative disease of the BRAIN characterized by…
           aka: Alzheimer Syndrome, Alzheimer-Type Dementia (ATD), …
           tree: C10.228.140.380.100, C10.574.945.249, F03.615.400.100
```

To also discover related terms (e.g. amyloid, tau):

```bash
uv run python -m data_download.mesh_lookup "Alzheimer Disease" --related -v
```

### Step 2: Find Subheading Qualifiers

Each MeSH descriptor has **subheading qualifiers** that specify the aspect of
the topic discussed in a paper. This is what makes your search specific.

```bash
uv run python -m data_download.mesh_lookup "Alzheimer Disease" --subheadings
```

Output:

```
  D000544  Alzheimer Disease
           qualifiers: blood, cerebrospinal fluid, diagnosis,
                       drug therapy, epidemiology, genetics,
                       metabolism, pathology, ...
```

Common qualifiers for biomarker research:

| Qualifier | Meaning |
|---|---|
| `/blood` | Blood-based measurements (serum, plasma) |
| `/cerebrospinal fluid` | CSF-based measurements |
| `/diagnosis` | Diagnostic methods and criteria |
| `/metabolism` | Metabolic pathways and processes |
| `/immunology` | Immune response and antibodies |
| `/genetics` | Genetic aspects |

Attach a qualifier to a term with `/` to narrow your search. For example,
`alzheimer disease/blood` restricts to papers where blood-related aspects
of Alzheimer Disease are discussed.

### Step 3: Preview Hit Counts

Before downloading, preview how many papers your query will return.
This lets you iterate on your terms until the result set is the right size.

```bash
uv run python -m data_download.mesh_lookup --preview \
    "alzheimer disease/blood" "biomarkers/blood"
```

Output:

```
  PubMed hit-count preview
  ────────────────────────────────────────────────────────────
     2,231  Broad  [MeSH Terms]
       535  Focused  [MAJR]
     3,235    solo: alzheimer disease/blood
    64,195    solo: biomarkers/blood
```

**`[MAJR]` vs `[MeSH Terms]`**: By default the downloader uses `[MAJR]`
(Major MeSH Heading), which only returns papers where the term is a *central*
focus — not just mentioned in passing. This is almost always what you want.
The preview shows you both counts so you can compare.

Try different combinations to dial in your specificity:

```bash
# Very focused: AD blood + amyloid/tau blood
uv run python -m data_download.mesh_lookup --preview \
    "alzheimer disease/blood" \
    "amyloid beta-peptides/blood" \
    "tau proteins/blood"

# Broader: AD diagnosis + blood biomarkers
uv run python -m data_download.mesh_lookup --preview \
    "alzheimer disease/diagnosis" \
    "biomarkers/blood"
```

### Step 4: Download Articles

Once you've settled on your terms, download the articles.

**Option A — Qualified terms (recommended)**

Terms are AND-joined and wrapped with `[MAJR]` automatically:

```bash
uv run python -m data_download.bioc_downloader \
    --terms "alzheimer disease/blood,biomarkers/blood" \
    --max-results 100
```

**Option B — Raw PubMed query (full control)**

For OR groups or complex boolean logic, pass the query string directly:

```bash
uv run python -m data_download.bioc_downloader --query \
    '"alzheimer disease/blood"[MAJR] AND ("amyloid beta-peptides/blood"[MAJR] OR "tau proteins/blood"[MAJR])'
```

**Option C — MeSH descriptor UIDs**

If you already know the UIDs:

```bash
uv run python -m data_download.bioc_downloader --mesh D000544,D015415
```

**Preview before downloading:**

Add `--preview` to any of the above to see the query and hit count without
actually downloading:

```bash
uv run python -m data_download.bioc_downloader \
    --terms "alzheimer disease/blood,biomarkers/blood" --preview
```

### Step 5: Verify in MinIO

Downloaded articles are stored as JSON files in your MinIO bucket under
`bioc_articles/{PMID}.json`. The bucket name is configured via the
`BIOC_DOWNLOAD_BUCKET` env var (default: `raw-pubmed-articles`).

## Configuration

All settings come from `.env.yaml` in the workspace root — the same file used by
the rest of the project. The downloader reads env vars at startup
(`BioCDownloadConfig` via pydantic-settings); the table below shows exactly which
`.env.yaml` keys map to which env vars.

### First-time setup

```bash
# from the workspace root
cp .env.yaml.example .env.yaml
# edit .env.yaml and fill in your values
```

### MinIO connection — required

Fill in the `minio:` block of `.env.yaml` (shared with the rest of the project):

```yaml
minio:
  url: "localhost:9000"
  access_key: "minioadmin"
  secret_key: "minioadmin"
  secure: false
  raw_articles_bucket: "raw-pubmed-articles"
```

### Download settings — `bioc_download` block

All BioC-specific settings live under the `bioc_download:` block. The defaults
work out of the box; set `mesh_terms` or `mesh_query` for your topic:

```yaml
bioc_download:
  # Bucket comes from minio.raw_articles_bucket — no separate download_bucket field.
  object_prefix: "run_2026_04"  # optional — organises objects into a sub-folder
  ncbi_api_key: ""          # optional — raises NCBI rate limit to 10 req/s
  ncbi_email: ""            # optional — recommended by NCBI usage policy
  mesh_query: ""            # raw PubMed query string (highest priority)
  mesh_terms:               # auto-joined with AND + wrapped with [MAJR]
    - "alzheimer disease/blood"
    - "biomarkers/blood"
  mesh_ids: []              # MeSH descriptor UIDs, e.g. ["D000544", "D015415"]
  major_topic_only: true
  max_results: 100
  batch_size: 25
  request_delay: 0.34
```

CLI flags (`--query`, `--terms`, `--mesh`, `--max-results`) override the YAML
values for one-off runs without editing the file.

## CLI Reference

### `mesh_lookup`

```
uv run python -m data_download.mesh_lookup [keywords...] [options]

Options:
  -v, --verbose       Show scope notes, synonyms, and tree numbers
  -s, --subheadings   Show available subheading qualifiers
  --related           Expand "see related" descriptors
  -p, --preview       Treat keywords as qualified terms; show PubMed hit counts
  --api-key KEY       NCBI API key
```

### `bioc_downloader`

```
uv run python -m data_download.bioc_downloader [options]

Query options (pick one or combine):
  --query QUERY       Raw PubMed query string
  --terms TERMS       Comma-separated qualified MeSH terms
  --mesh IDS          Comma-separated MeSH descriptor UIDs
  --no-major          Use [MeSH Terms] instead of [MAJR]

Other:
  --max-results N     Max PMIDs to retrieve
  --preview           Show query and hit count without downloading
```
