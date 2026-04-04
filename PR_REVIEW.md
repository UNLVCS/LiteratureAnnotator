# PR review: `pinecone-model-options`

This note is a reviewer-facing summary of the branch vs `main`. Use it together with the actual diff (`git diff origin/main...HEAD`).

## Summary

The branch expands **embedding and LLM configuration** (including **Gemini** for embeddings and chat), tightens the **uv workspace** so a root `uv sync` installs member dependencies, adds **LangChain-based embedding factories** and related **utilities**, improves **RAG / Label Studio** behavior (Gemini message content, batch queue drain on API startup), and includes **dependency/lockfile** updates.

## uv workspace (`pyproject.toml`, `uv.lock`)

- **`[tool.uv.sources]`** maps internal package names (`literature-annotator-config`, `literature-annotator-utilities`, etc.) to `{ workspace = true }` so members can depend on each other.
- The **root project lists all workspace members as `dependencies`** so `uv sync` installs the **union** of subpackage dependencies into the root `.venv` (avoids “missing `pydantic`” when only the root was synced with empty deps).

**Review focus:** Confirm lockfile and member list match how you deploy (CI, Docker).

## Config (`config/`)

- **`EmbeddingsConfig`** documents **voyage / openai / gemini** and related fields (`provider`, `api_key`, `model`, `dimensions`).
- **`LLMProvidersDictMixin.get_providers_dict()`** (`config/llm_providers_config.py`):
  - Instantiates **`GeminiProvider`** when `llm_providers.gemini` is configured (no provider-type-specific key checks in the mixin).
- **`GeminiProvider`**: raises **`ValueError`** in **`__init__`** if **`api_key`** is missing or blank (configure under **`llm_providers.gemini`** or per-model).
- **`LLMModelConfig.to_provider_kwargs`**: ensures **`api_key`** is always present in merged kwargs (default **`""`**) so providers can validate consistently.

**Review focus:** Ensure `.env.yaml` / secrets are **not** committed; use `.env.yaml.example` only for structure.

## Embeddings (`utilities/langchain_embeddings.py`, `vector_db.py`)

- Factory **`build_langchain_embeddings(EmbeddingsConfig)`** returns LangChain `Embeddings` for **Voyage** (custom wrapper), **OpenAI**, or **Gemini** (`GoogleGenerativeAIEmbeddings`).
- Pinecone / vector helpers respect configured **dimensions** where applicable.

## LLM providers (`llm_providers/`)

- New **`GeminiProvider`** using **`ChatGoogleGenerativeAI`** (`langchain-google-genai`), default chat model **`gemini-2.0-flash`** (override via per-model config).
- **`langchain-google-genai`** added to `llm_providers` dependencies.
- **`__init__.py`** exports `GeminiProvider`.

**Review focus:** Chat defaults and model IDs match your Google AI project and quotas.

## RAG pipelines (`data_generation/`)

- **`response_standardizer.py`**: **`coerce_llm_content_to_str()`** — Gemini / LangChain may return **`AIMessage.content` as a list**; content is flattened before `.strip()` / JSON parse (fixes `'list' object has no attribute 'strip'`).
- **`rag_labeling_script.py` / `rag_labeling_script_mp.py`**: No env-based throttling; rely on normal error handling.

## Label Studio API (`label_api/legacy_main.py`)

- **`import_all_pending_paper_tasks(project_id, max_rounds=500)`** drains the **main paper Redis queue** on startup (parallel to existing **`import_all_pending_human_tasks`** for the human queue).
- Startup calls the batch importer so seeding the queue does not wait for the periodic job alone.

**Review focus:** Large queues may create many LS tasks at once — acceptable for your UI/API limits?

## Utilities

- **`utilities/clear_dedup_sets.py`**: Deletes **`paper_dedup_set`** and **`human_dedup_set`** keys in Redis using app config (re-enqueue after clearing dedup).
- **`utilities/__init__.py`**, **`test_papers.txt`**, **`human_papers.txt`**: Adjust as needed for local testing.

## Docs / examples

- **`.env.yaml.example`**: Updated for embeddings options and **`llm_providers.gemini`** example block.

## Suggested reviewer checklist

- [ ] **Secrets:** No API keys in git; rotate any key that was ever exposed.
- [ ] **uv:** `uv lock` / `uv sync` succeed; app runs with root `.venv`.
- [ ] **Gemini:** Embeddings + chat work with your key; quotas sufficient for RAG + LS import volume.
- [ ] **Label Studio:** After API restart, batch import behavior matches expectations (queue drained up to `max_rounds`).
- [ ] **Redis / MinIO:** Classified JSON paths `{model}/{paper_id}.json` align with `llm_providers` non-skipped models.
- [ ] **Downstream packages:** `data_analysis`, `data_generation`, `label_api` dependency bumps are intentional.

## Files touched (high level)

| Area | Paths |
|------|--------|
| Workspace | `pyproject.toml`, `uv.lock` |
| Config | `config/app_config.py`, `config/llm_providers_config.py`, `.env.yaml.example` |
| LLM | `llm_providers/*` (incl. `gemini_provider.py`) |
| Embeddings / vectors | `utilities/langchain_embeddings.py`, `utilities/vector_db.py` |
| RAG | `data_generation/rag_labeling_script*.py`, `data_generation/response_standardizer.py` |
| Label API | `label_api/legacy_main.py`, `label_api/human_import.py` |
| Utilities | `utilities/clear_dedup_sets.py`, `utilities/__init__.py`, test data files |
| Other | `data_analysis/semantic_entropy/generate_samples.py`, assorted `pyproject.toml` / `requirements.txt` |

---

*Generated for hand-off review; adjust sections if the PR scope changes before merge.*
