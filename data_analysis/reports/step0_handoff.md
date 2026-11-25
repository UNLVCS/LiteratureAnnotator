# Step 0 answer generation

The study runner uses `config/study_models.yaml` for model selection, settings, sample
prefixes, and batch rates. It reads a local PMID list or JSON manifest with a
`papers` mapping. Each paper has six criterion requests. Results retain the labeler
schema and add batch and sample identifiers plus per-answer generation metadata.
The study writes to `synthetic-data-v3/<sample prefix>/<model ID>/<PMID>.json`.
State is kept separately for each model and sample prefix. Collection records
server-reported costs when available and otherwise prices reported usage with the
configured rates. Sample s01 onward requires collected s00 objects for every
selected PMID. This gate applies before any batch submission.

## Serving constraint

**Live study submission is blocked for all six routes.** OpenRouter Batch supports
`provider.only` and `allow_fallbacks: false` at the batch level. It rejects
`provider.order` and the other synchronous routing preferences, and a
`provider` field in a request body does not pin that batch. There is no documented
batch quantization filter or guarantee that a pinned provider uses a specific
quantization. Direct Anthropic and OpenAI batch requests likewise expose a model
ID but no enforceable serving host or quantization. Copying values into the study
YAML cannot supply these guarantees. [OpenRouter Batch Quickstart](https://openrouter.ai/docs/batch-quickstart#choose-a-provider)
[Anthropic Message Batches](https://platform.claude.com/docs/en/build-with-claude/batch-processing)
[OpenAI Batch API](https://developers.openai.com/api/docs/guides/batch)

The backend uses the documented JSON envelope and paginated results. The
response model, provider, quantization and timestamp are recorded when present.
The batch documentation does not guarantee quantization or a hardware host in
the results. Its completion window is 24 hours and result artifacts are kept
for 30 days. A completed batch can still contain failed requests. Collection
only uploads a paper if all six requests return, and a subsequent submit can
retry the whole paper. [OpenRouter Batch Quickstart](https://openrouter.ai/docs/batch-quickstart)

The prompts retain the shared paper prefix. OpenRouter lists batch cache-read
rates for the Together routes. Cache hits depend on the host and execution order.
The batch quickstart does not document an explicit cache-control mechanism, so
the backend relies on native automatic caching where the route supports it and
records observed cache usage. The cost estimate assumes no cache hits and uses the
configured completion cap. [OpenRouter caching documentation](https://openrouter.ai/docs/features/prompt-caching)

Opus 5.5 uses always-on thinking and the existing backend omits temperature,
top-p and top-k. The paper cache breakpoint is sent via the Messages payload.
The existing OpenAI temperature regex matches `gpt-6.2-sol`, so the request omits
temperature and uses the provider default. The YAML records the requested
temperature of 1 for the study, and metadata records whether it was sent.
[Opus 5.5 migration guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide)
[GPT-6.2 Sol model documentation](https://developers.openai.com/api/docs/models/gpt-6.2-sol)

## Model configuration

The priced routes below are planning candidates. They do not meet the full
pinning requirement. Rates are USD per million tokens.

| Alias | Batch candidate | Public provider slug | Input | Cached input | Output |
| --- | --- | --- | ---: | ---: | ---: |
| opus | claude-opus-5-5 | Anthropic direct | 2 | 0.20 | 10 |
| sol | gpt-6.2-sol | OpenAI direct | 1 | 0.05 | 5 |
| gemini | google/gemini-3.1-pro-preview | google-vertex | 1 | Unverified | 6 |
| deepseek | deepseek/deepseek-v4.1-flash-1017 | together | 0.035 | 0.0029 | 0.149 |
| qwen | qwen/qwen3.8-max-0902 | alibaba | 1.15 | Unverified | 3.45 |
| glm | z-ai/glm-5.3 | together | 0.28 | 0.061 | 1.249 |

DeepSeek Flash is the cost-oriented choice. The dated Qwen Max is a batch-listed
flagship-class snapshot. Together had the lowest listed input and output rates on
the selected DeepSeek and GLM batch pages. The config prices unverified cache
reads at the full input rate for estimates. Recheck prices for the selected
account and context length before launch. Confirm the exact revisions and obtain
enforceable `host` and `quantization` settings before running the study.
[Opus 5.5 pricing](https://platform.claude.com/docs/en/models/opus-5-5/overview)
[GPT-6.2 Sol pricing](https://developers.openai.com/api/docs/models/gpt-6.2-sol)
[Gemini batch model](https://openrouter.ai/google/gemini-3.1-pro-preview:batch)
[DeepSeek batch model](https://openrouter.ai/deepseek/deepseek-v4.1-flash-1017:batch)
[Qwen batch model](https://openrouter.ai/qwen/qwen3.8-max-0902:batch)
[GLM batch model](https://openrouter.ai/z-ai/glm-5.3:batch)

## Commands

Run from the worktree root. Here `pmids.txt` is the selected corpus and
`local-bioc/` contains local `<PMID>.json` BioC objects. To estimate each sample
for one direct model and one OpenRouter model, use the same command with the
appropriate alias. A local dry run uses no credentials or network connection.

```bash
batch() { uv run python data_generation/batch_labeler.py "$@" --study-config config/study_models.yaml; }
for model in opus gemini; do
  for sample in 0 1 2 3 4 5; do
    batch submit --model "$model" --sample "$sample" --pmids-file pmids.txt --papers-dir local-bioc --dry-run
  done
done
```

The following commands describe the live workflow once a route can enforce the
study pins. The current submission guard will stop them. Set `raw_prefix` to the
folder holding the selected BioC papers. To keep smoke outputs and state separate,
first make a smoke config with the same models and sample prefixes under
`study_v1_smoke/`.

```bash
raw_prefix=pilot_2026_10
for model in opus gemini; do
  uv run python data_generation/batch_labeler.py submit --study-config smoke_models.yaml \
    --model "$model" --sample 0 --raw-prefix "$raw_prefix" --pmids-file pmids.txt --limit 3
  uv run python data_generation/batch_labeler.py collect --study-config smoke_models.yaml \
    --model "$model" --sample 0 --wait
done

for model in opus sol gemini deepseek qwen glm; do
  batch run --model "$model" --sample 0 --raw-prefix "$raw_prefix" --pmids-file pmids.txt
  for sample in 1 2 3 4 5; do
    batch run --model "$model" --sample "$sample" --raw-prefix "$raw_prefix" --pmids-file pmids.txt
  done
done
```

Pass `--study-config config/study_models.yaml` to `samples.py`, `entropy.py` and
`semantic_entropy.py` to analyze the study prefixes. The pilot remains available
through the original commands. Study cross-model comparisons use the matching
sample index and both entropy analyses treat all six samples as repeated draws.

## Open questions

- Can the serving providers supply a contract or API that guarantees an exact
  weight revision, quantization and serving route for each batch?
- If the protocol permits a provider-only pin with undisclosed quantization,
  which direct and OpenRouter routes should be approved under that amendment?
- Do OpenRouter's native cache hits and result fields match the study's needs on
  the chosen routes? This requires a separately authorized small batch after
  the serving constraint is resolved.
