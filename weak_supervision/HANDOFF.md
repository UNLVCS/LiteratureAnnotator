# Handoff: labeling functions + weak-supervision model (no gold set yet)

Written 2026-10-20 for the agent picking up `weak_supervision/`. Read this whole file
before writing code.

## 1. The goal (read carefully, this is easy to get wrong)

We are **filtering LLM-generated synthetic training data**, not predicting the truth
of the inclusion criteria.

- One **data point** = one LLM output for one (paper, criterion, provider):
  the verdict (`satisfied`), its `reason`, and its cited `evidence`.
- **Labeling functions (LFs)** look at one data point and vote
  `VALID` / `INVALID` / `ABSTAIN`: is this output a trustworthy training sample?
- The **label model** (Snorkel-style) combines LF votes into a VALID/INVALID
  probability per data point. Downstream, INVALID samples are dropped.

Cross-model agreement is **one LF signal**, not a goal. Do not change prompts or
relabel to make models agree.

A human gold set (subject-matter experts reviewing VALID/INVALID in Label Studio) is
being set up by another agent. Until it exists, build everything so the gold set
plugs in later as a file of `(pmid, criterion, provider, label)`.

## 2. Data you have

Bucket `synthetic-data-v3` in MinIO, objects `<provider>/<pmid>.json`:

| provider (object prefix) | papers | how produced |
|---|---|---|
| `claude-sonnet-5-5` | 679 | Anthropic Message Batches |
| `gpt-6-luna` | 679 | OpenAI Batch API |
| `openai/gpt-oss-120b` | 678 | vLLM on the UNLV GPU cluster (PMID 40855644 missing: too long for 65k context; may be added later) |
| `qwen3.8:27b` | 679 | Ollama on falcon9 |

679 pilot papers × 6 criteria × 4 providers ≈ **16,290 data points**. All records are
`schema_version: whole-paper-v2`. Local copies of the Claude and gpt-6-luna records
are in `logs/results_batch_<model>/` (gitignored); read the others from MinIO.

Record shape (see `data_generation/labeling.py`):

```
paper_id, provider, title, labeled_at, paper{source_id, is_full_text, rendered_chars, truncated, ...},
errors[]                      # call failures; always [] in this bucket
criteria_results[6]:
  criterion                   # "criterion_1".."criterion_6"
  response                    # {"criterion_N": {"satisfied", "reason", "evidence"}} or None
  error                       # None, "Failed to parse JSON", or "Missing or malformed ..." (~20 of 16k)
  raw_response, cleaned_response, finish_reason (None for gpt-oss), usage (keys differ per provider)
  evidence[]: {quote, cited_section, matched, start, end, section, section_agrees}
```

- `satisfied` is `true` / `false`, or `"not_applicable"` for criteria 3 and 6 only
  (`utilities/criteria.py`: `CONDITIONAL_CRITERIA`).
- `matched=false` means the quote was not found verbatim in the paper (hallucinated or
  paraphrased). It is resolved by `papers/evidence.py`.
- Criterion texts: `utilities/criteria.py`. C1 original research, C2 AD focus,
  C3 sample size, C4 protein biomarkers, C5 no animal models, C6 blood as AD biomarker.

Paper metadata: `raw-pubmed-articles/pilot_2026_10/<pmid>.json` (BioC JSON: full text
plus DOI, year and authors, but **no MeSH terms or publication types**). The manifest
`raw-pubmed-articles/manifests/pilot_2026_10.json` gives each paper's sampling stratum
(`papers[pmid].sampled_from`) and each stratum's `expect` prior per criterion.

## 3. What we already measured (use as a sanity baseline)

- Pairwise verdict agreement between providers: 92–95%. All 4 agree on: C1 98%,
  C2 92%, C3 90%, C4 89%, C5 91%, **C6 65%**.
- C6 disagreement is mostly false vs not_applicable (122 cases): blood is mentioned but
  isn't a biomarker (blood-brain barrier, fetal bovine serum). The criterion text
  doesn't cover that case.
- Verbatim-quote match rate: Claude 99.9%, qwen 99.4%, gpt-6-luna 98.4%, gpt-oss 95.8%.
- **Stratum priors are noisy.** 48/150 `core_positive` papers fail C6 by majority, and
  the spot-checked reasons were correct (the search matched "blood" in "blood-brain
  barrier"). Treat `expect` as a weak LF, never as ground truth.

## 4. Suggested work plan

1. **Loader → sample table.** One row per (pmid, criterion, provider) with verdict,
   reason, evidence stats, error, finish_reason, stratum, plus the other providers'
   verdicts. Save a local Parquet/CSV snapshot so iteration doesn't hit MinIO each time.
2. **LFs** (starting list below). Each is a small pure function over a row, with a
   unit test. Report coverage / overlap / conflict per LF (Snorkel `LFAnalysis`).
3. **Label model** over the LF matrix (2 classes + abstain). Inspect the learned LF
   accuracies and the share of samples flagged INVALID, per provider and per criterion.
4. **Evaluation without gold** (be explicit in the write-up that these are proxies):
   - Spot-check: print about 30 samples the model scores most confidently INVALID and
     about 30 it scores VALID with their reason and evidence, for a human to read.
   - Stability: retrain with each LF left out; samples whose label flips are fragile.
   - Plausibility: e.g. INVALID rate should be higher for unmatched-quote samples and
     for the noisiest provider (gpt-oss by quote matching).
5. **Gold-set hook:** a function that loads `(pmid, criterion, provider, label)` and
   reports accuracy / F1 / calibration of LFs and label model. Leave it ready for when
   the SME data arrives.

## 5. Starting LF catalog

Votes are about the **sample**, not the criterion.

| LF | Votes | Notes |
|---|---|---|
| `parse_error` | INVALID if `error` is set or `response` is None | Clear-cut |
| `unmatched_quote` | INVALID if any evidence quote `matched=false` | Strongest per-sample signal |
| `quotes_all_matched` | VALID if ≥1 quote and all matched, else ABSTAIN | Rules allow empty evidence for "absence" decisions, so empty evidence must ABSTAIN, not INVALID |
| `section_mismatch` | INVALID if a matched quote has `section_agrees=false` | Weaker; check its conflict rate |
| `agrees_with_majority` | VALID if the verdict matches a strict majority of the *other* 3 providers, INVALID if a strict majority disagrees, ABSTAIN on a tie | Correlated with other providers' errors; see caution below |
| `stratum_prior` | VALID/INVALID versus the stratum `expect`, ABSTAIN where there is no prior | Weak and noisy (§3) |
| `pubmed_pubtype` (C1) | INVALID if C1=true but PubMed lists Review/Editorial/Comment/Preprint; VALID if consistent | Needs a PubMed efetch client (see §6) |
| `pubmed_mesh_animals` (C5) | INVALID if C5=true ("no animals") but MeSH has `Animals` without `Humans`, etc. | Same client |
| `na_consistency` (C3/C6) | e.g. INVALID if C6=`not_applicable` but the reason or evidence shows blood/serum/plasma mentioned | Heuristic over `reason` text |
| `reason_verdict_mismatch` | INVALID if the reason contradicts the verdict (e.g. "not used as a biomarker" with `satisfied: true`) | Start with keyword rules; an LLM-judge version costs API money, so **ask first** |
| `semantic_entropy` | VALID if the model's resampled answers are consistent and match this sample; INVALID if they scatter or this sample is the odd one out; ABSTAIN in between | Within-model uncertainty. Needs extra samples; see §5a |

### 5a. Semantic entropy LF (planned earlier, never run on this data)

Earlier design: branch `origin/SemanticEntropy` (March 2026, RAG era, no shared history
with `main`). See `data_analysis/semantic_entropy/README.md` there. Sample the same
prompt N≈10 times at temperature ~1.0, cluster the answers greedily with a cross-encoder
(`cross-encoder/stsb-roberta-large` in that notebook), turn cluster sizes into
probabilities, and compute Shannon entropy H = −Σ p log p. High H means the model is
guessing. The notebook only smoke-tests the cross-encoder; no entropy was ever computed.

Adapting it to whole-paper labeling:

- **Unit:** one (pmid, criterion, provider). The entropy of that model's resamples
  votes on the existing sample from the same model.
- **Two levels:** (a) *verdict entropy* over `satisfied` (true/false/not_applicable).
  It's discrete, so no cross-encoder is needed, and it's cheap to compute. (b) *Semantic
  entropy* over `reason` texts, clustered with the cross-encoder or with bidirectional
  NLI entailment (the Farquhar et al. 2024 method). This catches samples with the same
  verdict but inconsistent justifications. Start with (a).
- **Vote:** VALID if H is low and the sample's verdict equals the resampled mode;
  INVALID if H is high, or the sample disagrees with a confident mode; ABSTAIN between.
  Thresholds are placeholders until the gold set exists.
- **Decided (user, 2026-10-20): N = 5 resamples for all four models, paid ones
  included.** With the existing answer that makes 6 samples per (pmid, criterion,
  provider), about 81k extra LLM calls in total. Use N=5 everywhere so entropy is
  comparable across models.
- **Getting samples:** run the existing labelers 5 more times, each run writing to its
  own `minio.output_prefix`, `entropy/s01` … `entropy/s05` (objects become
  `entropy/s01/<provider>/<pmid>.json` in `synthetic-data-v3`). Set it through an
  `LA_CONFIG_OVERRIDE` YAML file. Don't edit `.env.yaml`, and don't write into the top
  level of the bucket. Skipping existing results works per prefix, so runs can resume.
  - **Claude / gpt-6-luna (batch APIs, half price):** per k in 01..05, with an override
    containing `minio: {output_prefix: "entropy/s<k>"}`:
    `LA_CONFIG_OVERRIDE=ovr_s<k>.yaml python data_generation/batch_labeler.py run --model <model> --from-prefix pilot_2026_10`
    (detach it with `setsid nohup`; see `logs/batch_run.sh`). Each prefix gets its own
    state file, `logs/batches/<model>__entropy_s<k>.json`. The OpenAI 20M enqueued-token
    limit is shared across a model's runs (the runner sums all its state files), so the
    five gpt-6-luna runs can start together and will queue behind each other. Each
    gpt-6-luna pass needs about 3 waves.
  - **qwen3.8:27b (falcon9):** `labeler.py --providers qwen3.8:27b --from-prefix pilot_2026_10 --upload --skip-existing`
    with the override also setting the ollama model's `temperature: 1.0`. About 7.5 h per
    pass sequentially. falcon9 is a shared server, so ask before running passes in
    parallel.
  - **gpt-oss-120b (UNLV cluster, the user submits):**
    `OUTPUT_PREFIX=entropy/s01 TEMPERATURE=1.0 sbatch slurm/label_vllm.sbatch`, once per k.
- **Temperature:** `gpt-6-luna` accepts only temperature 1 and `claude-sonnet-5-5` rejects
  sampling parameters; their default sampling is already stochastic, so leave them as they
  are. For Ollama/vLLM set 1.0 for these runs only (their labeling config uses 0.1).
- **Cost:** 5 extra passes of the batch cost for each paid model. Before launching,
  tell the user the per-pass cost from the first pass's `usage` fields, then run the rest.
  Claude batch requests also get prompt-cache reads on the paper text.
- **Not the same as agreement:** entropy is a model's uncertainty about its *own* answer;
  `agrees_with_majority` is consensus *across* models. They correlate, but a model can be
  confidently wrong together with the others (low entropy, agreement), and that case is
  exactly what the gold set will reveal.

Caution: label models assume LFs are roughly conditionally independent. Agreement LFs
built from the same other providers are correlated, and so are the two quote LFs.
Prefer one agreement LF over several pairwise ones, or model the dependency.

## 6. Environment and access (vostok)

- **Do not `uv sync --all-packages`** on vostok. It pulls torch/CUDA wheels and filled
  the 30 GB root disk on 2026-10-17 (5.5 GB free after cleanup; MinIO and Postgres share
  that disk). The local `.venv` is Python 3.14 and can't build the locked numpy. Use
  Python 3.12 (`/usr/bin/python3.12`) and a minimal env.
- `snorkel` 0.10 depends on **torch**. Install CPU-only torch
  (`--index-url https://download.pytorch.org/whl/cpu`) into a separate venv on a disk
  with room, or implement the label model directly (a small Dawid-Skene/EM in numpy is
  fine for 2 classes). Check `df -h /` before installing anything large.
- MinIO keys in `.env.yaml` are stale; the real ones come from Vault. Set these before
  `config.load_app_config()`:

  ```bash
  cd ~/research/LiteratureAnnotator
  export VAULT_ADDR=https://localhost:8200 VAULT_CACERT=$PWD/vault/tls/vault-cert.pem \
         VAULT_SECRET_ID_FILE=$PWD/.vault-secret-id \
         VAULT_ROLE_ID=$(grep -oP '^VAULT_ROLE_ID=\K.*' label_api/.env)
  export PYTHONPATH=$PWD:$PWD/data_generation
  uv run --no-project --with minio --with pyyaml --with pydantic --with pydantic-settings \
         --with hvac --with requests python your_script.py
  ```

- PubMed metadata: no efetch client exists yet (`data_download/mesh_lookup.py` only does
  MeSH search/counts, but its `_safe_request` retry helper is reusable). The NCBI key is
  `config.bioc_download.ncbi_api_key` (from Vault). Stay at or under 3 req/s, fetch PMIDs
  in batches of up to 200, and cache the results (679 papers, a few calls).

## 7. Boundaries

- **Don't touch** `label_api/`, Label Studio, or the Docker stack: another agent is
  reviving them.
- **API spending:** the 5 semantic-entropy resample passes for Claude and gpt-6-luna are
  approved (§5a; report the per-pass cost after the first one). Anything else (LLM-judge
  LFs, relabeling, more resamples) needs the user's OK first.
- **Don't edit** `.env.yaml` provider settings or the results in `synthetic-data-v3`.
  Write LF outputs and label-model scores somewhere new (local files, or a new prefix
  after asking).
- Work on branch `retire-rag` (or a branch off it). Commit only when the user asks.
- UNLV HPC: `srun` doesn't work there; use `sbatch`. You shouldn't need the cluster
  for this task.

## 8. Open questions for the user

- Exact SME rule for C6 when blood is mentioned but isn't a biomarker (affects the
  `na_consistency` LF and the gold-set guidelines).
- Where label-model outputs should live (a new MinIO prefix? a table Label Studio reads?).
- Whether an LLM-judge LF is in budget.
