# weak_supervision

Filters the LLM-generated training labels in `synthetic-data-v3`. A data point is one model's
answer for one (paper, criterion), with its verdict, reason and evidence. Labeling functions
(LFs) vote on whether that answer is a **trustworthy training sample** (VALID / INVALID /
ABSTAIN). A label model combines the votes into `p_valid`, and samples labeled INVALID are
dropped downstream. See `HANDOFF.md` for the background.

## Running

Everything goes through `ws.sh`, which sets up a throwaway Python 3.12 env with Vault
credentials. Don't `uv sync` on vostok (the workspace pulls torch; see HANDOFF §6).

```bash
weak_supervision/ws.sh samples.py    # MinIO -> data/samples.parquet (16,290 rows; records cached in data/raw/)
weak_supervision/ws.sh pubmed.py     # PubMed pub types + MeSH -> data/pubmed_meta.json
weak_supervision/ws.sh entropy.py    # entropy/s01..s05 resamples -> data/entropy.parquet (rerun as passes land)
weak_supervision/ws.sh run.py        # LFs + label model -> data/out/
weak_supervision/ws.sh run.py --gold gold.csv   # also score against SME labels
weak_supervision/ws.sh test          # unit tests
```

`data/` is gitignored. Outputs in `data/out/`:

- `scores.parquet` / `scores.csv`: `sample_id, pmid, criterion, provider, p_verdict, p_evidence, p_valid, label, fragile`
- `lf_matrix.parquet`: one column per LF
- `report.md`: LF coverage, overlap and conflict (overall and per criterion); learned LF accuracies; INVALID rates; leave-one-out stability
- `spot_check.md`: the 30 most confident INVALID samples and 30 random confident VALID ones, with reason, evidence and votes, for a human to read

## Files

| file | what |
|---|---|
| `samples.py` | loader: one row per (pmid, criterion, provider) with the verdict, reason, evidence stats, error, stratum and prior, the other providers' verdicts, and blood/serum/plasma mention counts from the paper text |
| `lfs.py` | the LFs plus `apply_lfs` / `lf_summary` (Snorkel `LFAnalysis`-style) |
| `label_model.py` | numpy EM label model, structured by facet (below) |
| `evaluate.py` | proxy evaluation: spot check, stability, plausibility |
| `gold.py` | gold-set hook: per-LF and label-model accuracy, INVALID precision/recall/F1, Brier, ECE; overall, per provider and per criterion |
| `pubmed.py` | efetch client (batches of 200, ≤3 req/s, cached) |
| `entropy.py` | verdict entropy over the semantic-entropy resamples |

## LFs

| LF | votes | notes |
|---|---|---|
| `parse_error` | INVALID | no parsable answer; a hard rule (p_valid = 0) |
| `quote_match` | both | the handoff's `unmatched_quote` + `quotes_all_matched`, merged because they are mutually exclusive functions of the same evidence; empty evidence abstains |
| `section_mismatch` | INVALID | a verbatim quote found in a different section than the one cited |
| `agrees_with_majority` | both | strict majority of the *other* providers |
| `stratum_prior` | both | C2/C4/C6 only. The C1/C5 strata were selected by PubMed pub type / MeSH, which the PubMed LFs read directly, so voting on them here would count that evidence twice. A prior of `false` accepts `not_applicable` |
| `pubmed_pubtype` | both | C1: Review/Editorial/Comment/... vs trial/observational types; a plain "Journal Article" abstains |
| `pubmed_mesh_animals` | both | C5: MeSH Animals without Humans, or the reverse |
| `na_consistency` | both | C6: `not_applicable` vs literal blood/serum/plasma mentions in the paper; blood-brain-barrier-only papers abstain (open SME question) |
| `reason_verdict_mismatch` | INVALID | keyword rules for a reason that contradicts its verdict. Tuned for precision: it fires 4 times, and each is a real contradiction |
| `semantic_entropy` | both | verdict entropy of the same model's 5 resamples. Abstains until they exist; thresholds are placeholders |

## Label model: two facets

The first design was one latent class with full Dawid-Skene (abstain as an outcome), as in
Snorkel. It failed two ways:

1. On C6 the latent "INVALID" class became "answered not_applicable". Which LFs can vote
   depends on the verdict, and EM fitted that coverage pattern.
2. An accuracy-only model fixed that, but then the class prior collapsed to 0.999 and every
   INVALID vote was treated as noise.

The cause: the LFs measure **two independent failure modes**. Verdict LFs' INVALID votes
co-occur 2–28× more often than chance. `quote_match`'s INVALID votes are independent of
theirs (lift ~1). So:

```
p_valid = p_verdict × p_evidence     (0 if parse_error fires)
```

- **verdict facet**: one-coin Dawid-Skene EM per criterion, with a Beta prior on each LF
  accuracy (mean 0.75, 20 pseudo-votes). Groups with fewer than 2 active LFs, or with no LF
  that votes VALID, fall back to fixed priors, because EM can't identify them.
- **evidence facet**: fixed accuracies (`quote_match` 0.98, `section_mismatch` 0.7, P(ok) 0.95).
  Two LFs aren't enough to fit. An unmatched quote makes a sample INVALID, and a section
  mismatch alone does not.

These priors are assumptions. The gold set should replace them with measured accuracies.

## Results (2026-10-20, no gold set; every number is a proxy)

1,326 of 16,290 samples are INVALID (8.1%).

| provider | INVALID | from verdict | from evidence |
|---|---|---|---|
| claude-sonnet-5-5 | 4.4% | 4.1% | 0.3% |
| gpt-6-luna | 6.7% | 4.4% | 2.4% |
| openai/gpt-oss-120b | 13.5% | 4.8% | 9.1% |
| qwen3.8:27b | 8.0% | 6.5% | 1.3% |

By criterion: C1 3.6%, C2 6.1%, C3 6.2%, C4 8.8%, C5 3.7%, **C6 20.4%**. These match the
expectations in HANDOFF §3: gpt-oss is the noisiest because of its quotes, and C6 is the
most contested criterion.

- **Spot check:** after parse failures, the top INVALID samples are mostly `not_applicable`
  answers for C6 on papers that do mention blood or plasma (often only blood-brain barrier
  or blood flow, which is the open SME question). Read `data/out/spot_check.md`.
- **Stability:** 9.7% of samples change label when some single LF is dropped. Most of this
  comes from `agrees_with_majority`: on C1–C5 it is the only high-coverage verdict LF until
  `semantic_entropy` has resamples.
- **Learned accuracies:** agreement 0.97–1.00 on C1–C5 and 0.97 on C6. `stratum_prior` 0.93 on
  C2/C4 and 0.75 on C6, which matches the noisy C6 strata in HANDOFF §3. `pubmed_pubtype` 0.98,
  `pubmed_mesh_animals` 0.95, `na_consistency` 0.94.

## Semantic-entropy resamples (HANDOFF §5a)

Overrides are `logs/entropy/ovr_s<k>.yaml`, which set `minio.output_prefix: entropy/s<k>`.
The launcher is `logs/entropy/batch_entropy.sh <model> <k>`, which takes `submit --dry-run` to
check a pass without sending anything. Both are local only, since `logs/` is gitignored.

- Measured from the existing pass's `usage`, one Claude pass is 1.44M input + 26.5M
  cache-write + 39.5M cache-read + 1.61M output tokens. At Sonnet 5.5 batch rates that is
  **~$47 per pass, ~$233 for 5**. One gpt-6-luna pass is 42.7M prompt tokens (no cache hits)
  + 1.37M completion tokens.
- qwen3.8:27b (falcon9, temperature 1.0): needs an override that also replaces the
  `llm_providers.ollama.models` list with `temperature: 1.0`, then
  `labeler.py --providers qwen3.8:27b --from-prefix pilot_2026_10 --upload --skip-existing`.
- gpt-oss-120b (the user submits): `OUTPUT_PREFIX=entropy/s01 TEMPERATURE=1.0 sbatch slurm/label_vllm.sbatch`, once per k.

## Open questions

- The C6 rule when blood is mentioned but isn't a biomarker sample (blood-brain barrier, FBS).
  This decides whether `na_consistency` should vote on those papers, and it drives most of the
  C6 INVALIDs.
- Where scores should live. They are local for now (`data/out/scores.parquet`).
- Whether an LLM-judge `reason_verdict_mismatch` is in budget. The keyword version is
  precise but covers only 4 samples.
