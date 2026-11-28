# study_v1

Measurement study of hallucination and self-consistency: six LLMs screen 100 papers
against six criteria through OpenRouter's batch API (`config/study_models.yaml`).

| Folder | Contents |
|---|---|
| `selection/` | How the papers were chosen. `select.py` (seed 20261126) reproduces the PMID lists and `selection_table.csv` (per-paper sampling weights) from the cached pool data; `SELECTION.md` documents the method. |
| `run/` | Submitting and collecting batches. `run.sh {submit,status,collect}` for one sample, `study_run.sh` for the full run in waves, `sync.sh` for a live one-paper test, `progress.py` for a per-sample summary. |
| `qa/` | `data_check.py` (integrity of all records), `reparse.py` (re-parse stored answers after a parser fix; no API calls). |

Analyses live in `data_analysis/study_v1/` (RQ1: `rq1/`).

## Samples

- `s00`: 100 papers (`selection/study_v1_s00_pmids.txt`) x 6 models, one answer per criterion.
- `s01`-`s05`: 25 of those papers (`selection/study_v1_resample_pmids.txt`) x 6 models, resampled.
- Results: MinIO `synthetic-data-v3/study_v1/s0K/<model>/<pmid>.json`; local copies and batch
  state stay in the gitignored `logs/` (`logs/results_batch_*`, `logs/batches/`).

## Commands (run from the repository root)

```bash
LA_STUDY_CONFIG=config/study_models.yaml ./study_v1/run/run.sh status
STUDY_MODELS=kimi ./study_v1/run/study_run.sh all      # one model, s00 then resamples
python3 study_v1/run/progress.py
python3 study_v1/qa/data_check.py
```

`run.sh` with no `LA_STUDY_CONFIG` runs the smoke test (`run/smoke_models.yaml`, 3 papers).
`LA_MAX_BATCH_USD` caps each batch's worst-case cost (OpenRouter holds it against the balance
and the API key's credit limit); `LA_CRITERIA=6` asks only criterion 6 and patches stored records.

## History

- GLM-5.3 was replaced by Kimi K3.1 Pro on 2026-11-27 after its `s04` batch stalled on Morph;
  GLM's other results remain in MinIO under `z-ai/glm-5.3/`.
- Criterion 6 was redefined on 2026-11-27 (schema `whole-paper-v3`); `run/c6test_models.yaml`
  and `selection/c6_test_pmids.txt` are the 34-paper test of the new wording.
- `run/original-submission.sh` is the original smoke-test command, kept for provenance.
