# study_v1 paper selection

Reproduce from the worktree root (offline, cached inputs only):

```bash
UV_CACHE_DIR=/tmp/la-step0-uv uv run --offline --no-project --python 3.12 \
  --with pandas --with pyarrow python -I study_v1/selection/select.py
```

**Seed** `20261126`. **Inputs**: `weak_supervision/data/samples.parquet` (4 pilot models' verdicts, paper length, search stratum) and `weak_supervision/data/pubmed_meta.json` (PubMed publication types), both in the main checkout.
**Outputs**: `study_v1_s00_pmids.txt` (100), `study_v1_resample_pmids.txt` (25, subset), `pilot_pmids.txt` (10, subset, no smoke PMIDs), `selection_table.csv` (every eligible paper with its cell, `pi`, `weight`, flags).

## Method
1. **Eligible pool, 647 of 679 pilot papers.** I excluded 29 papers over 80,000 rendered characters and 3 marked *Retracted Publication* in PubMed. Results therefore generalize to this 647-paper pool, not to the 29 longest papers.
2. **Rare flag.** A paper is flagged when the majority verdict of the four pilot models (Sonnet 5.5, GPT-6-luna, gpt-oss-120b, qwen3.8:27b) is C3 = false or C6 = not_applicable. 137 eligible papers have the flag. The flag comes from model votes, not human labels, so the realized s00 class counts will differ somewhat.
3. **Cells = search stratum × rare flag.** 42 papers come from the 137 flagged papers (rate 0.31) and 58 from the 510 unflagged papers (rate 0.11). Counts within each group follow proportional allocation by stratum (largest remainder). Papers are drawn by simple random sampling within each cell. The 3 smoke PMIDs (33386803, 37570596, 37252737) are placed in their cells first and count toward those cells' quotas. Strictly, that makes them certainty selections. With 3 of 100 papers, I treat them as ordinary draws.
4. **Resample subset, 25.** Drawn from the 81 of the 100 that are ≤ 50,000 characters. The flagged/unflagged split follows the 100 (11 flagged, 14 unflagged). P(resample | in s00) is 11/37 = 0.297 for flagged papers and 14/44 = 0.318 for unflagged ones; it is 0 for papers over 50k. This subset represents papers ≤ 50k, which is about 80% of the pool.
5. **Pilot subset, 10.** A greedy cover of the 8 minority classes (C1–C2 false, C3 false and not_applicable, C4–C6 false, C6 not_applicable), choosing the shortest qualifying paper first, with at most 2 papers per stratum. The rest is filled at random from the shorter half of the 100. It is a shakedown set, not a statistical sample.

## Inclusion probability and weight per cell (weight = N/n)

| stratum | flagged n/N | weight | unflagged n/N | weight |
|---|---|---:|---|---:|
| core_positive | 7/23 | 3.29 | 14/124 | 8.86 |
| unbiased_ad | 8/27 | 3.38 | 4/39 | 9.75 |
| c1_review | 0/0 | n/a | 6/54 | 9.00 |
| c1_editorial_preprint | 2/6 | 3.00 | 3/24 | 8.00 |
| c2_other_disease | 3/8 | 2.67 | 6/50 | 8.33 |
| c3_small_study | 3/11 | 3.67 | 1/12 | 12.00 |
| c4_nucleic_acid | 3/10 | 3.33 | 6/48 | 8.00 |
| c5_animal | 1/3 | 3.00 | 6/53 | 8.83 |
| c5_cell_culture | 5/17 | 3.40 | 3/22 | 7.33 |
| c6_blood_pressure | 2/5 | 2.50 | 6/54 | 9.00 |
| c6_csf_pet_only | 8/27 | 3.38 | 3/30 | 10.00 |

To estimate pool-level accuracy, weight each paper by `weight` in `selection_table.csv`; the weights sum to 647. For the resample set, multiply by 1/`pi_resample_given_s00`.

## Composition (pilot-model majority labels; true/false/not_applicable)

| set | n | total chars | mean | C1 | C2 | C3 | C4 | C5 | C6 | any model disagreement |
|---|---:|---:|---:|---|---|---|---|---|---|---:|
| s00 | 100 | 3,840,512 | 38,405 | 83/17 | 77/23 | 42/23/35 | 76/24 | 79/21 | 44/34/22 | 59 |
| resample | 25 | 748,339 | 29,934 | 19/6 | 20/5 | 11/4/10 | 21/4 | 21/4 | 9/9/7 | 15 |
| pilot | 10 | 220,464 | 22,046 | 6/4 | 9/1 | 3/3/4 | 9/1 | 7/3 | 6/2/2 | 5 |
| eligible pool | 647 | 25,556,458 | 39,500 | 518/129 | 507/140 | 309/75/263 | 489/158 | 466/181 | 286/289/72 | 370 |

Every minority class has at least 17 papers in s00. The target of about 20 is met for C3 false (23) and C6 not_applicable (22); C1 false (17) falls slightly short. I did not tune the seed.

## Estimated cost (all six models, no cache hits, from `select.py`)
Tokens per request ≈ chars/4.8 + 1,300 (fitted to the smoke run: Gemini estimated at $0.089 per paper against $0.084 billed). Output tokens per request are assumed: Opus 2,000, Sol 1,500, Qwen 1,300, GLM 2,000, Gemini and DeepSeek 800. The Opus, Sol and GLM figures are guesses.

| run | Opus | Sol | Gemini | DeepSeek | Qwen | GLM | **total** |
|---|---:|---:|---:|---:|---:|---:|---:|
| s00, 100 papers | 23.16 | 10.08 | 8.46 | 0.27 | 9.11 | 3.06 | **$54** |
| s01–s05, 25 × 5 | 26.30 | 11.28 | 9.25 | 0.29 | 9.86 | 3.46 | **$60** |
| pilot s00, 10 papers | 1.91 | 0.80 | 0.64 | 0.02 | 0.68 | 0.25 | **$4.30** |

**Budget warning.** s00 plus the resamples comes to about $114, above the $60–80 budget. Ways to close the gap:
- Run 3 resamples instead of 5: saves about $24.
- Lower the resample length cap to about 30k characters: saves only about $8, because at this length output tokens are roughly half the cost per paper, and the resample set then represents fewer papers.
- Leave Opus out of the resamples: saves $26.
- Count on prompt caching: Opus and Sol can reuse the cached paper prefix in 5 of 6 calls, which could cut their input cost substantially; this is not yet measured.

The pilot run should show the real output-token counts for Opus, Sol and GLM, which are the largest uncertainty in these estimates.
