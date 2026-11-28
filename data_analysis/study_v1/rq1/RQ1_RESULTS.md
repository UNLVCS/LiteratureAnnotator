# RQ1 Faithfulness: results (sample s00, 100 papers x 6 models)

Script: `data_analysis/study_v1/rq1/rq1_faithfulness.py` (output log `run_output.txt`, tables in `tables/`). No LLM/API calls; paper text was read from MinIO (read only) and only for classifying the 13 unmatched quotes. Unweighted numbers are the main ones; "w" = weighted by stratified-selection weight (s00 weights sum to 647). 95% CIs are cluster bootstraps over papers (5000 resamples, seed fixed).

## Headline

3,600 answers, 3,585 valid, 7,645 quotes. **Nearly all quotes are verbatim.** Only 13 of 7,645 quotes (0.17%) are unmatched, all from DeepSeek and Qwen.

| Model | valid ans. | quotes | unmatched | match rate (95% CI) | match rate w | answers with >=1 unmatched | section misattrib. (matched) | empty evidence |
|---|---|---|---|---|---|---|---|---|
| Opus 5.5 | 586 | 1459 | 0 | 100% (upper bound on unmatched ~0.2%*) | 100% | 0% | 1/1459 (0.07%) | 1.9% |
| GPT-6.2-sol | 600 | 1137 | 0 | 100% (~0.3%*) | 100% | 0% | 0 | 11.2% |
| Gemini 4-carbon | 599 | 1373 | 0 | 100% (~0.2%*) | 100% | 0% | 0 | 9.0% |
| DeepSeek v4.1 Flash | 600 | 1292 | 8 | 99.38% (98.94-99.77) | 99.51% | 1.5% (0.6-2.6) | 2/1284 (0.16%) | 11.8% |
| Qwen 3.8 Max | 600 | 1160 | 5 | 99.57% (99.15-99.91) | 99.72% | 0.9% (0.2-1.9) | 2/1155 (0.17%) | 11.2% |
| Kimi K3.1 Pro | 600 | 1224 | 0 | 100% (~0.25%*) | 100% | 0% | 0 | 13.5% |

\* With zero events the bootstrap CI is degenerate ([100%,100%]); the figure shown is the rule-of-three 95% upper bound on the unmatched rate (3/n, treating quotes as independent, so slightly optimistic). Weighted CIs, empty-evidence CIs and answer-level CIs are in `tables/per_model.csv`.

**Excluded as "no valid answer"** (not in any quote-based rate): Opus 14 (all `content_filter`, 4 papers), Gemini 1 (`length`), all others 0. Total 15 of 3,600. (The 2 malformed-JSON cases noted earlier are not in s00.)

**Pairwise (unmatched-quote rate, paired bootstrap, uncorrected, 15 comparisons):** DeepSeek (0.62%) and Qwen (0.43%) are each above the four zero-error models, with CIs that exclude 0 (e.g. DeepSeek minus Kimi 0.62 pp, CI 0.23-1.06; Qwen minus Kimi 0.43 pp, CI 0.09-0.85). DeepSeek vs Qwen: 0.19 pp, CI -0.32 to 0.69, no evidence of a difference. The four zero-error models cannot be separated. These CIs rest on 13 events and should be read as "DeepSeek/Qwen occasionally fail, the others did not here", not as a ranking.

## By criterion (all models pooled; `tables/per_criterion.csv`, `model_by_criterion.csv`)

| Criterion | quotes | match rate | section misattrib. | empty evidence |
|---|---|---|---|---|
| C1 | 1551 | 99.94% | 0 | 0.3% |
| C2 | 1514 | 99.80% | 0 | 0.3% |
| C3 | 1005 | 99.80% | 0 | 6.3% |
| C4 | 1522 | 99.67% | 0.07% | 6.0% |
| C5 | 1149 | 100% | 0.09% | 9.4% |
| C6 | 904 | 99.78% | 0.33% | **36.4%** |

Match-rate differences across criteria are 1-5 quotes each, i.e. noise. Empty evidence is the only measure with real structure:
- **C6 (blood biomarker)** is empty for 36% of valid answers (Opus 8%, Gemini 23%, GPT 48%, DeepSeek 42%, Qwen 45%, Kimi 51%).
- **Opus almost always quotes**: 1.9% empty overall vs 9-14% for the others. Gemini is the outlier on C5 (20% empty vs 1-11%).
- Empty evidence is concentrated in `false` verdicts (about 31-36% of false answers for GPT, DeepSeek, Qwen, Kimi; Opus 6%, Gemini 18%) versus 0.3-5% for `true` (`tables/empty_evidence_by_verdict.csv`). This matches the allowed "absence-based" answers and is not a faithfulness failure.

## Kinds of unmatched quote (n = 13)

Classifier (difflib `SequenceMatcher`, text normalized with `papers.evidence._normalize_plain`): a quote is split into fragments on ellipses and sentence boundaries; each fragment/quote is scored by the share of its characters aligned (in-order blocks >= 4 chars) to its best single passage.
(b) **stitched** if there are >=2 fragments, each scores >= 0.85, and they are not adjacent and in order in one passage; (a) **paraphrase** if not (b) and the whole quote scores >= 0.75 (sub-tier "near-verbatim edit" >= 0.90); (c) **no close passage** otherwise; quotes under 12 chars would be "too short".

| Kind | DeepSeek | Qwen | Total |
|---|---|---|---|
| (a) paraphrase / near-verbatim edit | 8 | 5 | **13** |
| (b) stitched fragments | 0 | 0 | 0 |
| (c) no close passage | 0 | 0 | 0 |

All 13 score 0.977-1.0 coverage (every one in the near-verbatim tier). I read all 13 against the paper text. They are **not invented**; each is a real sentence with a small surface edit that normalization does not absorb:
1. DeepSeek, PMID 22413000, C4: "...raising glutamine levels in the medium **protect** cultured..." (source "protects"; grammar fix).
2. DeepSeek, 36677925, C4: "Alzheimer's disease (AD) is a proteinopathy..." (source: "Alzheimer's disease is a proteinopathy...", model inserted "(AD)").
3. DeepSeek, 39121126, C3: dropped the "(ADC)" acronym after "Amsterdam dementia cohort".
4. DeepSeek, 41184364, C4: "amyloid- (A42)" and "(p-tau), and total tau" vs source "amyloid-(A42)" and "(p-tau), total tau" (spacing and an inserted "and").
5. DeepSeek, 40595248, C3: began with "The GSE37263 and GSE63061 datasets"; the source sentence starts "GSE37263 and ..." (inserted "The").
6. Qwen, 33386803, C2: "PICALM is a highly validated..." (source: "phosphatidylinositol-binding clathrin assembly protein (PICALM) is..."; truncated the subject).
7. Qwen, 33663013, C4: dropped a stray footnote digit "15" inside the source sentence ("1-42), 15 while ...").
8. Qwen, 34362322, C6: "amyloid rate" where the source says "amyloid ratio" (typo-level change).
9. Qwen, 37723371, C4: "Abeta42/Abeta40," vs source "Abeta42/Abeta40 ratio,"; dropped "ratio".
10. Qwen, 40418517, C1: "Meister und Slunecko; Duguay und Gold-Apel" vs source "and" (translated citation connective).

The remaining 3 (DeepSeek 29972206 C2: "supporting evidence that" vs source "supported by increasing evidence that"; DeepSeek 37723371 C6: "biomarker analysis" vs "biomarkers analysis"; DeepSeek 40296363 C2: dropped the "(MS)" acronym) are the same kind of small edit. Stitched and no-close-passage quotes were not observed, so the classifier's (b)/(c) branches are **untested on real examples** here. Full list with scores: `tables/unmatched_quotes_classified.csv`.

## Reason-verdict contradiction (optional)

Keyword rules (verdict true with a reason opening "No/Not/does not meet..."; verdict false with a reason saying "criterion met" and no negation) flagged **0 answers** after excluding C5. A first pass flagged 208 C5 answers, all false positives (C5 true means "no animal models", so "no animal models are used" is consistent). Zero hits shows only that these narrow rules do not fire; it is **not** evidence that no contradictions exist. Not validated beyond reading ~14 flagged examples.

## Caveats
- No gold labels: "matched" shows the quote exists in the paper text, not that it supports the verdict. Faithfulness here is citation fidelity only.
- Unmatched means "not found verbatim after normalization", which includes light edits (all 13 here) as well as invention. The normalizer already forgives case, dashes, quotes, whitespace, figure/table/citation markers, trailing punctuation and ellipsis-separated fragments, plus any fragment-edge trimming; a high match rate partly reflects that leniency. Fragments under 12 chars do not count as located.
- Models were handed the same rendered text (no s00 paper was truncated), which makes verbatim copying easy; results may not hold for PDFs or longer papers.
- 100 papers from a stratified sample (weighted results differ by <0.2 pp). Quotes cluster in papers; CIs account for that, but 13 events is a small base. Resamples s01-s05 are deliberately not analysed.
- Section misattribution is almost nil (5 of 7,632 matched quotes), partly because the resolver prefers a hit in the cited section; it cannot flag quotes that occur in several sections.
- CIs for the four zero-error models are rule-of-three bounds, not bootstrap.

## Draft text for Results -> Faithfulness
> Across the 3,585 valid answers in the 100-paper primary sample (15 of 3,600 had no valid answer: 14 Opus content-filter refusals and one Gemini truncation), models cited 7,645 evidence quotes, of which 7,632 (99.83%) were found verbatim in the paper after normalization. Four of six models (Opus 5.5, GPT-6.2-sol, Gemini 4-carbon, Kimi K3.1 Pro) produced no unmatched quote; DeepSeek v4.1 Flash and Qwen 3.8 Max had 8 (0.62%, 95% cluster-bootstrap CI for match rate 98.9-99.8%) and 5 (0.43%) unmatched quotes respectively. On inspection, all 13 were near-verbatim edits of a real sentence (inserted acronyms, number agreement, dropped words) rather than fabricated or stitched evidence. Section misattribution was negligible (5 of 7,632 matched quotes). Faithfulness differences between models were therefore concentrated in how often models cited evidence at all: empty evidence ranged from 1.9% of answers for Opus to 13.5% for Kimi, almost entirely on `false` verdicts, and reached 36% on C6 (blood biomarker). Results are unweighted; weighting by the sampling design changes rates by under 0.2 percentage points.
