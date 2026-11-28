"""Select the study_v1 paper sets from the cached pilot pool (no network, no paid calls).

Outputs (next to this script):
  study_v1_s00_pmids.txt       100 papers for s00 (all six models, one answer each)
  study_v1_resample_pmids.txt  25 of those 100 for s01-s05
  pilot_pmids.txt              10 of those 100 for a shakedown run (no smoke PMIDs)
  selection_table.csv          one row per eligible paper: cell, weights, flags, prior labels

Run from the worktree root:
  UV_CACHE_DIR=/tmp/la-step0-uv uv run --offline --no-project --python 3.12 \
      --with pandas --with pyarrow python -I study_v1/selection/select.py

Design (see SELECTION.md):
  1. Eligible = 679 pilot papers minus papers over MAX_CHARS rendered characters and
     papers whose PubMed publication types include "Retracted Publication".
  2. Each eligible paper gets a "rare" flag from the majority verdict of the four pilot
     models: C3 == false or C6 == not_applicable. These are the two classes too thin
     in a plain random draw.
  3. Sampling cells = search stratum x rare flag. Rare papers are sampled at a higher
     rate (N_RARE of the rare papers, the rest from the non-rare papers). Within each
     group the per-cell counts use proportional allocation (largest remainder), and
     papers are drawn by simple random sampling within a cell. The smoke PMIDs are
     placed in their cells first and count toward that cell's quota.
  4. Resample subset: drawn from the 100 among papers <= RESAMPLE_MAX_CHARS, stratified
     by the rare flag in proportion to the 100 and drawn at random within each part.
  5. Pilot subset: a greedy cover of every minority class, shortest papers first, at
     most two per search stratum, filled up to N_PILOT with random short papers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261126
MAX_CHARS = 80_000
RESAMPLE_MAX_CHARS = 50_000
N_TOTAL, N_RARE = 100, 42
N_RESAMPLE, N_PILOT = 25, 10
SMOKE = ["33386803", "37570596", "37252737"]

OUT = Path(__file__).resolve().parent
WS = Path("/home/abreham/research/LiteratureAnnotator/weak_supervision/data")
SAMPLES = WS / "samples.parquet"
PUBMED_META = WS / "pubmed_meta.json"
CRITERIA = [f"criterion_{i}" for i in range(1, 7)]
CLASSES = {c: ["true", "false", "not_applicable"] if c in ("criterion_3", "criterion_6")
           else ["true", "false"] for c in CRITERIA}
MINORITY = [("criterion_1", "false"), ("criterion_2", "false"), ("criterion_3", "false"),
            ("criterion_3", "not_applicable"), ("criterion_4", "false"),
            ("criterion_5", "false"), ("criterion_6", "false"),
            ("criterion_6", "not_applicable")]

# Cost model, calibrated on the smoke run (tokens per request ~ chars/4.8 + 1300;
# Gemini estimate $0.089/paper vs $0.084 billed). Output tokens per request are
# guesses for Opus/Sol/GLM. No cache hits assumed. Rates: config/study_models.yaml.
MODELS = {  # alias: (input $/M, output tokens per request, output $/M)
    "opus": (2.0, 2000, 10.0), "sol": (1.0, 1500, 5.0), "gemini": (1.0, 800, 6.0),
    "deepseek": (0.035, 800, 0.149), "qwen": (1.15, 1300, 3.45), "glm": (0.28, 2000, 1.249),
}


def paper_cost(chars: pd.Series) -> pd.Series:
    """Estimated USD for one sample (6 criteria) of all six models."""
    tokens_in = 6 * (chars / 4.8 + 1300)
    return sum(tokens_in * i / 1e6 + 6 * o * r / 1e6 for i, o, r in MODELS.values())


def majority(s: pd.Series):
    vc = s.dropna().astype(str).str.lower().value_counts()
    return vc.index[0] if len(vc) else None


def load_pool() -> pd.DataFrame:
    df = pd.read_parquet(SAMPLES)
    meta = json.loads(PUBMED_META.read_text())
    papers = df.groupby("pmid").agg(stratum=("stratum", "first"), chars=("rendered_chars", "first"))
    labels = df.groupby(["pmid", "criterion"]).verdict.agg(majority).unstack()[CRITERIA]
    agree = (df.groupby(["pmid", "criterion"]).verdict
             .agg(lambda s: s.astype(str).value_counts().iloc[0] / len(s)).unstack()[CRITERIA])
    papers = papers.join(labels)
    papers["n_disagree"] = (agree < 1).sum(axis=1)
    papers["retracted"] = [("Retracted Publication" in meta.get(p, {}).get("pub_types", []))
                           for p in papers.index]
    papers["cost_s00"] = paper_cost(papers.chars)
    return papers


def largest_remainder(sizes: pd.Series, n: int) -> pd.Series:
    quota = sizes / sizes.sum() * n
    alloc = np.floor(quota.to_numpy()).astype(int)
    order = np.argsort(-(quota.to_numpy() - alloc), kind="stable")
    alloc[order[: n - alloc.sum()]] += 1
    return pd.Series(alloc, index=sizes.index)


def draw(frame: pd.DataFrame, alloc: pd.Series, key, rng, forced=()) -> list[str]:
    chosen = []
    for cell, n in alloc.items():
        members = frame.index[frame[key] == cell] if isinstance(key, str) else \
            frame.index[(frame[list(key)] == cell).all(axis=1)]
        fixed = [p for p in forced if p in members]
        rest = sorted(set(members) - set(fixed))
        k = n - len(fixed)
        assert k >= 0, f"forced papers exceed quota in {cell}"
        chosen += fixed + list(rng.choice(rest, size=k, replace=False))
    return chosen


def class_counts(frame: pd.DataFrame) -> dict:
    return {c: "/".join(str((frame[c] == k).sum()) for k in CLASSES[c]) for c in CRITERIA}


def main() -> None:
    print(f"seed={SEED}\ninputs:\n  {SAMPLES}\n  {PUBMED_META}")
    rng = np.random.default_rng(SEED)
    pool = load_pool()
    pool["eligible"] = (pool.chars <= MAX_CHARS) & ~pool.retracted
    elig = pool[pool.eligible].copy()
    elig["rare"] = (elig.criterion_3 == "false") | (elig.criterion_6 == "not_applicable")
    print(f"pool={len(pool)} too_long={(pool.chars > MAX_CHARS).sum()} "
          f"retracted={pool.retracted.sum()} eligible={len(elig)} rare={elig.rare.sum()}")
    assert all(p in elig.index for p in SMOKE)

    # 1. Main 100: stratum x rare cells, proportional within the rare and non-rare groups.
    alloc = pd.concat([
        largest_remainder(elig[elig.rare == r].groupby("stratum").size(), n)
        .rename(lambda s, r=r: (s, r))
        for r, n in [(True, N_RARE), (False, N_TOTAL - N_RARE)]])
    sizes = elig.groupby(["stratum", "rare"]).size()
    elig["cell_N"] = [sizes[(s, r)] for s, r in zip(elig.stratum, elig.rare)]
    elig["cell_n"] = [alloc.get((s, r), 0) for s, r in zip(elig.stratum, elig.rare)]
    elig["pi"] = elig.cell_n / elig.cell_N
    s00 = draw(elig, alloc, ("stratum", "rare"), rng, forced=SMOKE)
    elig["in_s00"] = elig.index.isin(s00)
    main = elig.loc[s00]

    # 2. Resample 25: short papers of the 100, rare flag in proportion.
    short = main[main.chars <= RESAMPLE_MAX_CHARS]
    r_alloc = largest_remainder(main.groupby("rare").size(), N_RESAMPLE)
    r_alloc_d, rs_sizes_d = r_alloc.to_dict(), short.groupby("rare").size().to_dict()
    resample = draw(short, r_alloc, "rare", rng)
    elig["pi_resample_given_s00"] = [
        r_alloc_d[r] / rs_sizes_d[r] if s and c <= RESAMPLE_MAX_CHARS else 0.0
        for r, s, c in zip(elig.rare, elig.in_s00, elig.chars)]
    elig["in_resample"] = elig.index.isin(resample)

    # 3. Pilot 10: cover each minority class with short papers, <=2 per stratum.
    cand = main.drop(index=SMOKE).sort_values("chars", kind="stable")
    pilot, per_stratum = [], {}
    for crit, cls in MINORITY:
        if any(main.loc[p, crit] == cls for p in pilot):
            continue
        for p in cand.index:
            if p not in pilot and cand.loc[p, crit] == cls and per_stratum.get(cand.loc[p, "stratum"], 0) < 2:
                pilot.append(p)
                per_stratum[cand.loc[p, "stratum"]] = per_stratum.get(cand.loc[p, "stratum"], 0) + 1
                break
    filler = [p for p in cand.index[cand.chars <= cand.chars.median()]
              if p not in pilot and per_stratum.get(cand.loc[p, "stratum"], 0) < 2]
    pilot += list(rng.choice(filler, size=N_PILOT - len(pilot), replace=False))
    elig["in_pilot"] = elig.index.isin(pilot)

    for name, ids in [("study_v1_s00_pmids.txt", s00), ("study_v1_resample_pmids.txt", resample),
                      ("pilot_pmids.txt", pilot)]:
        (OUT / name).write_text("".join(f"{p}\n" for p in sorted(ids, key=int)))
    elig.assign(weight=1 / elig.pi).to_csv(OUT / "selection_table.csv")

    rs, pl = elig.loc[resample], elig.loc[pilot]
    print("\nper-cell allocation (stratum, rare): n / N, weight N/n")
    for (s, r), n in alloc.items():
        print(f"  {s:22s} rare={str(r):5s} {n:3d} / {sizes[(s, r)]:3d}  w={sizes[(s, r)] / n:.2f}"
              if n else f"  {s:22s} rare={str(r):5s}   0 / {sizes[(s, r)]:3d}")
    for name, f, k in [("s00", main, 1), ("resample", rs, 5), ("pilot", pl, 1), ("eligible pool", elig, 1)]:
        print(f"\n{name}: n={len(f)} total_chars={f.chars.sum():,} mean_chars={f.chars.mean():,.0f} "
              f"max={f.chars.max():,} cost(x{k} samples)=${k * f.cost_s00.sum():.2f} "
              f"any_disagreement={(f.n_disagree > 0).sum()}")
        print("  true/false/n.a.:", class_counts(f))
    print("\nresample per-paper inclusion | s00:", {r: round(r_alloc_d[r] / rs_sizes_d[r], 3) for r in r_alloc_d}, "short eligible:", rs_sizes_d)
    print("pilot strata:", pl.stratum.value_counts().to_dict())


if __name__ == "__main__":
    main()
