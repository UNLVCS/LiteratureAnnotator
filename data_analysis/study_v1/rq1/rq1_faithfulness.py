#!/usr/bin/env python
"""RQ1 (faithfulness) analysis on sample s00 (100 papers x 6 models). Read-only, no LLM calls.

Run from the worktree root. Paper text is needed only for classifying unmatched quotes; it is
read from MinIO (read only) and cached under $RQ1_PAPER_CACHE (default /tmp/rq1_paper_cache),
so reruns work offline. See study_v1/run/run.sh for the VAULT_*/LA_CONFIG_OVERRIDE env.

  PYTHONPATH=$PWD uv run --offline --no-project --python 3.12 --with pandas --with numpy \
     --with pydantic --with pyyaml --with requests --with minio --with hvac \
     python data_analysis/study_v1/rq1/rq1_faithfulness.py
"""
from __future__ import annotations

import difflib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]  # repository root
sys.path.insert(0, str(ROOT))
from study_v1.paths import main_checkout  # noqa: E402
OUT = ROOT / "data_analysis/study_v1/rq1/tables"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ.get("RQ1_PAPER_CACHE", "/tmp/rq1_paper_cache"))
CACHE.mkdir(parents=True, exist_ok=True)

MODELS = {  # slug -> label
    "anthropic_claude-opus-5.5": "Opus 5.5",
    "openai_gpt-6.2-sol": "GPT-6.2-sol",
    "google_gemini-4-carbon": "Gemini 4-carbon",
    "deepseek_deepseek-v4.1-flash-1017": "DeepSeek v4.1 Flash",
    "qwen_qwen3.8-max-0902": "Qwen 3.8 Max",
    "moonshotai_kimi-k3.1-pro": "Kimi K3.1 Pro",
}
ORDER = list(MODELS.values())
CRITS = [f"criterion_{i}" for i in range(1, 7)]
B = 5000
RAW_PREFIX = "pilot_2026_10"  # MinIO prefix of the study papers (as in weak_supervision/samples.py)
RNG = np.random.default_rng(20261127)

# Classifier thresholds (unmatched-quote kinds)
T_PIECE = 0.85      # a fragment counts as "found" if >=85% of its chars align to one passage
T_PARA = 0.75       # whole-quote coverage by one passage >= 0.75 -> paraphrase of nearby passage
T_NEAR = 0.90       # sub-tier of paraphrase: near-verbatim edit
MIN_PIECE = 20      # chars; shorter sentence pieces are merged into the previous piece
MIN_BLOCK = 4       # difflib blocks shorter than this are ignored when computing coverage


# ------------------------------------------------------------------ load records
pmids = (ROOT / "study_v1/selection/study_v1_s00_pmids.txt").read_text().split()
sel = pd.read_csv(ROOT / "study_v1/selection/selection_table.csv", dtype={"pmid": str})
W = sel.set_index("pmid")["weight"]
assert set(pmids) <= set(W.index), "pmid missing from selection table"
print(f"inputs: {len(pmids)} s00 pmids; weights for these sum to {W[pmids].sum():.1f}; "
      f"(table covers more papers; s00 weights should sum to the 647-paper eligible pool)")

rows, ans_rows = [], []
for slug, label in MODELS.items():
    d = ROOT / f"logs/results_batch_{slug}__study_v1_s00"
    print(f"inputs: {d.relative_to(ROOT)}")
    for p in pmids:
        rec = json.loads((d / f"{p}.json").read_text())
        assert rec["schema_version"] == "whole-paper-v3" and rec["sample_index"] == 0, (slug, p)
        assert len(rec["criteria_results"]) == 6
        for a in rec["criteria_results"]:
            valid = a["response"] is not None and not a["error"]
            sat = reason = None
            if valid:
                r = (a["response"] or {}).get(a["criterion"]) or {}
                sat, reason = r.get("satisfied"), r.get("reason")
            ev = a["evidence"] or []
            ans_rows.append(dict(model=label, pmid=p, criterion=a["criterion"], valid=valid,
                                 finish_reason=a["finish_reason"], n_quotes=len(ev) if valid else 0,
                                 n_unmatched=sum(not e["matched"] for e in ev) if valid else 0,
                                 verdict=str(sat), reason=reason or ""))
            if valid:
                for i, e in enumerate(ev):
                    rows.append(dict(model=label, pmid=p, criterion=a["criterion"], qi=i,
                                     quote=e["quote"], cited_section=e["cited_section"],
                                     matched=bool(e["matched"]), section=e["section"],
                                     section_agrees=e["section_agrees"]))
ans = pd.DataFrame(ans_rows)
q = pd.DataFrame(rows)
ans["w"] = ans.pmid.map(W)
q["w"] = q.pmid.map(W)
print(f"answers: {len(ans)} total, {int(ans.valid.sum())} valid; quotes: {len(q)}")

# ------------------------------------------------------------------ exclusions
excl = (ans.assign(invalid=~ans.valid)
        .groupby("model").agg(answers=("valid", "size"), no_valid_answer=("invalid", "sum"))
        .reindex(ORDER))
fr = ans[~ans.valid].groupby(["model", "finish_reason"]).size().unstack(fill_value=0).reindex(ORDER).fillna(0).astype(int)
excl = excl.join(fr).fillna(0).astype(int)
excl["papers_affected"] = ans[~ans.valid].groupby("model").pmid.nunique().reindex(ORDER).fillna(0).astype(int)
excl.to_csv(OUT / "exclusions.csv")
print("\nEXCLUSIONS (no valid answer)\n", excl)


# ------------------------------------------------------------------ bootstrap helpers
def paper_counts(df, models, num, den):
    """papers x models arrays of numerator/denominator sums, in `pmids` order."""
    n = np.zeros((len(pmids), len(models))); d = np.zeros_like(n)
    idx = {p: i for i, p in enumerate(pmids)}
    for j, m in enumerate(models):
        s = df[df.model == m]
        g = s.groupby("pmid").agg(n=(num, "sum"), d=(den, "sum"))
        for p, r in g.iterrows():
            n[idx[p], j], d[idx[p], j] = r.n, r.d
    return n, d


BOOT = RNG.multinomial(len(pmids), np.ones(len(pmids)) / len(pmids), size=B).astype(float)  # B x papers


def boot_rate(n, d, w=None):
    w = np.ones(len(pmids)) if w is None else w
    num = BOOT @ (n * w[:, None]); den = BOOT @ (d * w[:, None])
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den  # B x models


def point(n, d, w=None):
    w = np.ones(len(pmids)) if w is None else w
    return (n * w[:, None]).sum(0) / (d * w[:, None]).sum(0)


def ci(x):
    return np.nanpercentile(x, [2.5, 97.5], axis=0)


wvec = np.array([W[p] for p in pmids])
q["unmatched"] = ~q.matched
q["one"] = 1
q["matched_i"] = q.matched.astype(int)
q["unm_i"] = q.unmatched.astype(int)
ans["has_unm"] = (ans.n_unmatched > 0).astype(int)
ans["empty"] = (ans.valid & (ans.n_quotes == 0)).astype(int)
ans["valid_i"] = ans.valid.astype(int)
ans["has_quotes"] = (ans.valid & (ans.n_quotes > 0)).astype(int)

# ------------------------------------------------------------------ 1. per model headline
res = pd.DataFrame(index=ORDER)
res["valid_answers"] = ans.groupby("model").valid_i.sum().reindex(ORDER)
res["quotes"] = q.groupby("model").one.sum().reindex(ORDER)
res["unmatched_quotes"] = q.groupby("model").unm_i.sum().reindex(ORDER)

n, d = paper_counts(q, ORDER, "matched_i", "one")
bm = boot_rate(n, d); lo, hi = ci(bm)
res["match_rate"] = point(n, d); res["match_lo"] = lo; res["match_hi"] = hi
bwm = boot_rate(n, d, wvec); wl, wh = ci(bwm)
res["match_rate_weighted"] = point(n, d, wvec); res["wmatch_lo"] = wl; res["wmatch_hi"] = wh

n2, d2 = paper_counts(ans[ans.has_quotes == 1], ORDER, "has_unm", "has_quotes")
b2 = boot_rate(n2, d2); l2, h2 = ci(b2)
res["answers_with_quotes"] = d2.sum(0)
res["share_answers_any_unmatched"] = point(n2, d2); res["any_unm_lo"] = l2; res["any_unm_hi"] = h2
res["share_answers_any_unmatched_weighted"] = point(n2, d2, wvec)
bw2 = boot_rate(n2, d2, wvec); res["any_unm_w_lo"], res["any_unm_w_hi"] = ci(bw2)

# section misattribution among matched quotes with a cited section
qm = q[q.matched & q.section_agrees.notna()].copy()
qm["mis"] = (~qm.section_agrees.astype(bool)).astype(int)
n3, d3 = paper_counts(qm, ORDER, "mis", "one")
res["matched_with_cited_section"] = d3.sum(0)
res["section_misattrib"] = point(n3, d3)
res["misattrib_lo"], res["misattrib_hi"] = ci(boot_rate(n3, d3))
res["section_misattrib_weighted"] = point(n3, d3, wvec)

# empty evidence among valid answers
v = ans[ans.valid]
n4, d4 = paper_counts(v, ORDER, "empty", "valid_i")
res["empty_evidence"] = point(n4, d4)
res["empty_lo"], res["empty_hi"] = ci(boot_rate(n4, d4))
res["empty_evidence_weighted"] = point(n4, d4, wvec)
res["mean_quotes_per_valid_answer"] = (res.quotes / res.valid_answers)
res.round(4).to_csv(OUT / "per_model.csv", index_label="model")
print("\nPER MODEL\n", res.round(4).T.to_string())

# ------------------------------------------------------------------ pairwise on unmatched-quote rate
n5, d5 = paper_counts(q, ORDER, "unm_i", "one")
bu = boot_rate(n5, d5)  # paired: same paper resample for every model
pu = point(n5, d5)
pw = []
for i in range(6):
    for j in range(i + 1, 6):
        diff = bu[:, i] - bu[:, j]; l, h = np.nanpercentile(diff, [2.5, 97.5])
        pw.append(dict(model_a=ORDER[i], model_b=ORDER[j], unmatched_a=pu[i], unmatched_b=pu[j],
                       diff_a_minus_b=pu[i] - pu[j], ci_lo=l, ci_hi=h, ci_excludes_0=bool(l > 0 or h < 0)))
pw = pd.DataFrame(pw)
pw.round(4).to_csv(OUT / "pairwise_unmatched_rate.csv", index=False)
print("\nPAIRWISE unmatched-rate difference (paired cluster bootstrap, 95% CI, 15 comparisons, uncorrected)\n",
      pw.round(4).to_string())

# ------------------------------------------------------------------ per criterion and model x criterion
def grid(df, num, den, by):
    return df.groupby(by).apply(lambda g: g[num].sum() / g[den].sum() if g[den].sum() else np.nan, include_groups=False)

pc = pd.DataFrame({
    "quotes": q.groupby("criterion").one.sum(),
    "match_rate": grid(q, "matched_i", "one", "criterion"),
    "match_rate_weighted": q.groupby("criterion").apply(lambda g: (g.w * g.matched_i).sum() / g.w.sum(), include_groups=False),
    "section_misattrib": grid(qm, "mis", "one", "criterion"),
    "empty_evidence": grid(v, "empty", "valid_i", "criterion"),
    "share_answers_any_unmatched": grid(ans[ans.has_quotes == 1], "has_unm", "has_quotes", "criterion"),
}).reindex(CRITS)
# cluster bootstrap CI for per-criterion match rate
for c in CRITS:
    s = q[q.criterion == c]
    nn, dd = paper_counts(s.assign(model=ORDER[0]), ORDER[:1], "matched_i", "one")
    pc.loc[c, "match_lo"], pc.loc[c, "match_hi"] = ci(boot_rate(nn, dd))[:, 0]
pc.round(4).to_csv(OUT / "per_criterion.csv", index_label="criterion")
print("\nPER CRITERION\n", pc.round(4).to_string())

mc = pd.DataFrame({
    "quotes": q.groupby(["model", "criterion"]).one.sum(),
    "unmatched": q.groupby(["model", "criterion"]).unm_i.sum(),
    "match_rate": grid(q, "matched_i", "one", ["model", "criterion"]),
    "section_misattrib": grid(qm, "mis", "one", ["model", "criterion"]),
    "empty_evidence": grid(v, "empty", "valid_i", ["model", "criterion"]),
    "valid_answers": v.groupby(["model", "criterion"]).valid_i.sum(),
    "share_answers_any_unmatched": grid(ans[ans.has_quotes == 1], "has_unm", "has_quotes", ["model", "criterion"]),
}).reset_index()
mc["model"] = pd.Categorical(mc.model, ORDER, ordered=True)
mc = mc.sort_values(["model", "criterion"])
mc.round(4).to_csv(OUT / "model_by_criterion.csv", index=False)
print("\nMODEL x CRITERION match rate\n",
      mc.pivot(index="model", columns="criterion", values="match_rate").round(3).to_string())
print("\nMODEL x CRITERION empty evidence\n",
      mc.pivot(index="model", columns="criterion", values="empty_evidence").round(3).to_string())

# empty evidence by verdict
ev = v.assign(verdict_c=v.verdict).groupby(["model", "verdict_c"]).agg(
    answers=("valid_i", "sum"), empty=("empty", "sum")).reset_index()
ev["empty_share"] = ev["empty"] / ev["answers"]
ev.round(4).to_csv(OUT / "empty_evidence_by_verdict.csv", index=False)

# ------------------------------------------------------------------ 4. classify unmatched quotes
sys.path.insert(0, str(ROOT))
from papers.evidence import _ELLIPSIS, _FRAGMENT_EDGE, _normalize_plain  # noqa: E402

_app = None


def passages_for(pmid):
    f = CACHE / f"{pmid}.json"
    if f.exists():
        return json.loads(f.read_text())
    global _app
    from config.app_config import load_app_config
    from papers import load_paper_from_config
    if _app is None:
        _app = load_app_config((main_checkout() / ".env.yaml"))
        _app.papers.raw_prefix = RAW_PREFIX  # the study papers live under pilot_2026_10
    paper = load_paper_from_config(pmid, _app)  # read-only MinIO get
    data = [(p.section, _normalize_plain(p.text)) for p in paper.passages if p.text.strip()]
    f.write_text(json.dumps(data))
    return data


def coverage(piece, text):
    """Share of `piece` chars aligned (in order, blocks >= MIN_BLOCK) to `text`; plus span in text."""
    sm = difflib.SequenceMatcher(None, text, piece, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size >= MIN_BLOCK]
    if not blocks:
        return 0.0, (0, 0)
    return sum(b.size for b in blocks) / len(piece), (blocks[0].a, blocks[-1].a + blocks[-1].size)


def best_match(piece, passages):
    best = (0.0, -1, (0, 0))
    for k, (_, txt) in enumerate(passages):
        if len(piece) > 40 and len(set(piece.split()) & set(txt.split())) < 3:
            continue  # cheap prefilter: no shared vocabulary
        cov, span = coverage(piece, txt)
        if cov > best[0]:
            best = (cov, k, span)
    return best


def split_pieces(quote):
    parts = [_normalize_plain(p).strip(_FRAGMENT_EDGE) for p in _ELLIPSIS.split(quote or "")]
    out = []
    for part in parts:
        for s in re.split(r"(?<=[.;!?])\s+(?=[a-z0-9\[(\"'])", part):  # text is lower-cased by _normalize
            s = s.strip(_FRAGMENT_EDGE)
            if not s:
                continue
            if out and len(s) < MIN_PIECE:
                out[-1] += " " + s
            else:
                out.append(s)
    if len(out) > 1 and len(out[0]) < MIN_PIECE:
        out[1] = out[0] + " " + out[1]; out = out[1:]
    return [o for o in out if o]


def classify(quote, passages):
    qn = _normalize_plain(quote or "").strip(_FRAGMENT_EDGE)
    if len(qn) < 12:
        return dict(kind="too_short", whole=np.nan, min_piece=np.nan, n_pieces=0)
    whole, wk, _ = best_match(qn, passages)
    pieces = split_pieces(quote)
    found = [best_match(p, passages) for p in pieces] if len(pieces) > 1 else []
    info = dict(whole=round(whole, 3), n_pieces=len(pieces),
                min_piece=round(min(f[0] for f in found), 3) if found else np.nan)
    if len(pieces) > 1 and all(f[0] >= T_PIECE for f in found):
        # separate fragments all exist; stitched unless they lie in order, in one passage, adjacent
        contiguous = all(f[1] == found[0][1] for f in found) and all(
            found[i + 1][2][0] - found[i][2][1] <= 20 and found[i + 1][2][0] >= found[i][2][0]
            for i in range(len(found) - 1))
        if not contiguous:
            same = len({f[1] for f in found}) == 1
            return dict(kind="b_stitched", sub="same_passage_gap_or_order" if same else "different_passages", **info)
    if whole >= T_PARA:
        return dict(kind="a_paraphrase", sub="near_verbatim_edit" if whole >= T_NEAR else "paraphrase", **info)
    return dict(kind="c_no_close_passage", sub="", **info)


um = q[q.unmatched].copy()
print(f"\nclassifying {len(um)} unmatched quotes over {um.pmid.nunique()} papers "
      f"(thresholds: piece>={T_PIECE}, paraphrase whole>={T_PARA}, near-verbatim>={T_NEAR}, blocks>={MIN_BLOCK} chars)")
cls = []
cache_q = {}
for r in um.itertuples():
    key = (r.pmid, r.quote)
    if key not in cache_q:
        cache_q[key] = classify(r.quote, passages_for(r.pmid))
    cls.append(cache_q[key])
cl = pd.DataFrame(cls, index=um.index)
um = um.join(cl)
um["sub"] = um["sub"].fillna("")
um.drop(columns=["one", "matched_i", "unm_i", "matched", "unmatched"]).to_csv(OUT / "unmatched_quotes_classified.csv", index=False)

kinds = um.groupby(["model", "kind"]).size().unstack(fill_value=0).reindex(ORDER).fillna(0).astype(int)
kinds["total_unmatched"] = kinds.sum(1)
tot = kinds.sum(0); tot.name = "ALL"
kinds = pd.concat([kinds, tot.to_frame().T])
share = kinds.drop(columns="total_unmatched").div(kinds.total_unmatched, axis=0).round(3).add_suffix("_share")
kinds = kinds.join(share)
kinds.to_csv(OUT / "unmatched_kinds.csv", index_label="model")
print("\nUNMATCHED KINDS\n", kinds.to_string())
sub = um.groupby(["kind", "sub"]).size()
sub.to_csv(OUT / "unmatched_kinds_sub.csv")
print("\n", sub.to_string())
kc = um.groupby(["criterion", "kind"]).size().unstack(fill_value=0)
kc.to_csv(OUT / "unmatched_kinds_by_criterion.csv")
print("\nUNMATCHED KINDS by criterion\n", kc.to_string())

# ------------------------------------------------------------------ 5. reason-verdict contradiction (keyword rules)
NEG = re.compile(r"\b(not|no|n't|neither|nor|without|fail(s|ed)?|lack(s|ed|ing)?|insufficient)\b", re.I)
CLAIM_NO = re.compile(r"(^|[.;:]\s*)(no|not)\b|\b(does|do|did) not (meet|satisfy|qualify|fulfil)|\bnot (met|satisfied|fulfilled)\b"
                      r"|\bfails? to (meet|satisfy|qualify)|\bdoes not (qualify|count)|\bcriterion (is )?not\b", re.I)
CLAIM_YES = re.compile(r"\b(criterion|requirement|condition|threshold) (is |are )?(met|satisfied|fulfilled)\b"
                       r"|\b(meets|satisfies|fulfils|fulfills) (the |this )?(criterion|requirement|condition|threshold)\b", re.I)
contr = []
# C5 is skipped: a true verdict there means "no animal model", so a negative-sounding reason
# ("no animal models are used") is consistent, not contradictory (first pass flagged 208 such cases).
for r in v[v.criterion != "criterion_5"].itertuples():
    if r.verdict == "True" and CLAIM_NO.search(r.reason[:160]) and not CLAIM_YES.search(r.reason):
        contr.append((r.model, r.pmid, r.criterion, r.verdict, "verdict true, reason negative", r.reason[:240]))
    elif r.verdict == "False" and CLAIM_YES.search(r.reason) and not NEG.search(r.reason):
        contr.append((r.model, r.pmid, r.criterion, r.verdict, "verdict false, reason positive", r.reason[:240]))
cdf = pd.DataFrame(contr, columns=["model", "pmid", "criterion", "verdict", "rule", "reason_start"])
cdf.to_csv(OUT / "reason_verdict_flags.csv", index=False)
print(f"\nREASON-VERDICT keyword flags: {len(cdf)} of {len(v)} valid answers (unvalidated; see CSV)")
print(cdf.groupby(["model", "rule"]).size().to_string() if len(cdf) else "none")

print("\nDONE. Tables in", OUT.relative_to(ROOT))
