"""
Evaluation without gold labels. Every number here is a proxy, not an accuracy:

- spot_check: the most confident INVALID and a sample of confident VALID outputs, with
  reason, evidence and LF votes, for a human to read;
- stability: refit with each LF left out; samples whose label flips are fragile;
- plausibility: INVALID rates where we expect them to differ (by provider, by criterion,
  for unmatched quotes), plus agreement with the plain LF majority vote.
"""

from __future__ import annotations

import json
from typing import Callable, Dict

import numpy as np
import pandas as pd

from label_model import majority_vote
from lfs import ABSTAIN, INVALID, VALID

VOTE = {VALID: "VALID", INVALID: "INVALID", ABSTAIN: "·"}


def invalid_rates(samples: pd.DataFrame, is_invalid: pd.Series) -> Dict[str, pd.DataFrame]:
    d = samples.assign(invalid=is_invalid)
    rates = {
        "provider": d.groupby("provider")["invalid"].agg(["mean", "sum", "size"]),
        "criterion": d.groupby("criterion")["invalid"].agg(["mean", "sum", "size"]),
        "unmatched quote": d.groupby(d["n_unmatched"] > 0)["invalid"].agg(["mean", "sum", "size"]),
        "provider x criterion": d.pivot_table(index="provider", columns="criterion", values="invalid", aggfunc="mean"),
    }
    return rates


def stability(L: pd.DataFrame, fit: Callable[[pd.DataFrame], pd.Series], threshold: float = 0.5) -> pd.DataFrame:
    """Leave-one-LF-out: how many labels flip when each LF is dropped and the model refitted."""
    base = fit(L) < threshold
    rows, flipped_any = [], pd.Series(False, index=L.index)
    for lf in L.columns:
        if (L[lf] == ABSTAIN).all():
            continue
        alt = fit(L.drop(columns=lf)) < threshold
        flips = alt != base
        flipped_any |= flips
        rows.append({"left_out": lf, "flips": int(flips.sum()),
                     "to_invalid": int((flips & alt).sum()), "to_valid": int((flips & ~alt).sum()),
                     "invalid_rate_without": float(alt.mean())})
    table = pd.DataFrame(rows).set_index("left_out")
    table.attrs["fragile"] = flipped_any
    return table


def majority_vote_agreement(L: pd.DataFrame, p_valid: pd.Series, threshold: float = 0.5) -> Dict[str, float]:
    mv = majority_vote(L.to_numpy())
    decided = mv != 0.5
    lm_invalid = (p_valid < threshold).to_numpy()
    return {
        "mv_decided": float(decided.mean()),
        "agreement_where_mv_decided": float(((mv[decided] == 0) == lm_invalid[decided]).mean()),
        "mv_invalid_rate": float((mv == 0).mean()),
        "lm_invalid_rate": float(lm_invalid.mean()),
    }


def _sample_md(row: pd.Series, lf_names) -> str:
    evidence = json.loads(row["evidence_json"] or "[]")
    votes = ", ".join(f"{lf}={VOTE[int(row[lf])]}" for lf in lf_names if row[lf] != ABSTAIN) or "none"
    others = ", ".join(f"{c[2:]}={row[c]}" for c in ("v_claude", "v_luna", "v_gptoss", "v_qwen"))
    lines = [
        f"### {row['pmid']} · {row['criterion']} · {row['provider']}",
        f"- p_valid={row['p_valid']:.3f} (verdict {row['p_verdict']:.3f}, evidence {row['p_evidence']:.3f}); "
        f"stratum `{row['stratum']}`",
        f"- **verdict:** `{row['verdict']}` · other providers: {others}",
        f"- **reason:** {row['reason']}",
        f"- **LF votes:** {votes}",
    ]
    if row.get("error"):
        lines.append(f"- **error:** {row['error']}")
    for e in evidence:
        flag = "" if e.get("matched") else " **[UNMATCHED]**"
        sec = "" if e.get("section_agrees") is not False else f" **[cited {e.get('cited_section')!r}, found in {e.get('section')!r}]**"
        lines.append(f"  - > {e.get('quote')}{flag}{sec}")
    return "\n".join(lines)


def spot_check(samples: pd.DataFrame, L: pd.DataFrame, scores: pd.DataFrame, n: int = 30, seed: int = 0) -> str:
    """Markdown: the n most confident INVALID samples, then n random ones from the confident VALID pool.

    Parse failures (p_valid = 0 by rule, nothing to read) are counted, not listed. Confident VALID is sampled at random (p_valid >= 0.99) rather than taken from the top,
    where thousands of near-identical scores would all come from the same LF pattern.
    """
    d = samples.join(L).join(scores)
    no_answer = d["parse_error"] == INVALID if "parse_error" in d else pd.Series(False, index=d.index)
    invalid = d[~no_answer].sort_values("p_valid").head(n)
    pool = d[d["p_valid"] >= 0.99]
    valid = pool.sample(min(n, len(pool)), random_state=seed)
    parts = ["# Spot check (proxy evaluation: read and judge, these are not gold labels)", "",
             f"## {len(invalid)} most confident INVALID",
             "", f"({int(no_answer.sum())} parse failures, INVALID by rule, are not listed.)", ""]
    parts += [_sample_md(r, L.columns) + "\n" for _, r in invalid.iterrows()]
    parts += [f"## {len(valid)} random confident VALID (p_valid >= 0.99, {len(pool)} in pool)", ""]
    parts += [_sample_md(r, L.columns) + "\n" for _, r in valid.iterrows()]
    return "\n".join(parts)
