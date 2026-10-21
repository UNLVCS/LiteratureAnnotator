"""
Apply the LFs, fit the label model, and write scores and reports.

    weak_supervision/ws.sh samples.py          # once: MinIO -> data/samples.parquet
    weak_supervision/ws.sh pubmed.py           # once: PubMed pub types / MeSH cache
    weak_supervision/ws.sh entropy.py          # whenever resample passes or the cluster job finish
    weak_supervision/ws.sh run.py [--gold gold.csv]

Outputs (weak_supervision/data/out/, gitignored):
    lf_matrix.parquet   sample_id + one column per LF (1 VALID, 0 INVALID, -1 ABSTAIN)
    scores.parquet/.csv sample_id, pmid, criterion, provider, p_verdict, p_evidence, p_valid, label
    report.md           LF analysis, learned parameters, proxy evaluation
    spot_check.md       samples for a human to read
    gold_report.md      only with --gold
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parent)]

from entropy import load_entropy, load_semantic_entropy  # noqa: E402
from evaluate import invalid_rates, majority_vote_agreement, spot_check, stability  # noqa: E402
from label_model import fit_facets, lf_params_table  # noqa: E402
from lfs import apply_lfs, lf_summary  # noqa: E402
from pubmed import load_meta  # noqa: E402
from samples import DATA_DIR, load_samples  # noqa: E402

OUT_DIR = DATA_DIR / "out"


def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return df.to_markdown(floatfmt=floatfmt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", type=Path, help="Gold labels: pmid, criterion, provider, label.")
    parser.add_argument("--threshold", type=float, default=0.5, help="INVALID if p_valid < threshold.")
    parser.add_argument("--no-stability", action="store_true", help="Skip the leave-one-LF-out refits.")
    args = parser.parse_args()

    samples = load_samples()
    pubmed, entropy, sem_entropy = load_meta(), load_entropy(), load_semantic_entropy()
    print(f"{len(samples)} samples, PubMed metadata for {len(pubmed)} papers, entropy for {len(entropy)} samples")
    L = apply_lfs(samples, pubmed=pubmed, entropy=entropy, sem_entropy=sem_entropy)
    groups = samples["criterion"]
    scores, models = fit_facets(L, groups)
    is_invalid = scores["p_valid"] < args.threshold

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    L.assign(sample_id=samples["sample_id"]).to_parquet(OUT_DIR / "lf_matrix.parquet", index=False)
    out = samples[["sample_id", "pmid", "criterion", "provider"]].join(scores)
    out["label"] = is_invalid.map({True: "INVALID", False: "VALID"})
    out.to_parquet(OUT_DIR / "scores.parquet", index=False)
    out.to_csv(OUT_DIR / "scores.csv", index=False)

    report = [
        "# Weak-supervision report",
        "",
        "Every number below is a proxy computed without gold labels.",
        "",
        f"{len(samples)} samples; INVALID if p_valid < {args.threshold}: "
        f"**{int(is_invalid.sum())} ({is_invalid.mean():.1%})**. "
        f"Entropy resamples available for {len(entropy)} samples.",
        "",
        "## LF analysis", "", _md(lf_summary(L)), "",
        "## LF analysis per criterion", "",
    ]
    for crit, idx in groups.groupby(groups).groups.items():
        summ = lf_summary(L.loc[idx])
        report += [f"### {crit}", "", _md(summ[summ["coverage"] > 0]), ""]
    report += ["## Learned LF parameters", "",
               "Verdict facet: EM per criterion (`fitted`). Evidence facet and groups too small for EM: "
               "fixed prior accuracies and class prior.", "",
               _md(lf_params_table(models)), ""]
    report += ["## INVALID rates (plausibility)", ""]
    for name, table in invalid_rates(samples, is_invalid).items():
        report += [f"### by {name}", "", _md(table), ""]
    for facet in ("verdict", "evidence"):
        report += [f"### verdict vs evidence facet: INVALID rate from the {facet} facet alone", "",
                   _md(invalid_rates(samples, scores[f"p_{facet}"] < args.threshold)["provider"]), ""]
    report += ["## Agreement with the plain LF majority vote", "",
               _md(pd.DataFrame([majority_vote_agreement(L, scores["p_valid"], args.threshold)])), ""]

    if not args.no_stability:
        def refit(Lsub):
            return fit_facets(Lsub, groups)[0]["p_valid"]
        stab = stability(L, refit, args.threshold)
        fragile = stab.attrs["fragile"]
        report += ["## Stability: leave one LF out", "",
                   f"{int(fragile.sum())} samples ({fragile.mean():.1%}) change label when some single LF is "
                   "dropped. They are listed in scores.parquet as `fragile`.", "", _md(stab), ""]
        out["fragile"] = fragile
        out.to_parquet(OUT_DIR / "scores.parquet", index=False)
        out.to_csv(OUT_DIR / "scores.csv", index=False)

    (OUT_DIR / "report.md").write_text("\n".join(report))
    (OUT_DIR / "spot_check.md").write_text(spot_check(samples, L, scores))
    print(f"INVALID {int(is_invalid.sum())}/{len(samples)}; wrote {OUT_DIR}/report.md, spot_check.md, scores.*")

    if args.gold:
        from gold import evaluate, load_gold

        res = evaluate(samples, L, scores, load_gold(args.gold), args.threshold)
        text = ["# Gold-set evaluation", ""]
        for name, table in res.items():
            text += [f"## {name}", "", _md(table), ""]
        (OUT_DIR / "gold_report.md").write_text("\n".join(text))
        print(res["label_model"].to_string())


if __name__ == "__main__":
    main()
