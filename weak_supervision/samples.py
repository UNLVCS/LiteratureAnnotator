"""
Sample table: one row per (pmid, criterion, provider) LLM output in synthetic-data-v3.

    python weak_supervision/samples.py            # MinIO -> data/samples.parquet
    python weak_supervision/samples.py --refresh  # re-download cached records

Raw records are cached under weak_supervision/data/raw/ (gitignored), so rebuilding the
table after a code change does not hit MinIO again. Each row carries the verdict, reason,
evidence statistics, errors, the paper's sampling strata and stratum priors, paper-text
features used by heuristic LFs, and every other provider's verdict for the same
(pmid, criterion).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "weak_supervision")]

from utilities.criteria import CRITERION_NAMES, NOT_APPLICABLE  # noqa: E402

DATA_DIR = ROOT / "weak_supervision" / "data"
SAMPLES_PATH = DATA_DIR / "samples.parquet"
MANIFEST = "manifests/pilot_2026_10.json"
RAW_PREFIX = "pilot_2026_10"

# Object prefix in synthetic-data-v3 -> short column-safe alias.
PROVIDERS = {
    "claude-sonnet-5-5": "claude",
    "gpt-6-luna": "luna",
    "openai/gpt-oss-120b": "gptoss",
    "qwen3.8:27b": "qwen",
}

# Verdicts are stored as strings so parquet columns have one type; None = no answer.
TRUE, FALSE = "true", "false"


def verdict_str(satisfied: Any) -> Optional[str]:
    if satisfied is True:
        return TRUE
    if satisfied is False:
        return FALSE
    if satisfied == NOT_APPLICABLE:
        return NOT_APPLICABLE
    return None


# ---------------------------------------------------------------------------
# Paper-text features (computed once per paper, used by na_consistency)
# ---------------------------------------------------------------------------

BLOOD_TERM = re.compile(r"\b(blood|serum|sera|plasma)\b", re.I)
# Mentions that are not blood as a sample: vascular measures (criterion 6 excludes them
# explicitly), the blood-brain barrier, cell-culture serum, plasma membranes, etc.
NON_SAMPLE_BLOOD = re.compile(
    r"\bblood[\s-]+(brain|csf|cerebrospinal|retina[l]?|nerve)[\s-]+barrier"
    r"|\bblood[\s-]+(pressure|flow|vessels?|supply|perfusion|glucose|sugar|oxygen\w*|volume|cells? count)"
    r"|\b(cerebral|regional)\s+blood\b"
    r"|\b(fetal|foetal)\s+(bovine|calf)\s+serum|\bFBS\b|\b(horse|goat|donkey|bovine|calf|normal)\s+serum"
    r"|\bserum[\s-]+free\b|\bplasma\s+membranes?\b|\bplasma\s+cells?\b|\bplasmalogen",
    re.I,
)


def blood_mentions(text: str) -> Dict[str, int]:
    """Literal blood/serum/plasma mentions, split into sample-like and other uses."""
    total = len(BLOOD_TERM.findall(text))
    other = sum(len(BLOOD_TERM.findall(m.group(0))) for m in NON_SAMPLE_BLOOD.finditer(text))
    return {"blood_mentions": total, "blood_sample_mentions": max(total - other, 0)}


# ---------------------------------------------------------------------------
# Record -> rows
# ---------------------------------------------------------------------------


def evidence_stats(evidence: List[Dict[str, Any]]) -> Dict[str, int]:
    matched = [e for e in evidence if e.get("matched")]
    return {
        "n_quotes": len(evidence),
        "n_matched": len(matched),
        "n_unmatched": len(evidence) - len(matched),
        # section_agrees is None when the model gave no section; only False is a mismatch.
        "n_section_mismatch": sum(1 for e in matched if e.get("section_agrees") is False),
    }


def record_rows(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per criterion of one result record (without cross-provider columns)."""
    paper = record.get("paper") or {}
    rows = []
    for cr in record["criteria_results"]:
        name = cr["criterion"]
        answer = (cr.get("response") or {}).get(name)
        answer = answer if isinstance(answer, dict) else {}
        evidence = cr.get("evidence") or []
        usage = cr.get("usage") or {}
        rows.append(
            {
                "pmid": str(record["paper_id"]),
                "criterion": name,
                "provider": record["provider"],
                "verdict": verdict_str(answer.get("satisfied")) if cr.get("error") is None else None,
                "reason": answer.get("reason") if isinstance(answer.get("reason"), str) else None,
                "error": cr.get("error"),
                "has_response": cr.get("response") is not None,
                "finish_reason": cr.get("finish_reason"),
                "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
                **evidence_stats(evidence),
                "evidence_json": json.dumps(
                    [{k: e.get(k) for k in ("quote", "cited_section", "matched", "section", "section_agrees")}
                     for e in evidence],
                    ensure_ascii=False,
                ),
                "is_full_text": paper.get("is_full_text"),
                "truncated": paper.get("truncated"),
                "rendered_chars": paper.get("rendered_chars"),
                "record_errors": len(record.get("errors") or []),
            }
        )
    return rows


def add_cross_provider(df: pd.DataFrame) -> pd.DataFrame:
    """Columns v_<alias>: each provider's verdict on the same (pmid, criterion)."""
    wide = df.pivot_table(index=["pmid", "criterion"], columns="provider", values="verdict", aggfunc="first")
    wide = wide.reindex(columns=list(PROVIDERS)).rename(columns=lambda p: f"v_{PROVIDERS[p]}")
    wide.columns.name = None
    return df.merge(wide.reset_index(), on=["pmid", "criterion"], how="left")


def add_strata(df: pd.DataFrame, manifest: Dict[str, Any]) -> pd.DataFrame:
    """Stratum columns and the combined stratum prior for each (pmid, criterion).

    ``prior`` is the shared ``expect`` value over every stratum the paper's query matched
    (``in_strata``); None when no stratum states one or the strata disagree.
    """
    expects = {s["name"]: s.get("expect") or {} for s in manifest["strata"]}
    papers = manifest["papers"]

    def prior(pmid: str, criterion: str) -> Optional[bool]:
        votes = {expects[s][criterion] for s in papers.get(pmid, {}).get("in_strata", []) if criterion in expects[s]}
        return votes.pop() if len(votes) == 1 else None

    df["stratum"] = df["pmid"].map(lambda p: papers.get(p, {}).get("sampled_from"))
    df["in_strata"] = df["pmid"].map(lambda p: ",".join(papers.get(p, {}).get("in_strata", [])))
    df["prior"] = [prior(p, c) for p, c in zip(df["pmid"], df["criterion"])]
    return df


def build_table(records: Iterable[Dict[str, Any]], manifest: Dict[str, Any],
                paper_features: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = [row for record in records for row in record_rows(record)]
    df = pd.DataFrame(rows)
    df["alias"] = df["provider"].map(PROVIDERS)
    df = add_cross_provider(df)
    df = add_strata(df, manifest)
    feats = pd.DataFrame.from_dict(paper_features, orient="index")
    feats.index.name = "pmid"
    df = df.merge(feats.reset_index(), on="pmid", how="left")
    df["sample_id"] = df["pmid"] + "|" + df["criterion"] + "|" + df["provider"]
    order = {c: i for i, c in enumerate(CRITERION_NAMES)}
    return df.sort_values(["pmid", "criterion", "provider"], key=lambda s: s.map(order) if s.name == "criterion" else s)\
             .reset_index(drop=True)


# ---------------------------------------------------------------------------
# MinIO access with a local cache
# ---------------------------------------------------------------------------


def minio_client(app_config):
    from minio import Minio

    m = app_config.minio
    return Minio(m.url, access_key=m.access_key, secret_key=m.secret_key, secure=m.secure)


def cached_get(client, bucket: str, name: str, cache: Path, refresh: bool = False) -> bytes:
    if cache.exists() and not refresh:
        return cache.read_bytes()
    resp = client.get_object(bucket, name)
    try:
        data = resp.read()
    finally:
        resp.close()
        resp.release_conn()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(data)
    return data


def list_results(client, bucket: str, prefix: str = "") -> List[str]:
    """Result objects ``[<prefix>/]<provider>/<pmid>.json`` for the known providers."""
    base = f"{prefix.strip('/')}/" if prefix else ""
    names = []
    for provider in PROVIDERS:
        names += [o.object_name for o in client.list_objects(bucket, prefix=f"{base}{provider}/", recursive=True)
                  if o.object_name.endswith(".json")]
    return names


def fetch_records(client, bucket: str, prefix: str = "", refresh: bool = False, workers: int = 16
                  ) -> List[Dict[str, Any]]:
    names = list_results(client, bucket, prefix)
    cache_root = DATA_DIR / "raw" / bucket

    def get(name: str) -> Dict[str, Any]:
        return json.loads(cached_get(client, bucket, name, cache_root / name, refresh))

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(get, names))


def fetch_paper_features(app_config, client, pmids: Iterable[str], refresh: bool = False, workers: int = 16
                         ) -> Dict[str, Dict[str, Any]]:
    """blood_mentions / blood_sample_mentions per paper, over the text the models saw."""
    from papers import parse_bioc, render_paper

    bucket = app_config.minio.raw_articles_bucket
    cache_root = DATA_DIR / "raw" / bucket

    def features(pmid: str) -> Dict[str, Any]:
        name = f"{RAW_PREFIX}/{pmid}.json"
        data = json.loads(cached_get(client, bucket, name, cache_root / name, refresh))
        paper = parse_bioc(pmid, data, app_config.papers.exclude_section_types)
        text = render_paper(paper, max_chars=app_config.papers.max_chars).text
        return blood_mentions(text)

    pmids = sorted(set(pmids))
    with ThreadPoolExecutor(workers) as pool:
        return dict(zip(pmids, pool.map(features, pmids)))


def load_manifest(app_config, client, refresh: bool = False) -> Dict[str, Any]:
    bucket = app_config.minio.raw_articles_bucket
    return json.loads(cached_get(client, bucket, MANIFEST, DATA_DIR / "raw" / bucket / MANIFEST, refresh))


def snapshot(refresh: bool = False, out: Path = SAMPLES_PATH) -> pd.DataFrame:
    from config.app_config import load_app_config

    app_config = load_app_config()
    client = minio_client(app_config)
    records = fetch_records(client, app_config.minio.synthetic_data_bucket, refresh=refresh)
    manifest = load_manifest(app_config, client, refresh)
    feats = fetch_paper_features(app_config, client, (str(r["paper_id"]) for r in records), refresh)
    df = build_table(records, manifest, feats)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"{len(df)} samples from {len(records)} records -> {out}")
    return df


def load_samples(path: Path = SAMPLES_PATH) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{path} missing; run `weak_supervision/ws.sh samples.py` first")
    return pd.read_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="Re-download records instead of using the cache.")
    args = parser.parse_args()
    snapshot(refresh=args.refresh)


if __name__ == "__main__":
    main()
