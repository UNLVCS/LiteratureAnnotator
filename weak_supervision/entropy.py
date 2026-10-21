"""
Verdict entropy over the semantic-entropy resamples (HANDOFF §5a, level (a)).

    weak_supervision/ws.sh entropy.py   # entropy/s01..s05 in MinIO -> data/entropy.parquet

Resample k of a provider lives at ``entropy/s<k>/<provider>/<pmid>.json`` in
synthetic-data-v3, written by the labelers with ``minio.output_prefix`` overridden.
For each (pmid, criterion, provider) this computes, over the resamples that parsed:

    n        number of resampled verdicts
    entropy  Shannon entropy (nats) of the verdict distribution
    mode     most common resampled verdict (ties broken by first seen)
    counts   JSON {verdict: count}

The original sample is not included, so the LF can ask whether it agrees with the mode.
Prefixes that do not exist yet are skipped; rerun as passes finish.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "weak_supervision")]

from samples import DATA_DIR, fetch_records, minio_client, record_rows  # noqa: E402

ENTROPY_PATH = DATA_DIR / "entropy.parquet"
PREFIXES = [f"entropy/s{k:02d}" for k in range(1, 6)]


def verdict_entropy(verdicts: Iterable[str]) -> Dict:
    verdicts = [v for v in verdicts if v is not None]
    counts = Counter(verdicts)
    n = len(verdicts)
    h = -sum(c / n * math.log(c / n) for c in counts.values()) if n else float("nan")
    mode = counts.most_common(1)[0][0] if n else None
    return {"n": n, "entropy": h, "mode": mode, "counts": json.dumps(dict(counts))}


def entropy_table(resample_rows: List[Dict]) -> pd.DataFrame:
    """One row per sample_id from the per-criterion rows of every resample record."""
    df = pd.DataFrame(resample_rows)
    if df.empty:
        return pd.DataFrame(columns=["sample_id", "n", "entropy", "mode", "counts"])
    df["sample_id"] = df["pmid"] + "|" + df["criterion"] + "|" + df["provider"]
    out = df.groupby("sample_id")["verdict"].apply(lambda s: pd.Series(verdict_entropy(s))).unstack()
    return out.reset_index()


def load_entropy(path: Path = ENTROPY_PATH) -> Dict[str, Dict]:
    """{sample_id: {"n", "entropy", "mode"}} for lfs.apply_lfs; empty if not computed yet."""
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return {r["sample_id"]: {"n": int(r["n"]), "entropy": float(r["entropy"]), "mode": r["mode"]}
            for r in df.to_dict("records")}


SEMANTIC_PATH = DATA_DIR / "semantic_entropy.parquet"
SEMANTIC_OBJECT = "entropy/semantic_entropy.parquet"  # written by semantic_entropy.py on the cluster


def fetch_semantic_entropy(path: Path = SEMANTIC_PATH) -> bool:
    """Download the cluster job's table from MinIO; False if it does not exist yet."""
    from config.app_config import load_app_config
    from minio.error import S3Error

    app_config = load_app_config()
    try:
        minio_client(app_config).fget_object(app_config.minio.synthetic_data_bucket, SEMANTIC_OBJECT, str(path))
    except S3Error as exc:
        if exc.code in ("NoSuchKey", "NoSuchObject"):
            return False
        raise
    return True


def load_semantic_entropy(path: Path = SEMANTIC_PATH) -> Dict[str, Dict]:
    """{sample_id: {"n", "semantic_entropy", "orig_in_largest", "largest_share"}}; empty if absent."""
    if not path.exists():
        return {}
    cols = ["n", "semantic_entropy", "orig_in_largest", "largest_share"]
    return {r["sample_id"]: {c: r[c] for c in cols} for r in pd.read_parquet(path).to_dict("records")}


def main() -> None:
    from config.app_config import load_app_config

    app_config = load_app_config()
    client = minio_client(app_config)
    rows = []
    for prefix in PREFIXES:
        records = fetch_records(client, app_config.minio.synthetic_data_bucket, prefix=prefix)
        print(f"{prefix}: {len(records)} records")
        rows += [r for rec in records for r in record_rows(rec)]
    table = entropy_table(rows)
    table.to_parquet(ENTROPY_PATH, index=False)
    print(f"{len(table)} samples with resamples -> {ENTROPY_PATH}")
    found = fetch_semantic_entropy()
    print(f"semantic entropy table: {'downloaded to ' + str(SEMANTIC_PATH) if found else 'not in MinIO yet'}")


if __name__ == "__main__":
    main()
