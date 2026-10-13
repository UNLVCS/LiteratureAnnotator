"""
Download a stratified corpus: several PubMed queries, a sample from each, one manifest.

A strata file (YAML, see data_download/strata/) lists named queries with a target count.
For each stratum the full hit list is fetched, shuffled with a fixed seed, and walked
until ``n`` new full-text articles have been uploaded. A paper picked by an earlier
stratum is not picked again, but the manifest records every stratum whose hit list
contains it, so stratum membership can later serve as a noisy prior per criterion.

Outputs:
- ``{raw_articles_bucket}/{object_prefix}/{pmid}.json``  BioC JSON (PMC Open Access only)
- ``{raw_articles_bucket}/manifests/{object_prefix}.json`` and a local copy
  ``data_download/manifests/{object_prefix}.json``

Re-running is safe: papers already in the bucket are not fetched again.

Usage (from workspace root):
    python -m data_download.stratified_download data_download/strata/pilot_2026_10.yaml
    python -m data_download.stratified_download data_download/strata/pilot_2026_10.yaml --preview
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import time

import requests
import yaml
from minio.error import S3Error

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import load_app_config
from data_download.bioc_downloader import (
    _ensure_bucket,
    _get_minio_client,
    fetch_bioc_article,
    search_pubmed,
    upload_article,
)

log = logging.getLogger(__name__)

# esearch returns at most 9,999 IDs per request without the history server.
MAX_HITS = 9999
MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"


def _search(query: str, api_key, email) -> tuple[list[str], int]:
    """esearch with retries: NCBI returns sporadic 5xx under load."""
    for attempt in range(5):
        try:
            return search_pubmed(query, max_results=MAX_HITS, api_key=api_key, email=email)
        except requests.RequestException as exc:
            if attempt == 4:
                raise
            log.warning("esearch failed (%s), retrying", exc)
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def load_strata(path: str) -> dict[str, Any]:
    """Read a strata file and expand ``{fragment}`` references in each query."""
    spec = yaml.safe_load(Path(path).read_text())
    fragments = spec.get("fragments") or {}
    common = spec.get("common_filter")
    for stratum in spec["strata"]:
        query = stratum["query"].format(**fragments)
        stratum["full_query"] = f"({query}) AND {common}" if common else query
    return spec


def _object_exists(client, bucket: str, name: str) -> bool:
    try:
        client.stat_object(bucket, name)
        return True
    except S3Error as exc:
        if exc.code in ("NoSuchKey", "NoSuchObject"):
            return False
        raise


def _is_full_text(article: dict) -> bool:
    """BioC from pmcoa.cgi carries body passages; the abstract endpoint does not."""
    collections = article if isinstance(article, list) else [article]
    for collection in collections:
        for doc in collection.get("documents", []) if isinstance(collection, dict) else []:
            for p in doc.get("passages", []):
                if p.get("infons", {}).get("section_type") not in (None, "TITLE", "ABSTRACT"):
                    return True
    return False


def run(strata_path: str, preview: bool = False) -> dict[str, Any]:
    app = load_app_config()
    bd = app.bioc_download
    spec = load_strata(strata_path)
    prefix = spec["object_prefix"].strip("/")
    rng = random.Random(spec.get("seed", 0))

    hits: dict[str, list[str]] = {}
    for stratum in spec["strata"]:
        ids, total = _search(stratum["full_query"], bd.ncbi_api_key, bd.ncbi_email)
        hits[stratum["name"]] = ids
        stratum["total_hits"] = total
        print(f"{stratum['name']:24s} hits={total:>6}  target={stratum['n']}")
    if preview:
        return {}

    bucket = app.minio.raw_articles_bucket
    client = _get_minio_client(app.minio)
    _ensure_bucket(client, bucket)

    hit_sets = {name: set(ids) for name, ids in hits.items()}
    papers: dict[str, dict[str, Any]] = {}
    failed: dict[str, str] = {}

    for stratum in spec["strata"]:
        name = stratum["name"]
        candidates = sorted(hits[name])
        rng.shuffle(candidates)
        got = 0
        for pmid in candidates:
            if got >= stratum["n"]:
                break
            if pmid in papers or pmid in failed:
                continue
            object_name = f"{prefix}/{pmid}.json"
            if _object_exists(client, bucket, object_name):
                log.info("[%s] %s already in bucket", name, pmid)
            else:
                article = fetch_bioc_article(pmid, delay=spec.get("request_delay", bd.request_delay), full_text=True, full_text_fallback=False)
                if article is None:
                    failed[pmid] = "not in PMC Open Access BioC"
                    continue
                if not _is_full_text(article):
                    failed[pmid] = "no body text"
                    continue
                upload_article(client, bucket, pmid, article, prefix=prefix)
            papers[pmid] = {"sampled_from": name}
            got += 1
        stratum["downloaded"] = got
        print(f"{name:24s} downloaded {got}/{stratum['n']}")

    for pmid, info in papers.items():
        info["in_strata"] = sorted(s for s, ids in hit_sets.items() if pmid in ids)

    manifest = {
        "object_prefix": prefix,
        "bucket": bucket,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strata_file": Path(strata_path).name,
        "seed": spec.get("seed"),
        "strata": [
            {k: s.get(k) for k in ("name", "description", "full_query", "n", "total_hits", "downloaded", "expect")}
            for s in spec["strata"]
        ],
        "papers": papers,
        "failed": failed,
    }
    payload = json.dumps(manifest, indent=2).encode("utf-8")
    client.put_object(bucket, f"manifests/{prefix}.json", BytesIO(payload), len(payload), content_type="application/json")
    MANIFEST_DIR.mkdir(exist_ok=True)
    (MANIFEST_DIR / f"{prefix}.json").write_bytes(payload)
    print(f"\n{len(papers)} papers under {bucket}/{prefix}/ ({len(failed)} skipped); manifest at {bucket}/manifests/{prefix}.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a stratified PubMed corpus into MinIO.")
    parser.add_argument("strata_file", help="YAML strata spec, e.g. data_download/strata/pilot_2026_10.yaml")
    parser.add_argument("--preview", action="store_true", help="Only print hit counts per stratum.")
    args = parser.parse_args()
    run(args.strata_file, preview=args.preview)


if __name__ == "__main__":
    main()
