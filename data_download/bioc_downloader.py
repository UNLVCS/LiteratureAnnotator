"""
Download PubMed articles via the NCBI BioC RESTful API and persist them to MinIO.

Workflow
--------
1. Build a PubMed query from the configured MeSH terms / IDs / raw query.
2. Run esearch to obtain PMIDs, then fetch each article's BioC-JSON.
3. Upload each article to the configured MinIO bucket.

Usage
-----
    # Raw query — full control
    python -m data_download.bioc_downloader \
        --query '"alzheimer disease/blood"[MAJR] AND "biomarkers/blood"[MAJR]'

    # Qualified terms — auto-builds query with [MAJR] by default
    python -m data_download.bioc_downloader \
        --terms "alzheimer disease/blood,biomarkers/blood" --max-results 50

    # Descriptor IDs
    python -m data_download.bioc_downloader --mesh D000544,D015415

    # Preview hit count without downloading
    python -m data_download.bioc_downloader \
        --terms "alzheimer disease/blood,biomarkers/blood" --preview
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any
import requests
from minio import Minio

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import load_app_config, BiocDownloadSettings
from config.app_config import MinioConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

# Abstract-only — all PubMed articles
BIOC_PUBMED_URL = (
    "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pubmed.cgi/BioC_json/{pmid}/unicode"
)
# Full text — PMC Open Access subset only (returns 404 for paywalled articles)
BIOC_PMCOA_URL = (
    "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmid}/unicode"
)


# ------------------------------------------------------------------
# NCBI helpers
# ------------------------------------------------------------------

def build_query(bd: BiocDownloadSettings) -> str:
    """Build the PubMed query string from *bd*.

    Priority order:
    1. ``bd.mesh_query`` — used verbatim if set.
    2. ``bd.mesh_terms`` — each term (with optional ``/subheading``)
       is wrapped with the appropriate field tag and AND-joined.
    3. ``bd.mesh_ids`` — same as above but for raw descriptor UIDs.

    When ``bd.major_topic_only`` is True the field tag is ``[MAJR]``;
    otherwise it falls back to ``[MeSH Terms]``.
    """
    if bd.mesh_query:
        return bd.mesh_query

    tag = "[MAJR]" if bd.major_topic_only else "[MeSH Terms]"
    parts: list[str] = []

    for term in bd.mesh_terms:
        parts.append(f'"{term}"{tag}')

    for uid in bd.mesh_ids:
        parts.append(f"{uid}{tag}")

    if not parts:
        return ""

    return " AND ".join(parts)


def search_pubmed(
    query: str,
    max_results: int = 100,
    api_key: str | None = None,
    email: str | None = None,
) -> tuple[list[str], int]:
    """Run *query* against PubMed esearch.

    Returns ``(id_list, total_count)`` where *total_count* is the full
    number of hits in PubMed (useful for previewing before download).
    """
    log.info("PubMed query: %s", query)

    params: dict[str, Any] = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email

    resp = requests.get(ESEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    result = data.get("esearchresult", {})
    id_list: list[str] = result.get("idlist", [])
    total_count = int(result.get("count", 0))
    log.info("esearch returned %d total hits; fetched %d PMIDs", total_count, len(id_list))
    return id_list, total_count


def _fetch_url(url: str) -> dict | None:
    """GET *url* and return parsed JSON, or None on error."""
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        log.error("Request failed (%s): %s", url, exc)
        return None


def fetch_bioc_article(
    pmid: str,
    delay: float = 0.34,
    full_text: bool = False,
    full_text_fallback: bool = True,
) -> dict | None:
    """Fetch a single article in BioC-JSON format from the NCBI BioC API.

    When *full_text* is True the PMC Open Access endpoint (``pmcoa.cgi``) is
    used, which returns the complete article body for articles in the PMC OA
    subset.  If the article is not in PMC OA (404) and *full_text_fallback* is
    True, the PubMed abstract endpoint (``pubmed.cgi``) is tried instead.

    Returns the parsed JSON dict, or ``None`` if the article is unavailable.
    """
    time.sleep(delay)

    if full_text:
        result = _fetch_url(BIOC_PMCOA_URL.format(pmid=pmid))
        if result is not None:
            return result
        if full_text_fallback:
            log.warning(
                "PMID %s not in PMC Open Access — falling back to PubMed abstract", pmid
            )
            return _fetch_url(BIOC_PUBMED_URL.format(pmid=pmid))
        log.warning("PMID %s not in PMC Open Access (404) — skipping", pmid)
        return None

    result = _fetch_url(BIOC_PUBMED_URL.format(pmid=pmid))
    if result is None:
        log.warning("PMID %s not available in BioC (404)", pmid)
    return result


# ------------------------------------------------------------------
# MinIO helpers
# ------------------------------------------------------------------

def _get_minio_client(minio: MinioConfig) -> Minio:
    return Minio(
        minio.url,
        access_key=minio.access_key,
        secret_key=minio.secret_key,
        secure=minio.secure,
    )


def _ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        log.info("Creating MinIO bucket: %s", bucket)
        client.make_bucket(bucket)


def upload_article(client: Minio, bucket: str, pmid: str, article: dict, prefix: str = "") -> None:
    """Serialize *article* to JSON and upload to MinIO.

    *prefix* is an optional key prefix (folder path) within the bucket,
    e.g. "run_2026_04" → objects land at ``run_2026_04/{pmid}.json``.
    Bucket names cannot contain slashes — use this field for sub-folder layout.
    """
    payload = json.dumps(article).encode("utf-8")
    folder = prefix.strip("/") + "/" if prefix.strip("/") else ""
    object_name = f"{folder}{pmid}.json"
    client.put_object(
        bucket_name=bucket,
        object_name=object_name,
        data=BytesIO(payload),
        length=len(payload),
        content_type="application/json",
    )
    log.info("Uploaded → %s/%s", bucket, object_name)


# ------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------

def preview_query(bd: BiocDownloadSettings) -> int:
    """Print the query and total hit count without downloading anything."""
    query = build_query(bd)
    if not query:
        log.error("No MeSH query could be built from config.")
        return 0

    _, total = search_pubmed(
        query=query,
        max_results=0,
        api_key=bd.ncbi_api_key,
        email=bd.ncbi_email,
    )
    print(f"\n  Query:  {query}")
    print(f"  Hits:   {total:,}\n")
    return total


def download_articles(bd: BiocDownloadSettings, minio: MinioConfig) -> list[str]:
    """End-to-end: search PubMed by MeSH, fetch BioC JSON, upload to MinIO.

    Returns the list of PMIDs that were successfully uploaded.
    """
    query = build_query(bd)
    if not query:
        log.error("No MeSH query could be built — set mesh_query, mesh_terms, or mesh_ids in .env.yaml.")
        return []

    pmids, total = search_pubmed(
        query=query,
        max_results=bd.max_results,
        api_key=bd.ncbi_api_key,
        email=bd.ncbi_email,
    )
    if not pmids:
        log.warning("No PMIDs returned for the given MeSH query.")
        return []

    bucket = minio.raw_articles_bucket
    log.info("Target bucket (minio.raw_articles_bucket): %s", bucket)
    minio_client = _get_minio_client(minio)
    _ensure_bucket(minio_client, bucket)

    uploaded: list[str] = []
    for i, pmid in enumerate(pmids, 1):
        log.info("[%d/%d] Fetching PMID %s …", i, len(pmids), pmid)
        article = fetch_bioc_article(
            pmid,
            delay=bd.request_delay,
            full_text=bd.full_text,
            full_text_fallback=bd.full_text_fallback,
        )
        if article is None:
            continue
        try:
            upload_article(minio_client, bucket, pmid, article, prefix=bd.object_prefix)
            uploaded.append(pmid)
        except Exception as exc:
            log.error("Upload failed for PMID %s: %s", pmid, exc)

    log.info(
        "Done — %d / %d articles uploaded to %s",
        len(uploaded),
        len(pmids),
        bucket,
    )
    return uploaded


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PubMed articles via BioC API and store in MinIO.",
    )

    query_group = parser.add_argument_group("query options (pick one or combine)")
    query_group.add_argument(
        "--query",
        type=str,
        default=None,
        help="Raw PubMed query string (passed verbatim to esearch).",
    )
    query_group.add_argument(
        "--terms",
        type=str,
        default=None,
        help=(
            "Comma-separated MeSH term names with optional /subheading qualifiers. "
            "Example: 'alzheimer disease/blood,biomarkers/blood'"
        ),
    )
    query_group.add_argument(
        "--mesh",
        type=str,
        default=None,
        help="Comma-separated MeSH descriptor UIDs (e.g. D000544,D015415).",
    )
    query_group.add_argument(
        "--no-major",
        action="store_true",
        help="Use [MeSH Terms] instead of [MAJR] (broader, less focused results).",
    )

    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Max PMIDs to retrieve (overrides BIOC_MAX_RESULTS env var).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Only show the query and hit count — don't download anything.",
    )
    return parser.parse_args()


def main() -> None:
    app = load_app_config()
    # Take a mutable copy so CLI args can override YAML values without mutating
    # the cached AppConfig.
    bd = app.bioc_download.model_copy()

    args = _parse_args()
    if args.query:
        bd.mesh_query = args.query
    if args.terms:
        bd.mesh_terms = [t.strip() for t in args.terms.split(",") if t.strip()]
    if args.mesh:
        bd.mesh_ids = [m.strip() for m in args.mesh.split(",") if m.strip()]
    if args.no_major:
        bd.major_topic_only = False
    if args.max_results is not None:
        bd.max_results = args.max_results

    if args.preview:
        preview_query(bd)
        return

    uploaded = download_articles(bd, app.minio)
    if not uploaded:
        sys.exit(1)


if __name__ == "__main__":
    main()
