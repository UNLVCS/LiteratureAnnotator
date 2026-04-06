"""
Look up MeSH descriptor IDs from NCBI by keyword and preview PubMed hit counts.

Provides both a programmatic API and a standalone CLI so you can discover
the right MeSH IDs — and test how many papers they match — before downloading.

Examples
--------
    # Search by keyword
    python -m data_download.mesh_lookup "Alzheimer Disease" -v

    # Expand related terms (e.g. amyloid, tau)
    python -m data_download.mesh_lookup "Alzheimer Disease" --related -v

    # Show available subheading qualifiers
    python -m data_download.mesh_lookup "Alzheimer Disease" --subheadings

    # Preview PubMed hit counts for term combinations
    python -m data_download.mesh_lookup --preview \
        "alzheimer disease/blood" "biomarkers/blood"

    # Programmatic
    from data_download.mesh_lookup import search_mesh, pubmed_count
    hits = search_mesh("Alzheimer Disease")
    count = pubmed_count('"alzheimer disease/blood"[MAJR]')
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import requests
from requests import Response

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
NCBI_TOOL_NAME = "literature_annotator_mesh_lookup"


def _safe_request(
    url: str,
    params: dict[str, Any],
    *,
    min_delay: float,
    max_retries: int,
    last_request_time: list[float],
) -> Response:
    """Rate-limited GET with retry/backoff for transient NCBI failures."""
    # Basic client-side rate limit to avoid bursts.
    elapsed = time.monotonic() - last_request_time[0]
    if elapsed < min_delay:
        time.sleep(min_delay - elapsed)

    retry = 0
    while True:
        resp = requests.get(url, params=params, timeout=30)
        last_request_time[0] = time.monotonic()

        if resp.status_code != 429 and resp.status_code < 500:
            resp.raise_for_status()
            return resp

        retry += 1
        if retry > max_retries:
            resp.raise_for_status()

        # Honor Retry-After if present, otherwise exponential backoff with jitter.
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            wait_s = float(retry_after)
        else:
            wait_s = min(2**retry, 30.0) + 0.1 * retry
        time.sleep(wait_s)


def search_mesh(
    keyword: str,
    max_results: int = 10,
    api_key: str | None = None,
    email: str | None = None,
    min_delay: float = 0.34,
    max_retries: int = 5,
    last_request_time: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Search the MeSH database by keyword and return matching descriptors.

    Each result dict contains ``uid``, ``mesh_ui`` (e.g. D000544),
    ``name``, ``scope_note``, and ``synonyms``.
    """
    params: dict[str, Any] = {
        "db": "mesh",
        "term": keyword,
        "retmax": max_results,
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email
    params["tool"] = NCBI_TOOL_NAME

    clock = last_request_time or [0.0]
    resp = _safe_request(
        ESEARCH_URL,
        params,
        min_delay=min_delay,
        max_retries=max_retries,
        last_request_time=clock,
    )
    uids = resp.json().get("esearchresult", {}).get("idlist", [])

    if not uids:
        return []

    return _fetch_summaries(
        uids,
        api_key=api_key,
        email=email,
        min_delay=min_delay,
        max_retries=max_retries,
        last_request_time=clock,
    )


def get_mesh_details(
    uid: str,
    api_key: str | None = None,
    email: str | None = None,
    min_delay: float = 0.34,
    max_retries: int = 5,
    last_request_time: list[float] | None = None,
) -> dict[str, Any] | None:
    """Fetch full details for a single MeSH UID (e.g. ``68000544``)."""
    results = _fetch_summaries(
        [uid],
        api_key=api_key,
        email=email,
        min_delay=min_delay,
        max_retries=max_retries,
        last_request_time=last_request_time or [0.0],
    )
    return results[0] if results else None


def _fetch_summaries(
    uids: list[str],
    api_key: str | None = None,
    email: str | None = None,
    min_delay: float = 0.34,
    max_retries: int = 5,
    last_request_time: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Call esummary for a batch of MeSH UIDs and normalise the output."""
    params: dict[str, Any] = {
        "db": "mesh",
        "id": ",".join(uids),
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email
    params["tool"] = NCBI_TOOL_NAME

    resp = _safe_request(
        ESUMMARY_URL,
        params,
        min_delay=min_delay,
        max_retries=max_retries,
        last_request_time=last_request_time or [0.0],
    )
    data = resp.json().get("result", {})

    results: list[dict[str, Any]] = []
    for uid in uids:
        entry = data.get(uid)
        if not entry:
            continue
        results.append({
            "uid": uid,
            "mesh_ui": entry.get("ds_meshui", ""),
            "name": (entry.get("ds_meshterms") or [""])[0],
            "scope_note": entry.get("ds_scopenote", ""),
            "synonyms": entry.get("ds_meshterms", []),
            "subheadings": entry.get("ds_subheading", []),
            "see_related_uids": entry.get("ds_seerelated", []),
            "tree_numbers": [
                link["treenum"] for link in entry.get("ds_idxlinks", [])
            ],
        })
    return results


def find_mesh_ids_for_topic(
    *keywords: str,
    include_related: bool = False,
    api_key: str | None = None,
    email: str | None = None,
    min_delay: float = 0.34,
    max_retries: int = 5,
) -> list[dict[str, Any]]:
    """High-level helper: search multiple keywords and optionally expand
    related terms. Returns a deduplicated list of descriptors."""
    seen: set[str] = set()
    all_results: list[dict[str, Any]] = []
    clock = [0.0]

    for kw in keywords:
        hits = search_mesh(
            kw,
            api_key=api_key,
            email=email,
            min_delay=min_delay,
            max_retries=max_retries,
            last_request_time=clock,
        )
        for hit in hits:
            if hit["uid"] not in seen:
                seen.add(hit["uid"])
                all_results.append(hit)

            if include_related:
                for rel_uid in hit.get("see_related_uids", []):
                    if rel_uid not in seen:
                        detail = get_mesh_details(
                            rel_uid,
                            api_key=api_key,
                            email=email,
                            min_delay=min_delay,
                            max_retries=max_retries,
                            last_request_time=clock,
                        )
                        if detail:
                            seen.add(rel_uid)
                            all_results.append(detail)

    return all_results


# ------------------------------------------------------------------
# PubMed hit-count preview
# ------------------------------------------------------------------

def pubmed_count(
    query: str,
    api_key: str | None = None,
    email: str | None = None,
    min_delay: float = 0.34,
    max_retries: int = 5,
    last_request_time: list[float] | None = None,
) -> int:
    """Return the total PubMed hit count for *query* without fetching IDs."""
    params: dict[str, Any] = {
        "db": "pubmed",
        "term": query,
        "retmax": 0,
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email
    params["tool"] = NCBI_TOOL_NAME

    resp = _safe_request(
        ESEARCH_URL,
        params,
        min_delay=min_delay,
        max_retries=max_retries,
        last_request_time=last_request_time or [0.0],
    )
    return int(resp.json().get("esearchresult", {}).get("count", 0))


def preview_queries(
    terms: list[str],
    api_key: str | None = None,
    email: str | None = None,
    min_delay: float = 0.34,
    max_retries: int = 5,
) -> None:
    """Print PubMed hit counts for several query strategies built from *terms*.

    Helps you compare broad vs. focused queries before committing to a download.
    """
    mesh_tag = "[MeSH Terms]"
    majr_tag = "[MAJR]"

    broad = " AND ".join(f'"{t}"{mesh_tag}' for t in terms)
    major = " AND ".join(f'"{t}"{majr_tag}' for t in terms)

    queries = [
        ("Broad  [MeSH Terms]", broad),
        ("Focused  [MAJR]", major),
    ]

    # Also show each individual term's MAJR count
    if len(terms) > 1:
        for t in terms:
            queries.append((f"  solo: {t}", f'"{t}"{majr_tag}'))

    clock = [0.0]
    print("\n  PubMed hit-count preview\n  " + "─" * 60)
    for label, q in queries:
        try:
            count = pubmed_count(
                q,
                api_key=api_key,
                email=email,
                min_delay=min_delay,
                max_retries=max_retries,
                last_request_time=clock,
            )
            print(f"  {count:>8,}  {label}")
        except Exception as exc:
            print(f"    ERROR  {label}: {exc}")

    print(f"\n  Suggested MESH_TERMS env var:")
    print(f"  MESH_TERMS={','.join(terms)}\n")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _print_results(
    results: list[dict[str, Any]],
    verbose: bool = False,
    show_subheadings: bool = False,
) -> None:
    for r in results:
        print(f"  {r['mesh_ui']}  {r['name']}")
        if verbose:
            scope = r.get("scope_note", "")
            if scope:
                truncated = scope[:200] + ("…" if len(scope) > 200 else "")
                print(f"           {truncated}")
            syns = r.get("synonyms", [])[1:6]
            if syns:
                print(f"           aka: {', '.join(syns)}")
            trees = r.get("tree_numbers", [])
            if trees:
                print(f"           tree: {', '.join(trees)}")
        if show_subheadings:
            subs = r.get("subheadings", [])
            if subs:
                print(f"           qualifiers: {', '.join(subs)}")
                name = r["name"].lower()
                examples = [f"{name}/{s}" for s in subs[:5]]
                print(f"           usage:  --terms \"{','.join(examples)}\"")
        if verbose or show_subheadings:
            print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Look up MeSH descriptor IDs and preview PubMed hit counts.",
    )
    parser.add_argument(
        "keywords", nargs="+",
        help="One or more search terms (or qualified terms for --preview).",
    )
    parser.add_argument(
        "--related", action="store_true",
        help="Also fetch 'see related' descriptors for each hit.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show scope notes, synonyms, and tree numbers.",
    )
    parser.add_argument(
        "--subheadings", "-s", action="store_true",
        help="Show available subheading qualifiers (e.g. /blood, /diagnosis).",
    )
    parser.add_argument(
        "--preview", "-p", action="store_true",
        help=(
            "Treat keywords as qualified MeSH terms and show PubMed hit counts. "
            "Example: --preview 'alzheimer disease/blood' 'biomarkers/blood'"
        ),
    )
    parser.add_argument("--api-key", default=None, help="NCBI API key.")
    parser.add_argument(
        "--email",
        default=None,
        help="NCBI contact email (recommended by NCBI eutils policy).",
    )
    parser.add_argument(
        "--min-delay",
        type=float,
        default=0.34,
        help="Minimum delay (seconds) between API calls. Increase to reduce 429s.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Retry attempts for 429/5xx responses.",
    )
    args = parser.parse_args()

    if args.preview:
        preview_queries(
            args.keywords,
            api_key=args.api_key,
            email=args.email,
            min_delay=args.min_delay,
            max_retries=args.max_retries,
        )
        return

    results = find_mesh_ids_for_topic(
        *args.keywords,
        include_related=args.related,
        api_key=args.api_key,
        email=args.email,
        min_delay=args.min_delay,
        max_retries=args.max_retries,
    )

    if not results:
        print("No MeSH descriptors found.")
        sys.exit(1)

    print(f"\nFound {len(results)} MeSH descriptor(s):\n")
    _print_results(results, verbose=args.verbose, show_subheadings=args.subheadings)

    mesh_ids = [r["mesh_ui"] for r in results if r["mesh_ui"]]
    print(f"MESH_IDS={','.join(mesh_ids)}")


if __name__ == "__main__":
    main()
