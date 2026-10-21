"""
PubMed metadata (publication types and MeSH headings) for the pilot papers.

    weak_supervision/ws.sh pubmed.py   # fetch missing PMIDs into data/pubmed_meta.json

The BioC JSON in MinIO has neither field, so the pubmed_pubtype (C1) and
pubmed_mesh_animals (C5) LFs read this cache. efetch is called in batches of up to
200 PMIDs at <= 3 requests/s; the result is cached, so a rerun only fetches PMIDs
not seen before. PMIDs PubMed returns nothing for are cached as empty entries.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "weak_supervision")]

from data_download.mesh_lookup import _safe_request  # noqa: E402
from samples import DATA_DIR, SAMPLES_PATH  # noqa: E402

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
NCBI_TOOL_NAME = "literature_annotator_weak_supervision"
META_PATH = DATA_DIR / "pubmed_meta.json"
BATCH = 200


def parse_efetch(xml_text: str) -> Dict[str, Dict[str, List[str]]]:
    """{pmid: {"pub_types": [...], "mesh": [descriptor names]}} from efetch XML."""
    out = {}
    for article in ET.fromstring(xml_text).iter("PubmedArticle"):
        pmid = article.findtext("MedlineCitation/PMID")
        out[pmid] = {
            "pub_types": [e.text for e in article.iter("PublicationType") if e.text],
            "mesh": [e.text for e in article.iter("DescriptorName") if e.text],
        }
    return out


def fetch(pmids: Iterable[str], api_key: str | None = None) -> Dict[str, Dict[str, List[str]]]:
    pmids = list(pmids)
    last = [0.0]
    out: Dict[str, Dict[str, List[str]]] = {}
    for i in range(0, len(pmids), BATCH):
        chunk = pmids[i:i + BATCH]
        params = {"db": "pubmed", "id": ",".join(chunk), "retmode": "xml", "tool": NCBI_TOOL_NAME}
        if api_key:
            params["api_key"] = api_key
        resp = _safe_request(EFETCH_URL, params, min_delay=0.34, max_retries=5, last_request_time=last)
        got = parse_efetch(resp.text)
        out.update({p: got.get(p, {"pub_types": [], "mesh": []}) for p in chunk})
        print(f"efetch {i + len(chunk)}/{len(pmids)}")
    return out


def load_meta(path: Path = META_PATH) -> Dict[str, Dict[str, List[str]]]:
    return json.loads(path.read_text()) if path.exists() else {}


def update_cache(pmids: Iterable[str], api_key: str | None = None, path: Path = META_PATH) -> Dict[str, Dict]:
    meta = load_meta(path)
    missing = sorted(set(pmids) - set(meta))
    if missing:
        meta.update(fetch(missing, api_key))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=1))
    print(f"{len(meta)} PMIDs cached in {path} ({len(missing)} fetched)")
    return meta


def main() -> None:
    import pandas as pd
    from config.app_config import load_app_config

    pmids = pd.read_parquet(SAMPLES_PATH, columns=["pmid"])["pmid"].unique()
    update_cache(pmids, load_app_config().bioc_download.ncbi_api_key)


if __name__ == "__main__":
    main()
