"""
Parse raw BioC JSON (as stored by data_download) into a Paper with passage offsets.

BioC passages carry ``offset`` (character position in the source document),
``infons.section_type`` (TITLE, ABSTRACT, INTRO, METHODS, ..., REF) and
``infons.type`` (front, abstract, title_1, title_2, paragraph, ref, ...).

``section_type`` is unreliable for PMC full text: many articles label every body
passage INTRO. Heading passages (``title_1``/``title_2``/...) are therefore used
to name sections, falling back to the mapped ``section_type``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

# BioC section types that carry no evidence for the screening criteria.
# References in particular mislead absence checks (cited "mouse model" / "review" titles).
DEFAULT_EXCLUDED_SECTION_TYPES = ("REF", "AUTH_CONT", "COMP_INT", "ACK_FUND")

# Captions/tables BioC places inline wherever they float in the body.
FLOATING_SECTION_TYPES = frozenset({"FIG", "TABLE"})

SECTION_NAMES = {
    "TITLE": "Title",
    "ABSTRACT": "Abstract",
    "INTRO": "Introduction",
    "METHODS": "Methods",
    "RESULTS": "Results",
    "DISCUSS": "Discussion",
    "CONCL": "Conclusion",
    "FIG": "Figure",
    "TABLE": "Table",
    "SUPPL": "Supplementary",
    "ABBR": "Abbreviations",
    "APPENDIX": "Appendix",
    "CASE": "Case report",
}


@dataclass(frozen=True)
class Passage:
    offset: int  # BioC document offset of text[0]
    text: str
    section_type: str  # raw BioC section_type, upper-cased ("" if absent)
    passage_type: str  # raw BioC infons.type ("" if absent)
    section: str  # human-readable section label, e.g. "Methods > Participants"

    @property
    def is_heading(self) -> bool:
        return self.passage_type.startswith("title") or self.passage_type == "abstract_title_1"


@dataclass
class Paper:
    pmid: str
    source_id: str  # BioC document id (PMC id for full text, PMID for abstracts)
    source: str  # BioC collection source, e.g. "PMC" or "PubMed"
    title: str
    passages: List[Passage] = field(default_factory=list)

    @property
    def is_full_text(self) -> bool:
        return any(
            p.section_type not in ("TITLE", "ABSTRACT", "") for p in self.passages if not p.is_heading
        )

    @property
    def sections(self) -> List[str]:
        seen: List[str] = []
        for p in self.passages:
            if p.section not in seen:
                seen.append(p.section)
        return seen


def _heading_level(passage_type: str) -> Optional[int]:
    if passage_type.startswith("title_"):
        try:
            return int(passage_type.split("_", 1)[1])
        except ValueError:
            return None
    return None


def parse_bioc(
    pmid: str,
    bioc_data: object,
    exclude_section_types: Iterable[str] = DEFAULT_EXCLUDED_SECTION_TYPES,
) -> Paper:
    """Convert raw BioC JSON from the NCBI API into a Paper.

    Args:
        pmid: Paper ID (MinIO object stem / queue ID).
        bioc_data: Parsed JSON. The API returns a list of collections; the first is used.
        exclude_section_types: BioC section types dropped entirely (References etc.).
    """
    if isinstance(bioc_data, list):
        bioc_data = bioc_data[0] if bioc_data else {}
    if not isinstance(bioc_data, dict):
        raise ValueError(f"Unexpected BioC payload type for {pmid}: {type(bioc_data).__name__}")

    documents = bioc_data.get("documents") or []
    paper = Paper(pmid=str(pmid), source_id="", source=str(bioc_data.get("source", "")), title="")
    if not documents:
        paper.title = paper.pmid
        return paper

    doc = documents[0]
    paper.source_id = str(doc.get("id", ""))
    excluded = {s.upper() for s in exclude_section_types}

    # Heading stack: level -> heading text. Reset when the body moves to a different
    # top-level section type; figures and tables are floated inline and must not reset it.
    headings: dict[int, str] = {}
    current_type = ""

    for raw in doc.get("passages", []):
        infons = raw.get("infons") or {}
        section_type = str(infons.get("section_type", "")).upper()
        passage_type = str(infons.get("type", ""))
        text = raw.get("text") or ""
        if section_type in excluded or not text.strip():
            continue

        floating = section_type in FLOATING_SECTION_TYPES
        if not floating and section_type != current_type:
            headings = {}
            current_type = section_type

        level = None if floating else _heading_level(passage_type)
        if level is not None:
            headings = {k: v for k, v in headings.items() if k < level}
            headings[level] = text.strip()

        base = SECTION_NAMES.get(section_type, section_type.title() or "Body")
        path = [headings[k] for k in sorted(headings)]
        if level is not None:
            path = path[:-1]  # a heading belongs to its parent section
        # Body text under generic INTRO tagging is usually not the introduction;
        # prefer the heading path whenever one exists.
        if floating:
            section = " > ".join(path + [base]) if path else base
        else:
            section = " > ".join(path) if path else base

        passage = Passage(
            offset=int(raw.get("offset", 0) or 0),
            text=text,
            section_type=section_type,
            passage_type=passage_type,
            section=section,
        )
        paper.passages.append(passage)
        if section_type == "TITLE" and not paper.title:
            paper.title = text.strip()

    if not paper.title:
        paper.title = paper.pmid
    return paper
