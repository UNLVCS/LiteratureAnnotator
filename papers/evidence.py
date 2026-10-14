"""
Resolve model-cited quotes back to exact locations in the source paper.

The labeler asks models for ``{"section": ..., "quote": <verbatim span>}`` instead of
page numbers (BioC has none) or offsets (models report numbers unreliably). A quote is
located by string match against the parsed passages after normalizing whitespace,
typographic quotes and dashes, ignoring in-text figure/table/citation markers and
punctuation at the ends of the quote. A quote that cannot be located is reported as
unmatched; that is a hallucinated or paraphrased citation and a useful quality signal.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from papers.bioc import Paper, Passage

_CHAR_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
}
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…|\[\.\.\.\])\s*")
# In-text figure/table pointers and numeric citations. Models routinely drop these when
# quoting ("... p < 0.05 (Fig. 2a)." -> "... p < 0.05."), so both the source and the quote
# are matched with them removed.
_REFERENCE_MARKERS = re.compile(
    r"\s*\((?:see\s+|cf\.\s+)?(?:fig|figs|figure|figures|table|tables|suppl|supplementary|additional\s+file|online\s+resource)\b[^()]*\)"
    r"|\s*\[\d+(?:\s*[,\u2013\u2014-]\s*\d+)*\]",
    re.IGNORECASE,
)
# Trimmed from the ends of each quote fragment: models close a quote with a period or
# quotation mark even when the source sentence continues.
_FRAGMENT_EDGE = " \"'.,;:"
# Fragments shorter than this are too generic to count as a located citation.
MIN_FRAGMENT_CHARS = 12


@dataclass(frozen=True)
class ResolvedEvidence:
    quote: str
    cited_section: Optional[str]
    matched: bool
    start: Optional[int] = None  # BioC document offset of the first matched char
    end: Optional[int] = None  # BioC document offset one past the last matched char
    section: Optional[str] = None  # section of the passage the quote was found in
    section_agrees: Optional[bool] = None  # cited section == located section

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize(text: str) -> Tuple[str, List[int]]:
    """Normalize text, returning it with a map from each output char to its source index."""
    out: List[str] = []
    index: List[int] = []
    prev_space = True
    after_dash = False
    skip = set()
    for m in _REFERENCE_MARKERS.finditer(text):
        skip.update(range(m.start(), m.end()))
    for i, ch in enumerate(text):
        if i in skip:
            continue
        ch = _CHAR_MAP.get(ch, ch)
        if ch.isspace():
            # Collapse runs, and drop spacing around dashes ("a - b" == "a—b").
            if prev_space or after_dash:
                continue
            ch = " "
            prev_space = True
        else:
            if ch == "-" and out and out[-1] == " ":
                out.pop()
                index.pop()
            prev_space = False
        after_dash = ch == "-"
        out.append(ch.lower())
        index.append(i)
    while out and out[-1] == " ":
        out.pop()
        index.pop()
    return "".join(out), index


def _normalize_plain(text: str) -> str:
    return _normalize(text)[0]


def _find_in_passage(passage: Passage, fragments: List[str]) -> Optional[Tuple[int, int]]:
    """Locate all fragments, in order, within one passage. Returns source (start, end)."""
    norm, index = _normalize(passage.text)
    pos = 0
    first = last = None
    for frag in fragments:
        hit = norm.find(frag, pos)
        if hit < 0:
            return None
        if first is None:
            first = hit
        last = hit + len(frag) - 1
        pos = hit + len(frag)
    return passage.offset + index[first], passage.offset + index[last] + 1


def _same_section(cited: Optional[str], located: str) -> Optional[bool]:
    if not cited:
        return None
    c, l = _normalize_plain(cited), _normalize_plain(located)
    return c == l or c in l or l in c


def resolve_quote(paper: Paper, quote: str, cited_section: Optional[str] = None) -> ResolvedEvidence:
    """Locate one quote. Ellipses split it into fragments that must appear in order."""
    fragments = [f for f in (_normalize_plain(p) for p in _ELLIPSIS.split(quote or "")) if f]
    fragments = [f.strip(_FRAGMENT_EDGE) for f in fragments]
    fragments = [f for f in fragments if f]
    if not fragments or max(len(f) for f in fragments) < MIN_FRAGMENT_CHARS:
        return ResolvedEvidence(quote=quote, cited_section=cited_section, matched=False)

    hits = []
    for passage in paper.passages:
        span = _find_in_passage(passage, fragments)
        if span:
            hits.append((passage, span))
    if not hits:
        return ResolvedEvidence(quote=quote, cited_section=cited_section, matched=False)

    # Prefer a hit inside the section the model cited.
    passage, (start, end) = next(
        (h for h in hits if _same_section(cited_section, h[0].section)), hits[0]
    )
    return ResolvedEvidence(
        quote=quote,
        cited_section=cited_section,
        matched=True,
        start=start,
        end=end,
        section=passage.section,
        section_agrees=_same_section(cited_section, passage.section),
    )


def resolve_evidence(paper: Paper, evidence: Any) -> List[ResolvedEvidence]:
    """Resolve a model's ``evidence`` field (list of {section, quote}) leniently."""
    if not isinstance(evidence, list):
        return []
    resolved = []
    for item in evidence:
        if isinstance(item, str):
            resolved.append(resolve_quote(paper, item))
        elif isinstance(item, dict):
            quote = item.get("quote") or item.get("text") or ""
            resolved.append(resolve_quote(paper, str(quote), item.get("section")))
    return resolved
