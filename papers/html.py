"""Render a Paper as HTML for Label Studio, optionally highlighting cited spans."""

from __future__ import annotations

import html
from typing import Iterable, List, Optional, Tuple

from papers.bioc import Paper

# (start, end, label) in BioC document offsets, e.g. from ResolvedEvidence.
Highlight = Tuple[int, int, str]

_MARK_STYLE = "background:#fff3a3;padding:0 2px;border-radius:3px;"
_LABEL_STYLE = "font-size:11px;font-weight:700;color:#8a6d00;vertical-align:super;margin-left:2px;"


def _highlight_text(text: str, offset: int, highlights: List[Highlight]) -> str:
    """Escape ``text`` and wrap the parts covered by highlights in <mark>."""
    end_of_text = offset + len(text)
    # Several criteria often cite the same span: merge their labels into one mark.
    merged: dict[tuple[int, int], list[str]] = {}
    for s, e, label in highlights:
        if s < end_of_text and e > offset:
            labels = merged.setdefault((max(s, offset) - offset, min(e, end_of_text) - offset), [])
            if label not in labels:
                labels.append(label)
    spans = sorted((start, end, " ".join(labels)) for (start, end), labels in merged.items())
    out, pos = [], 0
    for start, end, label in spans:
        if start < pos:  # overlapping citation: fold into the previous mark
            start = pos
        if start >= end:
            continue
        out.append(html.escape(text[pos:start]))
        out.append(
            f'<mark style="{_MARK_STYLE}" title="{html.escape(label)}">{html.escape(text[start:end])}</mark>'
            f'<span style="{_LABEL_STYLE}">{html.escape(label)}</span>'
        )
        pos = end
    out.append(html.escape(text[pos:]))
    return "".join(out)


def render_paper_html(paper: Paper, highlights: Optional[Iterable[Highlight]] = None) -> str:
    """Whole paper as HTML with section headings; highlighted spans carry their label."""
    marks = [h for h in (highlights or []) if h[0] is not None and h[1] is not None]
    parts = [f"<h3 style='margin:0 0 12px 0;'>{html.escape(paper.title)}</h3>"]
    current_section = None
    for passage in paper.passages:
        if passage.section_type == "TITLE" or passage.is_heading:
            continue
        if passage.section != current_section:
            current_section = passage.section
            parts.append(
                "<h4 style='margin:18px 0 6px 0;color:#2c3e50;border-bottom:1px solid #e0e0e0;'>"
                f"{html.escape(passage.section)}</h4>"
            )
        parts.append(
            f"<p style='margin:0 0 10px 0;white-space:pre-wrap;'>"
            f"{_highlight_text(passage.text, passage.offset, marks)}</p>"
        )
    return "".join(parts)
