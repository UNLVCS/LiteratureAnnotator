"""Render a Paper as plain text for an LLM prompt or a human reviewer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from papers.bioc import Paper

TRUNCATION_MARKER = "[... remainder of paper truncated ...]"


@dataclass(frozen=True)
class RenderedPaper:
    text: str
    truncated: bool
    passages_included: int
    passages_total: int


def render_paper(paper: Paper, max_chars: Optional[int] = None) -> RenderedPaper:
    """Render the whole paper with a ``## <section>`` header at each section change.

    Section labels in the output are the same strings the model is asked to cite,
    so ``evidence[].section`` can be compared against ``Paper.sections``.

    Args:
        max_chars: Optional hard cap. Truncation happens at a passage boundary and is
            reported via ``RenderedPaper.truncated`` so callers can record it.
    """
    parts = [f"Title: {paper.title}"]
    used = len(parts[0])
    current_section = None
    included = 0
    body = [p for p in paper.passages if p.section_type != "TITLE" and not p.is_heading]
    truncated = False

    for passage in body:
        block = ""
        if passage.section != current_section:
            block += f"\n## {passage.section}\n"
        block += passage.text.strip()
        if max_chars is not None and used + len(block) + 2 > max_chars:
            truncated = True
            break
        if passage.section != current_section:
            current_section = passage.section
        parts.append(block)
        used += len(block) + 2
        included += 1

    if truncated:
        parts.append(f"\n{TRUNCATION_MARKER}")
    return RenderedPaper(
        text="\n\n".join(parts).replace("\n\n\n## ", "\n\n## "),
        truncated=truncated,
        passages_included=included,
        passages_total=len(body),
    )
