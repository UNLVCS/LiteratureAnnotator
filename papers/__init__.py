"""
Whole-paper access for LiteratureAnnotator.

Replaces the retired RAG path (data_vectorize + Pinecone): papers are read straight
from the raw BioC JSON in MinIO, rendered in full for the model, and model-cited
quotes are resolved back to BioC offsets.
"""

from papers.bioc import DEFAULT_EXCLUDED_SECTION_TYPES, Paper, Passage, parse_bioc
from papers.html import Highlight, render_paper_html
from papers.evidence import ResolvedEvidence, resolve_evidence, resolve_quote
from papers.render import RenderedPaper, render_paper
from papers.store import PaperNotFoundError, load_paper, load_paper_from_config, raw_object_name

__all__ = [
    "DEFAULT_EXCLUDED_SECTION_TYPES",
    "Highlight",
    "Paper",
    "PaperNotFoundError",
    "Passage",
    "RenderedPaper",
    "ResolvedEvidence",
    "load_paper",
    "load_paper_from_config",
    "parse_bioc",
    "raw_object_name",
    "render_paper",
    "render_paper_html",
    "resolve_evidence",
    "resolve_quote",
]
