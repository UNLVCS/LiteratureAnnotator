"""
Task payload for the model-review project: a reviewer judges each LLM criterion answer
(CORRECT / INCORRECT / ABSTAIN) against the full paper, with the model's cited quotes
highlighted in place.
"""

from __future__ import annotations

import html
import json
from typing import Any, Dict, List, Optional

from papers import Paper, render_paper_html
from utilities.criteria import CRITERIA_INSTRUCTIONS, CRITERION_NAMES

# Reviewers see the criterion definition without the model-facing output-format rules.
_DEFINITIONS = dict(zip(CRITERION_NAMES, CRITERIA_INSTRUCTIONS))

_CARD = (
    "margin-bottom: 25px; padding: 18px; background: #ffffff; border: 2px solid #e0e0e0; "
    "border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);"
)
_BANNER = (
    "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 14px 18px; "
    "border-radius: 8px; margin-bottom: 14px; color: white;"
)
_BOX = "padding: 12px 14px; border-radius: 8px; font-size: 13px; line-height: 1.7; white-space: pre-wrap;"


def _criterion_label(criterion: str) -> str:
    """'criterion_3' -> 'C3' (used to tag highlights in the paper)."""
    return "C" + criterion.rsplit("_", 1)[-1] if criterion.startswith("criterion_") else criterion


def _parse_response(criteria_res: Dict[str, Any]) -> Dict[str, Any]:
    response = criteria_res.get("response")
    if response is None:
        cleaned = criteria_res.get("cleaned_response") or "{}"
        try:
            response = json.loads(cleaned) if isinstance(cleaned, str) else cleaned
        except json.JSONDecodeError:
            response = {}
    return (response or {}).get(criteria_res.get("criterion", ""), {}) or {}


def build_criteria_list(paper_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten a labeling result into one entry per criterion for the review UI.

    Works for whole-paper results (with resolved ``evidence``) and for legacy
    pre-retirement (RAG-era) results, which have no evidence and show the reason only.
    """
    criteria_list: List[Dict[str, Any]] = []
    for criteria_res in paper_data.get("criteria_results", []):
        criterion = criteria_res.get("criterion", "")
        answer = _parse_response(criteria_res)
        criteria_list.append(
            {
                "criterion": criterion,
                "satisfied": answer.get("satisfied", "unknown"),
                "reason": answer.get("reason", "NO LLM ANSWER"),
                "class_criteria": _DEFINITIONS.get(criterion) or criteria_res.get("prompt", "NO CLASS CRITERIA"),
                "error": criteria_res.get("error"),
                "evidence": criteria_res.get("evidence") or [],
            }
        )
    return criteria_list


def _evidence_html(evidence: List[Dict[str, Any]]) -> str:
    if not evidence:
        return "<p style='color:#6c757d;margin:0;font-size:13px;'>No passages cited (decision based on absence, or none given).</p>"
    items = []
    for ev in evidence:
        quote = html.escape(str(ev.get("quote", "")))
        cited = html.escape(str(ev.get("cited_section") or "?"))
        if ev.get("matched"):
            where = html.escape(str(ev.get("section") or ""))
            status = f"<span style='color:#2e7d32;'>&#10003; found in <b>{where}</b></span>"
            if ev.get("section_agrees") is False:
                status += f" <span style='color:#b26a00;'>(model said: {cited})</span>"
        else:
            status = f"<span style='color:#c62828;'>&#10007; not found in paper</span> (model said: {cited})"
        items.append(f"<li style='margin-bottom:8px;'>&ldquo;{quote}&rdquo;<br><small>{status}</small></li>")
    return f"<ul style='margin:0;padding-left:18px;font-size:13px;line-height:1.6;'>{''.join(items)}</ul>"


def build_rag_criteria_html(
    criteria_list: List[Dict[str, Any]], paper: Optional[Paper] = None
) -> tuple[str, List[str]]:
    """HTML for the review task: one card per criterion, then the full paper once."""
    criteria_html = "<div style='font-family: Arial, sans-serif;'>"
    criteria_html += "<h2 style='color: #2c3e50; margin-bottom: 20px;'>Model answers</h2>"
    criterion_names: List[str] = []
    highlights = []

    for idx, item in enumerate(criteria_list, 1):
        criterion = str(item.get("criterion", f"criterion_{idx}"))
        criterion_names.append(html.escape(criterion))
        label = _criterion_label(criterion)
        for ev in item.get("evidence", []):
            if ev.get("matched"):
                highlights.append((ev.get("start"), ev.get("end"), label))
        error = item.get("error")
        error_html = (
            f"<p style='color:#c62828;margin:6px 0 0 0;font-size:12px;'>Output error: {html.escape(str(error))}</p>"
            if error else ""
        )
        criteria_html += f"""
        <div id="criterion_{idx}" style='{_CARD}'>
            <div style='{_BANNER}'>
                <h3 style='color: white; margin: 0 0 6px 0; font-size: 16px; font-weight: 700;'>{html.escape(criterion)} ({label})</h3>
                <p style='margin: 0; font-size: 13px; font-weight: 500;'>LLM result: <strong>{html.escape(str(item.get("satisfied")))}</strong></p>
            </div>
            <details style='margin-bottom: 12px;'>
                <summary style='cursor:pointer;color:#1565c0;font-size:13px;font-weight:600;'>Criterion definition</summary>
                <div style='{_BOX} background:#e3f2fd; color:#1565c0; margin-top:8px;'>{html.escape(str(item.get("class_criteria", "")))}</div>
            </details>
            <h4 style='color: #2c3e50; margin: 0 0 6px 0; font-size: 14px;'>Reason</h4>
            <div style='{_BOX} background:#f1f8f4; color:#1b5e20; margin-bottom: 12px;'>{html.escape(str(item.get("reason", "")))}</div>
            <h4 style='color: #2c3e50; margin: 0 0 6px 0; font-size: 14px;'>Cited evidence</h4>
            {_evidence_html(item.get("evidence", []))}
            {error_html}
        </div>
        """

    criteria_html += "<h2 style='color: #2c3e50; margin: 30px 0 12px 0;'>Full paper</h2>"
    if paper is None:
        criteria_html += "<p style='color:#c62828;'>Raw article not found in MinIO; only the model answers are shown.</p>"
    else:
        criteria_html += (
            "<p style='color:#6c757d;font-size:12px;margin:0 0 10px 0;'>Highlighted spans are passages a model cited; "
            "the tag shows which criterion. References are omitted.</p>"
            "<div style='max-height: 900px; overflow-y: auto; background: #fafafa; padding: 18px; "
            "border-radius: 8px; border: 1px solid #dee2e6; font-size: 13.5px; color: #2c3e50; line-height: 1.8;'>"
            f"{render_paper_html(paper, highlights)}</div>"
        )
    criteria_html += "</div>"
    return criteria_html, criterion_names
