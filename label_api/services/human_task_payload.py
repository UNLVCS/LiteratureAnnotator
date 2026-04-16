from __future__ import annotations

import html
from typing import Any, Dict, List


def resolve_title(current_title: str, chunks: List[Any]) -> str:
    if not chunks or current_title != "Title N/A":
        return current_title
    meta = getattr(chunks[0], "metadata", None) or {}
    title = meta.get("title", "Title N/A")
    if isinstance(title, list):
        return title[0] if title else "Title N/A"
    return str(title)


def build_context(chunks: List[Any]) -> str:
    if not chunks:
        return "No chunks retrieved."
    context_parts: List[str] = []
    for i, doc in enumerate(chunks):
        content = getattr(doc, "page_content", str(doc))
        context_parts.append(f"=== Chunk {i + 1} ===\n{content}\n")
    return "\n".join(context_parts)


def build_human_criteria_html(criteria_list: List[Dict[str, Any]]) -> tuple[str, List[str]]:
    criteria_html = '<div style="font-family: Arial, sans-serif;">'
    criteria_html += '<h2 style="color: #2c3e50; margin-bottom: 20px;">All Criteria</h2>'
    criterion_names: List[str] = []
    for idx, item in enumerate(criteria_list, 1):
        criterion = html.escape(str(item["criterion"]))
        criterion_names.append(criterion)
        class_criteria = html.escape(str(item["class_criteria"]))
        num_chunks = item["num_chunks"]
        full_context = html.escape(str(item["full_context"]))
        criteria_html += f"""
        <div id="criterion_{idx}" style="margin-bottom: 25px; padding: 18px; background: #ffffff; border: 2px solid #e0e0e0; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 14px 18px; border-radius: 8px; margin-bottom: 18px; color: white;">
                <h3 style="color: white; margin: 0 0 6px 0; font-size: 16px; font-weight: 700;">{criterion}</h3>
            </div>
            <div style="margin-bottom: 18px;">
                <h4 style="color: #2c3e50; margin: 0 0 8px 0; font-size: 14px; font-weight: 600;">Classification Criteria</h4>
                <div style="background: #e3f2fd; padding: 14px; border-radius: 8px; border-left: 4px solid #2196f3;">
                    <p style="color: #1565c0; margin: 0; line-height: 1.7; white-space: pre-wrap; font-size: 13px;">{class_criteria}</p>
                </div>
            </div>
            <div>
                <h4 style="color: #2c3e50; margin: 0 0 10px 0; font-size: 14px; font-weight: 600; border-bottom: 2px solid #e0e0e0; padding-bottom: 8px;">Retrieved Chunks ({num_chunks})</h4>
                <div style="max-height: 500px; overflow-y: auto; background: #fafafa; padding: 18px; border-radius: 8px; border: 1px solid #dee2e6;">
                    <div style="font-size: 13.5px; color: #2c3e50; line-height: 1.8; white-space: pre-wrap; word-wrap: break-word;">{full_context}</div>
                </div>
            </div>
        </div>
        """
    criteria_html += "</div>"
    return criteria_html, criterion_names
