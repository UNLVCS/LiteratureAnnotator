from __future__ import annotations

import html
from typing import Any, Dict, List

from papers import Paper, render_paper_html


def build_human_criteria_html(criteria_list: List[Dict[str, Any]], paper: Paper) -> tuple[str, List[str]]:
    """Criteria definitions followed by the full paper (no model output shown)."""
    criteria_html = '<div style="font-family: Arial, sans-serif;">'
    criteria_html += '<h2 style="color: #2c3e50; margin-bottom: 20px;">Criteria</h2>'
    criterion_names: List[str] = []
    for idx, item in enumerate(criteria_list, 1):
        criterion = html.escape(str(item["criterion"]))
        criterion_names.append(criterion)
        class_criteria = html.escape(str(item["class_criteria"]))
        criteria_html += f"""
        <div id="criterion_{idx}" style="margin-bottom: 14px; padding: 14px; background: #ffffff; border: 2px solid #e0e0e0; border-radius: 10px;">
            <h3 style="color: #2c3e50; margin: 0 0 8px 0; font-size: 15px; font-weight: 700;">{criterion}</h3>
            <div style="background: #e3f2fd; padding: 12px; border-radius: 8px; border-left: 4px solid #2196f3;">
                <p style="color: #1565c0; margin: 0; line-height: 1.7; white-space: pre-wrap; font-size: 13px;">{class_criteria}</p>
            </div>
        </div>
        """
    criteria_html += (
        '<h2 style="color: #2c3e50; margin: 30px 0 12px 0;">Full paper</h2>'
        '<p style="color:#6c757d;font-size:12px;margin:0 0 10px 0;">References are omitted.</p>'
        '<div style="max-height: 900px; overflow-y: auto; background: #fafafa; padding: 18px; '
        'border-radius: 8px; border: 1px solid #dee2e6; font-size: 13.5px; color: #2c3e50; line-height: 1.8;">'
        f"{render_paper_html(paper)}</div>"
    )
    criteria_html += "</div>"
    return criteria_html, criterion_names
