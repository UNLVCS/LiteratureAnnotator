from __future__ import annotations

import html
import json
from typing import Any, Dict, List


def build_criteria_list(paper_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    criteria_list: List[Dict[str, Any]] = []
    for criteria_res in paper_data.get("criteria_results", []):
        criterion = criteria_res.get("criterion", "")
        cleaned_response = criteria_res.get("cleaned_response", "{}")
        try:
            response_obj = json.loads(cleaned_response) if isinstance(cleaned_response, str) else cleaned_response
        except json.JSONDecodeError:
            response_obj = {}
        criterion_data = response_obj.get(criterion, {})
        criteria_list.append(
            {
                "criterion": criterion,
                "satisfied": criterion_data.get("satisfied", "unknown"),
                "paper_text": criterion_data.get("reason", "NO LLM ANSWER"),
                "retrieved_chunks": criteria_res.get("chunks_used", 0),
                "class_criteria": criteria_res.get("prompt", "NO CLASS CRITERIA"),
                "num_chunks": criteria_res.get("chunks_used", 0),
                "full_context": criteria_res.get("full_context", "Full context not available"),
            }
        )
    return criteria_list


def build_rag_criteria_html(criteria_list: List[Dict[str, Any]]) -> tuple[str, List[str]]:
    criteria_html = "<div style='font-family: Arial, sans-serif;'>"
    criteria_html += "<h2 style='color: #2c3e50; margin-bottom: 20px;'>📋 All Criteria</h2>"
    criterion_names: List[str] = []
    for idx, criterion_data in enumerate(criteria_list, 1):
        criterion = html.escape(str(criterion_data.get("criterion", f"criterion_{idx}")))
        criterion_names.append(criterion)
        satisfied = html.escape(str(criterion_data.get("satisfied", "unknown")))
        paper_text = html.escape(str(criterion_data.get("paper_text", "")))
        class_criteria = html.escape(str(criterion_data.get("class_criteria", "")))
        num_chunks = criterion_data.get("num_chunks", 0)
        full_context = html.escape(str(criterion_data.get("full_context", "")))
        criteria_html += f"""
        <div id="criterion_{idx}" style='margin-bottom: 25px; padding: 18px; background: #ffffff; border: 2px solid #e0e0e0; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);'>
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 14px 18px; border-radius: 8px; margin-bottom: 18px; color: white;'>
                <h3 style='color: white; margin: 0 0 6px 0; font-size: 16px; font-weight: 700;'>📋 {criterion}</h3>
                <p style='color: rgba(255,255,255,0.95); margin: 0; font-size: 13px; font-weight: 500;'>✓ LLM Result: <strong>{satisfied}</strong></p>
            </div>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 18px;'>
                <div>
                    <h4 style='color: #2c3e50; margin: 0 0 8px 0; font-size: 14px; font-weight: 600;'>📋 Classification Criteria</h4>
                    <div style='background: #e3f2fd; padding: 14px; border-radius: 8px; border-left: 4px solid #2196f3;'>
                        <p style='color: #1565c0; margin: 0; line-height: 1.7; white-space: pre-wrap; font-size: 13px;'>{class_criteria}</p>
                    </div>
                </div>
                <div>
                    <h4 style='color: #2c3e50; margin: 0 0 8px 0; font-size: 14px; font-weight: 600;'>🤖 LLM Generated Answer</h4>
                    <div style='background: #f1f8f4; padding: 14px; border-radius: 8px; border-left: 4px solid #4caf50;'>
                        <p style='color: #1b5e20; margin: 0; line-height: 1.7; white-space: pre-wrap; font-size: 13px;'>{paper_text}</p>
                    </div>
                </div>
            </div>
            <div>
                <h4 style='color: #2c3e50; margin: 0 0 10px 0; font-size: 14px; font-weight: 600; border-bottom: 2px solid #e0e0e0; padding-bottom: 8px;'>📄 Retrieved Chunks ({num_chunks} chunks) - Full Content</h4>
                <div style='max-height: 500px; overflow-y: auto; background: #fafafa; padding: 18px; border-radius: 8px; border: 1px solid #dee2e6;'>
                    <div style='font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-size: 13.5px; color: #2c3e50; line-height: 1.8; white-space: pre-wrap; word-wrap: break-word;'>{full_context}</div>
                </div>
            </div>
        </div>
        """
    criteria_html += "</div>"
    return criteria_html, criterion_names
