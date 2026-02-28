"""
Human Labeling Project: chunk retrieval and task import.
Retrieves chunks from Pinecone and creates Label Studio tasks.
"""

from __future__ import annotations

import html
import json
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from label_api.criteria import CRITERIA_PROMPTS, CRITERION_NAMES

if TYPE_CHECKING:
    from label_api.human_labeller_sdk import HumanLabellerSDK

# Lazy-initialized vector store
_vector_store = None
_vdb = None
_embedder = None


def _get_vector_store():
    """Lazy-init vector store for chunk retrieval."""
    global _vector_store, _vdb, _embedder
    if _vector_store is None:
        from utilities.vector_db import VectorDb

        _vdb = VectorDb()
        _embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
        namespace = os.getenv("PINECONE_HUMAN_NAMESPACE", "article_upload_test_2")
        _vector_store = PineconeVectorStore(
            index=_vdb.__get_index__(),
            embedding=_embedder,
            namespace=namespace,
        )
    return _vector_store


def return_relevant_chunks(paper_id: str, criteria_query: str, k: int = 5) -> List[Any]:
    """
    Retrieve the most relevant chunks for a criteria query using vector similarity.
    Filter by paper_id, return top k chunks.
    """
    vs = _get_vector_store()
    filtered_retriever = vs.as_retriever(
        search_kwargs={
            "filter": {"doc": paper_id},
            "k": k,
        }
    )
    # Use invoke instead of deprecated get_relevant_documents
    docs = filtered_retriever.invoke(criteria_query)
    return docs


def import_next_human_tasks(ls_human: "HumanLabellerSDK") -> None:
    """
    Pull next paper from human queue, retrieve chunks per criterion, create task.
    """
    from utilities.queue_helpers import (
        ack_paper_human,
        claim_next_paper_human,
        requeue_inflight_human,
    )

    claim_token = claim_next_paper_human()
    if not claim_token:
        print("[Human] No paper ID found in queue")
        return

    paper_id = claim_token  # str shape: paper_id is the claim token

    try:
        criteria_list = []
        title = "Title N/A"

        for idx, (criterion_name, criteria_prompt) in enumerate(
            zip(CRITERION_NAMES, CRITERIA_PROMPTS)
        ):
            chunks = return_relevant_chunks(paper_id, criteria_prompt, k=5)

            if chunks and title == "Title N/A":
                meta = getattr(chunks[0], "metadata", None) or {}
                title = meta.get("title", "Title N/A")
                if isinstance(title, list):
                    title = title[0] if title else "Title N/A"

            context_parts = []
            for i, doc in enumerate(chunks):
                content = getattr(doc, "page_content", str(doc))
                context_parts.append(f"=== Chunk {i + 1} ===\n{content}\n")
            full_context = "\n".join(context_parts) if context_parts else "No chunks retrieved."

            criteria_list.append({
                "criterion": criterion_name,
                "class_criteria": criteria_prompt,
                "full_context": full_context,
                "num_chunks": len(chunks),
            })

        # If no chunks at all for any criterion, ack and skip
        if all(c["num_chunks"] == 0 for c in criteria_list):
            print(f"[Human] No chunks found for paper {paper_id}, skipping")
            ack_paper_human(claim_token)
            return

        # Build criteria_html (no LLM output)
        criteria_html = '<div style="font-family: Arial, sans-serif;">'
        criteria_html += '<h2 style="color: #2c3e50; margin-bottom: 20px;">All Criteria</h2>'
        criterion_names = []

        for idx, c in enumerate(criteria_list, 1):
            criterion = html.escape(str(c["criterion"]))
            criterion_names.append(criterion)
            class_criteria = html.escape(str(c["class_criteria"]))
            num_chunks = c["num_chunks"]
            full_context = html.escape(str(c["full_context"]))

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

        task_data = {
            "paper_id": paper_id,
            "title": title,
            "criteria_html": criteria_html,
            "criteria_json": json.dumps(criteria_list),
            "criterion_names": json.dumps(criterion_names),
        }
        task = {"data": task_data}
        ls_human.import_tasks([task])

        ack_paper_human(claim_token)
        print(f"[Human] Imported task for paper {paper_id}")

    except Exception as e:
        print(f"[Human] Error importing paper {paper_id}: {e}")
        requeue_inflight_human(claim_token)
        raise
