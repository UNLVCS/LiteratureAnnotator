"""
Human Labeling Project: chunk retrieval and task import.
Retrieves chunks from Pinecone and creates Label Studio tasks.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, List

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from label_api.services.human_task_payload import (
    build_context,
    build_human_criteria_html,
    resolve_title,
)
from utilities.criteria import CRITERIA_PROMPTS, CRITERION_NAMES

if TYPE_CHECKING:
    from label_api.clients.label_studio.human import HumanLabellerSDK

_vector_store = None
_vdb = None
_embedder = None


def _get_vector_store() -> PineconeVectorStore:
    global _vector_store, _vdb, _embedder
    if _vector_store is None:
        from config.app_config import load_app_config
        from data_vectorize.vector_db import VectorDb

        app_config = load_app_config()
        _vdb = VectorDb(pinecone_config=app_config.pinecone)
        _embedder = OpenAIEmbeddings(
            model=app_config.embeddings.model,
            api_key=app_config.embeddings.api_key,
            dimensions=None
            if app_config.embeddings.model == "text-embedding-ada-002"
            else app_config.embeddings.dimensions,
        )
        _vector_store = PineconeVectorStore(
            index=_vdb.__get_index__(),
            embedding=_embedder,
            namespace=app_config.pinecone.namespace,
        )
    return _vector_store


def return_relevant_chunks(paper_id: str, criteria_query: str, k: int = 5) -> List[Any]:
    vs = _get_vector_store()
    filtered_retriever = vs.as_retriever(
        search_kwargs={
            "filter": {"doc": paper_id},
            "k": k,
        }
    )
    return filtered_retriever.invoke(criteria_query)


def import_next_human_tasks(ls_human: "HumanLabellerSDK") -> None:
    from utilities.queue_helpers import (
        ack_paper_human,
        claim_next_paper_human,
        requeue_inflight_human,
    )

    paper_id = claim_next_paper_human()
    if not paper_id:
        print("[Human] No paper ID found in queue")
        return

    try:
        criteria_list = []
        title = "Title N/A"

        for criterion_name, criteria_prompt in zip(CRITERION_NAMES, CRITERIA_PROMPTS):
            chunks = return_relevant_chunks(paper_id, criteria_prompt, k=5)
            title = resolve_title(title, chunks)
            full_context = build_context(chunks)

            criteria_list.append(
                {
                    "criterion": criterion_name,
                    "class_criteria": criteria_prompt,
                    "full_context": full_context,
                    "num_chunks": len(chunks),
                }
            )

        if all(c["num_chunks"] == 0 for c in criteria_list):
            print(f"[Human] No chunks found for paper {paper_id}, skipping")
            ack_paper_human(paper_id)
            return

        criteria_html, criterion_names = build_human_criteria_html(criteria_list)

        task_data = {
            "paper_id": paper_id,
            "title": title,
            "criteria_html": criteria_html,
            "criteria_json": json.dumps(criteria_list),
            "criterion_names": json.dumps(criterion_names),
        }
        ls_human.import_tasks([{"data": task_data}])

        ack_paper_human(paper_id)
        print(f"[Human] Imported task for paper {paper_id}")

    except Exception as e:
        print(f"[Human] Error importing paper {paper_id}: {e}")
        requeue_inflight_human(paper_id)
        raise


def import_all_pending_human_tasks(ls_human: "HumanLabellerSDK", max_rounds: int = 500) -> int:
    from utilities.queue_helpers import human_paper_queue_len

    rounds = 0
    while human_paper_queue_len() > 0 and rounds < max_rounds:
        import_next_human_tasks(ls_human)
        rounds += 1
    remaining = human_paper_queue_len()
    print(f"[Human] Batch import finished: {rounds} task(s) created, human queue length now {remaining}")
    return rounds
