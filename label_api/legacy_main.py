from __future__ import annotations

from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel
from label_api.lstudio_interfacer_sdk import LabellerSDK
# from label_api.lstudio_interfacer import Labeller
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA 
from langchain import hub
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
# from utilities.vector_db import VectorDb
import json
from typing import Any, Dict, Optional, Tuple   
from minio import Minio
 
# Queue helpers (use these instead of any local Redis calls)
from utilities.queue_helpers import (
    enqueue_paper_id,
    pop_paper_id,             # simple pattern
    claim_next_paper,         # safer pattern (claim + ack/requeue)
    ack_paper,
    requeue_inflight,
    push_completed_annotation,
)
client = Minio(
    "localhost:5000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)
bucket_name = "criteria-classified-articles"

# --------------------
# RAG / Model setup
# --------------------

# vdb = VectorDb()
# embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
# vector_store = PineconeVectorStore(
#     index=vdb.__get_index__(), embedding=embedder, namespace="article_upload_test_2"
# )

# llm = ChatOpenAI(model="gpt-4o")
# prompt = hub.pull("rlm/rag-prompt")

# qa_chain = RetrievalQA.from_chain_type(
#     llm,
#     retriever=vector_store.as_retriever(),
#     chain_type_kwargs={"prompt": prompt},
# )

# prompts = [
#     "What part of the paper talk about sample size ....",
#     "etc...",
# ]


LS = LabellerSDK()
app = FastAPI()

LS.create_webhook(
    endpoint="http://localhost:8000/webhook"
) 

# --------------------
# Schemas
# --------------------

class CriteriaRequest(BaseModel):
    criteria: str


# --------------------
# Webhooks 
# --------------------

@app.post("/webhook")
async def ls_webhook(req: Request, bg: BackgroundTasks):
    payload = await req.json()
    action = payload.get("action")

    print("==========================================")
    print("WEBHOOK RECEIVED: ", action)
    print("==========================================")
    # print(payload.get("event", "no_event"))


    if  action == "PROJECT_CREATED": 
        # If a new project is created, we can start importing tasks
        proj_id = payload["project"]["id"]   
        bg.add_task(import_next_paper_tasks, proj_id)
        return {"status": "ok", "event": "project_created"}

        # return {"status": "ignored", "event": payload.get("event")}

    proj_id = payload["project"]["id"]

    # if  action == "TASK_CREATED":
    if  action == "ANNOTATION_COMPLETED":
        task_info = payload["tasks"]
        annotation = payload["annotation"]
        bg.add_task(handle_completed_task, task_info, annotation)
        return {"status": "ok", "event": "annotation_completed"}

    # Handle the completed task in the background
    # bg.add_task(handle_completed_task, task_info, annotation)e

    # If there are no new tasks left in the project, pull the next paper from the queue
    new_count = LS.count_new_tasks(proj_id)
    if new_count == 0:
        bg.add_task(import_next_paper_tasks, proj_id)

    return {"status": "ok"} 


# --------------------
# Queue-driven import
# --------------------

USE_SAFE_QUEUE = True  # toggle to False to use simple pop() pattern
 

def _unpack_claim(claim: Any) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort unpack for different possible claim shapes.
    Returns (paper_id, claim_token).
    """
    if claim is None:
        return None, None
    # Dict shape
    if isinstance(claim, dict):
        pid = (
            claim.get("paper_id")
            or claim.get("id")
            or claim.get("item_id")
            or claim.get("article_id")
        )
        token = claim.get("claim_id") or claim.get("receipt") or claim.get("token")
        return pid, token
    # Tuple or list shape
    if isinstance(claim, (tuple, list)) and len(claim) >= 2:
        return str(claim[0]) if claim[0] is not None else None, str(claim[1]) if claim[1] is not None else None
    # Str shape (just an id, no claim token)
    if isinstance(claim, str):
        return claim, None
    return None, None
 

def import_next_paper_tasks(project_id: int) -> None:
    """Pull the next paper from the queue and create Label Studio tasks.

    Uses the safer claim/ack pattern when available, with a fallback to simple pop.
    Creates a paper-specific RAG pipeline to ensure responses only come from the relevant paper.
    Retrieves ALL chunks for the paper to allow the LLM to reason over the complete content.
    """
    paper_id: Optional[str] = None
    claim_token: Optional[str] = None

    if USE_SAFE_QUEUE:
        claim = claim_next_paper()
        paper_id, claim_token = _unpack_claim(claim)
    else:
        paper_id = pop_paper_id()

    if not paper_id:
        return

    try: 
        tasks = []

        providers = ['gpt-4o', 'gpt-oss:20b', 'qwen:235b']
        paper_data = None
        for provider in providers:
            try:
                object_name = f"{provider}/{paper_id}.json"
                response = client.get_object(bucket_name = bucket_name, object_name = object_name)
                data = response.data.decode('utf-8')
                paper_data = json.loads(data)
                print(f"Paper data for {provider}: {paper_data}")
            except Exception as e:
                print(f"Error getting paper {paper_id} data for {provider}: {e}")
                continue

            if not paper_data:
                print(f"No paper data found for {paper_id}")
                if claim_token:
                    ack_paper(claim_token)
                return            
            for criteria_res in paper_data.get("criteria_results", []):
                tasks.append({
                    "data": {
                        "paper_id": paper_id,
                        "title": paper_data.get("title", "Title N/A"),
                        "paper_text": criteria_res.get("response", {}).get("reason", "NO LLM ANSWER"),
                        "retrieved_chunks": criteria_res.get("chunks_used", 0),
                        "class_criteria": criteria_res.get("prompt", "NO CLASS CRITERIA"),
                        "num_chunks": criteria_res.get("chunks_used", 0),
                        "full_context": criteria_res.get("full_context", "NO FULL CONTEXT")
                    }
                })

        if len(tasks) > 0:
            LS.import_tasks(tasks)

        # Acknowledge the claimed item only if we used the claim pattern
        if claim_token:
            ack_paper(claim_token)

    except Exception:
        # If something failed after claiming, requeue the inflight item
        if claim_token:
            requeue_inflight(claim_token)
        raise


# --------------------
# Completed task handling
# --------------------

def handle_completed_task(task: Dict[str, Any], annotation: Dict[str, Any]) -> None:
    record: Dict[str, Any] = {}

    for prefix, d in (("task", task), ("ann", annotation)):
        for k, v in d.items():
            record[f"data_{k}"] = v

    # Persist the completed annotation in the queue-backed sink
    push_completed_annotation(json.dumps(record))
