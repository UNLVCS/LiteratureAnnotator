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
from vector_db import VectorDb
import json
from typing import Any, Dict, Optional, Tuple

# Queue helpers (use these instead of any local Redis calls)
from queue_helpers import (
    enqueue_paper_id,
    pop_paper_id,             # simple pattern
    claim_next_paper,         # safer pattern (claim + ack/requeue)
    ack_paper,
    requeue_inflight,
    push_completed_annotation,
)


# --------------------
# RAG / Model setup
# --------------------

vdb = VectorDb()
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
vector_store = PineconeVectorStore(
    index=vdb.__get_index__(), embedding=embedder, namespace="article_upload_test_2"
)

llm = ChatOpenAI(model="gpt-4o")
prompt = hub.pull("rlm/rag-prompt")

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vector_store.as_retriever(),
    chain_type_kwargs={"prompt": prompt},
)

# prompts = [
#     "What part of the paper talk about sample size ....",
#     "etc...",
# ]


prompts = [
    # 1) Original research
    """Criterion 1 – Original Research
    Decide if the paper is an original research article (not a review, perspective, poster, or preprint).
    - Positive signals: data collection + statistical analysis (often in Methods).
    - Negative signals: clear mentions of review, perspective, poster, preprint.
    Return JSON only:
    {"criterion_1": {"satisfied": true/false, "reason": "<brief reason>"}}""",

    # 2) AD focus
    """Criterion 2 – AD Focus
    Decide if Alzheimer's Disease (AD) is the main focus (diagnosis, treatment, biomarkers, pathology; AD patients incl. MCI/at risk).
    - Include AD biomarkers: amyloid-beta, tau.
    - Exclude if focus is general neurodegeneration markers without AD specificity.
    Return JSON only:
    {"criterion_2": {"satisfied": true/false, "reason": "<brief reason>"}}""",

    # 3) Sample size >= 50 (leniency note)
    """Criterion 3 – Sample Size
    If human study: determine if sample size n >= 50.
    - If stated n >= 50 → satisfied=true.
    - If < 50 → satisfied=false (note: can be relaxed later if other criteria are very strong).
    Return JSON only:
    {"criterion_3": {"satisfied": true/false, "reason": "<brief reason; include n if found>"}}""",

    # 4) Proteins as biomarkers (exclude gene/RNA/transcript/fragment focus)
    """Criterion 4 – Protein Biomarkers
    Decide if the study’s biomarker focus is on proteins (e.g., protein, amyloid, tau; beta-amyloid).
    - Satisfied if protein focus is central and recurrent.
    - Not satisfied if focus is genes/RNA/transcripts/fragments.
    Return JSON only:
    {"criterion_4": {"satisfied": true/false, "reason": "<brief reason>"}}""",

    # 5) Animal models exclusion (use human; flag patient cell-cultures)
    """Criterion 5 – Animal Models Exclusion
    Determine if the study uses animal models.
    - If animal models are used → satisfied=false.
    - If human data only → satisfied=true.
    - If using patient-derived cell cultures (not animals), note that explicitly.
    Return JSON only:
    {"criterion_5": {"satisfied": true/false, "reason": "<brief reason; note 'patient cell cultures' if applicable>"}}""",

    # 6) Blood as AD biomarker (not blood pressure)
    """Criterion 6 – Blood as AD Biomarker
    If 'blood' appears, decide if it is used as an AD biomarker (e.g., serum/plasma for amyloid/tau).
    - Exclude circulatory measures (e.g., blood pressure, hypertension, vascular health).
    Return JSON only:
    {"criterion_6": {"satisfied": true/false, "reason": "<brief reason>"}}""",

    # Final aggregation (single-shot option)
    """Final Classification – Aggregate
    Evaluate Criteria 1–6 and produce the final binary classification:
    - "relevant" if criteria strongly indicate AD-relevant original research (consider leniency on n<50 if others are strong).
    - "irrelevant" otherwise.
    Return JSON only and exactly this shape:
    {
    "criteria": {
        "criterion_1": {"satisfied": true/false, "reason": "<...>"},
        "criterion_2": {"satisfied": true/false, "reason": "<...>"},
        "criterion_3": {"satisfied": true/false, "reason": "<...>"},
        "criterion_4": {"satisfied": true/false, "reason": "<...>"},
        "criterion_5": {"satisfied": true/false, "reason": "<...>"},
        "criterion_6": {"satisfied": true/false, "reason": "<...>"}
    },
    "final_classification": "relevant/irrelevant",
    "justification": "<overall reasoning>"
    }"""
]

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
        # Create a retriever with metadata filter for this specific paper
        # and increase the number of results to get all chunks
        filtered_retriever = vector_store.as_retriever(
            search_kwargs={
                "filter": {"doc": paper_id},  # Filter to only search within this paper
                "k": 10  # Increase this if papers have more chunks
            }
        )
        
        # Create a new QA chain with the filtered retriever
        paper_qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=filtered_retriever,
            return_source_documents=True,
            chain_type="stuff",  # Explicitly set to use stuff chain 
            chain_type_kwargs={
                "prompt": prompt  
            }
        )

        for q in prompts:
            result = paper_qa_chain.invoke({
                "query": q, 
                "context": "Consider ALL provided chunks of the paper when answering. Synthesize information from all relevant sections."
            })
            llm_answer = result.get("result") if isinstance(result, dict) else result
            
            # Access the retrieved chunks (source documents)
            source_docs = result.get("source_documents", []) if isinstance(result, dict) else []
            
            # Format retrieved chunks for Label Studio
            retrieved_chunks_text = ""
            for i, doc in enumerate(source_docs):
                retrieved_chunks_text += f"=== Chunk {i+1} ===\n"
                retrieved_chunks_text += f"{doc.page_content}\n"
                retrieved_chunks_text += f"Metadata: {doc.metadata}\n\n"
            
            # print(f"\n=== QUERY: {q} ===")
            # print(f"LLM ANSWER: {llm_answer}")
            # print(f"RETRIEVED {len(source_docs)} CHUNKS")

            tasks.append(
                {
                    "data": {
                        "paper_id": paper_id,
                        "title": "SOME TITLE | REPLACE LATER",
                        "paper_text": llm_answer or "NO LLM ANSWER",
                        "retrieved_chunks": retrieved_chunks_text,
                        "class_criteria": q,
                        "num_chunks": len(source_docs)
                    }
                }
            )

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
