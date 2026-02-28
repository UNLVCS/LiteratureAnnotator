from __future__ import annotations

import html
# from label_api.lstudio_interfacer import Labeller
# from langchain.chains import RetrievalQA
# from langchain import hub
# from utilities.vector_db import VectorDb
import json
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, Request, BackgroundTasks
from minio import Minio
from pydantic import BaseModel

from config import load_config
from label_api.config_def import LabelApiConfig
from label_api.human_import import import_next_human_tasks
from label_api.human_labeller_sdk import HumanLabellerSDK
from label_api.lstudio_interfacer_sdk import LabellerSDK
from utilities.queue_helpers import (
    ack_paper,
    claim_next_paper,
    pop_paper_id,
    requeue_inflight,
)

load_dotenv(find_dotenv(), override=True)

# Load config at startup; validates required env vars
_label_config = load_config(LabelApiConfig)

# MinIO client configuration (from config)
client = Minio(
    _label_config.minio_endpoint,
    access_key=_label_config.minio_access_key,
    secret_key=_label_config.minio_secret_key,
    secure=_label_config.minio_secure,
)
bucket_name = _label_config.minio_bucket

# Scheduler for background tasks
scheduler = BackgroundScheduler()
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


LS = LabellerSDK(
    base_url=_label_config.label_studio_url or None,
    api_key=_label_config.label_studio_api_key or None,
)
LS_Human = HumanLabellerSDK(
    base_url=_label_config.label_studio_url or None,
    api_key=_label_config.label_studio_api_key or None,
)
app = FastAPI()

@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy"}

@app.on_event("startup")
async def startup_event():

    # Import initial tasks at startup instead of waiting for PROJECT_CREATED event
    # This ensures tasks are loaded even if the project already exists
    webhook_url = f"{_label_config.webhook_host}/webhook"
    LS.create_webhook(endpoint=webhook_url)
    LS_Human.create_webhook(endpoint=webhook_url)

    import_next_paper_tasks(LS.project_id)
    import_next_human_tasks(LS_Human)

    scheduler.add_job(periodic_paper_check, "interval", minutes=3, id="periodic_paper_check")
    scheduler.add_job(periodic_human_paper_check, "interval", minutes=3, id="periodic_human_paper_check")
    scheduler.start()
    print("[Startup] Scheduler started")
    return {"status": "ok"}

 
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
    proj_id = payload.get("project", {}).get("id")

    print("==========================================")
    print("WEBHOOK RECEIVED: ", action, "project_id:", proj_id)
    print("==========================================")
    # print(payload.get("event", "no_event"))


    if action == "PROJECT_CREATED":
        if proj_id == LS_Human.project_id:
            bg.add_task(import_next_human_tasks, LS_Human)
        else:
            bg.add_task(import_next_paper_tasks, proj_id)
        return {"status": "ok", "event": "project_created"}

    if action in ("ANNOTATION_CREATED", "ANNOTATION_UPDATED"):
        task_info = payload.get("task", {})
        annotation = payload.get("annotation", {})
        if proj_id == LS_Human.project_id:
            bg.add_task(handle_completed_human_task, task_info, annotation)
        else:
            bg.add_task(handle_completed_task, task_info, annotation)

    if proj_id == LS_Human.project_id:
        new_count = LS_Human.count_new_tasks(proj_id)
        if new_count == 0:
            bg.add_task(import_next_human_tasks, LS_Human)
    else:
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
    # Str shape (the paper_id IS the claim token for ack/requeue)
    if isinstance(claim, str):   
        return claim, claim
    return None, None
 
def periodic_paper_check():
    try:
        new_count = LS.count_new_tasks(LS.project_id)
        print(f"[Periodic Paper Check] New tasks: {new_count}")

        if new_count < 5:
            print("[Periodic Paper Check] No new tasks found, importing next paper")
            import_next_paper_tasks(LS.project_id)
    except Exception as e:
        print(f"[Periodic Paper Check] Error: {e}")


def periodic_human_paper_check():
    try:
        new_count = LS_Human.count_new_tasks(LS_Human.project_id)
        print(f"[Periodic Human Check] New tasks: {new_count}")

        if new_count < 5:
            print("[Periodic Human Check] Importing next paper")
            import_next_human_tasks(LS_Human)
    except Exception as e:
        print(f"[Periodic Human Check] Error: {e}")

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
        # claim = claim_next_paper_from_set()
        # print("Claim: ", claim)
        paper_id, claim_token = _unpack_claim(claim)
    else:
        paper_id = pop_paper_id()  

    if not paper_id: 
        print("No paper ID found") 
        return
   
    try: 
        # providers = ['gpt-4o']
        # providers = ['gpt-4o', 'gpt-oss:20b', 'qwen3:235b']
        providers = ['openai/gpt-oss-120b']
        paper_data = None 
        for provider in providers:
            try:
                object_name = f"{provider}/{paper_id}.json"
                response = client.get_object(bucket_name = bucket_name, object_name = object_name)
                data = response.data.decode('utf-8') 
                paper_data = json.loads(data) 
                print(f"Paper data for {provider}")
                # print(f"Paper data for {provider}: {paper_data}")
            except Exception as e:
                print(f"Error getting paper {paper_id} data for {provider}: {e}") 
                continue

            if not paper_data:
                print(f"No paper data found for {paper_id}")
                if claim_token:
                    ack_paper(claim_token)
                return             
            
            # Collect all criteria into a single array for one task per paper
            criteria_list = []
            for criteria_res in paper_data.get("criteria_results", []):
                # Extract the criterion name (e.g., "criterion_1")
                criterion = criteria_res.get("criterion", "")
                
                # Extract data from the cleaned_response structure (the actual LLM output)
                cleaned_response_str = criteria_res.get("cleaned_response", "{}")
                # Parse the cleaned_response JSON string
                try:
                    response_obj = json.loads(cleaned_response_str) if isinstance(cleaned_response_str, str) else cleaned_response_str
                except json.JSONDecodeError:
                    response_obj = {}
                
                criterion_data = response_obj.get(criterion, {})
                reason = criterion_data.get("reason", "NO LLM ANSWER") 
                satisfied = criterion_data.get("satisfied", "unknown")
                
                # Build criterion object for the array
                criteria_list.append({
                    "criterion": criterion,
                    "satisfied": satisfied,
                    "paper_text": reason,
                    "retrieved_chunks": criteria_res.get("chunks_used", 0),
                    "class_criteria": criteria_res.get("prompt", "NO CLASS CRITERIA"),
                    "num_chunks": criteria_res.get("chunks_used", 0),
                    "full_context": criteria_res.get("full_context", "Full context not available")
                })
            
            # Create one task per paper with all criteria bundled together
            if len(criteria_list) > 0:
                # Create formatted HTML string for displaying all criteria
                criteria_html = "<div style='font-family: Arial, sans-serif;'>"
                criteria_html += "<h2 style='color: #2c3e50; margin-bottom: 20px;'>📋 All Criteria</h2>"
                
                # Store criterion data with index for matching with evaluation sections
                criterion_names = []
                
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
                 
                # Structure data for Label Studio - include dummy criterion field for validation
                task_data = {
                    "paper_id": paper_id,
                    "provider": provider,
                    "title": paper_data.get("title", "Title N/A"),
                    "criterion": "multiple",  # Dummy field for validation compatibility
                    "criteria_html": criteria_html,  # Formatted HTML display
                    "criteria_json": json.dumps(criteria_list),  # Also keep JSON for reference
                    "criterion_names": json.dumps(criterion_names)  # List of criterion names for matching
                }
                task = {"data": task_data}
                LS.import_tasks([task])

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

# MinIO bucket for completed annotations

def _ensure_annotations_bucket():
    """Create the annotations bucket if it doesn't exist."""
    try:
        if not client.bucket_exists(_label_config.annotations_bucket):
            client.make_bucket(_label_config.annotations_bucket)
            print(f"Created bucket: {_label_config.annotations_bucket}")
    except Exception as e:
        print(f"Error checking/creating bucket {_label_config.annotations_bucket}: {e}")

def handle_completed_task(task: Dict[str, Any], annotation: Dict[str, Any]) -> None:
    """Save completed annotation to MinIO bucket."""
    record: Dict[str, Any] = {}

    for prefix, d in (("task", task), ("ann", annotation)):
        for k, v in d.items():
            record[f"{prefix}_{k}"] = v

    # Extract identifiers for the object name
    paper_id = task.get("data", {}).get("paper_id", "unknown")
    task_id = task.get("id", "unknown")
    annotation_id = annotation.get("id", "unknown")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    # Create object name: paper_id/task_id_annotation_id_timestamp.json
    object_name = f"{paper_id}/{task_id}_{annotation_id}_{timestamp}.json"
    
    # Serialize to JSON
    json_data = json.dumps(record, indent=2, default=str)
    data_bytes = json_data.encode("utf-8")
    
    try:
        _ensure_annotations_bucket()
        client.put_object(
            bucket_name=_label_config.annotations_bucket,
            object_name=object_name,
            data=BytesIO(data_bytes),
            length=len(data_bytes),
            content_type="application/json"
        )
        print(f"Saved annotation to MinIO: {_label_config.annotations_bucket}/{object_name}")
    except Exception as e:
        print(f"Error saving annotation to MinIO: {e}")
        raise


def _ensure_human_annotations_bucket():
    """Create the human annotations bucket if it doesn't exist."""
    try:
        if not client.bucket_exists(_label_config.human_annotations_bucket):
            client.make_bucket(_label_config.human_annotations_bucket)
            print(f"Created bucket: {_label_config.human_annotations_bucket}")
    except Exception as e:
        print(f"Error checking/creating bucket {_label_config.human_annotations_bucket}: {e}")


def handle_completed_human_task(task: Dict[str, Any], annotation: Dict[str, Any]) -> None:
    """Save completed human annotation to MinIO bucket."""
    record: Dict[str, Any] = {}
    for prefix, d in (("task", task), ("ann", annotation)):
        for k, v in d.items():
            record[f"{prefix}_{k}"] = v

    paper_id = task.get("data", {}).get("paper_id", "unknown")
    task_id = task.get("id", "unknown")
    annotation_id = annotation.get("id", "unknown")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    object_name = f"{paper_id}/{task_id}_{annotation_id}_{timestamp}.json"

    json_data = json.dumps(record, indent=2, default=str)
    data_bytes = json_data.encode("utf-8")

    try:
        _ensure_human_annotations_bucket()
        client.put_object(
            bucket_name=_label_config.human_annotations_bucket,
            object_name=object_name,
            data=BytesIO(data_bytes),
            length=len(data_bytes),
            content_type="application/json",
        )
        print(f"Saved human annotation to MinIO: {_label_config.human_annotations_bucket}/{object_name}")
    except Exception as e:
        print(f"Error saving human annotation to MinIO: {e}")
        raise

