#!/usr/bin/env python3
"""
RAG-based Data Labeling Script using LLM Providers (Global Functions)

This script generates labeled data using RAG chains with multiple LLM providers.
It processes papers from a queue and generates labeled and fetches 
semantically relevant chunks from a vector store.

This version uses global functions and variables for multiprocessing compatibility.
"""


import os
import json
import sys
from multiprocessing import Process, Queue, Manager, Lock
from typing import Any, Dict, List
from pathlib import Path
import signal
from dotenv import load_dotenv, find_dotenv

# Load environment variables


from response_standardizer import standardize_llm_response

# Add the parent directory to the path so we can import llm_providers
sys.path.append(str(Path(__file__).parent.parent))

# Import from the llm_providers package
from llm_providers.base import BaseLLMProvider, Query
from llm_providers.openai_provider import OpenAIProvider
from llm_providers.anthropic_provider import AnthropicProvider
from llm_providers.huggingface_provider import HuggingFaceProvider
from llm_providers.ollama_provider import OllamaProvider
from llm_providers.vllm_provider import VLLMProvider

# Import the existing components
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from utilities.vector_db import VectorDb
from utilities.queue_helpers import (
    claim_next_paper,
    ack_paper,
    paper_queue_len,
    push_completed_paper,
    completed_papers_count,
    export_completed_papers_to_file
)
from minio import Minio
from io import BytesIO


load_dotenv(find_dotenv(), override=True)

client = Minio(
    os.getenv("MINIO_URL"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    secure=False
)
bucket_name = os.getenv("MINIO_BUCKET_NAME")
if not client.bucket_exists(bucket_name):
    print(f"Bucket {bucket_name} does not exist. Creating it...")
    client.make_bucket(bucket_name)
else:
    print(f"Bucket {bucket_name} already exists.")

# Global variables for shared resources
_embedder = None
_vector_store = None
_vdb = None
_prompt = None
_criteria_prompts = None
_providers = {}

def initialize_shared_resources():
    """Initialize shared resources globally"""
    global _embedder, _vector_store, _vdb, _prompt, _criteria_prompts
    
    if _embedder is None:
        _vdb = VectorDb()
        _embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
        _vector_store = PineconeVectorStore(
            index=_vdb.__get_index__(), 
            embedding=_embedder, 
            namespace="V3_raw_pubmed_articles"
            # namespace="article_upload_test_2"
        )
        _prompt = hub.pull("rlm/rag-prompt")
        _criteria_prompts = [
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
            Decide if the study's biomarker focus is on proteins (e.g., protein, amyloid, tau; beta-amyloid).
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

def setup_providers(provider_configs: Dict[str, Dict[str, Any]]):
    """Setup LLM providers based on configuration"""
    global _providers
    
    for provider, models in provider_configs.items():
        for model in models:
            # print(type(model))
            if model['skip']: 
                continue
            if provider != "ollama" and not model.get("api_key"):
                print(f"Skipping {model['model']} - no API key found")

            if "openai" == provider:
                openai_params = model
                _providers[model['model']] = OpenAIProvider(**openai_params)
            if "anthropic" == provider:
                anthropic_params = model
                _providers[model['model']] = AnthropicProvider(**anthropic_params)
            if "huggingface" == provider:
                hf_params = model
                _providers[model['model']] = HuggingFaceProvider(**hf_params)
            if "vllm" == provider:
                vllm_params = model
                _providers[model['model']] = VLLMProvider(**vllm_params)
            if "ollama" == provider:
                ollama_params = model
                try:
                    temp_provider = OllamaProvider(**ollama_params)
                    if temp_provider.check_server_status():
                        _providers[model['model']] = OllamaProvider(**ollama_params)
                        print(f"OLLAMA server is running - {model['model']} provider available")
                    else:
                        print(f"Skipping {model['model']} - OLLAMA server not running (start with 'ollama serve')")
                except Exception as e:
                    print(f"Skipping {model['model']} - OLLAMA setup failed: {e}")

def get_paper_chunks(paper_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve all chunks for a specific paper from the vector store
    
    Args:
        paper_id: ID of the paper to retrieve chunks for
        
    Returns:
        List of document chunks with metadata
    """
    # Create a retriever with metadata filter for this specific paper
    filtered_retriever = _vector_store.as_retriever(
        search_kwargs={
            "filter": {"doc": paper_id},
            "k": 20  # Get more chunks to ensure we have the full paper
        }
    )
    
    # Retrieve documents
    docs = filtered_retriever.invoke("")
    return docs

def return_relevant_chunks(paper_id: str, criteria_query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve the most relevant chunks for a specific criteria query using vector similarity search
    
    Args:
        paper_id: ID of the paper to search within
        criteria_query: The specific criteria query to search for
        k: Number of most relevant chunks to retrieve
        
    Returns:
        List of most relevant document chunks for this criteria
    """
    # Create a retriever with metadata filter for this specific paper
    filtered_retriever = _vector_store.as_retriever(
        search_kwargs={
            "filter": {"doc": paper_id},
            "k": k  # Get top k most relevant chunks for this specific query
        }
    )
    
    # Use invoke instead of deprecated get_relevant_documents
    docs = filtered_retriever.invoke(criteria_query)
    return docs

def create_inference_query(full_context: str, criteria_prompt: str) -> str:
    """
    Create an inference query by combining paper chunks with the criteria prompt
    
    Args:
        paper_chunks: List of document chunks from the paper
        criteria_prompt: The criteria prompt to evaluate
        
    Returns:
        Combined query string for the LLM
    """
    # Create the RAG query with context instruction like in main.py
    inference_query = f"""
        Context from the research paper:
        {full_context}

        Task: {criteria_prompt}

        Consider ALL provided chunks of the paper when answering. Synthesize information from all relevant sections.
        """
    return inference_query

def process_paper_with_provider(paper_id: str, provider_name: str) -> Dict[str, Any]:
    """
    Process a single paper with a specific provider
    
    Args:
        paper_id: ID of the paper to process
        provider_name: Name of the provider to use
        
    Returns:
        Dictionary containing the results for this paper
    """
    if provider_name not in _providers:
        raise ValueError(f"Provider {provider_name} not available")
    
    provider = _providers[provider_name]
    results = {
        "paper_id": paper_id,
        "provider": provider_name,
        "title": None,  # Will be populated from chunk metadata
        "criteria_results": [],
        "final_classification": None,
        "chunks_processed": 0,
        "errors": []
    }
    
    # Process each criteria
    for i, criteria_prompt in enumerate(_criteria_prompts[:-1]):  # Exclude final aggregation
        full_context = ""  # Initialize to prevent UnboundLocalError in except block
        try:
            # Get relevant chunks for this specific criteria using vector similarity
            relevant_chunks = return_relevant_chunks(paper_id, criteria_prompt, k=5)
            
            # Extract title from the first chunk's metadata if not already set
            if relevant_chunks and results["title"] is None:
                if hasattr(relevant_chunks[0], 'metadata') and 'title' in relevant_chunks[0].metadata:
                    results["title"] = relevant_chunks[0].metadata['title']
            
            # Combine all chunks into a single context
            context_parts = []
            for chunk_idx, chunk in enumerate(relevant_chunks):
                context_parts.append(f"=== Chunk {chunk_idx+1} ===\n{chunk.page_content}\n")
            
            full_context = "\n".join(context_parts)

            if not relevant_chunks:
                results["errors"].append(f"No relevant chunks found for criterion {i+1}")
                continue

            inference_query = create_inference_query(full_context, criteria_prompt)

            query = Query(
                prompt=inference_query,
                system_message="You are an expert research analyst. Analyze the provided paper content and respond with valid JSON only.",
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=500
            )
            
            response = provider.call_api(query)
            
            # Use the standardizer to parse JSON response
            parsed_json, cleaned_content, success = standardize_llm_response(response.content)
            
            if success and parsed_json:
                results["criteria_results"].append({
                    "criterion": f"criterion_{i+1}",
                    "prompt": criteria_prompt,
                    "response": parsed_json,
                    "raw_response": response.content,
                    "cleaned_response": cleaned_content,
                    "chunks_used": len(relevant_chunks),
                    "full_context": full_context
                })
            else:
                results["criteria_results"].append({
                    "criterion": f"criterion_{i+1}",
                    "prompt": criteria_prompt,
                    "response": None,
                    "raw_response": response.content,
                    "cleaned_response": cleaned_content,
                    "error": "Failed to parse JSON",
                    "chunks_used": len(relevant_chunks),
                    "full_context": full_context
                })
                
        except Exception as e:
            results["errors"].append(f"Criterion {i+1} failed: {str(e)}")
            results["criteria_results"].append({
                "criterion": f"criterion_{i+1}",
                "prompt": criteria_prompt,
                "response": None,
                "error": str(e),
                "chunks_used": 0,
                "full_context": full_context
            })
            
    return results

def worker_process(provider_name: str, provider_config: BaseLLMProvider, 
                   result_queue: Queue, stop_event, shared_papers, paper_index):
                #    result_queue: Queue, stop_event, shared_papers, paper_index, index_lock):
    """
    Worker process that processes papers with a specific provider
    
    Args:
        provider_name: Name of the provider to use
        provider_config: Configuration for the provider
        result_queue: Queue to put results in
        stop_event: Event to signal when to stop
        shared_papers: List of paper IDs to process (shared across all workers)
        paper_index: Shared index to track which paper to process next
        index_lock: Lock for atomic operations on paper_index
    """
    try:
        # Initialize shared resources
        initialize_shared_resources()
        
        papers_processed = 0
        print(f"Worker for {provider_name} started")
        
        while not stop_event.is_set():
            # Get next paper from shared list atomically
            # with index_lock:
            #     if paper_index.value >= len(shared_papers):
            #         print(f"Worker {provider_name}: All papers assigned")
            #         break
            #     current_idx = paper_index.value
            #     paper_index.value += 1
            if paper_index >= len(shared_papers):
                print(f"Worker {provider_name}: All papers assigned")
                break
            current_idx = paper_index
            paper_index += 1
            
            paper_id = shared_papers[current_idx]
            print(f"Worker {provider_name}: Processing paper {paper_id} ({current_idx + 1}/{len(shared_papers)})")
            
            try:
                # Process with this provider
                result = process_paper_with_provider(paper_id, provider_name)
                result_queue.put(result)
                
                papers_processed += 1
                
            except Exception as e:
                print(f"Worker {provider_name}: Error processing paper {paper_id}: {e}")
                # Paper remains in shared list, no need to requeue
                continue
                
    except Exception as e:
        print(f"Worker {provider_name} failed: {e}")
        result_queue.put({
            "error": f"Worker {provider_name} failed: {str(e)}",
            "provider": provider_name
        })
    finally:
        print(f"Worker {provider_name} finished processing {papers_processed} papers")

def process_papers_multiprocessed(num_papers: int = 10, providers: List[str] = None, provider_configs: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Process papers using multiprocessing with one worker per provider
    
    Args:
        num_papers: Total number of papers to process across ALL providers (not per provider)
        providers: List of provider names to use (defaults to all available)
        provider_configs: Original provider configurations
        
    Returns:
        List of results for all processed papers (num_papers results, one per paper-provider combination)
    """
    if providers is None:
        providers = list(_providers.keys())
    
    if not providers:
        print("No providers available for processing")
        return []
    
    print(f"Starting multiprocessed batch processing of {num_papers} papers")
    print(f"Providers: {providers}")
    print(f"Queue length: {paper_queue_len()}")
    
    papers_to_process = []
    for _ in range(num_papers):
        paper_id = claim_next_paper(block_timeout=0)
        if paper_id:
            papers_to_process.append(paper_id)
        else:
            break
    
    if not papers_to_process:
        print("No papers available to process")
        return []
    
    print(f"Pre-fetched {len(papers_to_process)} papers: {papers_to_process}")
    
    # Create shared objects for inter-process communication
    manager = Manager()
    result_queue = Queue()
    stop_event = manager.Event()
    
    # Share the paper list with workers via Manager
    shared_papers = manager.list(papers_to_process)
    # paper_index = manager.Value('i', 0)
    paper_index = 0
    # paper_index_lock = Lock()
    
    # Start worker processes
    processes = []
    for provider_name, provider_object in _providers.items():
        process = Process(
            target=worker_process,
            args=(provider_name, provider_object, result_queue, stop_event, shared_papers, paper_index)
            # args=(provider_name, provider_object, result_queue, stop_event, shared_papers, paper_index, paper_index_lock)
        )
        process.start()
        processes.append(process)
        print(f"Started worker process for {provider_name} (PID: {process.pid})")
    
    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, stopping workers...")
        stop_event.set()
    
    original_sigint = signal.signal(signal.SIGINT, signal_handler)
    original_sigterm = signal.signal(signal.SIGTERM, signal_handler)
    
    # Collect results
    all_results = []
    completed_processes = set()
    
    try:
        while len(completed_processes) < len(processes):
            try:
                # Get result with timeout
                result = result_queue.get(timeout=5)
                all_results.append(result)
                
                # Save intermediate results periodically
                if len(all_results) % 10 == 0:
                    save_results(all_results, f"intermediate_results_{len(all_results)}.json")
                    # Also export completed papers periodically
                    export_completed_papers_to_file("data_generation/completed_papers.txt")
                    
            except Exception:
                # Check if any processes have finished
                for i, process in enumerate(processes):
                    if not process.is_alive() and i not in completed_processes:
                        completed_processes.add(i)
                        print(f"Process {i} ({process.name}) completed")
                
                # If all processes are done, break
                if len(completed_processes) >= len(processes):
                    break
                    
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, stopping workers...")
        stop_event.set()
        
    finally:
        # Restore original signal handlers
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        
        # Stop all processes
        stop_event.set()
        
        # Wait for processes to finish
        for process in processes:
            if process.is_alive():
                process.join(timeout=30)
                if process.is_alive():
                    print(f"Force terminating process {process.pid}")
                    process.terminate()
                    process.join()
    
    # Acknowledge all papers after all workers are done
    print(f"Acknowledging {len(papers_to_process)} papers from Redis...")
    for paper_id in papers_to_process:
        ack_paper(paper_id)
    
    # Track completed papers (deduplicated)
    completed_paper_ids = set()
    for result in all_results:
        if 'paper_id' in result:
            completed_paper_ids.add(result['paper_id'])

    for paper_id in completed_paper_ids:
        push_completed_paper(paper_id)
    
    print(f"Completed multiprocessed processing. Total results: {len(all_results)}")
    print(f"Unique papers processed: {len(completed_paper_ids)}")
    return all_results

def save_results(results: List[Dict[str, Any]], filename: str = None):
    """
    Save results to a JSON file and track completed papers
    
    Args:
        results: List of results to save
        filename: Optional filename (defaults to timestamped filename)
    """
    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rag_labeling_results_{timestamp}.json"
    
    saved_count = 0
    for result in results:
        # Skip results that are error messages from failed workers
        if 'error' in result and 'paper_id' not in result:
            print(f"Skipping worker error result: {result.get('error', 'Unknown error')}")
            continue
        
        # Skip results with processing errors
        if 'errors' in result and len(result['errors']) != 0:
            print(f"Result {result.get('paper_id', 'unknown')} has errors: {result['errors']}")
            continue
        
        # Skip results without required fields
        if 'paper_id' not in result or 'provider' not in result:
            print(f"Skipping invalid result: {result}")
            continue
            
        json_data = json.dumps(result).encode('utf-8')
        content_length = len(json_data)
        client.put_object(
            bucket_name = bucket_name,   
            object_name = f"{result['provider']}/{result['paper_id']}.json",
            data = BytesIO(json_data),
            length = content_length,
            content_type = "application/json"
        )
        saved_count += 1

    print(f"Results saved to {bucket_name}: {saved_count} papers")

def main():
    """
    Main function to run the RAG labeling script
    """
    # Configuration for different providers
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root_dir, "llm_params/llm_params3.json")) as f:
        provider_configs = json.load(f)
    
    # Initialize shared resources
    initialize_shared_resources()
    setup_providers(provider_configs)
    
    # Get current queue length to determine how many papers to process
    queue_size = paper_queue_len()
    print(f"Current queue size: {queue_size}")
    
    if queue_size == 0:
        print("Queue is empty. No papers to process.")
        return

    # Process papers using multiprocessing
    print("Starting RAG-based labeling generation with multiprocessing...")
    results = process_papers_multiprocessed(
        num_papers=queue_size,
        provider_configs=provider_configs
    )
    
    # Save final results
    save_results(results, "final_rag_labeling_results.json")
    
    # Export completed paper IDs to text file
    print(f"\n{'='*60}")
    print("Exporting completed paper IDs...")
    print(f"{'='*60}")
    
    completed_count = completed_papers_count()
    print(f"Total completed papers in queue: {completed_count}")
    
    if completed_count > 0:
        export_path = "data_generation/completed_papers.txt"
        exported = export_completed_papers_to_file(export_path)
        print(f"Exported {exported} unique completed paper IDs to {export_path}")
    
    print(f"\n{'='*60}")
    print(f"Completed processing {len(results)} paper-provider combinations")
    print("Results saved to final_rag_labeling_results.json")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
