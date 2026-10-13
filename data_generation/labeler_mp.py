#!/usr/bin/env python3
"""
Whole-paper Data Labeling Script using LLM Providers (Global Functions)

Claims paper IDs from the Redis queue, loads each paper's full text from the raw
BioC JSON in MinIO, and evaluates every criterion with every configured provider
(one worker process per provider). See labeling.py for the per-paper logic.

This version uses global functions and variables for multiprocessing compatibility.
"""


import json
import signal
import sys
from io import BytesIO
from multiprocessing import Process, Queue, Manager
from pathlib import Path
from typing import Any, Dict, List

from minio import Minio

# Add the parent directory to the path so we can import llm_providers
sys.path.append(str(Path(__file__).parent.parent))

from config.app_config import load_app_config
from config.llm_providers_config import instantiate_provider
from llm_providers.base import BaseLLMProvider
from papers import PaperNotFoundError, load_paper_from_config
from utilities.queue_helpers import (
    claim_next_paper,
    ack_paper,
    requeue_inflight,
    paper_queue_len,
    push_completed_paper,
    completed_papers_count,
    export_completed_papers_to_file,
)
from labeling import SkipPaper, label_paper, result_object_name

# Load config from .env.yaml
_app_config = load_app_config()

client = Minio(
    _app_config.minio.url,
    access_key=_app_config.minio.access_key,
    secret_key=_app_config.minio.secret_key,
    secure=_app_config.minio.secure,
)
bucket_name = _app_config.minio.synthetic_data_bucket  # labeling output
if not client.bucket_exists(bucket_name):
    print(f"Bucket {bucket_name} does not exist. Creating it...")
    client.make_bucket(bucket_name)
else:
    print(f"Bucket {bucket_name} already exists.")

def load_paper(paper_id: str):
    """Load and parse the raw BioC article for one paper ID from MinIO."""
    return load_paper_from_config(paper_id, _app_config, client=client)


def process_paper_with_provider(
    paper_id: str,
    provider_name: str,
    providers: Dict[str, BaseLLMProvider],
) -> Dict[str, Any]:
    """
    Label a single paper with a specific provider over its full text.

    Raises:
        PaperNotFoundError / SkipPaper: the paper cannot be labeled; callers ack it.
    """
    if provider_name not in providers:
        raise ValueError(f"Provider {provider_name} not available")
    paper = load_paper(paper_id)
    return label_paper(
        paper,
        provider_name,
        providers[provider_name],
        max_chars=_app_config.papers.max_chars,
    )

def worker_process(
    provider_name: str,
    provider_type: str,
    provider_kwargs: Dict[str, Any],
    result_queue: Queue,
    stop_event,
    shared_papers,
    paper_index,
):
    """
    Worker process that processes papers with a single provider.

    Each worker instantiates only its own provider via instantiate_provider(),
    avoiding the GPU-memory conflict that arises when every worker calls
    get_providers_dict() (which would load ALL providers, including vllm_native).

    Args:
        provider_name: Model name key (e.g. "meta-llama/Llama-3.1-8B-Instruct").
        provider_type: Provider type string (e.g. "vllm_native", "openai").
        provider_kwargs: Constructor kwargs for the provider.
        result_queue: Queue to put results in.
        stop_event: Event to signal when to stop.
        shared_papers: List of paper IDs to process (shared across all workers).
        paper_index: Starting index into shared_papers for this worker.
    """
    papers_processed = 0
    try:
        try:
            provider = instantiate_provider(provider_type, provider_kwargs)
        except Exception as e:
            print(f"Failed to setup {provider_name}: {e}")
            result_queue.put({"error": f"Provider setup failed: {e}", "provider": provider_name})
            return

        providers = {provider_name: provider}
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
                result = process_paper_with_provider(paper_id, provider_name, providers)
                result_queue.put(result)
                papers_processed += 1
            except (PaperNotFoundError, SkipPaper) as e:
                print(f"Worker {provider_name}: Skipping paper {paper_id}: {e}")
                result_queue.put({
                    "paper_id": paper_id,
                    "provider": provider_name,
                    "skipped": str(e),
                    "errors": [],
                    "criteria_results": [],
                })
            except Exception as e:
                print(f"Worker {provider_name}: Error processing paper {paper_id}: {e}")
                # Always emit a result so the coordinator can track this paper's outcome
                result_queue.put({
                    "paper_id": paper_id,
                    "provider": provider_name,
                    "errors": [f"Unhandled worker exception: {e}"],
                    "criteria_results": [],
                })
                
    except Exception as e:
        print(f"Worker {provider_name} failed: {e}")
        result_queue.put({
            "error": f"Worker {provider_name} failed: {str(e)}",
            "provider": provider_name
        })
    finally:
        print(f"Worker {provider_name} finished processing {papers_processed} papers")

def process_papers_multiprocessed(num_papers: int = 10, providers: List[str] = None) -> List[Dict[str, Any]]:
    """
    Process papers using multiprocessing with one worker per provider

    Args:
        num_papers: Total number of papers to process across ALL providers (not per provider)
        providers: List of provider names to use (defaults to all available)

    Returns:
        List of results for all processed papers (num_papers results, one per paper-provider combination)
    """
    # get_provider_specs() reads config only — no GPU/network side-effects.
    provider_specs = _app_config.get_provider_specs()

    if providers is None:
        provider_names = list(provider_specs.keys())
    else:
        provider_names = [name for name in providers if name in provider_specs]
        missing_providers = [name for name in providers if name not in provider_specs]
        for missing in missing_providers:
            print(f"Skipping unknown/unconfigured provider: {missing}")

    if not provider_names:
        print("No providers available for processing")
        return []

    print(f"Starting multiprocessed batch processing of {num_papers} papers")
    print(f"Providers: {provider_names}")
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
    
    # Start one worker process per provider, passing only that provider's spec so
    # each child instantiates exactly one provider (no shared GPU state conflicts).
    processes = []
    for provider_name in provider_names:
        provider_type, provider_kwargs = provider_specs[provider_name]
        process = Process(
            target=worker_process,
            args=(provider_name, provider_type, provider_kwargs,
                  result_queue, stop_event, shared_papers, paper_index),
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
    
    # Classify each paper as successful or failed.
    # A paper is successful only when EVERY provider produced a result with no errors.
    # Any provider failure (errors list non-empty) marks the paper as failed so it gets
    # requeued rather than silently skipped.
    papers_with_errors: set = set()
    papers_with_success: set = set()
    papers_skipped: set = set()
    for result in all_results:
        paper_id = result.get("paper_id")
        if paper_id is None:
            continue
        if result.get("skipped"):
            # Not downloaded / no text: retrying cannot help, so it is acked below.
            papers_skipped.add(paper_id)
        elif result.get("errors"):
            papers_with_errors.add(paper_id)
        else:
            papers_with_success.add(paper_id)

    # A paper is truly completed only if every provider succeeded (no errors from any provider).
    truly_completed = papers_with_success - papers_with_errors

    # Papers that never appeared in results (worker died before emitting) also need requeue.
    papers_missing = set(papers_to_process) - papers_with_success - papers_with_errors - papers_skipped

    papers_to_requeue = (papers_with_errors | papers_missing) - truly_completed - papers_skipped

    print(f"Acknowledging {len(truly_completed)} successfully completed papers from Redis...")
    for paper_id in truly_completed:
        ack_paper(paper_id)
        push_completed_paper(paper_id)

    if papers_skipped:
        print(f"Acknowledging {len(papers_skipped)} papers that cannot be labeled (see 'Skipping' above):")
        for paper_id in sorted(papers_skipped):
            ack_paper(paper_id)
            print(f"  Skipped: {paper_id}")

    print(f"Requeuing {len(papers_to_requeue)} papers that had provider failures...")
    for paper_id in papers_to_requeue:
        requeue_inflight(paper_id)
        print(f"  Requeued: {paper_id}")

    print(f"Completed multiprocessed processing. Total results: {len(all_results)}")
    print(f"Unique papers truly completed: {len(truly_completed)}")
    print(f"Unique papers skipped (not downloaded / empty): {len(papers_skipped)}")
    print(f"Unique papers requeued for retry: {len(papers_to_requeue)}")
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
        filename = f"labeling_results_{timestamp}.json"
    
    saved_count = 0
    for result in results:
        # Skip results that are error messages from failed workers
        if 'error' in result and 'paper_id' not in result:
            print(f"Skipping worker error result: {result.get('error', 'Unknown error')}")
            continue

        # Skip papers that were never labeled (missing raw article / empty text)
        if result.get('skipped'):
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
        object_name = result_object_name(result, _app_config.minio.output_prefix)
        client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=BytesIO(json_data),
            length=content_length,
            content_type="application/json",
        )
        saved_count += 1

    print(f"Results saved to {bucket_name}: {saved_count} papers")

def main():
    """
    Main function to run the whole-paper labeling script
    """
    # Get current queue length to determine how many papers to process
    queue_size = paper_queue_len()
    print(f"Current queue size: {queue_size}")
    
    if queue_size == 0:
        print("Queue is empty. No papers to process.")
        return

    # get_provider_specs() is config-only — no GPU/network side-effects here.
    provider_specs = _app_config.get_provider_specs()
    if not provider_specs:
        print(
            "No LLM providers configured. In .env.yaml set llm_providers with at least one "
            "model with skip: false (e.g. vllm with base_url, or ollama with server running, "
            "or openai/anthropic with api_key)."
        )
        return

    # Process papers using multiprocessing
    print("Starting whole-paper labeling with multiprocessing...")
    results = process_papers_multiprocessed(
        num_papers=queue_size,
        providers=list(provider_specs.keys()),
    )
    
    # Save final results
    save_results(results, "final_labeling_results.json")
    
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
    print(f"Results saved to MinIO bucket {bucket_name}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
