import os
import json
import redis
import atexit
import signal
import threading
from typing import Optional

from dotenv import load_dotenv, find_dotenv

from config import load_config
from config.queue_config import QueueConfig

load_dotenv(find_dotenv(), override=True)

# Load config at module init; validates required env vars at startup
_queue_config = load_config(QueueConfig)

# Guard against concurrent or re-entrant flushes (e.g., signal + atexit)
_flush_lock = threading.Lock()
_has_flushed_on_shutdown = False

print(_queue_config.redis_url)
r = redis.Redis.from_url(
    _queue_config.redis_url,
    decode_responses=True,
    health_check_interval=30,
    socket_connect_timeout=5,
    socket_timeout=5,
    retry_on_timeout=True,
    client_name="ls-app",
)

def enqueue_paper_id(paper_id: str) -> bool:
    """Add once; skip duplicates."""
    added = r.sadd(_queue_config.paper_dedup_set, paper_id)
    if added:
        r.rpush(_queue_config.paper_queue, paper_id)
    return bool(added) 

def pop_paper_id(block: bool = False, timeout: int = 0) -> str or None:
    """Simple pop (use when reliability is less critical)."""
    if block:
        item = r.blpop(_queue_config.paper_queue, timeout=timeout)
        if not item:
            return None
        _, pid = item
    else:
        pid = r.lpop(_queue_config.paper_queue)
    if pid:
        r.srem(_queue_config.paper_dedup_set, pid)
    return pid

def claim_next_paper(block_timeout: int = 0) -> str or None:
    """
    Safer: atomically move from main queue to 'processing' (BRPOPLPUSH).
    After successful processing, call `ack_paper(paper_id)` to remove it.
    Returns None if no papers available or timeout occurs.
    """
    try:
        # First check if queue is empty to avoid unnecessary blocking
        if block_timeout == 0 and r.llen(_queue_config.paper_queue) == 0:
            return None

        pid = r.brpoplpush(
            _queue_config.paper_queue,
            _queue_config.paper_processing,
            timeout=block_timeout,
        )
        if pid:
            enqueue_paper_id(pid) # Put it back in the queue just for now. REPLACE LATER
        return pid  # None on timeout
    except redis.exceptions.ConnectionError as e:
        print(f"Redis connection error while claiming paper: {e}")
        raise
    except redis.exceptions.RedisError as e:
        print(f"Redis error while claiming paper: {e}")
        raise

def claim_next_paper_from_set(block_timeout: int = 0) -> str | None:
    """Claim a paper from the deduplication set."""
    pid = r.spop(_queue_config.generated_set)
    if pid:
        r.rpush(_queue_config.paper_processing, pid)
    if pid:
        return pid
    return None

def ack_paper(paper_id: str) -> None:
    """Remove from processing + dedupe set after success."""
    r.lrem(_queue_config.paper_processing, 0, paper_id)
    r.srem(_queue_config.paper_dedup_set, paper_id)


def requeue_inflight(paper_id: str) -> None:
    """Put it back if processing fails."""
    r.lrem(_queue_config.paper_processing, 0, paper_id)
    r.lpush(_queue_config.paper_queue, paper_id)


def paper_queue_len() -> int:
    return r.llen(_queue_config.paper_queue)

def push_completed_annotation(record: dict) -> None:
    """Push a completed annotation to the queue and flush to disk if large.

    Accepts either a dict (will be json-serialized) or a pre-serialized JSON string.
    When the queue length exceeds ANN_FLUSH_THRESHOLD, all queued annotations are
    atomically drained and appended to ANN_PERSIST_PATH as JSONL for durability.
    """
    # Accept both dict objects and JSON strings
    try:
        payload = record if isinstance(record, str) else json.dumps(record)
        r.rpush(_queue_config.ann_queue, payload)
    except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
        print(f"Redis connection issue while pushing annotation: {e}")
        # Redis is potentially shutting down - try to flush what we have
        try:
            _flush_annotations_on_shutdown()
        except Exception as flush_err:
            print(f"Failed emergency flush during connection error: {flush_err}")
        raise  # Re-raise the original error

    try:
        qlen = r.llen(_queue_config.ann_queue)
    except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
        print(f"Redis connection issue while checking queue length: {e}")
        # Redis might be shutting down - try to flush
        try:
            _flush_annotations_on_shutdown()
        except Exception as flush_err:
            print(f"Failed emergency flush during connection error: {flush_err}")
        # If we can't read length, do not attempt flush; best-effort push already done
        return
    except redis.exceptions.RedisError as e:
        print(f"Redis error while checking queue length: {e}")
        return

    if _queue_config.ann_flush_threshold > 0 and qlen > _queue_config.ann_flush_threshold:
        flush_annotations_to_persistent()

def annotation_queue_len() -> int:
    """Return the current number of items in the annotations queue."""
    return r.llen(_queue_config.ann_queue)

def flush_annotations_to_persistent(max_items: Optional[int] = None) -> int:
    """Drain annotations from Redis to persistent storage (JSONL file).

    - max_items: if provided and > 0, drain up to this many items; otherwise drain all.
    Returns the number of flushed items.
    """
    # Atomically fetch and trim/delete using a pipeline transaction
    if max_items and max_items > 0:
        pipe = r.pipeline()
        pipe.lrange(_queue_config.ann_queue, 0, max_items - 1)
        pipe.ltrim(_queue_config.ann_queue, max_items, -1)
        results = pipe.execute()
        items = results[0]
    else:
        pipe = r.pipeline()
        pipe.lrange(_queue_config.ann_queue, 0, -1)
        pipe.delete(_queue_config.ann_queue)
        results = pipe.execute()
        items = results[0]

    if not items:
        return 0

    # Ensure directory exists if a directory component is provided
    dirname = os.path.dirname(_queue_config.ann_persist_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    try:
        with open(_queue_config.ann_persist_path, "a", encoding="utf-8") as f:
            for line in items:
                # Items are strings (already JSON); ensure newline-delimited
                text = line if isinstance(line, str) else json.dumps(line)
                if not text.endswith("\n"):
                    f.write(text + "\n")
                else:
                    f.write(text)
        return len(items)
    except Exception:
        # On persistence failure, push items back to the head preserving order
        # We deleted/trimmed them already; best-effort restore
        try:
            with r.pipeline() as p:
                for entry in reversed(items):
                    p.lpush(_queue_config.ann_queue, entry)
                p.execute()
        finally:
            pass
        raise

def _flush_annotations_on_shutdown() -> None:
    global _has_flushed_on_shutdown
    if _has_flushed_on_shutdown:
        return
    with _flush_lock:
        if _has_flushed_on_shutdown:
            return
        try:
            flushed = flush_annotations_to_persistent()
            if flushed:
                print(f"Flushed {flushed} annotations to {_queue_config.ann_persist_path} on shutdown")
        except Exception as e:
            # Best-effort; do not raise during shutdown
            print(f"Failed to flush annotations on shutdown: {e}")
        finally:
            _has_flushed_on_shutdown = True

def _make_signal_handler(prev_handler):
    def handler(signum, frame):
        try:
            _flush_annotations_on_shutdown()
        finally:
            if callable(prev_handler):
                try:
                    prev_handler(signum, frame)
                except Exception:
                    pass
            # If previous handler was default or ignore, exit gracefully
            raise SystemExit(0)
    return handler

# Register shutdown hooks based on env configuration
if _queue_config.ann_flush_on_exit:
    atexit.register(_flush_annotations_on_shutdown)

if _queue_config.ann_install_signal_handlers:
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            prev = signal.getsignal(sig)
            signal.signal(sig, _make_signal_handler(prev))
        except Exception:
            # Some environments may not allow setting handlers; ignore
            pass

def shutdown_annotations() -> None:
    """Public helper to proactively flush annotations before process exit."""
    _flush_annotations_on_shutdown()

def push_completed_paper(paper_id: str) -> None:
    """Add a paper ID to the completed papers queue."""
    r.sadd(_queue_config.generated_set, paper_id)
    # r.rpush(COMPLETED_PAPERS_QUEUE, paper_id)

def get_all_completed_papers() -> list:
    """Retrieve all completed paper IDs from the queue (non-destructive)."""
    return r.lrange(_queue_config.completed_papers_queue, 0, -1)


def completed_papers_count() -> int:
    """Return the count of completed papers in the queue."""
    return r.llen(_queue_config.completed_papers_queue)

def export_completed_papers_to_file(filepath: str = "completed_papers.txt") -> int:
    """Export all completed paper IDs to a text file (one per line).
    
    Returns the number of paper IDs exported.
    """
    paper_ids = get_all_completed_papers()
    
    if not paper_ids:
        print("No completed papers to export")
        return 0
    
    # Ensure directory exists
    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for paper_id in paper_ids:
            f.write(f"{paper_id}\n")
    
    print(f"Exported {len(paper_ids)} completed paper IDs to {filepath}")
    return len(paper_ids)
