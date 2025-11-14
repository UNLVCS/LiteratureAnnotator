import os, json, redis
import atexit, signal, threading
from typing import Optional

REDIS_URL       = os.getenv("REDIS_URL", "redis://localhost:6379/0")
PAPER_QUEUE     = os.getenv("PAPER_QUEUE", "q:papers:v1")
PROCESSING_Q    = os.getenv("PAPER_PROCESSING", "q:papers:processing:v1")
DEDUP_SET       = os.getenv("PAPER_DEDUP_SET", "s:papers:enqueued:v1")
ANN_QUEUE       = os.getenv("ANN_QUEUE", "q:annotations:completed:v1")

# When the annotations queue grows beyond this threshold, persist to disk
ANN_FLUSH_THRESHOLD = int(os.getenv("ANN_FLUSH_THRESHOLD", "1000"))
# Path to append persisted annotations (JSON Lines format)
ANN_PERSIST_PATH = os.getenv("ANN_PERSIST_PATH", "data_labeling/annotations.jsonl")
ANN_FLUSH_ON_EXIT = os.getenv("ANN_FLUSH_ON_EXIT", "1") not in ("0", "false", "False")
ANN_INSTALL_SIGNAL_HANDLERS = os.getenv("ANN_INSTALL_SIGNAL_HANDLERS", "1") not in ("0", "false", "False")

# Guard against concurrent or re-entrant flushes (e.g., signal + atexit)
_flush_lock = threading.Lock()
_has_flushed_on_shutdown = False

r = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,
    health_check_interval=30,
    socket_connect_timeout=5,
    socket_timeout=5,
    retry_on_timeout=True,
    client_name="ls-app",
)

def enqueue_paper_id(paper_id: str) -> bool:
    """Add once; skip duplicates."""
    added = r.sadd(DEDUP_SET, paper_id)
    if added:
        r.rpush(PAPER_QUEUE, paper_id)
    return bool(added)

def pop_paper_id(block: bool = False, timeout: int = 0) -> str or None:
    """Simple pop (use when reliability is less critical)."""
    if block:
        item = r.blpop(PAPER_QUEUE, timeout=timeout)
        if not item:
            return None
        _, pid = item
    else:
        pid = r.lpop(PAPER_QUEUE)
    if pid:
        r.srem(DEDUP_SET, pid)
    return pid

def claim_next_paper(block_timeout: int = 0) -> str or None:
    """
    Safer: atomically move from main queue to 'processing' (BRPOPLPUSH).
    After successful processing, call `ack_paper(paper_id)` to remove it.
    Returns None if no papers available or timeout occurs.
    """
    try:
        # First check if queue is empty to avoid unnecessary blocking
        if block_timeout == 0 and r.llen(PAPER_QUEUE) == 0:
            return None
            
        pid = r.brpoplpush(PAPER_QUEUE, PROCESSING_Q, timeout=block_timeout)
        return pid  # None on timeout
    except redis.exceptions.ConnectionError as e:
        print(f"Redis connection error while claiming paper: {e}")
        raise
    except redis.exceptions.RedisError as e:
        print(f"Redis error while claiming paper: {e}")
        raise

def ack_paper(paper_id: str) -> None:
    """Remove from processing + dedupe set after success."""
    r.lrem(PROCESSING_Q, 0, paper_id)
    r.srem(DEDUP_SET, paper_id)

def requeue_inflight(paper_id: str) -> None:
    """Put it back if processing fails."""
    r.lrem(PROCESSING_Q, 0, paper_id)
    r.lpush(PAPER_QUEUE, paper_id)

def paper_queue_len() -> int:
    return r.llen(PAPER_QUEUE)

def push_completed_annotation(record: dict) -> None:
    """Push a completed annotation to the queue and flush to disk if large.

    Accepts either a dict (will be json-serialized) or a pre-serialized JSON string.
    When the queue length exceeds ANN_FLUSH_THRESHOLD, all queued annotations are
    atomically drained and appended to ANN_PERSIST_PATH as JSONL for durability.
    """
    # Accept both dict objects and JSON strings
    try:
        payload = record if isinstance(record, str) else json.dumps(record)
        r.rpush(ANN_QUEUE, payload)
    except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
        print(f"Redis connection issue while pushing annotation: {e}")
        # Redis is potentially shutting down - try to flush what we have
        try:
            _flush_annotations_on_shutdown()
        except Exception as flush_err:
            print(f"Failed emergency flush during connection error: {flush_err}")
        raise  # Re-raise the original error

    try:
        qlen = r.llen(ANN_QUEUE)
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

    if ANN_FLUSH_THRESHOLD > 0 and qlen > ANN_FLUSH_THRESHOLD:
        flush_annotations_to_persistent()

def annotation_queue_len() -> int:
    """Return the current number of items in the annotations queue."""
    return r.llen(ANN_QUEUE)

def flush_annotations_to_persistent(max_items: Optional[int] = None) -> int:
    """Drain annotations from Redis to persistent storage (JSONL file).

    - max_items: if provided and > 0, drain up to this many items; otherwise drain all.
    Returns the number of flushed items.
    """
    # Atomically fetch and trim/delete using a pipeline transaction
    if max_items and max_items > 0:
        pipe = r.pipeline()
        pipe.lrange(ANN_QUEUE, 0, max_items - 1)
        pipe.ltrim(ANN_QUEUE, max_items, -1)
        results = pipe.execute()
        items = results[0]
    else:
        pipe = r.pipeline()
        pipe.lrange(ANN_QUEUE, 0, -1)
        pipe.delete(ANN_QUEUE)
        results = pipe.execute()
        items = results[0]

    if not items:
        return 0

    # Ensure directory exists if a directory component is provided
    dirname = os.path.dirname(ANN_PERSIST_PATH)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    try:
        with open(ANN_PERSIST_PATH, "a", encoding="utf-8") as f:
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
                    p.lpush(ANN_QUEUE, entry)
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
                print(f"Flushed {flushed} annotations to {ANN_PERSIST_PATH} on shutdown")
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
if ANN_FLUSH_ON_EXIT:
    atexit.register(_flush_annotations_on_shutdown)

if ANN_INSTALL_SIGNAL_HANDLERS:
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
