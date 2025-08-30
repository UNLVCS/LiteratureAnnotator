import os, json, redis

REDIS_URL       = os.getenv("REDIS_URL", "redis://localhost:6379/0")
PAPER_QUEUE     = os.getenv("PAPER_QUEUE", "q:papers:v1")
PROCESSING_Q    = os.getenv("PAPER_PROCESSING", "q:papers:processing:v1")
DEDUP_SET       = os.getenv("PAPER_DEDUP_SET", "s:papers:enqueued:v1")
ANN_QUEUE       = os.getenv("ANN_QUEUE", "q:annotations:completed:v1")

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
    r.rpush(ANN_QUEUE, json.dumps(record))
