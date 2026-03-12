"""
Queue helpers for paper processing pipelines.

Usage:
    # Use default instance (loads config from .env.yaml):
    from utilities.queue_helpers import claim_next_paper, ack_paper

    # Or create custom instance with explicit config:
    from utilities.queue_helpers import PaperQueue
    from config.app_config import load_app_config
    config = load_app_config()
    queue = PaperQueue.from_app_config(config)
    queue.claim_next_paper()
"""

import os
import json
import redis
import atexit
import signal
import threading
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from config.app_config import AppConfig, RedisConfig


class PaperQueueConfig:
    """Adapter to normalize different config formats."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        paper_queue: str = "q:papers:v1",
        paper_processing: str = "q:papers:processing:v1",
        paper_dedup_set: str = "s:papers:enqueued:v1",
        completed_papers_queue: str = "q:papers:completed:v1",
        generated_set: str = "s:papers:generated:v1",
        ann_queue: str = "q:annotations:completed:v1",
        ann_flush_threshold: int = 1000,
        ann_persist_path: str = "data_labeling/annotations.jsonl",
        ann_flush_on_exit: bool = True,
        ann_install_signal_handlers: bool = True,
        human_paper_queue: str = "q:papers:human:v1",
        human_processing_queue: str = "q:papers:human:processing:v1",
        human_dedup_set: str = "s:papers:human:enqueued:v1",
    ):
        self.redis_url = redis_url
        self.paper_queue = paper_queue
        self.paper_processing = paper_processing
        self.paper_dedup_set = paper_dedup_set
        self.completed_papers_queue = completed_papers_queue
        self.generated_set = generated_set
        self.ann_queue = ann_queue
        self.ann_flush_threshold = ann_flush_threshold
        self.ann_persist_path = ann_persist_path
        self.ann_flush_on_exit = ann_flush_on_exit
        self.ann_install_signal_handlers = ann_install_signal_handlers
        self.HUMAN_PAPER_QUEUE = human_paper_queue
        self.HUMAN_PROCESSING_Q = human_processing_queue
        self.HUMAN_DEDUP_SET = human_dedup_set

    @classmethod
    def from_redis_config(cls, redis_config: "RedisConfig") -> "PaperQueueConfig":
        """Create from AppConfig.redis (RedisConfig)."""
        return cls(
            redis_url=redis_config.url,
            paper_queue=redis_config.paper_queue,
            paper_processing=redis_config.paper_processing,
            paper_dedup_set=redis_config.paper_dedup_set,
            completed_papers_queue=redis_config.completed_papers_queue,
            generated_set=redis_config.generated_set,
            ann_queue=redis_config.ann_queue,
            ann_flush_threshold=redis_config.ann_flush_threshold,
            ann_persist_path=redis_config.ann_persist_path,
            ann_flush_on_exit=redis_config.ann_flush_on_exit,
            ann_install_signal_handlers=redis_config.ann_install_signal_handlers,
            human_paper_queue=redis_config.human_paper_queue,
            human_processing_queue=redis_config.human_processing_queue,
            human_dedup_set=redis_config.human_dedup_set,
        )


class PaperQueue:
    """
    Queue manager for paper processing pipelines.
    
    Accepts a PaperQueueConfig to allow custom Redis URLs and queue names.
    """

    def __init__(self, config: PaperQueueConfig):
        self.config = config
        self._flush_lock = threading.Lock()
        self._has_flushed_on_shutdown = False
        
        self.redis = redis.Redis.from_url(
            config.redis_url,
            decode_responses=True,
            health_check_interval=30,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            client_name="ls-app",
        )

    @classmethod
    def from_app_config(cls, app_config: "AppConfig") -> "PaperQueue":
        """Create PaperQueue from AppConfig."""
        config = PaperQueueConfig.from_redis_config(app_config.redis)
        return cls(config)

    # -------------------------------------------------------------------------
    # Paper queue operations
    # -------------------------------------------------------------------------

    def enqueue_paper_id(self, paper_id: str) -> bool:
        """Add once; skip duplicates."""
        added = self.redis.sadd(self.config.paper_dedup_set, paper_id)
        if added:
            self.redis.rpush(self.config.paper_queue, paper_id)
        return bool(added)

    def pop_paper_id(self, block: bool = False, timeout: int = 0) -> str | None:
        """Simple pop (use when reliability is less critical)."""
        if block:
            item = self.redis.blpop(self.config.paper_queue, timeout=timeout)
            if not item:
                return None
            _, pid = item
        else:
            pid = self.redis.lpop(self.config.paper_queue)
        if pid:
            self.redis.srem(self.config.paper_dedup_set, pid)
        return pid

    def claim_next_paper(self, block_timeout: int = 0) -> str | None:
        """
        Safer: atomically move from main queue to 'processing' (BRPOPLPUSH).
        After successful processing, call `ack_paper(paper_id)` to remove it.
        Returns None if no papers available or timeout occurs.
        """
        try:
            if block_timeout == 0 and self.redis.llen(self.config.paper_queue) == 0:
                return None

            pid = self.redis.brpoplpush(
                self.config.paper_queue,
                self.config.paper_processing,
                timeout=block_timeout,
            )
            if pid:
                self.enqueue_paper_id(pid)  # Put it back in the queue just for now. REPLACE LATER
            return pid
        except redis.exceptions.ConnectionError as e:
            print(f"Redis connection error while claiming paper: {e}")
            raise
        except redis.exceptions.RedisError as e:
            print(f"Redis error while claiming paper: {e}")
            raise

    def claim_next_paper_from_set(self, block_timeout: int = 0) -> str | None:
        """Claim a paper from the deduplication set."""
        pid = self.redis.spop(self.config.generated_set)
        if pid:
            self.redis.rpush(self.config.paper_processing, pid)
        return pid if pid else None

    def ack_paper(self, paper_id: str) -> None:
        """Remove from processing + dedupe set after success."""
        self.redis.lrem(self.config.paper_processing, 0, paper_id)
        self.redis.srem(self.config.paper_dedup_set, paper_id)

    def requeue_inflight(self, paper_id: str) -> None:
        """Put it back if processing fails."""
        self.redis.lrem(self.config.paper_processing, 0, paper_id)
        self.redis.lpush(self.config.paper_queue, paper_id)

    def paper_queue_len(self) -> int:
        return self.redis.llen(self.config.paper_queue)

    def push_completed_paper(self, paper_id: str) -> None:
        """Add a paper ID to the completed papers queue."""
        self.redis.sadd(self.config.generated_set, paper_id)

    def get_all_completed_papers(self) -> list:
        """Retrieve all completed paper IDs from the queue (non-destructive)."""
        return self.redis.lrange(self.config.completed_papers_queue, 0, -1)

    def completed_papers_count(self) -> int:
        """Return the count of completed papers in the queue."""
        return self.redis.llen(self.config.completed_papers_queue)

    def export_completed_papers_to_file(self, filepath: str = "completed_papers.txt") -> int:
        """Export all completed paper IDs to a text file (one per line)."""
        paper_ids = self.get_all_completed_papers()

        if not paper_ids:
            print("No completed papers to export")
            return 0

        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            for paper_id in paper_ids:
                f.write(f"{paper_id}\n")

        print(f"Exported {len(paper_ids)} completed paper IDs to {filepath}")
        return len(paper_ids)

    # -------------------------------------------------------------------------
    # Human labeling queue operations
    # -------------------------------------------------------------------------

    def enqueue_paper_id_human(self, paper_id: str) -> bool:
        """Add paper ID to human queue once; skip duplicates."""
        added = self.redis.sadd(self.config.HUMAN_DEDUP_SET, paper_id)
        if added:
            self.redis.rpush(self.config.HUMAN_PAPER_QUEUE, paper_id)
        return bool(added)

    def claim_next_paper_human(self, block_timeout: int = 0) -> str | None:
        """
        Claim next paper from human queue (atomically move to processing).
        After successful processing, call ack_paper_human(paper_id).
        Returns None if no papers available or timeout occurs.
        """
        try:
            if block_timeout == 0 and self.redis.llen(self.config.HUMAN_PAPER_QUEUE) == 0:
                return None
            pid = self.redis.brpoplpush(
                self.config.HUMAN_PAPER_QUEUE,
                self.config.HUMAN_PROCESSING_Q,
                timeout=block_timeout,
            )
            if pid:
                self.enqueue_paper_id_human(pid)
            return pid
        except redis.exceptions.ConnectionError as e:
            print(f"Redis connection error while claiming human paper: {e}")
            raise
        except redis.exceptions.RedisError as e:
            print(f"Redis error while claiming human paper: {e}")
            raise

    def ack_paper_human(self, paper_id: str) -> None:
        """Remove from human processing queue and dedup set after success."""
        self.redis.lrem(self.config.HUMAN_PROCESSING_Q, 0, paper_id)
        self.redis.srem(self.config.HUMAN_DEDUP_SET, paper_id)

    def requeue_inflight_human(self, paper_id: str) -> None:
        """Put paper back in human queue if processing fails."""
        self.redis.lrem(self.config.HUMAN_PROCESSING_Q, 0, paper_id)
        self.redis.lpush(self.config.HUMAN_PAPER_QUEUE, paper_id)

    def human_paper_queue_len(self) -> int:
        """Return the number of papers in the human queue."""
        return self.redis.llen(self.config.HUMAN_PAPER_QUEUE)

    # -------------------------------------------------------------------------
    # Annotation queue operations
    # -------------------------------------------------------------------------

    def push_completed_annotation(self, record: dict) -> None:
        """Push a completed annotation to the queue and flush to disk if large."""
        try:
            payload = record if isinstance(record, str) else json.dumps(record)
            self.redis.rpush(self.config.ann_queue, payload)
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
            print(f"Redis connection issue while pushing annotation: {e}")
            try:
                self._flush_annotations_on_shutdown()
            except Exception as flush_err:
                print(f"Failed emergency flush during connection error: {flush_err}")
            raise

        try:
            qlen = self.redis.llen(self.config.ann_queue)
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
            print(f"Redis connection issue while checking queue length: {e}")
            try:
                self._flush_annotations_on_shutdown()
            except Exception as flush_err:
                print(f"Failed emergency flush during connection error: {flush_err}")
            return
        except redis.exceptions.RedisError as e:
            print(f"Redis error while checking queue length: {e}")
            return

        if self.config.ann_flush_threshold > 0 and qlen > self.config.ann_flush_threshold:
            self.flush_annotations_to_persistent()

    def annotation_queue_len(self) -> int:
        """Return the current number of items in the annotations queue."""
        return self.redis.llen(self.config.ann_queue)

    def flush_annotations_to_persistent(self, max_items: Optional[int] = None) -> int:
        """Drain annotations from Redis to persistent storage (JSONL file)."""
        if max_items and max_items > 0:
            pipe = self.redis.pipeline()
            pipe.lrange(self.config.ann_queue, 0, max_items - 1)
            pipe.ltrim(self.config.ann_queue, max_items, -1)
            results = pipe.execute()
            items = results[0]
        else:
            pipe = self.redis.pipeline()
            pipe.lrange(self.config.ann_queue, 0, -1)
            pipe.delete(self.config.ann_queue)
            results = pipe.execute()
            items = results[0]

        if not items:
            return 0

        dirname = os.path.dirname(self.config.ann_persist_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        try:
            with open(self.config.ann_persist_path, "a", encoding="utf-8") as f:
                for line in items:
                    text = line if isinstance(line, str) else json.dumps(line)
                    if not text.endswith("\n"):
                        f.write(text + "\n")
                    else:
                        f.write(text)
            return len(items)
        except Exception:
            try:
                with self.redis.pipeline() as p:
                    for entry in reversed(items):
                        p.lpush(self.config.ann_queue, entry)
                    p.execute()
            finally:
                pass
            raise

    def _flush_annotations_on_shutdown(self) -> None:
        if self._has_flushed_on_shutdown:
            return
        with self._flush_lock:
            if self._has_flushed_on_shutdown:
                return
            try:
                flushed = self.flush_annotations_to_persistent()
                if flushed:
                    print(f"Flushed {flushed} annotations to {self.config.ann_persist_path} on shutdown")
            except Exception as e:
                print(f"Failed to flush annotations on shutdown: {e}")
            finally:
                self._has_flushed_on_shutdown = True

    def shutdown_annotations(self) -> None:
        """Public helper to proactively flush annotations before process exit."""
        self._flush_annotations_on_shutdown()

    def register_shutdown_hooks(self) -> None:
        """Register atexit and signal handlers for graceful shutdown."""
        if self.config.ann_flush_on_exit:
            atexit.register(self._flush_annotations_on_shutdown)

        if self.config.ann_install_signal_handlers:
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    prev = signal.getsignal(sig)
                    signal.signal(sig, self._make_signal_handler(prev))
                except Exception:
                    pass

    def _make_signal_handler(self, prev_handler):
        def handler(signum, frame):
            try:
                self._flush_annotations_on_shutdown()
            finally:
                if callable(prev_handler):
                    try:
                        prev_handler(signum, frame)
                    except Exception:
                        pass
                raise SystemExit(0)
        return handler


# -----------------------------------------------------------------------------
# Default instance for backward compatibility
# -----------------------------------------------------------------------------

def _create_default_queue() -> PaperQueue:
    """Create default queue instance from .env.yaml."""
    from config.app_config import load_app_config

    app_config = load_app_config()
    print(f"Redis URL: {app_config.redis.url}")
    
    queue = PaperQueue.from_app_config(app_config)
    queue.register_shutdown_hooks()
    return queue


# Lazy initialization to avoid connection at import time when not needed
_default_queue: Optional[PaperQueue] = None


def _get_default_queue() -> PaperQueue:
    global _default_queue
    if _default_queue is None:
        _default_queue = _create_default_queue()
    return _default_queue


# Backward-compatible module-level functions that delegate to default instance
def enqueue_paper_id(paper_id: str) -> bool:
    return _get_default_queue().enqueue_paper_id(paper_id)

def pop_paper_id(block: bool = False, timeout: int = 0) -> str | None:
    return _get_default_queue().pop_paper_id(block, timeout)

def claim_next_paper(block_timeout: int = 0) -> str | None:
    return _get_default_queue().claim_next_paper(block_timeout)

def claim_next_paper_from_set(block_timeout: int = 0) -> str | None:
    return _get_default_queue().claim_next_paper_from_set(block_timeout)

def ack_paper(paper_id: str) -> None:
    return _get_default_queue().ack_paper(paper_id)

def requeue_inflight(paper_id: str) -> None:
    return _get_default_queue().requeue_inflight(paper_id)

def paper_queue_len() -> int:
    return _get_default_queue().paper_queue_len()

def push_completed_paper(paper_id: str) -> None:
    return _get_default_queue().push_completed_paper(paper_id)

def get_all_completed_papers() -> list:
    return _get_default_queue().get_all_completed_papers()

def completed_papers_count() -> int:
    return _get_default_queue().completed_papers_count()

def export_completed_papers_to_file(filepath: str = "completed_papers.txt") -> int:
    return _get_default_queue().export_completed_papers_to_file(filepath)

def enqueue_paper_id_human(paper_id: str) -> bool:
    return _get_default_queue().enqueue_paper_id_human(paper_id)

def claim_next_paper_human(block_timeout: int = 0) -> str | None:
    return _get_default_queue().claim_next_paper_human(block_timeout)

def ack_paper_human(paper_id: str) -> None:
    return _get_default_queue().ack_paper_human(paper_id)

def requeue_inflight_human(paper_id: str) -> None:
    return _get_default_queue().requeue_inflight_human(paper_id)

def human_paper_queue_len() -> int:
    return _get_default_queue().human_paper_queue_len()

def push_completed_annotation(record: dict) -> None:
    return _get_default_queue().push_completed_annotation(record)

def annotation_queue_len() -> int:
    return _get_default_queue().annotation_queue_len()

def flush_annotations_to_persistent(max_items: Optional[int] = None) -> int:
    return _get_default_queue().flush_annotations_to_persistent(max_items)

def shutdown_annotations() -> None:
    return _get_default_queue().shutdown_annotations()
