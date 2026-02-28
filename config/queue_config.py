"""
Queue-related config schema.
Used by utilities.queue_helpers and seed scripts.
"""

from dataclasses import dataclass
from typing import Annotated

from config.tags import Default, Env


@dataclass
class QueueConfig:
    """Config for Redis queues and annotation persistence."""

    redis_url: Annotated[str, Env("REDIS_URL"), Default("redis://localhost:6379/0")]
    paper_queue: Annotated[str, Env("PAPER_QUEUE"), Default("q:papers:v1")]
    paper_processing: Annotated[str, Env("PAPER_PROCESSING"), Default("q:papers:processing:v1")]
    paper_dedup_set: Annotated[str, Env("PAPER_DEDUP_SET"), Default("s:papers:enqueued:v1")]
    ann_queue: Annotated[str, Env("ANN_QUEUE"), Default("q:annotations:completed:v1")]
    completed_papers_queue: Annotated[str, Env("COMPLETED_PAPERS_QUEUE"), Default("q:papers:completed:v1")]
    generated_set: Annotated[str, Env("GENERATED_SET"), Default("s:papers:generated:v1")]
    ann_flush_threshold: Annotated[int, Env("ANN_FLUSH_THRESHOLD"), Default(1000)]
    ann_persist_path: Annotated[str, Env("ANN_PERSIST_PATH"), Default("data_labeling/annotations.jsonl")]
    ann_flush_on_exit: Annotated[bool, Env("ANN_FLUSH_ON_EXIT"), Default(True)]
    ann_install_signal_handlers: Annotated[bool, Env("ANN_INSTALL_SIGNAL_HANDLERS"), Default(True)]
