"""
Queue-related config schema.
Used by utilities.queue_helpers and seed scripts.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class QueueConfig(BaseSettings):
    """Config for Redis queues and annotation persistence."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_url: str = Field(default="redis://localhost:6379/0", validation_alias="REDIS_URL")
    paper_queue: str = Field(default="q:papers:v1", validation_alias="PAPER_QUEUE")
    paper_processing: str = Field(
        default="q:papers:processing:v1",
        validation_alias="PAPER_PROCESSING",
    )
    paper_dedup_set: str = Field(
        default="s:papers:enqueued:v1",
        validation_alias="PAPER_DEDUP_SET",
    )
    ann_queue: str = Field(
        default="q:annotations:completed:v1",
        validation_alias="ANN_QUEUE",
    )
    completed_papers_queue: str = Field(
        default="q:papers:completed:v1",
        validation_alias="COMPLETED_PAPERS_QUEUE",
    )
    generated_set: str = Field(default="s:papers:generated:v1", validation_alias="GENERATED_SET")
    ann_flush_threshold: int = Field(default=1000, validation_alias="ANN_FLUSH_THRESHOLD")
    ann_persist_path: str = Field(
        default="data_labeling/annotations.jsonl",
        validation_alias="ANN_PERSIST_PATH",
    )
    ann_flush_on_exit: bool = Field(default=True, validation_alias="ANN_FLUSH_ON_EXIT")
    ann_install_signal_handlers: bool = Field(
        default=True,
        validation_alias="ANN_INSTALL_SIGNAL_HANDLERS",
    )
    HUMAN_PAPER_QUEUE: str = Field(
        default="q:papers:human:v1",
        validation_alias="HUMAN_PAPER_QUEUE",
    )
    HUMAN_PROCESSING_Q: str = Field(
        default="q:papers:human:processing:v1",
        validation_alias="HUMAN_PROCESSING_Q",
    )
    HUMAN_DEDUP_SET: str = Field(
        default="s:papers:human:enqueued:v1",
        validation_alias="HUMAN_DEDUP_SET",
    )
