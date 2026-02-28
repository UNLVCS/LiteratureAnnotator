"""
Config schemas for seed scripts.
Each seed script (seed_queue, seed_human_queue) loads only the vars it needs.
"""

from dataclasses import dataclass
from typing import Annotated

from config.tags import Default, Env


@dataclass
class SeedQueueConfig:
    """Config for seed_queue script."""

    redis_url: Annotated[str, Env("REDIS_URL"), Default("redis://localhost:6379/0")]
    file_path: Annotated[str, Env("SEED_QUEUE_FILE"), Default("test_papers.txt")]

@dataclass
class SeedHumanQueueConfig:
    """Config for seed_human_queue script."""

    redis_url: Annotated[str, Env("REDIS_URL"), Default("redis://localhost:6379/0")]
    human_papers_file: Annotated[str, Env("HUMAN_PAPERS_FILE"), Default("human_papers.txt")]
