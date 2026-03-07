"""
Config schemas for seed scripts.
Each seed script (seed_queue, seed_human_queue) loads only the vars it needs.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SeedQueueConfig(BaseSettings):
    """Config for seed_queue script."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_url: str = Field(default="redis://localhost:6379/0", validation_alias="REDIS_URL")
    file_path: str = Field(
        default="utilities/test_papers.txt",
        validation_alias="SEED_QUEUE_FILE",
    )


class SeedHumanQueueConfig(BaseSettings):
    """Config for seed_human_queue script."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_url: str = Field(default="redis://localhost:6379/0", validation_alias="REDIS_URL")
    human_papers_file: str = Field(
        default="utilities/human_papers.txt",
        validation_alias="HUMAN_PAPERS_FILE",
    )
