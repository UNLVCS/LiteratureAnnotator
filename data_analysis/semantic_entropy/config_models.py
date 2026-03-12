"""
Pydantic config models for generate_samples.py.
Combines LLM provider config with Minio, Pinecone, and Redis settings.
"""

from typing import Dict, Optional

from pydantic import BaseModel

from llm_providers.config_models import (
    LLMProviderConfig,
    LLMProvidersDictMixin,
)


class MinioConfig(BaseModel):
    """Minio object storage configuration."""

    url: str
    access_key: str
    secret_key: str
    bucket_name: str
    secure: bool = False


class PineconeConfig(BaseModel):
    """Pinecone vector store configuration."""

    api_key: str
    index_name: str = "adbm"
    namespace: str = "article_upload_test_2"


class RedisConfig(BaseModel):
    """Redis configuration for paper queues."""

    url: str = "redis://localhost:6379/0"
    paper_queue: str = "q:papers:v1"
    paper_processing: str = "q:papers:processing:v1"
    paper_dedup_set: str = "s:papers:enqueued:v1"
    generated_set: str = "s:papers:generated:v1"


class EmbeddingsConfig(BaseModel):
    """OpenAI embeddings configuration."""

    api_key: str
    model: str = "text-embedding-ada-002"


class GenerateSamplesConfig(LLMProvidersDictMixin, BaseModel):
    """
    Combined config for generate_samples.py.
    Use with load_config_from_yaml_file(GenerateSamplesConfig, path).
    """

    minio: MinioConfig
    pinecone: PineconeConfig
    redis: RedisConfig
    embeddings: EmbeddingsConfig
    llm_providers: Dict[str, LLMProviderConfig]

    def _get_providers_config(self) -> Dict[str, LLMProviderConfig]:
        return self.llm_providers
