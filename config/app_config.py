"""
Unified application configuration loaded from .env.yaml.

Optional: override.env.yaml merges over base (e.g. Docker mounts .env.docker-override.yaml.example there).

Usage:
    from config.app_config import load_app_config, AppConfig
    config = load_app_config()  # loads from .env.yaml
    config.redis.url
    config.minio.access_key
    config.llm_providers  # Dict[str, LLMProviderConfig]
"""

from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field

from config.llm_providers_config import LLMProviderConfig, LLMProvidersDictMixin


class RedisConfig(BaseModel):
    """Redis connection and queue configuration."""

    url: str = "redis://localhost:6379/0"
    paper_queue: str = "q:papers:v1"
    paper_processing: str = "q:papers:processing:v1"
    paper_dedup_set: str = "s:papers:enqueued:v1"
    completed_papers_queue: str = "q:papers:completed:v1"
    generated_set: str = "s:papers:generated:v1"
    ann_queue: str = "q:annotations:completed:v1"
    ann_flush_threshold: int = 1000
    ann_persist_path: str = "data_labeling/annotations.jsonl"
    ann_flush_on_exit: bool = True
    ann_install_signal_handlers: bool = True
    human_paper_queue: str = "q:papers:human:v1"
    human_processing_queue: str = "q:papers:human:processing:v1"
    human_dedup_set: str = "s:papers:human:enqueued:v1"


class MinioConfig(BaseModel):
    """MinIO object storage configuration."""

    url: str = "localhost:9000"
    access_key: str = ""
    secret_key: str = ""
    bucket_name: str = "v1-criteria-classified-articles"
    raw_articles_bucket: str = "raw-pubmed-articles"
    secure: bool = False
    human_annotations_bucket: str = "human-annotations"
    annotations_bucket: str = "completed-annotations"


class PineconeConfig(BaseModel):
    """Pinecone vector store configuration."""

    api_key: str = ""
    index_name: str = "adbm"
    namespace: str = "article_upload_test_2"


class LabelStudioConfig(BaseModel):
    """Label Studio configuration."""

    url: str = ""
    api_key: str = ""
    webhook_host: str = "http://localhost:8000"

    @property
    def label_studio_url(self) -> str:
        """Alias for compatibility with LabellerSDK protocol."""
        return self.url

    @property
    def label_studio_api_key(self) -> str:
        """Alias for compatibility with LabellerSDK protocol."""
        return self.api_key


class EmbeddingsConfig(BaseModel):
    """OpenAI embeddings configuration."""

    api_key: str = ""
    model: str = "text-embedding-ada-002"


class SeedConfig(BaseModel):
    """Seed script configuration."""

    queue_file: str = "utilities/test_papers.txt"
    human_papers_file: str = "utilities/human_papers.txt"


class AppConfig(LLMProvidersDictMixin, BaseModel):
    """
    Unified application configuration.
    
    Load with: load_app_config() or load_config_from_yaml_file(AppConfig, path)
    """

    redis: RedisConfig = Field(default_factory=RedisConfig)
    minio: MinioConfig = Field(default_factory=MinioConfig)
    pinecone: PineconeConfig = Field(default_factory=PineconeConfig)
    label_studio: LabelStudioConfig = Field(default_factory=LabelStudioConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    seed: SeedConfig = Field(default_factory=SeedConfig)
    llm_providers: Dict[str, LLMProviderConfig] = Field(default_factory=dict)

    def _get_providers_config(self) -> Dict[str, LLMProviderConfig]:
        return self.llm_providers


# Cached config instance
_app_config: Optional[AppConfig] = None


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override dict into base. Override values take precedence."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_app_config(config_path: Optional[Path] = None, reload: bool = False) -> AppConfig:
    """
    Load application config from .env.yaml, merging override.env.yaml if present.
    
    Args:
        config_path: Optional path to base config. Defaults to .env.yaml in project root.
        reload: Force reload even if already cached.
    
    Returns:
        AppConfig instance
    """
    global _app_config
    
    if _app_config is not None and not reload:
        return _app_config
    
    if config_path is None:
        config_path = Path(__file__).parent.parent / ".env.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Copy .env.yaml.example to .env.yaml and fill in your values."
        )
    
    import yaml
    from config.base import load_config_from_yaml_file
    
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    
    override_path = config_path.parent / "override.env.yaml"
    if override_path.exists():
        with open(override_path, "r", encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        data = _deep_merge(data, override)
    
    _app_config = AppConfig.model_validate(data)
    return _app_config


def get_app_config() -> AppConfig:
    """Get the cached config, loading if necessary."""
    return load_app_config()
