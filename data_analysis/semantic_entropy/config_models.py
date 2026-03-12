"""
Pydantic config models for generate_samples.py.
Combines LLM provider config with Minio settings.
"""

from typing import Dict

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


class GenerateSamplesConfig(LLMProvidersDictMixin, BaseModel):
    """
    Combined config for generate_samples.py.
    Use with load_config_from_yaml_file(GenerateSamplesConfig, path).
    """

    minio: MinioConfig
    llm_providers: Dict[str, LLMProviderConfig]

    def _get_providers_config(self) -> Dict[str, LLMProviderConfig]:
        return self.llm_providers
