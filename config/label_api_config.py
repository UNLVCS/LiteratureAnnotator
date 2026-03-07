"""
Config schema for the label API service.
Loads MinIO, Label Studio, and webhook-related env vars.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LabelApiConfig(BaseSettings):
    """Config for the label API (legacy_main)."""


    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # MinIO
    minio_endpoint: str = Field(default="localhost:9000", validation_alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", validation_alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", validation_alias="MINIO_SECRET_KEY")
    minio_secure: bool = Field(default=False, validation_alias="MINIO_SECURE")
    minio_bucket: str = Field(
        default="v4-criteria-classified-articles",
        validation_alias="MINIO_BUCKET_NAME",
    )

    # Label Studio (required by LabellerSDK - no defaults for prod)
    label_studio_url: str = Field(..., validation_alias="LABEL_STUDIO_URL")
    label_studio_api_key: str = Field(..., validation_alias="LABEL_STUDIO_API_KEY")

    # Webhook & buckets
    webhook_host: str = Field(default="http://localhost:8000", validation_alias="WEBHOOK_HOST")
    annotations_bucket: str = Field(
        default="completed-annotations",
        validation_alias="ANNOTATIONS_BUCKET",
    )
    human_annotations_bucket: str = Field(
        default="human-annotations",
        validation_alias="HUMAN_ANNOTATIONS_BUCKET",
    )
