"""
Config schema for the label API service.
Loads MinIO, Label Studio, and webhook-related env vars.
"""

from dataclasses import dataclass
from typing import Annotated

from config.tags import Default, Env


@dataclass
class LabelApiConfig:
    """Config for the label API (legacy_main)."""

    # MinIO
    minio_endpoint: Annotated[str, Env("MINIO_ENDPOINT"), Default("localhost:9000")]
    minio_access_key: Annotated[str, Env("MINIO_ACCESS_KEY"), Default("minioadmin")]
    minio_secret_key: Annotated[str, Env("MINIO_SECRET_KEY"), Default("minioadmin")]
    minio_secure: Annotated[bool, Env("MINIO_SECURE"), Default(False)]
    minio_bucket: Annotated[str, Env("MINIO_BUCKET"), Default("v4-criteria-classified-articles")]

    # Label Studio (required by LabellerSDK - no defaults for prod)
    label_studio_url: Annotated[str, Env("LABEL_STUDIO_URL"), Default("")]
    label_studio_api_key: Annotated[str, Env("LABEL_STUDIO_API_KEY"), Default("")]

    # Webhook & buckets
    webhook_host: Annotated[str, Env("WEBHOOK_HOST"), Default("http://localhost:8000")]
    annotations_bucket: Annotated[str, Env("ANNOTATIONS_BUCKET"), Default("completed-annotations")]
