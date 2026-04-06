"""
Config schema for the BioC article downloader.

Loads MinIO storage, NCBI E-utilities, and MeSH query settings
from environment variables (or a .env file).

Required env vars (no defaults):
    MINIO_URL, MINIO_ACCESS_KEY, MINIO_SECRET_KEY

Optional env vars (sensible defaults provided):
    MINIO_SECURE, BIOC_DOWNLOAD_BUCKET,
    MESH_IDS, MESH_TERMS, MESH_QUERY, MAJOR_TOPIC_ONLY,
    NCBI_API_KEY, NCBI_EMAIL,
    BIOC_MAX_RESULTS, BIOC_BATCH_SIZE, BIOC_REQUEST_DELAY

MeSH query construction
-----------------------
You can specify the PubMed query in three ways (highest to lowest priority):

1. ``MESH_QUERY``  – A raw PubMed query string; passed verbatim to esearch.
   Best when you need full control (OR groups, subheadings, etc.).
   Example::

       MESH_QUERY="alzheimer disease/blood"[MAJR] AND ("amyloid beta-peptides/blood"[MAJR] OR "tau proteins/blood"[MAJR])

2. ``MESH_TERMS``  – Comma-separated human-readable MeSH term names,
   optionally with ``/subheading`` qualifiers.  Joined with AND.
   Example::

       MESH_TERMS=alzheimer disease/blood,biomarkers/blood

3. ``MESH_IDS``    – Comma-separated MeSH descriptor UIDs (e.g. D000544).
   Useful if you already know the IDs.

When ``MAJOR_TOPIC_ONLY=true`` (the default), terms built from
``MESH_TERMS`` / ``MESH_IDS`` use ``[MAJR]`` instead of ``[MeSH Terms]``
so only papers where the topic is a *major* focus are returned.
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BioCDownloadConfig(BaseSettings):
    """Config for downloading PubMed articles via the BioC API."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ------------------------------------------------------------------
    # MinIO
    # ------------------------------------------------------------------
    minio_url: str = Field(..., validation_alias="MINIO_URL")
    minio_access_key: str = Field(..., validation_alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(..., validation_alias="MINIO_SECRET_KEY")
    minio_secure: bool = Field(default=False, validation_alias="MINIO_SECURE")

    download_bucket: str = Field(
        default="raw-pubmed-articles",
        validation_alias="BIOC_DOWNLOAD_BUCKET",
    )

    # ------------------------------------------------------------------
    # NCBI / E-utilities
    # ------------------------------------------------------------------
    ncbi_api_key: Optional[str] = Field(
        default=None,
        validation_alias="NCBI_API_KEY",
        description="Optional NCBI API key for higher rate limits (10 req/s vs 3 req/s).",
    )
    ncbi_email: Optional[str] = Field(
        default=None,
        validation_alias="NCBI_EMAIL",
        description="Email for NCBI E-utilities (recommended by NCBI usage policy).",
    )

    # ------------------------------------------------------------------
    # MeSH query
    # ------------------------------------------------------------------
    mesh_query: Optional[str] = Field(
        default=None,
        validation_alias="MESH_QUERY",
        description=(
            "Raw PubMed query string passed verbatim to esearch. "
            "Takes priority over mesh_terms / mesh_ids when set."
        ),
    )
    mesh_ids: list[str] = Field(
        default_factory=list,
        validation_alias="MESH_IDS",
        description=(
            "Comma-separated MeSH descriptor UIDs to search PubMed with. "
            "Example: D000544,D015415 (Alzheimer Disease, Biomarkers)."
        ),
    )
    mesh_terms: list[str] = Field(
        default_factory=list,
        validation_alias="MESH_TERMS",
        description=(
            "Comma-separated MeSH term names, optionally with /subheading qualifiers. "
            "Example: alzheimer disease/blood,biomarkers/blood"
        ),
    )
    major_topic_only: bool = Field(
        default=True,
        validation_alias="MAJOR_TOPIC_ONLY",
        description=(
            "When True, auto-built queries use [MAJR] (Major MeSH Heading) "
            "instead of [MeSH Terms], restricting to papers where the topic "
            "is a central focus. Ignored when mesh_query is set."
        ),
    )

    # ------------------------------------------------------------------
    # Download parameters
    # ------------------------------------------------------------------
    max_results: int = Field(
        default=100,
        validation_alias="BIOC_MAX_RESULTS",
        description="Maximum number of PMIDs to retrieve from the E-utilities search.",
    )
    batch_size: int = Field(
        default=25,
        validation_alias="BIOC_BATCH_SIZE",
        description="Number of articles to fetch per BioC API request batch.",
    )
    request_delay: float = Field(
        default=0.34,
        validation_alias="BIOC_REQUEST_DELAY",
        description="Seconds to wait between BioC API requests (NCBI rate limit).",
    )
