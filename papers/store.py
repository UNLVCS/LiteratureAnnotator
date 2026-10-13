"""Load raw BioC articles from MinIO (written by data_download) as Paper objects."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Iterable, Optional

from papers.bioc import DEFAULT_EXCLUDED_SECTION_TYPES, Paper, parse_bioc

if TYPE_CHECKING:
    from minio import Minio


class PaperNotFoundError(LookupError):
    """No raw article exists for this ID under the configured bucket/prefix."""


def raw_object_name(pmid: str, prefix: str = "") -> str:
    """Object key used by ``data_download.bioc_downloader.upload_article``."""
    folder = prefix.strip("/")
    return f"{folder}/{pmid}.json" if folder else f"{pmid}.json"


def load_paper(
    client: "Minio",
    bucket: str,
    pmid: str,
    prefix: str = "",
    exclude_section_types: Iterable[str] = DEFAULT_EXCLUDED_SECTION_TYPES,
) -> Paper:
    """Fetch ``{prefix}/{pmid}.json`` from MinIO and parse it.

    Raises:
        PaperNotFoundError: the object does not exist (paper was never downloaded).
    """
    from minio.error import S3Error

    object_name = raw_object_name(pmid, prefix)
    try:
        response = client.get_object(bucket_name=bucket, object_name=object_name)
    except S3Error as exc:
        if exc.code in ("NoSuchKey", "NoSuchObject"):
            raise PaperNotFoundError(f"{bucket}/{object_name}") from exc
        raise
    try:
        data = json.loads(response.read().decode("utf-8"))
    finally:
        response.close()
        response.release_conn()
    return parse_bioc(pmid, data, exclude_section_types)


def load_paper_from_config(pmid: str, app_config, client: Optional["Minio"] = None) -> Paper:
    """Convenience wrapper reading bucket, prefix and exclusions from AppConfig."""
    if client is None:
        from minio import Minio

        client = Minio(
            app_config.minio.url,
            access_key=app_config.minio.access_key,
            secret_key=app_config.minio.secret_key,
            secure=app_config.minio.secure,
        )
    prefix = app_config.papers.raw_prefix
    if prefix is None:
        prefix = app_config.bioc_download.object_prefix or ""
    return load_paper(
        client,
        app_config.minio.raw_articles_bucket,
        pmid,
        prefix=prefix,
        exclude_section_types=app_config.papers.exclude_section_types,
    )
