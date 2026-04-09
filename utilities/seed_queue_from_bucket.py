"""
Seed a labeling queue from the object names in a MinIO bucket.

Paper IDs are derived from the filename stem of each object
(e.g. ``raw-pubmed-articles/38291045.json`` → ``38291045``).
Duplicate IDs are silently skipped — the queue's dedup set prevents
double-enqueuing the same paper regardless of how many times this
script is run.

When ``--bucket`` and ``--prefix`` are omitted the script falls back to
``minio.raw_articles_bucket`` and ``bioc_download.object_prefix`` from
``.env.yaml``, so running it right after a download job requires no extra flags.

Usage (from workspace root):

    # Use bucket + prefix from .env.yaml bioc_download section:
    uv run python utilities/seed_queue_from_bucket.py

    # Override the bucket explicitly:
    uv run python utilities/seed_queue_from_bucket.py --bucket raw-pubmed-articles

    # Enqueue into the human-labeling queue:
    uv run python utilities/seed_queue_from_bucket.py --queue human

    # Limit to objects under a specific path prefix:
    uv run python utilities/seed_queue_from_bucket.py --prefix 2024/

    # Dry-run: list what would be enqueued without touching Redis:
    uv run python utilities/seed_queue_from_bucket.py --dry-run
"""

import argparse
import sys
from minio import Minio

from config.app_config import load_app_config
from utilities.queue_helpers import PaperQueue


def _extract_paper_id(object_name: str) -> str:
    """Return the filename stem of an object path as the paper ID.

    ``some/prefix/38291045.json`` → ``38291045``
    """
    return object_name.split("/")[-1].split(".")[0]


def seed_from_bucket(
    bucket: str,
    queue_target: str = "labeler",
    prefix: str = "",
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    List every object in *bucket* (optionally filtered by *prefix*) and
    enqueue its paper ID into the chosen queue.

    Args:
        bucket:       MinIO bucket name.
        queue_target: ``"labeler"`` (default) or ``"human"``.
        prefix:       Optional object-name prefix filter.
        dry_run:      If True, print what would be enqueued but don't write
                      anything to Redis.

    Returns:
        ``(enqueued, skipped)`` counts.
    """
    app_config = load_app_config()

    minio_client = Minio(
        app_config.minio.url,
        access_key=app_config.minio.access_key,
        secret_key=app_config.minio.secret_key,
        secure=app_config.minio.secure,
    )

    if not dry_run:
        queue = PaperQueue.from_app_config(app_config)

    enqueued = 0
    skipped = 0

    print(f"Scanning bucket: {bucket}" + (f"  prefix: {prefix!r}" if prefix else ""))
    print(f"Target queue:    {queue_target}")
    if dry_run:
        print("DRY RUN — Redis will not be modified.\n")

    objects = list(minio_client.list_objects(bucket, prefix=prefix or None, recursive=True))

    if not objects:
        print("No objects found.")
        return 0, 0

    for obj in objects:
        paper_id = _extract_paper_id(obj.object_name)
        if not paper_id:
            continue

        if dry_run:
            print(f"  would enqueue: {paper_id}  (from {obj.object_name})")
            enqueued += 1
            continue

        if queue_target == "human":
            added = queue.enqueue_paper_id_human(paper_id)
        else:
            added = queue.enqueue_paper_id(paper_id)

        if added:
            enqueued += 1
            print(f"  enqueued: {paper_id}")
        else:
            skipped += 1
            print(f"  skipped (already in queue): {paper_id}")

    print(f"\nDone. enqueued={enqueued}  skipped={skipped}")
    return enqueued, skipped


def _parse_args(argv=None) -> argparse.Namespace:
    # Load config first so we can use config values as defaults.
    app_config = load_app_config()
    default_bucket = app_config.minio.raw_articles_bucket
    default_prefix = app_config.bioc_download.object_prefix or ""

    parser = argparse.ArgumentParser(
        description=(
            "Seed a labeling queue from the contents of a MinIO bucket. "
            "Defaults to minio.raw_articles_bucket / bioc_download.object_prefix from .env.yaml."
        ),
    )
    parser.add_argument(
        "--bucket",
        default=default_bucket,
        help=(
            f"MinIO bucket to read paper IDs from "
            f"(default from .env.yaml minio: {default_bucket!r})."
        ),
    )
    parser.add_argument(
        "--queue",
        choices=["labeler", "human"],
        default="labeler",
        help="Which queue to write to (default: labeler).",
    )
    parser.add_argument(
        "--prefix",
        default=default_prefix,
        help=(
            f"Only consider objects whose names start with this prefix "
            f"(default from .env.yaml bioc_download: {default_prefix!r})."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be enqueued without touching Redis.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    enqueued, skipped = seed_from_bucket(
        bucket=args.bucket,
        queue_target=args.queue,
        prefix=args.prefix,
        dry_run=args.dry_run,
    )
    sys.exit(0 if enqueued > 0 or skipped >= 0 else 1)
