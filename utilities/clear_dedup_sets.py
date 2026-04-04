"""
Delete Redis deduplication SET keys so paper IDs can be enqueued again.

Uses ``.env.yaml`` (``load_app_config``) for ``redis.url`` and key names.

Run from project root: ``python -m utilities.clear_dedup_sets``
"""

from __future__ import annotations

import redis

from config.app_config import load_app_config


def main() -> None:
    cfg = load_app_config()
    r = redis.Redis.from_url(cfg.redis.url, decode_responses=True)
    keys = (cfg.redis.paper_dedup_set, cfg.redis.human_dedup_set)
    for k in keys:
        deleted = r.delete(k)
        status = "cleared" if deleted else "was absent or already empty"
        print(f"{k}: {status}")


if __name__ == "__main__":
    main()
