"""
Seed the paper queue from a file of paper IDs.
Uses config manager for env vars (validates at startup).
"""
import redis

from config import load_config
from config.seed_configs import SeedQueueConfig
from utilities.queue_helpers import enqueue_paper_id

# Load config at startup; validates required env vars
config = load_config(SeedQueueConfig)
print(f"Redis URL: {config.redis_url}")
r = redis.Redis.from_url(config.redis_url, decode_responses=True)


def seed_queue_from_file(file_path: str) -> None:
    """Read paper IDs from file and enqueue for processing."""
    with open(file_path, "r") as f:
        for line in f:
            paper_id = line.strip()
            if paper_id:
                success = enqueue_paper_id(paper_id)
                if success:
                    print(f"Enqueued: {paper_id}")
                else:
                    print(f"Duplicate or failed to enqueue: {paper_id}")


if __name__ == "__main__":
    seed_queue_from_file(config.file_path)
