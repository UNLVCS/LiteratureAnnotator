"""
Seed the paper queue from a file of paper IDs.
Uses config manager for .env.yaml (validates at startup).
run from project root with python -m utilities.seed_queue
"""

from config.app_config import load_app_config
from utilities.queue_helpers import enqueue_paper_id

# Load config at startup from .env.yaml
config = load_app_config()
print(f"Redis URL: {config.redis.url}")


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
    seed_queue_from_file(config.seed.queue_file)
