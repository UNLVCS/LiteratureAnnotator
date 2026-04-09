"""
Queue status dashboard for the LiteratureAnnotator paper processing pipelines.

Usage:
    python -m utilities.queue_status           # summary
    python -m utilities.queue_status --ids     # also print paper IDs in each queue
    python -m utilities.queue_status --fix     # move stuck in-flight papers back to pending
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.app_config import load_app_config
from utilities.queue_helpers import PaperQueue, PaperQueueConfig


def _bar(n: int, total: int, width: int = 20) -> str:
    if total == 0:
        filled = 0
    else:
        filled = round(n / total * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def print_status(queue: PaperQueue, show_ids: bool = False) -> None:
    r = queue.redis
    cfg = queue.config

    pending       = r.llen(cfg.paper_queue)
    in_flight     = r.llen(cfg.paper_processing)
    dedup_size    = r.scard(cfg.paper_dedup_set)
    generated     = r.scard(cfg.generated_set)
    completed     = r.llen(cfg.completed_papers_queue)
    human_pending = r.llen(cfg.HUMAN_PAPER_QUEUE)
    human_flight  = r.llen(cfg.HUMAN_PROCESSING_Q)
    human_dedup   = r.scard(cfg.HUMAN_DEDUP_SET)
    ann_queue     = r.llen(cfg.ann_queue)

    total_rag = pending + in_flight
    total_human = human_pending + human_flight

    print()
    print("=" * 56)
    print("  LiteratureAnnotator — Queue Status")
    print("=" * 56)

    print("\n── RAG Labeling Pipeline ──────────────────────────────")
    print(f"  Pending       {pending:>6}  {_bar(pending, max(total_rag, 1))}")
    print(f"  In-flight     {in_flight:>6}  {_bar(in_flight, max(total_rag, 1))}")
    print(f"  Dedup set     {dedup_size:>6}  (papers ever enqueued & not yet acked)")
    print(f"  Generated set {generated:>6}  (completed successfully)")
    print(f"  Completed log {completed:>6}  (legacy completed-papers list)")

    print("\n── Human Labeling Pipeline ────────────────────────────")
    print(f"  Pending       {human_pending:>6}  {_bar(human_pending, max(total_human, 1))}")
    print(f"  In-flight     {human_flight:>6}  {_bar(human_flight, max(total_human, 1))}")
    print(f"  Dedup set     {human_dedup:>6}")

    print("\n── Annotations ────────────────────────────────────────")
    print(f"  Buffered      {ann_queue:>6}  (pending flush to disk)")

    print("\n── Redis connection ───────────────────────────────────")
    print(f"  URL           {cfg.redis_url}")
    try:
        info = r.info("server")
        print(f"  Version       {info.get('redis_version', '?')}")
        print(f"  Uptime        {info.get('uptime_in_days', '?')}d")
    except Exception:
        print("  (could not fetch server info)")

    if show_ids:
        _print_ids(r, "RAG pending",    cfg.paper_queue,       "LRANGE")
        _print_ids(r, "RAG in-flight",  cfg.paper_processing,  "LRANGE")
        _print_ids(r, "Human pending",  cfg.HUMAN_PAPER_QUEUE, "LRANGE")
        _print_ids(r, "Human in-flight",cfg.HUMAN_PROCESSING_Q,"LRANGE")
        _print_ids(r, "Generated set",  cfg.generated_set,     "SMEMBERS")

    print()


def _print_ids(r, label: str, key: str, method: str) -> None:
    if method == "LRANGE":
        ids = r.lrange(key, 0, -1)
    else:
        ids = list(r.smembers(key))
    if not ids:
        return
    print(f"\n  {label} ({len(ids)}):")
    for pid in ids:
        print(f"    {pid}")


def fix_inflight(queue: PaperQueue) -> None:
    """Move all stuck in-flight papers back to the pending queue."""
    r = queue.redis
    cfg = queue.config
    moved = 0
    while True:
        pid = r.rpoplpush(cfg.paper_processing, cfg.paper_queue)
        if pid is None:
            break
        print(f"  Requeued: {pid}")
        moved += 1
    if moved:
        print(f"\n  ✓ Moved {moved} paper(s) from in-flight → pending.")
    else:
        print("  Nothing stuck in in-flight queue.")


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------

_CLEAR_TARGETS = {
    "labeler": "RAG labeler queue (pending, in-flight, dedup set)",
    "human":   "human labeling queue (pending, in-flight, dedup set)",
    "generated": "generated/completed set",
    "annotations": "annotations buffer",
    "all":     "ALL queues and sets listed above",
}


def _clear_summary(queue: PaperQueue, target: str) -> dict[str, int]:
    """Return key→count for every Redis key that *target* would delete."""
    r   = queue.redis
    cfg = queue.config
    counts: dict[str, int] = {}

    def _add(key: str, kind: str) -> None:
        n = r.llen(key) if kind == "list" else r.scard(key)
        if n:
            counts[key] = n

    if target in ("labeler", "all"):
        _add(cfg.paper_queue,       "list")
        _add(cfg.paper_processing,  "list")
        _add(cfg.paper_dedup_set,   "set")

    if target in ("human", "all"):
        _add(cfg.HUMAN_PAPER_QUEUE, "list")
        _add(cfg.HUMAN_PROCESSING_Q,"list")
        _add(cfg.HUMAN_DEDUP_SET,   "set")

    if target in ("generated", "all"):
        _add(cfg.generated_set,          "set")
        _add(cfg.completed_papers_queue, "list")

    if target in ("annotations", "all"):
        _add(cfg.ann_queue, "list")

    return counts


def clear_queues(queue: PaperQueue, target: str, confirmed: bool = False) -> None:
    """Delete the Redis keys belonging to *target*.

    Prints a summary and prompts for confirmation unless *confirmed* is True
    (i.e. ``--yes`` was passed on the CLI).
    """
    counts = _clear_summary(queue, target)

    if not counts:
        print(f"  Nothing to clear for target '{target}'.")
        return

    print(f"\n  The following Redis keys will be permanently deleted:\n")
    total = 0
    for key, n in counts.items():
        print(f"    {key}  ({n} item{'s' if n != 1 else ''})")
        total += n
    print(f"\n  Total: {total} item{'s' if total != 1 else ''} across {len(counts)} key{'s' if len(counts) != 1 else ''}.")

    if not confirmed:
        try:
            answer = input("\n  Type 'yes' to confirm: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Aborted.")
            return
        if answer != "yes":
            print("  Aborted.")
            return

    r   = queue.redis
    cfg = queue.config

    if target in ("labeler", "all"):
        r.delete(cfg.paper_queue, cfg.paper_processing, cfg.paper_dedup_set)

    if target in ("human", "all"):
        r.delete(cfg.HUMAN_PAPER_QUEUE, cfg.HUMAN_PROCESSING_Q, cfg.HUMAN_DEDUP_SET)

    if target in ("generated", "all"):
        r.delete(cfg.generated_set, cfg.completed_papers_queue)

    if target in ("annotations", "all"):
        r.delete(cfg.ann_queue)

    print(f"\n  ✓ Cleared '{target}'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show LiteratureAnnotator queue status."
    )
    parser.add_argument(
        "--ids", action="store_true",
        help="Print the actual paper IDs in each queue."
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="Move stuck in-flight papers back to the pending queue."
    )
    parser.add_argument(
        "--clear",
        choices=list(_CLEAR_TARGETS),
        metavar="TARGET",
        help=(
            "Delete all Redis keys for the given target. "
            f"Choices: {', '.join(_CLEAR_TARGETS)}. "
            "Prompts for confirmation unless --yes is also passed."
        ),
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the confirmation prompt when used with --clear."
    )
    args = parser.parse_args()

    config = load_app_config()
    queue  = PaperQueue.from_app_config(config)

    print_status(queue, show_ids=args.ids)

    if args.fix:
        print("── Fixing stuck in-flight papers ──────────────────────")
        fix_inflight(queue)
        print()
        print("Updated status:")
        print_status(queue, show_ids=args.ids)

    if args.clear:
        print(f"── Clearing: {_CLEAR_TARGETS[args.clear]} {'─' * 10}")
        clear_queues(queue, args.clear, confirmed=args.yes)
        print()
        print("Updated status:")
        print_status(queue)


if __name__ == "__main__":
    main()
