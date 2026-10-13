"""
Human Labeling Project: task import.
Loads the full paper from MinIO and creates a Label Studio task in which the labeler
decides every criterion directly, without seeing any model output.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from label_api.services.human_task_payload import build_human_criteria_html
from papers import PaperNotFoundError, load_paper_from_config
from utilities.criteria import CRITERIA_INSTRUCTIONS, CRITERION_NAMES

if TYPE_CHECKING:
    from label_api.clients.label_studio.human import HumanLabellerSDK


def import_next_human_tasks(ls_human: "HumanLabellerSDK") -> None:
    from label_api.services.context import app_config, client
    from utilities.queue_helpers import (
        ack_paper_human,
        claim_next_paper_human,
        requeue_inflight_human,
    )

    paper_id = claim_next_paper_human()
    if not paper_id:
        print("[Human] No paper ID found in queue")
        return

    try:
        try:
            paper = load_paper_from_config(paper_id, app_config, client=client)
        except PaperNotFoundError as e:
            print(f"[Human] Raw article not found for paper {paper_id} ({e}), skipping")
            ack_paper_human(paper_id)
            return

        if not paper.passages:
            print(f"[Human] Paper {paper_id} has no text after parsing, skipping")
            ack_paper_human(paper_id)
            return

        criteria_list = [
            {"criterion": name, "class_criteria": instructions}
            for name, instructions in zip(CRITERION_NAMES, CRITERIA_INSTRUCTIONS)
        ]
        criteria_html, criterion_names = build_human_criteria_html(criteria_list, paper)

        task_data = {
            "paper_id": paper_id,
            "title": paper.title,
            "criteria_html": criteria_html,
            "criteria_json": json.dumps(criteria_list),
            "criterion_names": json.dumps(criterion_names),
        }
        ls_human.import_tasks([{"data": task_data}])

        ack_paper_human(paper_id)
        print(f"[Human] Imported task for paper {paper_id}")

    except Exception as e:
        print(f"[Human] Error importing paper {paper_id}: {e}")
        requeue_inflight_human(paper_id)
        raise


def import_all_pending_human_tasks(ls_human: "HumanLabellerSDK", max_rounds: int = 500) -> int:
    from utilities.queue_helpers import human_paper_queue_len

    rounds = 0
    while human_paper_queue_len() > 0 and rounds < max_rounds:
        import_next_human_tasks(ls_human)
        rounds += 1
    remaining = human_paper_queue_len()
    print(f"[Human] Batch import finished: {rounds} task(s) created, human queue length now {remaining}")
    return rounds
