from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import BackgroundTasks

from label_api.services.task_dispatch import ensure_next_task_if_empty, handlers_for_project


def webhook_handler(payload: Dict[str, Any], background_tasks: BackgroundTasks) -> Dict[str, str]:
    action = payload.get("action")
    project_id: Optional[int] = payload.get("project", {}).get("id")

    print("==========================================")
    print("WEBHOOK RECEIVED: ", action, "project_id:", project_id)
    print("==========================================")

    handlers = handlers_for_project(project_id)

    if action == "PROJECT_CREATED":
        background_tasks.add_task(handlers.import_next)
        return {"status": "ok", "event": "project_created"}

    if action in ("ANNOTATION_CREATED", "ANNOTATION_UPDATED"):
        task_info = payload.get("task", {})
        annotation = payload.get("annotation", {})
        background_tasks.add_task(handlers.handle_completed, task_info, annotation)

    background_tasks.add_task(ensure_next_task_if_empty, handlers, project_id)
    return {"status": "ok"}
