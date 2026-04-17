from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from label_api.services.importers.human import import_all_pending_human_tasks, import_next_human_tasks
from label_api.services.annotation_store import handle_completed_human_task, handle_completed_task
from label_api.services.context import ls_human, ls_rag
from label_api.services.paper_queue_import import import_next_paper_tasks


@dataclass(frozen=True)
class ProjectTaskHandlers:
    sdk: Any
    import_initial: Callable[[], None]
    import_next: Callable[[], None]
    handle_completed: Callable[[Dict[str, Any], Dict[str, Any]], None]
    periodic_log_name: str


def rag_handlers() -> ProjectTaskHandlers:
    return ProjectTaskHandlers(
        sdk=ls_rag,
        import_initial=lambda: import_next_paper_tasks(ls_rag.project_id),
        import_next=lambda: import_next_paper_tasks(ls_rag.project_id),
        handle_completed=handle_completed_task,
        periodic_log_name="Paper",
    )


def human_handlers() -> ProjectTaskHandlers:
    return ProjectTaskHandlers(
        sdk=ls_human,
        import_initial=lambda: import_all_pending_human_tasks(ls_human),
        import_next=lambda: import_next_human_tasks(ls_human),
        handle_completed=handle_completed_human_task,
        periodic_log_name="Human",
    )


def handlers_for_project(project_id: Optional[int]) -> ProjectTaskHandlers:
    return human_handlers() if project_id == ls_human.project_id else rag_handlers()


def ensure_next_task_if_empty(handlers: ProjectTaskHandlers, project_id: Optional[int]) -> None:
    if project_id is None:
        return
    if handlers.sdk.count_new_tasks(project_id) == 0:
        handlers.import_next()
