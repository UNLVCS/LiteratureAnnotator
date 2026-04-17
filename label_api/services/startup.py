from __future__ import annotations

from label_api.services.context import app_config, scheduler
from label_api.services.task_dispatch import human_handlers, rag_handlers


def run_periodic_task_check(handlers, *, threshold: int = 5) -> None:
    try:
        new_count = handlers.sdk.count_new_tasks(handlers.sdk.project_id)
        print(f"[Periodic {handlers.periodic_log_name} Check] New tasks: {new_count}")
        if new_count < threshold:
            print(f"[Periodic {handlers.periodic_log_name} Check] Importing next task(s)")
            handlers.import_next()
    except Exception as e:
        print(f"[Periodic {handlers.periodic_log_name} Check] Error: {e}")


def periodic_paper_check() -> None:
    run_periodic_task_check(rag_handlers())


def periodic_human_paper_check() -> None:
    run_periodic_task_check(human_handlers())


def startup_services() -> None:
    webhook_url = f"{app_config.label_studio.webhook_host}/webhook"
    for handlers in (rag_handlers(), human_handlers()):
        handlers.sdk.create_webhook(endpoint=webhook_url)
        handlers.import_initial()

    scheduler.add_job(periodic_paper_check, "interval", minutes=3, id="periodic_paper_check")
    scheduler.add_job(periodic_human_paper_check, "interval", minutes=3, id="periodic_human_paper_check")
    scheduler.start()
    print("[Startup] Scheduler started")
