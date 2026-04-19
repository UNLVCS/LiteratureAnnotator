from fastapi import APIRouter, BackgroundTasks, Request

from label_api.api.handlers.health import health_check_handler
from label_api.api.handlers.webhook import webhook_handler

router = APIRouter()


@router.get("/health")
async def health_check():
    return health_check_handler()


@router.post("/webhook")
async def ls_webhook(req: Request, bg: BackgroundTasks):
    payload = await req.json()
    return webhook_handler(payload, bg)


@router.post("/import")
async def trigger_import(bg: BackgroundTasks):
    """Manually trigger the next paper import for both RAG and human projects."""
    from label_api.services.task_dispatch import rag_handlers, human_handlers
    bg.add_task(rag_handlers().import_next)
    bg.add_task(human_handlers().import_next)
    return {"status": "import triggered"}
