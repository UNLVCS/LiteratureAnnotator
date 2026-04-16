from fastapi import FastAPI

from label_api.api.routes import router
from label_api.services.startup import startup_services

app = FastAPI()
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    startup_services()
    return {"status": "ok"}
