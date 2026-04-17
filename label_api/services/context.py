from __future__ import annotations

from apscheduler.schedulers.background import BackgroundScheduler
from minio import Minio

from config.app_config import AppConfig, load_app_config
from label_api.clients.label_studio.human import HumanLabellerSDK
from label_api.clients.label_studio.rag import LabellerSDK

app_config = load_app_config()


def _minio_client(config: AppConfig) -> Minio:
    return Minio(
        config.minio.url,
        access_key=config.minio.access_key,
        secret_key=config.minio.secret_key,
        secure=config.minio.secure,
    )


client = _minio_client(app_config)
scheduler = BackgroundScheduler()
ls_rag = LabellerSDK(app_config.label_studio)
ls_human = HumanLabellerSDK(app_config.label_studio)
