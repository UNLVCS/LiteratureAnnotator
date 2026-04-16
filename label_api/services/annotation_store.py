from __future__ import annotations

import json
from datetime import datetime
from io import BytesIO
from typing import Any, Dict

from label_api.services.context import app_config, client


def _ensure_bucket(bucket_name: str) -> None:
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            print(f"Created bucket: {bucket_name}")
    except Exception as e:
        print(f"Error checking/creating bucket {bucket_name}: {e}")


def _build_annotation_record(task: Dict[str, Any], annotation: Dict[str, Any]) -> Dict[str, Any]:
    record: Dict[str, Any] = {}
    for prefix, data in (("task", task), ("ann", annotation)):
        for key, value in data.items():
            record[f"{prefix}_{key}"] = value
    return record


def _annotation_object_name(task: Dict[str, Any], annotation: Dict[str, Any]) -> str:
    paper_id = task.get("data", {}).get("paper_id", "unknown")
    task_id = task.get("id", "unknown")
    annotation_id = annotation.get("id", "unknown")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{paper_id}/{task_id}_{annotation_id}_{timestamp}.json"


def save_annotation_record(
    *,
    bucket_name: str,
    task: Dict[str, Any],
    annotation: Dict[str, Any],
    log_label: str,
) -> None:
    record = _build_annotation_record(task, annotation)
    object_name = _annotation_object_name(task, annotation)
    data_bytes = json.dumps(record, indent=2, default=str).encode("utf-8")
    _ensure_bucket(bucket_name)
    client.put_object(
        bucket_name=bucket_name,
        object_name=object_name,
        data=BytesIO(data_bytes),
        length=len(data_bytes),
        content_type="application/json",
    )
    print(f"Saved {log_label} to MinIO: {bucket_name}/{object_name}")


def handle_completed_task(task: Dict[str, Any], annotation: Dict[str, Any]) -> None:
    save_annotation_record(
        bucket_name=app_config.minio.annotations_bucket,
        task=task,
        annotation=annotation,
        log_label="annotation",
    )


def handle_completed_human_task(task: Dict[str, Any], annotation: Dict[str, Any]) -> None:
    save_annotation_record(
        bucket_name=app_config.minio.human_annotations_bucket,
        task=task,
        annotation=annotation,
        log_label="human annotation",
    )
