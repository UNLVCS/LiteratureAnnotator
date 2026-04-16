from __future__ import annotations

import json
from typing import Optional

from config.app_config import AppConfig
from label_api.services.context import app_config, client, ls_rag
from label_api.services.rag_task_payload import build_criteria_list, build_rag_criteria_html
from utilities.queue_helpers import ack_paper, claim_next_paper, requeue_inflight


def _classified_minio_prefixes(config: AppConfig) -> list[str]:
    seen: list[str] = []
    for provider_cfg in config.llm_providers.values():
        for model in provider_cfg.models:
            if model.skip:
                continue
            if model.model not in seen:
                seen.append(model.model)
    return seen


def _load_classified_paper_data(paper_id: str, providers: list[str]) -> tuple[Optional[dict], Optional[str]]:
    for provider in providers:
        try:
            object_name = f"{provider}/{paper_id}.json"
            response = client.get_object(bucket_name=app_config.minio.bucket_name, object_name=object_name)
            print(f"Paper data for {provider}")
            return json.loads(response.data.decode("utf-8")), provider
        except Exception as e:
            print(f"Error getting paper {paper_id} data for {provider}: {e}")
    return None, None


def import_next_paper_tasks(_project_id: int) -> None:
    paper_id: Optional[str] = claim_next_paper()
    if not paper_id:
        print("No paper ID found")
        return

    try:
        providers = _classified_minio_prefixes(app_config)
        if not providers:
            print("import_next_paper_tasks: no LLM models available; cannot resolve MinIO paths")
            requeue_inflight(paper_id)
            return

        paper_data, provider = _load_classified_paper_data(paper_id, providers)
        if paper_data is None or provider is None:
            print(f"No classified JSON for paper {paper_id} (tried: {providers}); requeueing")
            requeue_inflight(paper_id)
            return

        criteria_list = build_criteria_list(paper_data)
        if criteria_list:
            criteria_html, criterion_names = build_rag_criteria_html(criteria_list)
            task_data = {
                "paper_id": paper_id,
                "provider": provider,
                "title": paper_data.get("title", "Title N/A"),
                "criterion": "multiple",
                "criteria_html": criteria_html,
                "criteria_json": json.dumps(criteria_list),
                "criterion_names": json.dumps(criterion_names),
            }
            ls_rag.import_tasks([{"data": task_data}])

        ack_paper(paper_id)
    except Exception:
        requeue_inflight(paper_id)
        raise
