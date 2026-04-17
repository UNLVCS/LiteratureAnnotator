"""Shared Label Studio SDK helpers for label_api projects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from label_studio_sdk.client import LabelStudio


class LabelStudioConfigLike(Protocol):
    """Config protocol exposing Label Studio connection settings."""

    label_studio_url: str
    label_studio_api_key: str


class BaseLabelStudioProject:
    """Reusable SDK wrapper for a single Label Studio project."""

    def __init__(
        self,
        config: LabelStudioConfigLike,
        *,
        project_title: str,
        project_description: str,
        config_filename: str,
        log_prefix: str = "",
    ) -> None:
        self.client = LabelStudio(
            base_url=config.label_studio_url or None,
            api_key=config.label_studio_api_key or None,
        )
        self._log_prefix = log_prefix

        config_path = Path(__file__).resolve().parents[2] / "services" / "ui-config" / config_filename
        interface_config = config_path.read_text(encoding="utf-8")

        matching_project = self._find_project_by_title(project_title)
        if matching_project is not None:
            self._log(f"Reusing existing project: {matching_project.id}")
            self.project_id = matching_project.id
            return

        self._log("Creating new project")
        project = self.client.projects.create(
            title=project_title,
            description=project_description,
            label_config=interface_config,
        )
        self.project_id = project.id

    def _log(self, message: str) -> None:
        if self._log_prefix:
            print(f"{self._log_prefix} {message}")
        else:
            print(message)

    def _find_project_by_title(self, title: str) -> Any:
        for project in self.client.projects.list():
            if project.title == title:
                return project
        return None

    def import_tasks(self, tasks: List[Dict[str, Any]], *, commit: Optional[bool] = None) -> Any:
        """Bulk-import tasks via SDK."""
        return self.client.projects.import_tasks(
            id=self.project_id,
            request=tasks,
            commit_to_project=commit,
            return_task_ids=True,
        )

    @staticmethod
    def _status_filter_json(status_value: str) -> str:
        """Build Data Manager filters JSON for task status."""
        is_labeled_value = status_value == "completed"
        filters = {
            "filters": {
                "conjunction": "and",
                "items": [
                    {
                        "filter": "filter:tasks:is_labeled",
                        "operator": "equal",
                        "type": "Boolean",
                        "value": is_labeled_value,
                    }
                ],
            }
        }
        return json.dumps(filters)

    def count_new_tasks(self, project_id: Optional[int] = None, *, page_size: int = 1000) -> int:
        """Count tasks with status='new'."""
        pid = project_id or self.project_id
        pager = self.client.tasks.list(
            project=pid,
            query=self._status_filter_json("new"),
            page_size=page_size,
        )
        return sum(1 for _ in pager)

    def get_completed_tasks(self, project_id: Optional[int] = None) -> List[Any]:
        """Return completed tasks with annotations."""
        pid = project_id or self.project_id
        pager = self.client.tasks.list(
            project=pid,
            query=self._status_filter_json("completed"),
        )
        return list(pager)

    def create_webhook(self, endpoint: str, actions: Optional[List[str]] = None) -> Any:
        """Create webhook on this project. Skips if URL already exists."""
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("Endpoint URL must start with http:// or https://")
        if not endpoint.endswith("/webhook"):
            raise ValueError("Endpoint URL must end with /webhook for FastAPI handler")

        existing_webhooks = list(self.client.webhooks.list(project=self.project_id))
        for webhook in existing_webhooks:
            if webhook.url == endpoint:
                self._log(f"Webhook already exists: {endpoint}")
                return webhook

        if actions is None:
            actions = [
                "ANNOTATION_CREATED",
                "ANNOTATION_UPDATED",
                "ANNOTATIONS_CREATED",
                "ANNOTATIONS_DELETED",
                "TASKS_CREATED",
                "TASKS_DELETED",
            ]

        self._log(f"Creating webhook with actions: {actions}")
        return self.client.webhooks.create(
            url=endpoint,
            project=self.project_id,
            send_payload=True,
            is_active=True,
            headers={"Content-Type": "application/json"},
            actions=actions,
        )
