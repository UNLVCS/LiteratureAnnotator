"""
Label Studio SDK for the Human Labeling Project.
Users make classification decisions directly from chunks (no LLM pre-labeling).
"""

import os
import json
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

from label_studio_sdk.client import LabelStudio


class HumanLabellerSDK:
    """Label Studio interfacer for Human Labeling Project.

    Same structure as LabellerSDK but uses:
    - Project title: "Human Labeling Project"
    - Config: human_label_interface_config.xml
    """

    def __init__(self, *, base_url: Optional[str] = None, api_key: Optional[str] = None):
        load_dotenv()

        self.client = LabelStudio(
            base_url=base_url if base_url is not None else os.getenv("LABEL_STUDIO_URL"),
            api_key=api_key if api_key is not None else os.getenv("LABEL_STUDIO_API_KEY"),
        )

        config_path = os.path.join(os.path.dirname(__file__), "human_label_interface_config.xml")
        with open(config_path, "r") as f:
            interface_config = f.read()

        project_title = "Human Labeling Project"
        existing_projects = list(self.client.projects.list())
        matching_project = None
        for proj in existing_projects:
            if proj.title == project_title:
                matching_project = proj
                break

        if matching_project:
            print(f"[Human] Reusing existing project: {matching_project.id}")
            self.project_id = matching_project.id
        else:
            print("[Human] Creating new project")
            project = self.client.projects.create(
                title=project_title,
                description="Direct human classification from chunks for fine-tuning",
                label_config=interface_config,
            )
            self.project_id = project.id

    def import_tasks(self, tasks: List[Dict[str, Any]], *, commit: Optional[bool] = None) -> Any:
        """Bulk-import tasks via SDK."""
        response = self.client.projects.import_tasks(
            id=self.project_id,
            request=tasks,
            commit_to_project=commit,
            return_task_ids=True,
        )
        return response

    @staticmethod
    def _status_filter_json(status_value: str) -> str:
        """Build Data Manager filters JSON for task status."""
        is_labeled_value = True if status_value == "completed" else False
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
        """Count tasks with status=new."""
        pid = project_id or self.project_id
        pager = self.client.tasks.list(
            project=pid,
            query=self._status_filter_json("new"),
            page_size=page_size,
        )
        total = 0
        for _ in pager:
            total += 1
        return total

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
        for wh in existing_webhooks:
            if wh.url == endpoint:
                print(f"[Human] Webhook already exists: {endpoint}")
                return wh

        if actions is None:
            actions = [
                "ANNOTATION_CREATED",
                "ANNOTATION_UPDATED",
                "ANNOTATIONS_CREATED",
                "ANNOTATIONS_DELETED",
                "TASKS_CREATED",
                "TASKS_DELETED",
            ]
        print("[Human] Creating webhook with actions:", actions)
        webhook = self.client.webhooks.create(
            url=endpoint,
            project=self.project_id,
            send_payload=True,
            is_active=True,
            headers={"Content-Type": "application/json"},
            actions=actions,
        )
        return webhook
