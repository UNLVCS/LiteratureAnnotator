import os
import json
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

from label_studio_sdk.client import LabelStudio


class LabellerSDK:
    """Label Studio interfacer built on the typed SDK (no raw requests).

    Responsibilities:
    - Initialize SDK client and create a project from XML config
    - Build tasks and import in bulk
    - Count tasks by status (e.g., "new") using Data Manager filters
    - Retrieve completed tasks (including annotations)
    - Create webhooks with configured actions
    """

    def __init__(self):
        load_dotenv()

        self.client = LabelStudio(
            base_url=os.getenv("LABEL_STUDIO_URL"),
            api_key=os.getenv("LABEL_STUDIO_API_KEY"),
        )

        # Load XML label config from file co-located with this module by default
        default_config_path = os.path.join(os.path.dirname(__file__), "label_interface_config.xml")
        with open(default_config_path, "r") as config_file:
            interface_config = config_file.read()

        project = self.client.projects.create(
            title="RAG Annotation Project",
            description="Labeling sections relevant to questions using a RAG pipeline",
            label_config=interface_config,
        )
        self.project_id: int = project.id

    # -----------------------------
    # Task creation & import
    # -----------------------------
    def create_tasks_payload(self, article: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build a list of task dicts compliant with LS bulk import.

        Expected `article` structure matches existing code:
        - article['paper_id']
        - article['title']
        - article['queries']: List[str]
        - article[query]["retrieved_chunks"] for each query
        """
        tasks: List[Dict[str, Any]] = []
        for query in article.get("queries", []):
            task: Dict[str, Any] = {
                "data": {
                    "paper_id": article.get("paper_id"),
                    "title": article.get("title"),
                    "paper_text": article.get(query, {}).get("retrieved_chunks"),
                    "class_criteria": query,
                }
            }
            tasks.append(task)
        return tasks

    def import_tasks(self, tasks: List[Dict[str, Any]], *, commit: Optional[bool] = None) -> Any:
        """Bulk-import tasks via SDK. Returns the import response object."""
        response = self.client.projects.import_tasks(
            id=self.project_id,
            request=tasks,
            commit_to_project=commit,
            return_task_ids=True,
        )
        return response

    # -----------------------------
    # Query helpers
    # -----------------------------
    @staticmethod
    def _status_filter_json(status_value: str) -> str:
        """Build Data Manager filters JSON string for a task status.
        
        Note: Label Studio uses 'is_labeled' field, not 'status'.
        - "new" tasks = is_labeled = false
        - "completed" tasks = is_labeled = true
        """
        # Map status values to is_labeled boolean
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

    # -----------------------------
    # Counting & retrieval
    # -----------------------------
    def count_new_tasks(self, project_id: Optional[int] = None, *, page_size: int = 1000) -> int:
        """Count tasks with status="new" using SDK pagination."""
        pid = project_id or self.project_id
        pager = self.client.tasks.list(
            project=pid,
            query=self._status_filter_json("new"),
            page_size=page_size,
        )
        total = 0
        # Iterate through individual tasks instead of pages
        for task in pager:
            total += 1
        return total

    def get_completed_tasks(self, project_id: Optional[int] = None) -> List[Any]:
        """Return tasks with status="completed" (annotations included by default)."""
        pid = project_id or self.project_id
        pager = self.client.tasks.list(
            project=pid,
            query=self._status_filter_json("completed"),
        )
        return list(pager)

    # -----------------------------
    # Webhooks
    # -----------------------------
    def create_webhook(self, endpoint: str, actions: Optional[List[str]] = None) -> Any:
        """Create a webhook on the current project via SDK (validates URL).

        By default, uses project-level actions only (no organization-only events).
        """
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("Endpoint URL must start with http:// or https://")
        if not endpoint.endswith("/webhook"):
            raise ValueError("Endpoint URL must end with /webhook for FastAPI handler")

        # Default to project-level webhook actions only
        if actions is None:
            actions = [
                "ANNOTATION_CREATED",
                "ANNOTATION_UPDATED",
                "ANNOTATIONS_CREATED",
                "ANNOTATIONS_DELETED",
                "TASKS_CREATED",
                "TASKS_DELETED",
            ]

        webhook = self.client.webhooks.create(
            url=endpoint,
            project=self.project_id,
            send_payload=True,
            is_active=True,
            headers={"Content-Type": "application/json"},
            actions=actions,
        )
        return webhook


