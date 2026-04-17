from typing import Any, Dict, List, Optional, Protocol

from label_api.clients.label_studio.base import BaseLabelStudioProject


class LabelStudioConfig(Protocol):
    """Protocol for config objects that provide Label Studio URL and API key."""

    label_studio_url: str
    label_studio_api_key: str


class LabellerSDK(BaseLabelStudioProject):
    def __init__(self, config: LabelStudioConfig):
        super().__init__(
            config,
            project_title="RAG Annotation Project",
            project_description="Labeling sections relevant to questions using a RAG pipeline",
            config_filename="label_interface_config.xml",
        )

    def create_tasks_payload(self, article: Dict[str, Any]) -> List[Dict[str, Any]]:
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
        return super().import_tasks(tasks, commit=commit)
