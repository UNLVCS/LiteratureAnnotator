"""Label Studio SDK for the Human Labeling project."""

from typing import Any, Dict, List, Optional

from label_api.clients.label_studio.base import BaseLabelStudioProject
from label_api.clients.label_studio.rag import LabelStudioConfig


class HumanLabellerSDK(BaseLabelStudioProject):
    def __init__(self, config: LabelStudioConfig):
        super().__init__(
            config,
            project_title="Human Labeling Project",
            project_description="Direct human classification from chunks for fine-tuning",
            config_filename="human_label_interface_config.xml",
            log_prefix="[Human]",
        )

    def import_tasks(self, tasks: List[Dict[str, Any]], *, commit: Optional[bool] = None) -> Any:
        return super().import_tasks(tasks, commit=commit)

    def count_new_tasks(self, project_id: Optional[int] = None, *, page_size: int = 1000) -> int:
        return super().count_new_tasks(project_id=project_id, page_size=page_size)

    def get_completed_tasks(self, project_id: Optional[int] = None) -> List[Any]:
        return super().get_completed_tasks(project_id=project_id)

    def create_webhook(self, endpoint: str, actions: Optional[List[str]] = None) -> Any:
        return super().create_webhook(endpoint=endpoint, actions=actions)
