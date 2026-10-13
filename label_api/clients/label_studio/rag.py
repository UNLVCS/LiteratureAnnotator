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
            # Title kept as-is: existing projects are looked up by title.
            project_description="Review LLM criterion answers and their cited evidence against the full paper",
            config_filename="label_interface_config.xml",
        )

    def import_tasks(self, tasks: List[Dict[str, Any]], *, commit: Optional[bool] = None) -> Any:
        return super().import_tasks(tasks, commit=commit)
