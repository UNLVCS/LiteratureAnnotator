"""
Pydantic-based config loader.
Loads settings from environment or from files (e.g. YAML) with validation.
"""

from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

# Backward-compatible alias so existing code catching ConfigValidationError still works
ConfigValidationError = ValidationError

T = TypeVar("T", bound=BaseModel)


def load_config_from_env(schema_class: type[T]) -> T:
    """
    Load config from environment variables by instantiating the Pydantic Settings model.

    The model uses os.environ and any .env configuration defined in its model_config.
    """
    return schema_class()


def load_config_from_yaml_file(schema_class: type[T], yaml_file_path: str | Path) -> T:
    """
    Load config from a YAML file and validate with the given Pydantic model.

    Requires PyYAML: pip install pyyaml
    """
    import yaml

    with open(yaml_file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    return schema_class.model_validate(data)
