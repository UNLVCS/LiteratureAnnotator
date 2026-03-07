"""
Pydantic-based config loader.
Loads settings from environment (and optional .env) with validation.
"""

from typing import TypeVar

from pydantic import BaseModel, ValidationError

# Backward-compatible alias so existing code catching ConfigValidationError still works
ConfigValidationError = ValidationError

T = TypeVar("T", bound=BaseModel)


def load_config(schema_class: type[T]) -> T:
    """
    Load config from environment by instantiating the Pydantic Settings model.

    The model uses os.environ and any .env configuration defined in its model_config.
    """
    return schema_class()
