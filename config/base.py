"""
Pydantic-based config loader.
Loads settings from environment (and optional .env) with validation.
"""

from typing import TypeVar

from pydantic import BaseModel

from pydantic import ValidationError

# Backward-compatible alias so existing code catching ConfigValidationError still works
ConfigValidationError = ValidationError

T = TypeVar("T", bound=BaseModel)


def load_config(
    schema_class: type[T],
    env: dict[str, str] | None = None,
) -> T:
    """
    Load config from environment by instantiating the Pydantic Settings model.

    If env is provided, values are taken from that dict (e.g. for tests).
    Otherwise the model uses os.environ and optional .env file per its model_config.
    """
    if env is not None:
        return schema_class.model_validate(env)
    return schema_class()
