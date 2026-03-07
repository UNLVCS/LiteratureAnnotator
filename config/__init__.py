"""Config package: Pydantic Settings-based env config with load-time validation."""

from config.base import ConfigValidationError, load_config

__all__ = [
    "load_config",
    "ConfigValidationError",
]
