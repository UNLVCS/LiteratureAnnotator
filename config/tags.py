"""
Tag types for config schema definitions.
Used inside Annotated[type, ...] to specify env names, defaults, and validators.
"""

from typing import Callable


class Env:
    """Override the environment variable name (default: field_name -> FIELD_NAME)."""

    def __init__(self, name: str):
        self.name = name


class Default:
    """Explicit default value when env var is not set."""

    def __init__(self, value: object):
        self.value = value


class Required:
    """Mark field as required (no default)."""

    pass


class Coerce:
    """Custom coercion function: (raw: str) -> T."""

    def __init__(self, fn: Callable[[str], object]):
        self.fn = fn
