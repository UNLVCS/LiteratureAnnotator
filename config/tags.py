"""
Tag types for config schema definitions.
Used inside Annotated[type, ...] to specify env names and defaults.
"""


class Env:
    """Override the environment variable name (default: field_name -> FIELD_NAME)."""

    def __init__(self, name: str):
        self.name = name


class Default:
    """Explicit default value when env var is not set."""

    def __init__(self, value: object):
        self.value = value
