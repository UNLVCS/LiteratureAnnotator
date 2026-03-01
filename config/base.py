"""
Reflection-based config loader.
Introspects a schema class, resolves env vars, coerces types, validates at load time.

Supports only str, bool, int, and float. No Optional: missing value with no default errors.
"""

import os
from dataclasses import fields
from typing import Annotated, Any, get_args, get_origin

from config.tags import Default, Env

SUPPORTED_TYPES = frozenset({str, bool, int, float})


class ConfigValidationError(Exception):
    """Raised when required env vars are missing or type coercion fails."""

    def __init__(self, errors: list[tuple[str, str]]):
        self.errors = errors
        lines = [f"  {field}: {msg}" for field, msg in errors]
        super().__init__("Config validation failed:\n" + "\n".join(lines))


def _env_name(metadata: list[Any]) -> str | None:
    """Resolve env var name from metadata. Returns None if Env not in metadata."""
    for m in metadata:
        if isinstance(m, Env):
            return m.name
    return None


def _get_default_value(metadata: list[Any]) -> Any | None:
    """Resolve default value from Default tag in metadata."""
    for m in metadata:
        if isinstance(m, Default):
            return m.value
    return None


def _parse_bool(s: str) -> bool:
    v = s.strip().lower()
    if v in ("1", "true", "yes"):
        return True
    if v in ("0", "false", "no"):
        return False
    raise ValueError(f"Cannot coerce to bool: {s!r}")


def _parse_to_type(raw: str, typ: type) -> Any:
    if typ is str:
        return raw.strip() if raw else ""
    if typ is int:
        return int(raw.strip())
    if typ is float:
        return float(raw.strip())
    if typ is bool:
        return _parse_bool(raw)
    raise TypeError(f"Unsupported type: {typ}")


def load_config(
    schema_class: type,
    env: dict[str, str] | None = None,
) -> Any:
    """
    Load config from environment by introspecting the schema class.

    Supports only str, bool, int, float. If env var is missing and no default, raises ConfigValidationError.
    """
    if env is None:
        env = os.environ

    if not hasattr(schema_class, "__dataclass_fields__"):
        raise TypeError("Schema must be a dataclass")

    dc_fields = fields(schema_class)
    errors: list[tuple[str, str]] = []
    values: dict[str, Any] = {}

    for f in dc_fields:
        name = f.name
        hint = f.type
        if get_origin(hint) is not Annotated:
            errors.append((name, "Field must use Annotated[type, Env(...), Default(...)?]"))
            continue

        args = get_args(hint)
        hint = args[0]
        metadata = list(args[1:]) if len(args) > 1 else []

        typ = hint if hint in SUPPORTED_TYPES else str
        if hint not in SUPPORTED_TYPES:
            errors.append((name, f"Unsupported type {hint}. Use str, bool, int, or float."))
            continue

        env_name = _env_name(metadata)
        if env_name is None:
            errors.append((name, "Missing Env(...) in metadata"))
            continue
        default = _get_default_value(metadata)
        raw = env.get(env_name)
        is_empty = raw is None or (isinstance(raw, str) and raw.strip() == "")

        if is_empty:
            if default is not None:
                values[name] = default
                continue
            errors.append((name, f"Missing required env var: {env_name}"))
            continue

        raw_str = str(raw).strip()
        try:
            values[name] = _parse_to_type(raw_str, typ)
        except (ValueError, TypeError) as e:
            errors.append((name, f"Coercion failed for {env_name}: {e}"))

    if errors:
        raise ConfigValidationError(errors)

    return schema_class(**values)
