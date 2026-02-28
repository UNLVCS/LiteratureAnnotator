"""
Reflection-based config loader.
Introspects a schema class, resolves env vars, coerces types, validates at load time.
"""

import os
from dataclasses import MISSING, fields
from typing import Annotated, Any, Callable, get_args, get_origin

from config.tags import Coerce, Default, Env, Required


class ConfigValidationError(Exception):
    """Raised when required env vars are missing or type coercion fails."""

    def __init__(self, errors: list[tuple[str, str]]):
        self.errors = errors
        lines = [f"  {field}: {msg}" for field, msg in errors]
        super().__init__("Config validation failed:\n" + "\n".join(lines))


def _field_env_name(field_name: str, metadata: list[Any]) -> str:
    """Resolve env var name from metadata or derive from field name."""
    for m in metadata:
        if isinstance(m, Env):
            return m.name
    return field_name.upper()


def _field_default(field_default: Any, metadata: list[Any]) -> Any | None:
    """Resolve default value from Default tag or dataclass field default."""
    for m in metadata:
        if isinstance(m, Default):
            return m.value
    if field_default is not MISSING:
        return field_default
    return None


def _is_optional(hint: Any) -> bool:
    """True if type is Optional[T] (Union[T, None])."""
    origin = get_origin(hint)
    args = get_args(hint)
    return origin is not None and type(None) in (args or ())


def _field_required(field_default: Any, metadata: list[Any], hint: Any) -> bool:
    """True if field has no default and is required."""
    if _field_default(field_default, metadata) is not None:
        return False
    if _is_optional(hint):
        return False
    return True


def _field_custom_coerce(metadata: list[Any]) -> Callable[[str], Any] | None:
    for m in metadata:
        if isinstance(m, Coerce):
            return m.fn
    return None


def _coerce_bool(s: str) -> bool:
    v = s.strip().lower()
    if v in ("1", "true", "yes"):
        return True
    if v in ("0", "false", "no"):
        return False
    raise ValueError(f"Cannot coerce to bool: {s!r}")


def _coerce_to_type(raw: str, typ: type, custom_coerce: Callable[[str], Any] | None) -> Any:
    if custom_coerce:
        return custom_coerce(raw)
    if typ is str:
        return raw.strip() if raw else ""
    if typ is int:
        return int(raw.strip())
    if typ is float:
        return float(raw.strip())
    if typ is bool:
        return _coerce_bool(raw)
    return raw


def _get_origin_and_args(hint: Any) -> tuple[Any, tuple]:
    origin = get_origin(hint)
    args = get_args(hint) if origin else ()
    if origin is None:
        return hint, ()
    return origin, args


def _resolve_inner_type(hint: Any) -> type:
    """Get the non-Optional inner type for Optional[T]."""
    origin = get_origin(hint)
    args = get_args(hint)
    if origin is type(None):
        return str  # fallback
    if str(origin) == "typing.Union" and args and type(None) in args:
        inner = next((a for a in args if a is not type(None)), str)
        return inner if isinstance(inner, type) else str
    return hint if isinstance(hint, type) else str


def load_config(
    schema_class: type,
    env: dict[str, str] | None = None,
) -> Any:
    """
    Load config from environment by introspecting the schema class.

    - schema_class: a dataclass with type-annotated fields
    - env: dict to read from (default: os.environ). Pass a dict for tests.
    - Returns: instance of schema_class with populated fields
    - Raises: ConfigValidationError on missing required or coercion failure
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
        metadata = list(f.metadata.values()) if f.metadata else []
        # Support Annotated[X, Env(...), Default(...)]
        if get_origin(hint) is Annotated:
            args = get_args(hint)
            inner = args[0]
            metadata = list(args[1:]) if len(args) > 1 else []
            hint = inner

        origin, type_args = _get_origin_and_args(hint)
        env_name = _field_env_name(name, metadata)
        default = _field_default(f.default, metadata)
        required = _field_required(f.default, metadata, hint)
        custom_coerce = _field_custom_coerce(metadata)
        inner_type = _resolve_inner_type(hint)

        raw = env.get(env_name)
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            if default is not None:
                values[name] = default
                continue
            if required:
                errors.append((name, f"Missing required env var: {env_name}"))
                continue
            # Optional[T] with no value -> None
            if _is_optional(hint):
                values[name] = None
                continue
            errors.append((name, f"Missing env var: {env_name} (no default)"))
            continue

        raw_str = str(raw).strip()
        try:
            values[name] = _coerce_to_type(raw_str, inner_type, custom_coerce)
        except (ValueError, TypeError) as e:
            errors.append((name, f"Coercion failed for {env_name}: {e}"))

    if errors:
        raise ConfigValidationError(errors)

    return schema_class(**values)
