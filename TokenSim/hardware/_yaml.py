from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from TokenSim.errors import ConfigurationError


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ConfigurationError(
            "YAML catalogs require PyYAML; install project requirements"
        ) from exc
    path = Path(path)
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"catalog file not found: {path}") from exc
    except Exception as exc:
        raise ConfigurationError(f"failed to parse {path}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ConfigurationError(f"{path}: top level must be a mapping")
    return dict(document)


def require(mapping: Mapping[str, Any], field: str, context: str) -> Any:
    if field not in mapping:
        raise ConfigurationError(f"{context}: missing required field {field!r}")
    return mapping[field]


def positive_number(value: Any, field: str, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{context}: {field} must be numeric") from exc
    if not number > 0:
        raise ConfigurationError(f"{context}: {field} must be positive, got {value!r}")
    return number
