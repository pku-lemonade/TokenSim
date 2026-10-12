"""Load and validate conversation-to-replica routing mappings."""

import json
from collections.abc import Mapping
from pathlib import Path

from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError


def validate_router_mapping(mapping):
    if not isinstance(mapping, Mapping):
        raise ConfigurationError("router mapping must contain a JSON object")
    result = {}
    for key, value in mapping.items():
        name = f"router mapping rank for {key!r}"
        if isinstance(value, str):
            try:
                value = int(value)
            except ValueError as error:
                raise ConfigurationError(f"{name} must be an integer >= 0") from error
        result[str(key)] = integer(value, name)
    return result


def load_router_mapping(path):
    try:
        mapping = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ConfigurationError(f"invalid router mapping {path}: {error}") from error
    if not isinstance(mapping, Mapping):
        raise ConfigurationError("router mapping must contain a JSON object")
    return validate_router_mapping(mapping.get("mapping", mapping))
