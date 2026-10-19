"""Af common."""

from __future__ import annotations

from typing import Any, Mapping

from pydantic_core import ArgsKwargs

from TokenSim.dtypes import canonicalize_dtype
from TokenSim.errors import ConfigurationError


_AF_CONFIG_DTYPE_FIELDS = frozenset(
    {"af_dtype", "weight_dtype", "activation_dtype", "kv_cache_dtype"}
)


_AF_LINK_DTYPE_FIELDS = frozenset({"a_to_f_dtype", "f_to_a_dtype"})


_NEW_DTYPE_FIELDS = frozenset(
    {
        "weight_dtype",
        "activation_dtype",
        "kv_cache_dtype",
        "a_to_f_dtype",
        "f_to_a_dtype",
    }
)


_COLLECTIVE_OPERATIONS = frozenset({"all_reduce", "all_gather", "reduce_scatter"})


AF_LATENCY_PROVIDERS = ("injected", "operator_table")


AF_LATENCY_FALLBACKS = ("table_first", "table_only", "analytical_only")


def _capture_explicit_fields(
    value: Any,
    *,
    field_names: frozenset[str],
    positional_names: tuple[str, ...],
    init_var_name: str,
) -> Any:
    """Pass raw field presence through Pydantic default filling."""
    if isinstance(value, ArgsKwargs):
        args = tuple(value.args or ())
        kwargs = dict(value.kwargs or {})
        if init_var_name in kwargs:
            explicit = {name for name in field_names if kwargs.get(name) is not None}
        else:
            explicit = {
                positional_names[index]
                for index in range(min(len(args), len(positional_names)))
                if positional_names[index] in field_names
            }
            explicit.update(field_names.intersection(kwargs))
        kwargs[init_var_name] = tuple(sorted(explicit))
        return ArgsKwargs(args, kwargs)
    if isinstance(value, Mapping):
        values = dict(value)
        values[init_var_name] = tuple(sorted(field_names.intersection(values)))
        return values
    return value


def _canonical_config_dtype(value: str, role: str) -> str:
    try:
        registry_role = role if role in _NEW_DTYPE_FIELDS else None
        return canonicalize_dtype(value, role=registry_role)
    except ValueError as exc:
        message = str(exc)
        if role == "af_dtype":
            message = message.replace("dtype=", "af_dtype=", 1)
        raise ConfigurationError(message) from exc
