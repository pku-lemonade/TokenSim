from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


CANONICAL_DTYPE_ROLES = frozenset(
    {
        "weight_dtype",
        "activation_dtype",
        "kv_cache_dtype",
        "a_to_f_dtype",
        "f_to_a_dtype",
    }
)


@dataclass(frozen=True)
class DTypeSpec:
    canonical_name: str
    size_bytes: int
    aliases: tuple[str, ...]


_SPECS = (
    DTypeSpec("fp32", 4, ("fp32", "float32")),
    DTypeSpec("fp16", 2, ("fp16", "float16", "half")),
    DTypeSpec("bf16", 2, ("bf16", "bfloat16")),
    DTypeSpec(
        "fp8",
        1,
        (
            "fp8",
            "float8",
            "float8_e4m3fn",
            "float8_e4m3fnuz",
            "float8_e5m2",
            "float8_e5m2fnuz",
        ),
    ),
    DTypeSpec("int8", 1, ("int8",)),
)

DTYPE_REGISTRY: Mapping[str, DTypeSpec] = MappingProxyType(
    {spec.canonical_name: spec for spec in _SPECS}
)
_ALIAS_TO_CANONICAL: Mapping[str, str] = MappingProxyType(
    {
        alias: spec.canonical_name
        for spec in _SPECS
        for alias in spec.aliases
    }
)


def resolve_dtype(dtype: str, *, role: str | None = None) -> DTypeSpec:
    role_name = _validate_role(role)
    if not isinstance(dtype, str) or not dtype.strip():
        raise ValueError(f"{role_name} must be a non-empty dtype string")
    value = dtype.strip().lower()
    canonical = _ALIAS_TO_CANONICAL.get(value)
    if canonical is None:
        supported = ", ".join(DTYPE_REGISTRY)
        raise ValueError(
            f"{role_name}={dtype!r} has no explicit byte size; "
            f"supported canonical dtypes are [{supported}]"
        )
    return DTYPE_REGISTRY[canonical]


def canonicalize_dtype(dtype: str, *, role: str | None = None) -> str:
    return resolve_dtype(dtype, role=role).canonical_name


def dtype_size_bytes(dtype: str, *, role: str | None = None) -> int:
    return resolve_dtype(dtype, role=role).size_bytes


def _validate_role(role: str | None) -> str:
    if role is None:
        return "dtype"
    if not isinstance(role, str) or role not in CANONICAL_DTYPE_ROLES:
        raise ValueError(
            f"unknown dtype role {role!r}; expected one of "
            f"{sorted(CANONICAL_DTYPE_ROLES)}"
        )
    return role
