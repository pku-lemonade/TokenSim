"""Operator composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from TokenSim.af.providers.queries import AF_MATCH_TYPES
from TokenSim.latency.operator_table import OperatorEstimate, OperatorResolver


_MATCH_RANK = {name: rank for rank, name in enumerate(AF_MATCH_TYPES)}


_COLLECTIVE_MATCH = {"measured": "exact"}


def _worst_match(match_types: Iterable[str]) -> str:
    """The least trustworthy match type of a composition (exact < ... < analytical)."""
    worst = "exact"
    for match_type in match_types:
        if _MATCH_RANK.get(match_type, len(AF_MATCH_TYPES)) > _MATCH_RANK[worst]:
            worst = match_type if match_type in _MATCH_RANK else "analytical"
    return worst


def _compose_source_id(source_ids: Iterable[str]) -> str:
    unique = tuple(dict.fromkeys(source_ids))
    if not unique:
        return "composition:empty"
    if len(unique) == 1:
        return unique[0]
    return "compose:" + ":".join(unique)


def _compose_dataset_version(versions: Iterable[str]) -> str:
    unique = tuple(dict.fromkeys(v for v in versions if v))
    if not unique:
        return "unknown"
    if len(unique) == 1:
        return unique[0]
    return "compose:" + ":".join(unique)


def _estimate_dataset_version(
    resolver: OperatorResolver, estimate: OperatorEstimate
) -> str:
    """Label analytical estimates separately from a package row."""
    if estimate.match_type == "analytical":
        return f"analytical:{resolver.device.device_id}"
    return resolver.dataset_version or f"unknown:{resolver.device.device_id}"


@dataclass(frozen=True)
class OperatorComponent:
    """One operator of an AF stage and how it was priced."""

    name: str
    table: str
    key: Mapping[str, Any]
    latency_us: float
    match_type: str
    source_id: str
    dataset_version: str = ""
    covered_operations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "table": self.table,
            "key": dict(self.key),
            "latency_us": self.latency_us,
            "match_type": self.match_type,
            "source_id": self.source_id,
            "dataset_version": self.dataset_version,
            "covered_operations": self.covered_operations,
        }
