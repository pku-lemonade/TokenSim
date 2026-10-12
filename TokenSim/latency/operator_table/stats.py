"""Operator estimate records, provenance counters and workload candidates."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any
from TokenSim.operator_data.coverage import MissingShapeReport
from TokenSim.operator_data.lookup import MissingOperatorDataError


logger = logging.getLogger(__name__)

COMPONENTS = ("gemm", "attention", "moe", "elementwise", "lm_head", "comm", "overhead")

_MOE_DISTRIBUTION_CANDIDATES = {
    "uniform": ("uniform", "balanced", "power_law_1.01"),
    "balanced": ("balanced", "uniform", "power_law_1.01"),
    "skew": ("power_law_1.2", "skew", "uniform", "balanced"),
    "hot": ("power_law_1.2", "hot", "uniform", "balanced"),
    "burst": ("power_law_1.2", "burst", "uniform", "balanced"),
}


@dataclass
class OperatorStats:
    """Aggregated provenance and time breakdown of every operator estimate.

    ``missing`` is the collection checklist of shapes no table answered. The
    worker shares the same report with its collective model, so communication
    misses (collective, ep_all2all) land in the same list as compute misses.
    """

    match_type_counts: Counter = field(default_factory=Counter)
    table_match_counts: dict[str, Counter] = field(default_factory=dict)
    component_seconds: Counter = field(default_factory=Counter)
    missing: MissingShapeReport = field(default_factory=MissingShapeReport)
    step_count: int = 0

    def record(self, table: str, match_type: str) -> None:
        self.match_type_counts[match_type] += 1
        self.table_match_counts.setdefault(table, Counter())[match_type] += 1

    def record_missing(self, error: MissingOperatorDataError) -> None:
        self.missing.record_error(error)

    def record_component(self, component: str, seconds: float) -> None:
        self.component_seconds[component] += seconds

    def aggregate(self, other: "OperatorStats") -> "OperatorStats":
        result = OperatorStats(missing=self.missing.merge(other.missing))
        result.match_type_counts = self.match_type_counts + other.match_type_counts
        for table, counter in list(self.table_match_counts.items()) + list(
            other.table_match_counts.items()
        ):
            result.table_match_counts[table] = (
                result.table_match_counts.get(table, Counter()) + counter
            )
        result.component_seconds = self.component_seconds + other.component_seconds
        result.step_count = self.step_count + other.step_count
        return result

    def as_dict(self) -> dict[str, Any]:
        total = sum(self.match_type_counts.values())
        return {
            "operator_query_count": total,
            "operator_match_type_counts": dict(self.match_type_counts),
            "operator_table_match_counts": {
                table: dict(counter)
                for table, counter in sorted(self.table_match_counts.items())
            },
            "operator_component_seconds": {
                k: self.component_seconds.get(k, 0.0) for k in COMPONENTS
            },
            "operator_missing_shape_count": self.missing.shape_count,
            "operator_missing_shape_groups": self.missing.group_count,
            "operator_missing_per_table": self.missing.per_table(),
            "operator_step_count": self.step_count,
        }

    def missing_shape_records(self) -> list[dict[str, Any]]:
        """Missing shapes grouped by table and discrete key, most frequent first."""
        return self.missing.records()


@dataclass(frozen=True)
class OperatorEstimate:
    latency_us: float
    match_type: str
    source_id: str
    workload_distribution: str | None = None
