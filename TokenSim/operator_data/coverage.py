"""Coverage reports: which queries a package answers, and what it could not.

:func:`coverage_report` checks a shape manifest offline; :class:`MissingShapeReport`
is filled at simulation time by every consumer of the tables (operator backend
and collective model) and turns the individual misses into a collection
checklist grouped by table and discrete key.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

from TokenSim.operator_data.lookup import LookupPolicy, MissingOperatorDataError, OperatorLookup
from TokenSim.operator_data.manifest import ShapeManifest
from TokenSim.operator_data.package import OperatorDataPackage
from TokenSim.operator_data.schema import TABLE_SPECS


@dataclass
class CoverageReport:
    per_table: dict[str, Counter] = field(default_factory=dict)
    missing: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        totals: Counter = Counter()
        for counter in self.per_table.values():
            totals.update(counter)
        return {
            "tables": {table: dict(counter) for table, counter in sorted(self.per_table.items())},
            "totals": dict(totals),
            "missing_count": len(self.missing),
        }


@dataclass
class MissingShapeGroup:
    """All misses of one table that share a discrete key (dtype, heads, mode, ...)."""

    table: str
    key: dict[str, Any]
    axes: dict[str, list[Any]] = field(default_factory=dict)
    query_count: int = 0
    kinds: Counter = field(default_factory=Counter)
    reason: str = ""
    examples: list[dict[str, Any]] = field(default_factory=list)
    _shapes: set[tuple] = field(default_factory=set, repr=False)

    @property
    def shape_count(self) -> int:
        return len(self._shapes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "key": dict(self.key),
            "axes": {axis: list(bounds) for axis, bounds in sorted(self.axes.items())},
            "query_count": self.query_count,
            "shape_count": self.shape_count,
            "kinds": dict(self.kinds),
            "reason": self.reason,
            "examples": list(self.examples),
        }


class MissingShapeReport:
    """Aggregates operator queries that no measured table could answer.

    Misses are grouped by ``(table, discrete key)``; within a group the report
    keeps the requested range of every interpolation axis, how often the group
    was hit, why (``MissingOperatorDataError.kind``) and a few example keys.
    Sharing one instance between the operator backend and the collective model
    of a worker gives a single checklist covering compute and communication.
    """

    def __init__(self, *, max_groups: int = 4096, max_examples: int = 3, max_shapes_per_group: int = 256) -> None:
        self.max_groups = max_groups
        self.max_examples = max_examples
        self.max_shapes_per_group = max_shapes_per_group
        self.groups: dict[tuple, MissingShapeGroup] = {}
        self.dropped_queries = 0

    def record(self, table: str, key: Mapping[str, Any], reason: str, kind: str = "discrete_key") -> None:
        spec = TABLE_SPECS.get(table)
        axis_names = spec.axis_names if spec is not None else ()
        discrete = {name: value for name, value in key.items() if name not in axis_names}
        identity = (table, tuple(sorted(discrete.items())))
        group = self.groups.get(identity)
        if group is None:
            if len(self.groups) >= self.max_groups:
                self.dropped_queries += 1
                return
            group = self.groups[identity] = MissingShapeGroup(table=table, key=discrete, reason=reason)
        group.query_count += 1
        group.kinds[kind] += 1
        if len(group._shapes) < self.max_shapes_per_group:
            group._shapes.add(tuple(sorted(key.items())))
        for name in axis_names:
            value = key.get(name)
            if value is None:
                continue
            bounds = group.axes.get(name)
            if bounds is None:
                group.axes[name] = [value, value]
            else:
                bounds[0], bounds[1] = min(bounds[0], value), max(bounds[1], value)
        if len(group.examples) < self.max_examples and dict(key) not in group.examples:
            group.examples.append(dict(key))

    def record_error(self, error: MissingOperatorDataError) -> None:
        self.record(error.table_name, error.key, error.reason, error.kind)

    # -- aggregation ------------------------------------------------------------

    @property
    def group_count(self) -> int:
        return len(self.groups)

    @property
    def shape_count(self) -> int:
        return sum(group.shape_count for group in self.groups.values())

    @property
    def query_count(self) -> int:
        return sum(group.query_count for group in self.groups.values()) + self.dropped_queries

    def merge(self, other: "MissingShapeReport") -> "MissingShapeReport":
        """Union of two reports (per-worker reports are merged for the result file)."""
        result = MissingShapeReport(
            max_groups=max(self.max_groups, other.max_groups),
            max_examples=max(self.max_examples, other.max_examples),
            max_shapes_per_group=max(self.max_shapes_per_group, other.max_shapes_per_group),
        )
        for report in (self, other):
            result.dropped_queries += report.dropped_queries
            for identity, group in report.groups.items():
                target = result.groups.get(identity)
                if target is None:
                    target = result.groups[identity] = MissingShapeGroup(
                        table=group.table, key=dict(group.key), reason=group.reason
                    )
                target.query_count += group.query_count
                target.kinds.update(group.kinds)
                for name, bounds in group.axes.items():
                    existing = target.axes.get(name)
                    target.axes[name] = (
                        list(bounds) if existing is None else [min(existing[0], bounds[0]), max(existing[1], bounds[1])]
                    )
                for shape in group._shapes:
                    if len(target._shapes) >= result.max_shapes_per_group:
                        break
                    target._shapes.add(shape)
                for example in group.examples:
                    if len(target.examples) >= result.max_examples:
                        break
                    if example not in target.examples:
                        target.examples.append(example)
        return result

    def records(self) -> list[dict[str, Any]]:
        """Groups as plain dicts, most frequently missed first."""
        ordered = sorted(self.groups.values(), key=lambda g: (-g.query_count, g.table, sorted(g.key.items())))
        return [group.to_dict() for group in ordered]

    def per_table(self) -> dict[str, dict[str, int]]:
        summary: dict[str, dict[str, int]] = {}
        for group in self.groups.values():
            entry = summary.setdefault(group.table, {"groups": 0, "shapes": 0, "queries": 0})
            entry["groups"] += 1
            entry["shapes"] += group.shape_count
            entry["queries"] += group.query_count
        return dict(sorted(summary.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_count": self.group_count,
            "shape_count": self.shape_count,
            "query_count": self.query_count,
            "per_table": self.per_table(),
            "groups": self.records(),
        }


def coverage_report(
    package: OperatorDataPackage,
    manifest: ShapeManifest,
    *,
    policy: LookupPolicy | None = None,
) -> CoverageReport:
    lookup = OperatorLookup(package, policy)
    report = CoverageReport()
    for table, keys in manifest.keys.items():
        counter = report.per_table.setdefault(table, Counter())
        if not lookup.has_table(table):
            counter["missing"] += len(keys)
            for key in keys:
                report.missing.append({"table": table, "key": dict(key), "reason": "table absent"})
            continue
        for key in keys:
            try:
                result = lookup.lookup(table, key)
            except MissingOperatorDataError as exc:
                counter["missing"] += 1
                report.missing.append({"table": table, "key": dict(key), "reason": exc.reason})
                continue
            counter[result.match_type] += 1
    return report
