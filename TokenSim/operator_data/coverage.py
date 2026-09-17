"""Coverage report: which manifest queries a package answers, and how."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from TokenSim.operator_data.lookup import LookupPolicy, MissingOperatorDataError, OperatorLookup
from TokenSim.operator_data.manifest import ShapeManifest
from TokenSim.operator_data.package import OperatorDataPackage


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
