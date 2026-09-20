from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from typing import Callable

from TokenSim.hardware.device import normalize_dtype
from TokenSim.operator_data.package import OperatorDataPackage
from TokenSim.operator_data.schema import TABLE_SPECS, AxisSpec, TableSpec

MATCH_TYPES = ("exact", "interpolated", "extrapolated")
EXTRAPOLATION_MODES = ("none", "hold", "scale", "analytical")
# How a consumer of the tables (operator backend, collective model) treats a
# query that no measured row answers: fall back to the analytical model and
# report the miss, raise, or never consult tables at all.
FALLBACK_POLICIES = ("table_first", "table_only", "analytical_only")
# Why a lookup failed; the missing-shape report groups by these.
MISSING_KINDS = (
    "table_absent",          # the package ships no rows for this table
    "discrete_key",          # rows exist, none for this dtype/heads/... combination
    "out_of_range",          # an axis value lies beyond the extrapolation bound
    "extrapolation_disabled",
    "interpolation_disabled",
    "no_measured_mode",      # the configured algorithm/mode is analytical by design
    "malformed_key",
    "unknown_table",
)

# Returns the analytical (model-based) latency in microseconds for a fully
# specified key, or ``None`` when the model cannot price that key.
AnalyticalScaler = Callable[[str, Mapping[str, Any]], "float | None"]


class MissingOperatorDataError(LookupError):
    def __init__(
        self, table_name: str, key: Mapping[str, Any], reason: str, kind: str = "discrete_key"
    ) -> None:
        if kind not in MISSING_KINDS:
            raise ValueError(f"unknown missing-data kind {kind!r}")
        self.table_name = table_name
        self.key = dict(key)
        self.reason = reason
        self.kind = kind
        super().__init__(f"{table_name} lookup failed for key={self.key}: {reason}")


@dataclass(frozen=True)
class LookupResult:
    latency_us: float
    match_type: str
    source_id: str
    key: Mapping[str, Any]
    table: str
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LookupPolicy:
    """Controls how far a query may stray from measured points.

    ``interpolate``: allow interpolation along the table's declared axes when
    all discrete keys match.

    ``extrapolate``: what to do when a query lies outside the measured range on
    an axis (the boundary row is always the starting point):

    * ``analytical`` (default): keep the boundary row's measured efficiency and
      let the analytical model carry the growth, i.e.
      ``latency = measured(boundary) * analytical(target) / analytical(boundary)``.
      This is the strategy AIConfigurator uses past its collected range. It
      needs an :data:`AnalyticalScaler`; without one it behaves like ``scale``.
    * ``scale``: multiply the boundary latency by ``target / boundary`` on
      logarithmic axes when the target is larger (work grows with the axis).
    * ``hold``: return the boundary latency unchanged.
    * ``none``: raise :class:`MissingOperatorDataError`.

    ``max_extrapolation_ratio`` bounds how far past the boundary a query may go
    in either direction before it is treated as missing.
    """

    interpolate: bool = True
    extrapolate: str = "analytical"
    max_extrapolation_ratio: float = 16.0

    def __post_init__(self) -> None:
        if self.extrapolate not in EXTRAPOLATION_MODES:
            raise ValueError(f"extrapolate must be one of {EXTRAPOLATION_MODES}")


class OperatorLookup:
    """Exact-first lookup with controlled multi-axis interpolation.

    Rows are indexed by their discrete key; within one discrete key the axis
    values form a (possibly ragged) grid. Interpolation walks the axes in the
    order declared by the :class:`TableSpec`, blending in log or linear space
    per axis, so a GEMM query resolves ``k`` then ``n`` then ``m``.
    """

    def __init__(
        self,
        package: OperatorDataPackage,
        policy: LookupPolicy | None = None,
        analytical_scaler: AnalyticalScaler | None = None,
    ) -> None:
        self.package = package
        self.policy = policy or LookupPolicy()
        self.analytical_scaler = analytical_scaler
        # The index is owned by the (immutable) package and shared by every
        # lookup created for it; see OperatorDataPackage.lookup_index.
        index = package.lookup_index
        self._exact: Mapping[str, Mapping[tuple[Any, ...], Mapping[str, Any]]] = {
            name: table.exact for name, table in index.items()
        }
        self._grids: Mapping[str, Mapping[tuple[Any, ...], Any]] = {
            name: table.grids for name, table in index.items()
        }

    # -- public API --------------------------------------------------------

    def has_table(self, table_name: str) -> bool:
        return bool(self._exact.get(table_name))

    def row_count(self, table_name: str) -> int:
        return len(self._exact.get(table_name, {}))

    def lookup(
        self,
        table_name: str,
        key: Mapping[str, Any],
        scaler: AnalyticalScaler | None = None,
    ) -> LookupResult:
        """Resolve ``key``; ``scaler`` overrides the instance-level analytical scaler."""
        if table_name not in TABLE_SPECS:
            raise MissingOperatorDataError(table_name, key, "unknown table", kind="unknown_table")
        spec = TABLE_SPECS[table_name]
        normalized = self._normalize_key(spec, key)
        exact = self._exact.get(table_name)
        if not exact:
            raise MissingOperatorDataError(
                table_name, normalized, "table has no rows in this package", kind="table_absent"
            )
        row = exact.get(tuple(normalized[f] for f in spec.key_fields))
        if row is not None:
            return LookupResult(
                latency_us=float(row["latency_us"]),
                match_type="exact",
                source_id=str(row["source_id"]),
                key=normalized,
                table=table_name,
            )
        if not self.policy.interpolate or not spec.axes:
            raise MissingOperatorDataError(
                table_name,
                normalized,
                "exact record missing and interpolation is disabled",
                kind="interpolation_disabled",
            )
        discrete = tuple(normalized[f] for f in spec.discrete_fields)
        grid = self._grids[table_name].get(discrete)
        if grid is None:
            available = self._describe_discrete(table_name, spec)
            raise MissingOperatorDataError(
                table_name,
                normalized,
                f"no rows share the discrete key {dict(zip(spec.discrete_fields, discrete))}; "
                f"available discrete keys: {available}",
                kind="discrete_key",
            )
        latency, sources, flags = self._resolve(
            spec, list(spec.axes), grid, normalized, table_name, scaler or self.analytical_scaler
        )
        match_type = "extrapolated" if "extrapolated" in flags else "interpolated"
        source_id = "+".join(sorted(sources)) if len(sources) > 1 else next(iter(sources))
        return LookupResult(
            latency_us=latency,
            match_type=match_type,
            source_id=source_id,
            key=normalized,
            table=table_name,
            detail={"flags": sorted(flags)},
        )

    def axis_values(self, table_name: str, key: Mapping[str, Any], axis: str) -> list[Any]:
        """Measured values of ``axis`` for the discrete part of ``key`` (diagnostics)."""
        spec = TABLE_SPECS[table_name]
        normalized = self._normalize_key(spec, {**key, axis: key.get(axis, 0)})
        discrete = tuple(normalized[f] for f in spec.discrete_fields)
        values: set[Any] = set()
        for row in self.package.tables.get(table_name, ()):
            if tuple(row[f] for f in spec.discrete_fields) == discrete:
                values.add(row[axis])
        return sorted(values)

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _normalize_key(spec: TableSpec, key: Mapping[str, Any]) -> dict[str, Any]:
        missing = [f for f in spec.key_fields if f not in key]
        if missing:
            raise MissingOperatorDataError(
                spec.name, key, f"key is missing fields {missing}", kind="malformed_key"
            )
        normalized: dict[str, Any] = {}
        for f in spec.key_fields:
            value = key[f]
            if f in spec.dtype_fields:
                value = normalize_dtype(str(value))
            elif isinstance(value, str):
                value = value.strip().lower()
            elif isinstance(value, float) and value.is_integer():
                value = int(value)
            normalized[f] = value
        return normalized

    def _describe_discrete(self, table_name: str, spec: TableSpec) -> list[dict[str, Any]]:
        keys = list(self._grids[table_name])[:8]
        return [dict(zip(spec.discrete_fields, key)) for key in keys]

    def _resolve(
        self,
        spec: TableSpec,
        axes: list[AxisSpec],
        node: Any,
        key: Mapping[str, Any],
        table_name: str,
        scaler: AnalyticalScaler | None,
    ) -> tuple[float, set[str], set[str]]:
        if not axes:
            row = node
            return float(row["latency_us"]), {str(row["source_id"])}, set()
        axis, rest = axes[0], axes[1:]
        target = key[axis.name]
        if target in node:
            return self._resolve(spec, rest, node[target], key, table_name, scaler)
        values = sorted(node)
        lower = [v for v in values if v < target]
        upper = [v for v in values if v > target]
        if lower and upper:
            lo, hi = lower[-1], upper[0]
            lat_lo, src_lo, flag_lo = self._resolve(spec, rest, node[lo], key, table_name, scaler)
            lat_hi, src_hi, flag_hi = self._resolve(spec, rest, node[hi], key, table_name, scaler)
            weight = _blend_weight(axis, lo, hi, target)
            latency = lat_lo + weight * (lat_hi - lat_lo)
            return latency, src_lo | src_hi, flag_lo | flag_hi | {"interpolated"}
        # Out of range on this axis.
        mode = self.policy.extrapolate
        if mode == "none":
            raise MissingOperatorDataError(
                table_name,
                key,
                f"{axis.name}={target} outside measured range {values[0]}..{values[-1]} "
                "and extrapolation is disabled",
                kind="extrapolation_disabled",
            )
        boundary = values[-1] if upper == [] else values[0]
        ratio = float(target) / float(boundary) if boundary else 1.0
        max_ratio = axis.max_extrapolation_ratio or self.policy.max_extrapolation_ratio
        if ratio > max_ratio or ratio < 1.0 / max_ratio:
            raise MissingOperatorDataError(
                table_name,
                key,
                f"{axis.name}={target} is {ratio:.2f}x the measured boundary {boundary}; "
                f"exceeds max_extrapolation_ratio={max_ratio}",
                kind="out_of_range",
            )
        latency, sources, flags = self._resolve(spec, rest, node[boundary], key, table_name, scaler)
        flags = flags | {"extrapolated"}
        if mode == "hold":
            return latency, sources, flags | {"held"}
        if mode == "analytical" and scaler is not None:
            growth = self._analytical_growth(table_name, key, axis.name, boundary, scaler)
            if growth is not None:
                return latency * growth, sources, flags | {"analytical_scaled"}
        # ``scale`` (or ``analytical`` without a usable model): work grows with
        # the axis value on logarithmic axes; below the boundary the fixed cost
        # dominates, so only scale upward.
        if axis.scale == "log" and ratio > 1.0:
            latency *= ratio
        return latency, sources, flags | {"linear_scaled"}

    @staticmethod
    def _analytical_growth(
        table_name: str,
        key: Mapping[str, Any],
        axis_name: str,
        boundary: Any,
        scaler: AnalyticalScaler,
    ) -> float | None:
        """``analytical(target) / analytical(boundary)`` or ``None`` if unavailable."""
        boundary_key = {**key, axis_name: boundary}
        try:
            at_target = scaler(table_name, key)
            at_boundary = scaler(table_name, boundary_key)
        except Exception:
            return None
        if at_target is None or at_boundary is None:
            return None
        if not (math.isfinite(at_target) and math.isfinite(at_boundary)) or at_boundary <= 0 or at_target <= 0:
            return None
        return at_target / at_boundary


def _blend_weight(axis: AxisSpec, lo: Any, hi: Any, target: Any) -> float:
    if axis.scale == "log" and lo > 0 and hi > 0 and target > 0:
        return (math.log(target) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return (float(target) - float(lo)) / (float(hi) - float(lo))
