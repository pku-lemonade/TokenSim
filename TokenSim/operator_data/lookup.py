from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from TokenSim.dtypes import canonicalize_dtype
from TokenSim.operator_data.package import OperatorDataPackage
from TokenSim.operator_data.schema import TABLE_SPECS, dtype_role_for_field, validate_batch_key


class MissingOperatorDataError(LookupError):
    def __init__(self, table_name: str, key: Mapping[str, Any], reason: str) -> None:
        self.table_name = table_name
        self.key = dict(key)
        self.reason = reason
        super().__init__(f"{table_name} lookup failed for key={self.key}: {reason}")


@dataclass(frozen=True)
class LookupResult:
    latency_us: float
    match_type: Literal["exact", "interpolated"]
    source_id: str
    key: Mapping[str, Any]
    interpolation: Mapping[str, Any] | None = None
    values: Mapping[str, Any] = field(default_factory=dict)


class OperatorDataLookup:
    def __init__(self, package: OperatorDataPackage) -> None:
        self.package = package
        self._exact: dict[str, dict[tuple[Any, ...], Mapping[str, Any]]] = {}
        for table_name, rows in package.tables.items():
            spec = TABLE_SPECS[table_name]
            self._exact[table_name] = {
                tuple(row[field] for field in spec.key_fields): row for row in rows
            }

    def lookup(self, table_name: str, key: Mapping[str, Any]) -> LookupResult:
        if table_name not in TABLE_SPECS:
            raise MissingOperatorDataError(table_name, key, "unknown table")
        spec = TABLE_SPECS[table_name]
        normalized = self._normalize_key(table_name, key)
        exact_key = tuple(normalized[field] for field in spec.key_fields)
        if table_name not in self._exact:
            raise MissingOperatorDataError(table_name, normalized,
                "batch table missing; regenerate operator data for the execution shapes")
        row = self._exact[table_name].get(exact_key)
        if row is not None:
            return self._result_from_row(normalized, row)
        if spec.interpolation_axis is None:
            raise MissingOperatorDataError(
                table_name,
                normalized,
                "exact record missing and interpolation is disabled",
            )
        return self._interpolate(table_name, normalized)

    def link_profile(self, key: Mapping[str, Any]) -> LookupResult:
        result = self.lookup("communication", key)
        values = result.values
        return LookupResult(
            latency_us=float(values["fixed_latency_us"]),
            match_type=result.match_type,
            source_id=result.source_id,
            key=result.key,
            interpolation=result.interpolation,
            values=values,
        )

    @staticmethod
    def _normalize_key(table_name: str, key: Mapping[str, Any]) -> dict[str, Any]:
        spec = TABLE_SPECS[table_name]
        missing = [field for field in spec.key_fields if field not in key]
        extra = sorted(set(key) - set(spec.key_fields))
        if missing or extra:
            raise MissingOperatorDataError(
                table_name,
                key,
                f"key fields mismatch; missing={missing}, extra={extra}",
            )
        normalized: dict[str, Any] = {}
        for field in spec.key_fields:
            value = key[field]
            dtype_role = dtype_role_for_field(field, key)
            if dtype_role is not None:
                try:
                    value = canonicalize_dtype(str(value), role=dtype_role)
                except ValueError as exc:
                    raise MissingOperatorDataError(
                        table_name,
                        key,
                        f"invalid dtype field {field!r}: {exc}",
                    ) from exc
            if isinstance(value, str):
                value = value.strip().lower()
            normalized[field] = value
        try:
            validate_batch_key(table_name, normalized)
        except ValueError as exc:
            raise MissingOperatorDataError(table_name, normalized, str(exc)) from exc
        return normalized

    @staticmethod
    def _result_from_row(
        key: Mapping[str, Any], row: Mapping[str, Any]
    ) -> LookupResult:
        latency_us = float(row.get("latency_us", row.get("fixed_latency_us", 0.0)))
        return LookupResult(
            latency_us=latency_us,
            match_type="exact",
            source_id=str(row["source_id"]),
            key=dict(key),
            values=dict(row),
        )

    def _interpolate(self, table_name: str, key: Mapping[str, Any]) -> LookupResult:
        spec = TABLE_SPECS[table_name]
        axis = spec.interpolation_axis
        target = key[axis]
        peers = [
            row
            for row in self.package.tables[table_name]
            if all(row[field] == key[field] for field in spec.discrete_fields)
        ]
        lower = [row for row in peers if row[axis] < target]
        upper = [row for row in peers if row[axis] > target]
        if not lower or not upper:
            known = sorted({row[axis] for row in peers})
            raise MissingOperatorDataError(
                table_name,
                key,
                f"no bounded {axis} interpolation; available values={known}",
            )
        left = max(lower, key=lambda row: row[axis])
        right = min(upper, key=lambda row: row[axis])
        weight = (target - left[axis]) / (right[axis] - left[axis])
        latency_us = float(left["latency_us"]) + weight * (
            float(right["latency_us"]) - float(left["latency_us"])
        )
        source_id = f"interpolate:{left['source_id']}:{right['source_id']}"
        interpolation = {
            "axis": axis,
            "left_value": left[axis],
            "right_value": right[axis],
            "left_source_id": left["source_id"],
            "right_source_id": right["source_id"],
            "left_weight": 1.0 - weight,
            "right_weight": weight,
        }
        values = {"latency_us": latency_us, "source_id": source_id}
        # Preserve decomposition only when both endpoints have the same method.
        for name in ("timing_method", "calibration_applied"):
            if name in left and left.get(name) == right.get(name):
                values[name] = left[name]
        if (values.get("calibration_applied") is True
                and left.get("fixed_overhead_us") is not None
                and right.get("fixed_overhead_us") is not None):
            values["fixed_overhead_us"] = (1 - weight) * left["fixed_overhead_us"] + weight * right["fixed_overhead_us"]
        return LookupResult(
            latency_us=latency_us,
            match_type="interpolated",
            source_id=source_id,
            key=dict(key),
            interpolation=interpolation,
            values=values,
        )
