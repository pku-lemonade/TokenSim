"""Primitive operator resolution shared by AF compute providers."""

from typing import Any, Iterable, Mapping
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import DeviceSpec
from TokenSim.operator_data.analytical import Calibration, analytical_model_for
from TokenSim.operator_data.coverage import MissingShapeReport
from TokenSim.operator_data.lookup import (
    FALLBACK_POLICIES,
    LookupPolicy,
    MissingOperatorDataError,
    OperatorLookup,
)
from TokenSim.operator_data.package import OperatorDataPackage
from TokenSim.operator_data.schema import TABLE_SPECS, normalize_table_key_value
from .stats import OperatorStats, OperatorEstimate


class OperatorResolver:
    """Answer AF primitive queries with operator-table provenance and fallback."""

    _CACHE_LIMIT = 262144

    def __init__(
        self,
        *,
        device: DeviceSpec,
        package: OperatorDataPackage | None = None,
        fallback: str = "table_first",
        lookup_policy: LookupPolicy | None = None,
        gated: bool = True,
        missing_report: MissingShapeReport | None = None,
    ) -> None:
        if fallback not in FALLBACK_POLICIES:
            raise ConfigurationError(
                f"fallback must be one of {FALLBACK_POLICIES}, got {fallback!r}"
            )
        if fallback == "table_only" and package is None:
            raise ConfigurationError(
                f"latency backend 'table_only' requires operator data for device {device.device_id!r}"
            )
        self.device = device
        self.fallback = fallback
        self.gated = gated
        self.package = package if fallback != "analytical_only" else None
        self.analytical = analytical_model_for(device.family)
        # Out-of-range table queries keep the boundary row's measured efficiency
        # and follow the analytical model's growth (see LookupPolicy).
        self.lookup = (
            OperatorLookup(
                self.package,
                lookup_policy,
                analytical_scaler=self.analytical_reference_us,
            )
            if self.package is not None
            else None
        )
        self.stats = OperatorStats(missing=missing_report or MissingShapeReport())
        self._cache: dict[tuple, OperatorEstimate] = {}
        self._calibrations = {
            table: Calibration.from_mapping(values)
            for table, values in (
                self.package.meta.calibration.items() if self.package else []
            )
        }

    @property
    def dataset_version(self) -> str | None:
        return self.package.dataset_version if self.package else None

    @property
    def operator_backend(self) -> str:
        return self.package.backend if self.package else "analytical"

    def query(self, table: str, key: Mapping[str, Any]) -> OperatorEstimate:
        if table not in TABLE_SPECS:
            raise MissingOperatorDataError(table, key, "unknown table")
        normalized_key = dict(key)
        for field in TABLE_SPECS[table].key_fields:
            if field in normalized_key:
                normalized_key[field] = normalize_table_key_value(
                    table, field, normalized_key[field]
                )
        identity = (table, tuple(sorted(normalized_key.items())))
        cached = self._cache.get(identity)
        if cached is not None:
            self.stats.record(table, cached.match_type)
            return cached
        estimate: OperatorEstimate | None = None
        if self.lookup is not None and self.fallback != "analytical_only":
            try:
                result = self.lookup.lookup(table, normalized_key)
                estimate = OperatorEstimate(
                    result.latency_us, result.match_type, result.source_id
                )
            except MissingOperatorDataError as exc:
                if self.fallback == "table_only":
                    raise
                self.stats.record_missing(exc)
        if estimate is None and self.fallback == "table_only":
            raise MissingOperatorDataError(
                table, normalized_key, "no operator package for strict table lookup"
            )
        if estimate is None:
            estimate = self.analytical_estimate(table, normalized_key, gated=self.gated)
        self._remember(identity, estimate)
        self.stats.record(table, estimate.match_type)
        return estimate

    def query_moe(
        self, base_key: Mapping[str, Any], distributions: Iterable[str]
    ) -> OperatorEstimate:
        """Look up a MoE row trying each routing ``distribution`` label in turn."""
        distributions = tuple(distributions) or ("uniform",)
        last_error: MissingOperatorDataError | None = None
        if self.lookup is not None and self.fallback != "analytical_only":
            for distribution in distributions:
                key = {**base_key, "distribution": distribution}
                cached = self._cache.get(("moe", tuple(sorted(key.items()))))
                if cached is not None:
                    self.stats.record("moe", cached.match_type)
                    return cached
                try:
                    result = self.lookup.lookup("moe", key)
                except MissingOperatorDataError as exc:
                    last_error = exc
                    continue
                estimate = OperatorEstimate(
                    result.latency_us, result.match_type, result.source_id
                )
                self._remember(("moe", tuple(sorted(key.items()))), estimate)
                self.stats.record("moe", estimate.match_type)
                return estimate
        key = {**base_key, "distribution": distributions[0]}
        if last_error is None:
            last_error = MissingOperatorDataError(
                "moe", key, "no moe table", kind="table_absent"
            )
        if self.fallback == "table_only":
            raise last_error
        self.stats.record_missing(last_error)
        estimate = self.analytical_estimate("moe", key, gated=self.gated)
        self.stats.record("moe", estimate.match_type)
        return estimate

    def analytical_estimate(
        self, table: str, key: Mapping[str, Any], **context: Any
    ) -> OperatorEstimate:
        calibration = self._calibrations.get(table) or self._calibrations.get("default")
        result = self.analytical.estimate(
            table, key, self.device, calibration, **context
        )
        return OperatorEstimate(result.latency_us, "analytical", result.source_id)

    def analytical_reference_us(
        self, table: str, key: Mapping[str, Any]
    ) -> float | None:
        """Uncalibrated analytical latency used as the growth reference for extrapolation."""
        try:
            return self.analytical.estimate(
                table, key, self.device, Calibration(), gated=self.gated
            ).latency_us
        except Exception:
            return None

    def _remember(self, identity: tuple, estimate: OperatorEstimate) -> None:
        if len(self._cache) >= self._CACHE_LIMIT:
            self._cache.clear()
        self._cache[identity] = estimate
