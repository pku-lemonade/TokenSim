"""Operator dataset metadata, provenance and validation errors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from TokenSim.errors import ConfigurationError
from TokenSim.operator_data.schema import LATENCY_UNIT, SCHEMA_VERSION, TABLE_SPECS


META_FILENAME = "generation_meta.yaml"


class OperatorDataValidationError(ConfigurationError):
    pass


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    grade: str
    method: str  # measured | imported | public_benchmark | analytical | assumption
    reference: str = ""
    notes: str = ""
    device: str = ""
    backend: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "grade": self.grade,
            "method": self.method,
            "reference": self.reference,
            "notes": self.notes,
            "device": self.device,
            "backend": self.backend,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], context: str) -> "SourceRecord":
        for required in ("source_id", "grade", "method"):
            if required not in raw:
                raise OperatorDataValidationError(
                    f"{context}: source is missing {required!r}"
                )
        grade = str(raw["grade"]).upper()
        if grade not in {"A", "B", "C", "D"}:
            raise OperatorDataValidationError(f"{context}: grade must be A/B/C/D")
        method = str(raw["method"])
        if method not in {
            "measured",
            "imported",
            "public_benchmark",
            "analytical",
            "assumption",
        }:
            raise OperatorDataValidationError(
                f"{context}: unsupported method {method!r}"
            )
        return cls(
            source_id=str(raw["source_id"]),
            grade=grade,
            method=method,
            reference=str(raw.get("reference", "")),
            notes=str(raw.get("notes", "")),
            device=str(raw.get("device", "")),
            backend=str(raw.get("backend", "")),
        )


@dataclass(frozen=True)
class PackageMeta:
    dataset_version: str
    device_id: str
    backend: str
    framework_version: str = ""
    sources: Mapping[str, SourceRecord] = field(default_factory=dict)
    calibration: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    table_rows: Mapping[str, int] = field(default_factory=dict)
    notes: str = ""
    schema_version: int = SCHEMA_VERSION
    latency_unit: str = LATENCY_UNIT
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "latency_unit": self.latency_unit,
            "dataset_version": self.dataset_version,
            "device_id": self.device_id,
            "backend": self.backend,
            "framework_version": self.framework_version,
            "notes": self.notes,
            "sources": [source.to_dict() for source in self.sources.values()],
            "calibration": {k: dict(v) for k, v in self.calibration.items()},
            "tables": {k: {"rows": v} for k, v in self.table_rows.items()},
            **dict(self.extra),
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], path: Path) -> "PackageMeta":
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise OperatorDataValidationError(
                f"{path}: unsupported schema_version={raw.get('schema_version')!r}; expected {SCHEMA_VERSION}"
            )
        if raw.get("latency_unit", LATENCY_UNIT) != LATENCY_UNIT:
            raise OperatorDataValidationError(
                f"{path}: latency_unit must be {LATENCY_UNIT!r}"
            )
        for required in ("dataset_version", "device_id", "backend"):
            if not raw.get(required):
                raise OperatorDataValidationError(f"{path}: {required} is required")
        sources_raw = raw.get("sources") or []
        if isinstance(sources_raw, Mapping):
            sources_raw = [{"source_id": k, **v} for k, v in sources_raw.items()]
        sources = {}
        for item in sources_raw:
            record = SourceRecord.from_mapping(item, str(path))
            if record.source_id in sources:
                raise OperatorDataValidationError(
                    f"{path}: duplicate source_id {record.source_id!r}"
                )
            sources[record.source_id] = record
        tables = raw.get("tables") or {}
        if not isinstance(tables, Mapping):
            raise OperatorDataValidationError(
                f"{path}: tables must be a manifest object"
            )
        for name, description in tables.items():
            count = (
                description.get("rows") if isinstance(description, Mapping) else None
            )
            if (
                name not in TABLE_SPECS
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                raise OperatorDataValidationError(
                    f"{path}: invalid table manifest entry {name!r}"
                )
        return cls(
            dataset_version=str(raw["dataset_version"]),
            device_id=str(raw["device_id"]),
            backend=str(raw["backend"]),
            framework_version=str(raw.get("framework_version", "")),
            sources=sources,
            calibration={
                str(k): dict(v) for k, v in (raw.get("calibration") or {}).items()
            },
            table_rows={str(k): int(v.get("rows", 0)) for k, v in tables.items()},
            notes=str(raw.get("notes", "")),
            extra={
                str(k): v
                for k, v in raw.items()
                if k
                not in {
                    "schema_version",
                    "latency_unit",
                    "dataset_version",
                    "device_id",
                    "backend",
                    "framework_version",
                    "notes",
                    "sources",
                    "calibration",
                    "tables",
                }
            },
        )
