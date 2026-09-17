from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from TokenSim.errors import ConfigurationError
from TokenSim.hardware._yaml import load_yaml_mapping
from TokenSim.hardware.device import normalize_dtype
from TokenSim.operator_data.schema import (
    INTEGER_FIELDS,
    LATENCY_UNIT,
    SCHEMA_VERSION,
    TABLE_SPECS,
    TableSpec,
)

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
                raise OperatorDataValidationError(f"{context}: source is missing {required!r}")
        grade = str(raw["grade"]).upper()
        if grade not in {"A", "B", "C", "D"}:
            raise OperatorDataValidationError(f"{context}: grade must be A/B/C/D")
        method = str(raw["method"])
        if method not in {"measured", "imported", "public_benchmark", "analytical", "assumption"}:
            raise OperatorDataValidationError(f"{context}: unsupported method {method!r}")
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
            raise OperatorDataValidationError(f"{path}: latency_unit must be {LATENCY_UNIT!r}")
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
                raise OperatorDataValidationError(f"{path}: duplicate source_id {record.source_id!r}")
            sources[record.source_id] = record
        tables = raw.get("tables") or {}
        return cls(
            dataset_version=str(raw["dataset_version"]),
            device_id=str(raw["device_id"]),
            backend=str(raw["backend"]),
            framework_version=str(raw.get("framework_version", "")),
            sources=sources,
            calibration={str(k): dict(v) for k, v in (raw.get("calibration") or {}).items()},
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


@dataclass(frozen=True)
class OperatorDataPackage:
    root: Path | None
    meta: PackageMeta
    tables: Mapping[str, tuple[Mapping[str, Any], ...]]
    analysis: Mapping[str, tuple[Mapping[str, Any], ...]] = field(default_factory=dict)

    @property
    def dataset_version(self) -> str:
        return self.meta.dataset_version

    @property
    def device_id(self) -> str:
        return self.meta.device_id

    @property
    def backend(self) -> str:
        return self.meta.backend

    # -- loading -----------------------------------------------------------

    @classmethod
    def load(cls, root: str | Path, *, strict_sources: bool = True) -> "OperatorDataPackage":
        root_path = Path(root)
        if not root_path.is_dir():
            raise OperatorDataValidationError(f"operator-data package directory not found: {root_path}")
        meta_path = root_path / META_FILENAME
        if not meta_path.is_file():
            raise OperatorDataValidationError(f"missing {META_FILENAME} in {root_path}")
        meta = PackageMeta.from_mapping(load_yaml_mapping(meta_path), meta_path)
        tables: dict[str, tuple[Mapping[str, Any], ...]] = {}
        analysis: dict[str, tuple[Mapping[str, Any], ...]] = {}
        for table_name, spec in TABLE_SPECS.items():
            rows = _read_table(root_path, spec.perf_filename)
            if rows is None:
                continue
            checked = validate_rows(table_name, rows, meta, strict_sources=strict_sources, path=root_path / spec.perf_filename)
            tables[table_name] = tuple(checked)
            analysis_rows = _read_table(root_path, spec.analysis_filename)
            if analysis_rows is not None:
                analysis[table_name] = tuple(analysis_rows)
        if not tables:
            raise OperatorDataValidationError(f"{root_path}: package has no *_perf tables")
        return cls(root=root_path, meta=meta, tables=tables, analysis=analysis)

    @classmethod
    def from_rows(
        cls,
        meta: PackageMeta,
        tables: Mapping[str, Iterable[Mapping[str, Any]]],
        analysis: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
        *,
        strict_sources: bool = True,
    ) -> "OperatorDataPackage":
        checked = {
            name: tuple(validate_rows(name, list(rows), meta, strict_sources=strict_sources))
            for name, rows in tables.items()
        }
        return cls(
            root=None,
            meta=meta,
            tables=checked,
            analysis={name: tuple(rows) for name, rows in (analysis or {}).items()},
        )

    # -- writing -----------------------------------------------------------

    def write(self, root: str | Path) -> Path:
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        table_rows = {}
        for table_name, rows in self.tables.items():
            spec = TABLE_SPECS[table_name]
            _write_table(root_path / spec.perf_filename, rows, spec.required_fields)
            table_rows[table_name] = len(rows)
            analysis_rows = self.analysis.get(table_name)
            if analysis_rows:
                _write_table(root_path / spec.analysis_filename, analysis_rows, None)
        meta_dict = self.meta.to_dict()
        meta_dict["tables"] = {k: {"rows": v} for k, v in table_rows.items()}
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise OperatorDataValidationError("writing packages requires PyYAML") from exc
        (root_path / META_FILENAME).write_text(
            yaml.safe_dump(meta_dict, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        return root_path

    def summary(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "backend": self.backend,
            "dataset_version": self.dataset_version,
            "tables": {name: len(rows) for name, rows in self.tables.items()},
            "sources": sorted(self.meta.sources),
        }


# -- row validation -----------------------------------------------------------


def validate_rows(
    table_name: str,
    rows: list[Mapping[str, Any]],
    meta: PackageMeta,
    *,
    strict_sources: bool = True,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    spec: TableSpec = TABLE_SPECS[table_name]
    where = str(path) if path else table_name
    checked: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for index, row in enumerate(rows):
        missing = [f for f in spec.required_fields if f not in row]
        if missing:
            raise OperatorDataValidationError(f"{where}: row {index} is missing fields {missing}")
        clean: dict[str, Any] = {}
        for f in spec.key_fields:
            value = row[f]
            if f in INTEGER_FIELDS:
                try:
                    number = int(value)
                except (TypeError, ValueError) as exc:
                    raise OperatorDataValidationError(f"{where}: row {index} field {f} must be an integer") from exc
                if number < 0:
                    raise OperatorDataValidationError(f"{where}: row {index} field {f} must be non-negative")
                clean[f] = number
            elif f in spec.dtype_fields:
                clean[f] = normalize_dtype(str(value))
            else:
                clean[f] = str(value).strip().lower()
        try:
            latency = float(row["latency_us"])
        except (TypeError, ValueError) as exc:
            raise OperatorDataValidationError(f"{where}: row {index} latency_us must be numeric") from exc
        if not math.isfinite(latency) or latency < 0:
            raise OperatorDataValidationError(f"{where}: row {index} latency_us must be finite and non-negative")
        clean["latency_us"] = latency
        source_id = str(row["source_id"])
        if strict_sources and meta.sources and source_id not in meta.sources:
            raise OperatorDataValidationError(
                f"{where}: row {index} references unknown source_id {source_id!r}"
            )
        clean["source_id"] = source_id
        for extra_field in ("kernel", "grade"):
            if extra_field in row and row[extra_field] is not None:
                clean[extra_field] = row[extra_field]
        key = tuple(clean[f] for f in spec.key_fields)
        if key in seen:
            raise OperatorDataValidationError(
                f"{where}: duplicate key {dict(zip(spec.key_fields, key))}"
            )
        seen.add(key)
        checked.append(clean)
    return checked


# -- parquet / csv IO ----------------------------------------------------------


def _read_table(root: Path, filename: str) -> list[dict[str, Any]] | None:
    parquet_path = root / filename
    csv_path = parquet_path.with_suffix(".csv")
    if parquet_path.is_file():
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise OperatorDataValidationError(
                "reading parquet operator tables requires pyarrow; install project requirements"
            ) from exc
        try:
            return pq.read_table(parquet_path).to_pylist()
        except Exception as exc:
            raise OperatorDataValidationError(f"failed to read {parquet_path}: {exc}") from exc
    if csv_path.is_file():
        import csv

        with csv_path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    return None


def _write_table(path: Path, rows: Iterable[Mapping[str, Any]], columns: tuple[str, ...] | None) -> None:
    rows = [dict(row) for row in rows]
    if not rows:
        return
    if columns is None:
        columns = tuple(dict.fromkeys(k for row in rows for k in row))
    else:
        extra = tuple(dict.fromkeys(k for row in rows for k in row if k not in columns))
        columns = columns + extra
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:  # pragma: no cover - fall back to CSV
        import csv

        with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns))
            writer.writeheader()
            for row in rows:
                writer.writerow({c: row.get(c) for c in columns})
        return
    table = pa.Table.from_pylist([{c: row.get(c) for c in columns} for row in rows])
    pq.write_table(table, path, compression="snappy")
