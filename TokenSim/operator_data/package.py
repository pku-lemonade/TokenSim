from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from TokenSim.operator_data.schema import (
    INTEGER_FIELDS,
    LATENCY_UNIT,
    SCHEMA_VERSION,
    STRING_FIELDS,
    TABLE_SPECS,
    analysis_filename,
    perf_filename,
)


class OperatorDataValidationError(ValueError):
    pass


def _load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise OperatorDataValidationError(
            "operator-data YAML support requires PyYAML; install project requirements"
        ) from exc
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OperatorDataValidationError(f"failed to parse {path}: {exc}") from exc


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise OperatorDataValidationError(
            "operator-data Parquet support requires pyarrow; install project requirements"
        ) from exc
    try:
        return pq.read_table(path).to_pylist()
    except Exception as exc:
        raise OperatorDataValidationError(f"failed to read {path}: {exc}") from exc


@dataclass(frozen=True)
class OperatorDataPackage:
    root: Path
    schema_version: int
    dataset_version: str
    hardware: Mapping[str, Any]
    metadata: Mapping[str, Any]
    manifests: tuple[Mapping[str, Any], ...]
    tables: Mapping[str, tuple[Mapping[str, Any], ...]]
    analysis_tables: Mapping[str, tuple[Mapping[str, Any], ...]]

    @classmethod
    def load(cls, root: str | Path) -> "OperatorDataPackage":
        root_path = Path(root)
        if not root_path.is_dir():
            raise OperatorDataValidationError(
                f"operator-data package directory does not exist: {root_path}"
            )
        hardware_path = root_path / "hardware.yaml"
        metadata_path = root_path / "generation_meta.yaml"
        for required in (hardware_path, metadata_path):
            if not required.is_file():
                raise OperatorDataValidationError(f"missing required file: {required}")

        hardware = _require_mapping(_load_yaml(hardware_path), hardware_path)
        metadata = _require_mapping(_load_yaml(metadata_path), metadata_path)
        schema_version = metadata.get("schema_version")
        if schema_version != SCHEMA_VERSION:
            raise OperatorDataValidationError(
                f"{metadata_path}: unsupported schema_version={schema_version!r}; "
                f"expected {SCHEMA_VERSION}"
            )
        if hardware.get("schema_version") != SCHEMA_VERSION:
            raise OperatorDataValidationError(
                f"{hardware_path}: unsupported schema_version="
                f"{hardware.get('schema_version')!r}; expected {SCHEMA_VERSION}"
            )
        dataset_version = metadata.get("dataset_version")
        if not isinstance(dataset_version, str) or not dataset_version:
            raise OperatorDataValidationError(
                f"{metadata_path}: dataset_version must be a non-empty string"
            )
        if metadata.get("latency_unit") != LATENCY_UNIT:
            raise OperatorDataValidationError(
                f"{metadata_path}: latency_unit must be {LATENCY_UNIT!r}"
            )
        devices = hardware.get("devices")
        if not isinstance(devices, Mapping) or not devices:
            raise OperatorDataValidationError(
                f"{hardware_path}: devices must be a non-empty mapping"
            )
        sources = metadata.get("sources")
        if not isinstance(sources, list) or not sources:
            raise OperatorDataValidationError(
                f"{metadata_path}: sources must be a non-empty list"
            )
        source_ids = _validate_sources(sources, metadata_path)

        manifest_dir = root_path / "shape_manifests"
        manifest_paths = (
            sorted(manifest_dir.glob("*.yaml")) if manifest_dir.is_dir() else []
        )
        if not manifest_paths:
            raise OperatorDataValidationError(
                f"{manifest_dir}: at least one shape manifest is required"
            )
        manifests = tuple(
            _validate_manifest(
                _require_mapping(_load_yaml(path), path), path, dataset_version
            )
            for path in manifest_paths
        )

        tables: dict[str, tuple[Mapping[str, Any], ...]] = {}
        analysis_tables: dict[str, tuple[Mapping[str, Any], ...]] = {}
        for table_name, spec in TABLE_SPECS.items():
            perf_path = root_path / perf_filename(table_name)
            analysis_path = root_path / analysis_filename(table_name)
            for required in (perf_path, analysis_path):
                if not required.is_file():
                    raise OperatorDataValidationError(
                        f"missing required file: {required}"
                    )
            rows = _read_parquet(perf_path)
            checked = _validate_rows(
                perf_path,
                rows,
                spec.required_fields,
                source_ids,
                require_latency="latency_us" in spec.value_fields,
            )
            _validate_unique_keys(perf_path, checked, spec.key_fields)
            tables[table_name] = tuple(checked)

            analysis_rows = _read_parquet(analysis_path)
            analysis_required = spec.key_fields + (
                "source_id",
                "method",
                "evidence_grade",
            )
            analysis_tables[table_name] = tuple(
                _validate_rows(
                    analysis_path,
                    analysis_rows,
                    analysis_required,
                    source_ids,
                    require_latency=False,
                )
            )
            _validate_unique_keys(
                analysis_path, list(analysis_tables[table_name]), spec.key_fields
            )
            perf_keys = {
                tuple(row[field] for field in spec.key_fields)
                for row in tables[table_name]
            }
            analysis_keys = {
                tuple(row[field] for field in spec.key_fields)
                for row in analysis_tables[table_name]
            }
            if perf_keys != analysis_keys:
                raise OperatorDataValidationError(
                    f"{analysis_path}: analysis keys must match {perf_path.name} exactly"
                )

        return cls(
            root=root_path.resolve(),
            schema_version=schema_version,
            dataset_version=dataset_version,
            hardware=hardware,
            metadata=metadata,
            manifests=manifests,
            tables=tables,
            analysis_tables=analysis_tables,
        )


def _require_mapping(value: Any, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OperatorDataValidationError(f"{path}: document root must be a mapping")
    return value


def _validate_sources(sources: list[Any], path: Path) -> set[str]:
    source_ids: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise OperatorDataValidationError(
                f"{path}: sources[{index}] must be a mapping"
            )
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise OperatorDataValidationError(
                f"{path}: sources[{index}].source_id must be a non-empty string"
            )
        if source_id in source_ids:
            raise OperatorDataValidationError(
                f"{path}: duplicate source_id={source_id!r}"
            )
        grade = source.get("evidence_grade")
        if grade not in {"A", "B", "C", "D"}:
            raise OperatorDataValidationError(
                f"{path}: source {source_id!r} has invalid evidence_grade={grade!r}"
            )
        method = source.get("method")
        if method not in {"measured", "public_benchmark", "analytical", "assumption"}:
            raise OperatorDataValidationError(
                f"{path}: source {source_id!r} has invalid method={method!r}"
            )
        if method == "measured" and grade not in {"A"}:
            raise OperatorDataValidationError(
                f"{path}: source {source_id!r} may be measured only with evidence grade A"
            )
        source_ids.add(source_id)
    return source_ids


def _validate_manifest(
    manifest: Mapping[str, Any], path: Path, dataset_version: str
) -> Mapping[str, Any]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise OperatorDataValidationError(f"{path}: invalid schema_version")
    if manifest.get("dataset_version") != dataset_version:
        raise OperatorDataValidationError(
            f"{path}: dataset_version does not match generation_meta.yaml"
        )
    if not isinstance(manifest.get("shapes"), list):
        raise OperatorDataValidationError(f"{path}: shapes must be a list")
    return manifest


def _validate_rows(
    path: Path,
    rows: list[dict[str, Any]],
    required_fields: tuple[str, ...],
    source_ids: set[str],
    require_latency: bool = True,
) -> list[dict[str, Any]]:
    if not rows:
        raise OperatorDataValidationError(
            f"{path}: table must contain at least one row"
        )
    checked: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        missing = [field for field in required_fields if field not in raw]
        if missing:
            raise OperatorDataValidationError(
                f"{path}: row {index} missing required fields {missing}"
            )
        row = dict(raw)
        for field in required_fields:
            value = row[field]
            if field in STRING_FIELDS and field != "source_id":
                value = value.strip().lower() if isinstance(value, str) else value
                row[field] = value
            if field in INTEGER_FIELDS and (type(value) is not int or value < 0):
                raise OperatorDataValidationError(
                    f"{path}: row {index} field {field} must be a non-negative integer"
                )
            if field in STRING_FIELDS and (
                not isinstance(value, str) or not value.strip()
            ):
                raise OperatorDataValidationError(
                    f"{path}: row {index} field {field} must be a non-empty string"
                )
        source_id = row.get("source_id")
        if source_id not in source_ids:
            raise OperatorDataValidationError(
                f"{path}: row {index} references unknown source_id={source_id!r}"
            )
        numeric_fields = (("latency_us",) if require_latency else ()) + tuple(
            field
            for field in ("fixed_latency_us", "effective_bandwidth_bytes_per_s")
            if field in required_fields
        )
        for field in numeric_fields:
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise OperatorDataValidationError(
                    f"{path}: row {index} field {field} must be numeric"
                )
            if not math.isfinite(float(value)) or float(value) < 0:
                raise OperatorDataValidationError(
                    f"{path}: row {index} field {field} must be finite and non-negative"
                )
        if (
            "effective_bandwidth_bytes_per_s" in required_fields
            and float(row["effective_bandwidth_bytes_per_s"]) <= 0
        ):
            raise OperatorDataValidationError(
                f"{path}: row {index} effective_bandwidth_bytes_per_s must be positive"
            )
        checked.append(row)
    return checked


def _validate_unique_keys(
    path: Path, rows: list[Mapping[str, Any]], key_fields: tuple[str, ...]
) -> None:
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key in seen:
            context = {field: row[field] for field in key_fields}
            raise OperatorDataValidationError(f"{path}: duplicate exact key {context}")
        seen.add(key)
