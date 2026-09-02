from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from TokenSim.operator_data.analytical import (
    AnalyticalEstimate,
    Calibration,
    HardwareProfile,
    estimate_operator,
)
from TokenSim.operator_data.schema import (
    LATENCY_UNIT,
    SCHEMA_VERSION,
    TABLE_SPECS,
    analysis_filename,
    perf_filename,
)


class EvidenceGrade(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    evidence_grade: EvidenceGrade
    method: str
    device: str
    backend: str
    reference: str
    input_hash: str

    def __post_init__(self) -> None:
        if self.method not in {
            "measured",
            "public_benchmark",
            "analytical",
            "assumption",
        }:
            raise ValueError(f"unsupported source method={self.method!r}")
        if self.method == "measured" and self.evidence_grade is not EvidenceGrade.A:
            raise ValueError("only grade A target-device data may be marked measured")
        if (
            self.method == "public_benchmark"
            and self.evidence_grade is not EvidenceGrade.B
        ):
            raise ValueError("public end-to-end benchmarks must use evidence grade B")
        if self.method == "analytical" and self.evidence_grade is not EvidenceGrade.C:
            raise ValueError("calibrated analytical data must use evidence grade C")
        if self.method == "assumption" and self.evidence_grade is not EvidenceGrade.D:
            raise ValueError("initial assumptions must use evidence grade D")

    def as_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence_grade"] = self.evidence_grade.value
        return value


@dataclass(frozen=True)
class Observation:
    table_name: str
    key: Mapping[str, Any]
    latency_us: float
    source_id: str
    split: str = "runtime"

    def __post_init__(self) -> None:
        if self.table_name not in TABLE_SPECS:
            raise ValueError(f"unknown table={self.table_name!r}")
        if self.split not in {"runtime", "calibration", "holdout"}:
            raise ValueError(f"unsupported observation split={self.split!r}")
        if not math.isfinite(float(self.latency_us)) or self.latency_us < 0:
            raise ValueError("observation latency_us must be finite and non-negative")


@dataclass(frozen=True)
class ValidationReport:
    count: int
    coverage: float
    mean_relative_error: float | None
    max_relative_error: float | None
    p50_relative_error: float | None
    p90_relative_error: float | None
    p95_relative_error: float | None
    bias: float | None
    bottleneck_consistency: float | None
    worst_shapes: tuple[Mapping[str, Any], ...]
    claim: str

    @classmethod
    def from_holdout(
        cls,
        expected_count: int,
        comparisons: Sequence[Mapping[str, Any]],
        has_target_ground_truth: bool,
    ) -> "ValidationReport":
        errors: list[float] = []
        signed: list[float] = []
        enriched: list[dict[str, Any]] = []
        bottleneck_hits: list[bool] = []
        for item in comparisons:
            actual = float(item["actual_us"])
            predicted = float(item["predicted_us"])
            if actual <= 0:
                continue
            relative = abs(predicted - actual) / actual
            errors.append(relative)
            signed.append((predicted - actual) / actual)
            enriched.append({**dict(item), "relative_error": relative})
            if "actual_bottleneck" in item and "predicted_bottleneck" in item:
                bottleneck_hits.append(
                    item["actual_bottleneck"] == item["predicted_bottleneck"]
                )
        errors.sort()
        worst = tuple(
            sorted(enriched, key=lambda row: row["relative_error"], reverse=True)[:10]
        )
        return cls(
            count=len(comparisons),
            coverage=(len(comparisons) / expected_count if expected_count else 1.0),
            mean_relative_error=mean(errors) if errors else None,
            max_relative_error=max(errors) if errors else None,
            p50_relative_error=_percentile(errors, 0.50),
            p90_relative_error=_percentile(errors, 0.90),
            p95_relative_error=_percentile(errors, 0.95),
            bias=mean(signed) if signed else None,
            bottleneck_consistency=(
                sum(bottleneck_hits) / len(bottleneck_hits) if bottleneck_hits else None
            ),
            worst_shapes=worst,
            claim=(
                "target-hardware holdout validation"
                if has_target_ground_truth
                else "mechanism and model self-consistency only; physical accuracy pending measurements"
            ),
        )


class OperatorDataGenerator:
    def __init__(
        self,
        dataset_version: str,
        hardware_profiles: Mapping[str, HardwareProfile],
        sources: Iterable[SourceRecord],
        calibrations: Mapping[str, Calibration] | None = None,
        tool_version: str = "tokensim-operator-data-v1",
    ) -> None:
        if not dataset_version:
            raise ValueError("dataset_version must be non-empty")
        self.dataset_version = dataset_version
        self.hardware_profiles = dict(hardware_profiles)
        source_values = tuple(sources)
        self.sources = {source.source_id: source for source in source_values}
        self.calibrations = dict(calibrations or {})
        self.tool_version = tool_version
        if not self.hardware_profiles:
            raise ValueError("at least one hardware profile is required")
        if len(self.sources) != len(source_values):
            raise ValueError("source_id values must be unique")

    def generate(
        self,
        output_dir: str | Path,
        experiment_id: str,
        shape_requests: Sequence[Mapping[str, Any]],
        observations: Sequence[Observation],
        holdout_comparisons: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    ) -> Mapping[str, ValidationReport]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        manifest_dir = output / "shape_manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        source_records = dict(self.sources)
        grouped_observations: dict[str, list[Observation]] = {
            name: [] for name in TABLE_SPECS
        }
        for observation in observations:
            if observation.source_id not in source_records:
                raise ValueError(f"unknown source_id={observation.source_id!r}")
            grouped_observations[observation.table_name].append(observation)

        request_by_table: dict[str, list[dict[str, Any]]] = {
            name: [] for name in TABLE_SPECS
        }
        for request in shape_requests:
            table_name = str(request.get("table"))
            if table_name not in TABLE_SPECS:
                raise ValueError(f"shape request has unknown table={table_name!r}")
            key = _canonical_key(table_name, request.get("key", {}))
            request_by_table[table_name].append(key)

        perf_tables: dict[str, list[dict[str, Any]]] = {}
        analysis_tables: dict[str, list[dict[str, Any]]] = {}
        for table_name in TABLE_SPECS:
            perf_rows, analysis_rows = self._generate_table(
                table_name,
                request_by_table[table_name],
                grouped_observations[table_name],
                source_records,
            )
            if not perf_rows:
                raise ValueError(
                    f"shape manifest must request at least one row for table={table_name!r}"
                )
            perf_tables[table_name] = perf_rows
            analysis_tables[table_name] = analysis_rows

        input_shapes = [
            {"table": str(item["table"]), "key": dict(item["key"])}
            for item in shape_requests
        ]
        input_observations = [
            {
                "table": item.table_name,
                "key": dict(item.key),
                "latency_us": item.latency_us,
                "source_id": item.source_id,
                "split": item.split,
            }
            for item in observations
        ]
        input_document = {
            "experiment_id": experiment_id,
            "shape_requests": sorted(
                input_shapes, key=lambda item: json.dumps(item, sort_keys=True)
            ),
            "observations": sorted(
                input_observations,
                key=lambda item: json.dumps(item, sort_keys=True),
            ),
        }
        # Canonical serialization makes generation metadata independent of input order.
        input_hash = hashlib.sha256(
            json.dumps(input_document, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

        reports: dict[str, ValidationReport] = {}
        comparisons = holdout_comparisons or {}
        for table_name in TABLE_SPECS:
            table_comparisons = comparisons.get(table_name, ())
            has_ground_truth = any(
                bool(item.get("target_hardware_ground_truth"))
                for item in table_comparisons
            )
            reports[table_name] = ValidationReport.from_holdout(
                len(table_comparisons), table_comparisons, has_ground_truth
            )

        hardware_document = {
            "schema_version": SCHEMA_VERSION,
            "devices": {
                name: {
                    "backend": profile.backend,
                    "peak_compute_by_dtype": dict(profile.peak_compute_by_dtype),
                    "memory_bandwidth_bytes_per_s": profile.memory_bandwidth_bytes_per_s,
                    "memory_capacity_bytes": profile.memory_capacity_bytes,
                    "interconnect_bandwidth_bytes_per_s": profile.interconnect_bandwidth_bytes_per_s,
                }
                for name, profile in sorted(self.hardware_profiles.items())
            },
        }
        metadata_document = {
            "schema_version": SCHEMA_VERSION,
            "dataset_version": self.dataset_version,
            "latency_unit": LATENCY_UNIT,
            "tool_version": self.tool_version,
            "input_hash": input_hash,
            "sources": [
                source.as_mapping()
                for source in sorted(
                    source_records.values(), key=lambda item: item.source_id
                )
            ],
            "calibration_ids": sorted(
                calibration.calibration_id for calibration in self.calibrations.values()
            ),
            "calibration_observation_ids": sorted(
                item.source_id for item in observations if item.split == "calibration"
            ),
            "holdout_observation_ids": sorted(
                item.source_id for item in observations if item.split == "holdout"
            ),
            "validation_reports": {
                name: asdict(report) for name, report in sorted(reports.items())
            },
        }
        manifest_document = {
            "schema_version": SCHEMA_VERSION,
            "dataset_version": self.dataset_version,
            "experiment_id": experiment_id,
            "shapes": sorted(
                (
                    {"table": table, "key": key}
                    for table, keys in request_by_table.items()
                    for key in keys
                ),
                key=lambda item: json.dumps(item, sort_keys=True),
            ),
        }
        _write_yaml(output / "hardware.yaml", hardware_document)
        _write_yaml(output / "generation_meta.yaml", metadata_document)
        _write_yaml(manifest_dir / f"{experiment_id}.yaml", manifest_document)
        for table_name in TABLE_SPECS:
            _write_parquet(output / perf_filename(table_name), perf_tables[table_name])
            _write_parquet(
                output / analysis_filename(table_name), analysis_tables[table_name]
            )
        return reports

    def _generate_table(
        self,
        table_name: str,
        requested_keys: Sequence[Mapping[str, Any]],
        observations: Sequence[Observation],
        source_records: dict[str, SourceRecord],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        observed_by_key: dict[tuple[Any, ...], list[Observation]] = {}
        spec = TABLE_SPECS[table_name]
        for observation in observations:
            key = _canonical_key(table_name, observation.key)
            key_tuple = tuple(key[field] for field in spec.key_fields)
            observed_by_key.setdefault(key_tuple, []).append(observation)

        perf_rows: list[dict[str, Any]] = []
        analysis_rows: list[dict[str, Any]] = []
        for key in sorted(requested_keys, key=lambda item: tuple(item.values())):
            key_tuple = tuple(key[field] for field in spec.key_fields)
            candidates = observed_by_key.get(key_tuple, [])
            eligible = [
                item
                for item in candidates
                if item.split != "holdout"
                and source_records[item.source_id].method != "public_benchmark"
            ]
            analytical: AnalyticalEstimate | None = None
            if table_name != "communication":
                formula_source_id = f"formula:{table_name}:{key['device']}"
                if formula_source_id not in source_records:
                    source_records[formula_source_id] = SourceRecord(
                        source_id=formula_source_id,
                        evidence_grade=EvidenceGrade.C,
                        method="analytical",
                        device=str(key["device"]),
                        backend=str(key["backend"]),
                        reference="operator-data analytical formula",
                        input_hash="generated",
                    )
                hardware = self.hardware_profiles[str(key["device"])]
                analytical = estimate_operator(
                    table_name,
                    key,
                    hardware,
                    self.calibrations.get(table_name),
                )
                eligible.append(
                    Observation(
                        table_name=table_name,
                        key=key,
                        latency_us=analytical.latency_us,
                        source_id=formula_source_id,
                    )
                )
            if not eligible:
                raise ValueError(
                    f"no eligible measured/profile data for {table_name} key={key}"
                )
            selected = min(
                eligible,
                key=lambda item: (
                    _source_priority(source_records[item.source_id]),
                    item.source_id,
                    float(item.latency_us),
                ),
            )
            source = source_records[selected.source_id]
            if table_name == "communication":
                raw = next(
                    item for item in candidates if item.source_id == selected.source_id
                )
                extra = {
                    name: raw.key[name]
                    for name in (
                        "fixed_latency_us",
                        "effective_bandwidth_bytes_per_s",
                    )
                    if name in raw.key
                }
                if set(extra) != {
                    "fixed_latency_us",
                    "effective_bandwidth_bytes_per_s",
                }:
                    raise ValueError(
                        f"communication observation missing link parameters for key={key}"
                    )
                perf_row = {**key, **extra, "source_id": selected.source_id}
            else:
                perf_row = {
                    **key,
                    "latency_us": float(selected.latency_us),
                    "source_id": selected.source_id,
                }
            perf_rows.append(perf_row)
            analysis_row = {
                **key,
                "latency_us": float(selected.latency_us),
                "source_id": selected.source_id,
                "method": source.method,
                "evidence_grade": source.evidence_grade.value,
                "candidate_count": len(eligible),
                "selected_reason": "source precedence",
            }
            if analytical is not None:
                analysis_row.update(asdict(analytical))
                analysis_row["latency_us"] = float(selected.latency_us)
            analysis_rows.append(analysis_row)
        return perf_rows, analysis_rows


def _canonical_key(table_name: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    spec = TABLE_SPECS[table_name]
    missing = [field for field in spec.key_fields if field not in raw]
    if missing:
        raise ValueError(f"{table_name} key missing fields {missing}: {dict(raw)}")
    key: dict[str, Any] = {}
    for field in spec.key_fields:
        value = raw[field]
        key[field] = value.strip().lower() if isinstance(value, str) else value
    return key


def _source_priority(source: SourceRecord) -> int:
    return {
        ("measured", EvidenceGrade.A): 0,
        ("analytical", EvidenceGrade.C): 1,
        ("assumption", EvidenceGrade.D): 2,
        ("public_benchmark", EvidenceGrade.B): 3,
    }[(source.method, source.evidence_grade)]


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _write_yaml(path: Path, document: Mapping[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("generation requires PyYAML") from exc
    path.write_text(
        yaml.safe_dump(document, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("generation requires pyarrow") from exc
    table = pa.Table.from_pylist([dict(row) for row in rows])
    pq.write_table(table, path)
