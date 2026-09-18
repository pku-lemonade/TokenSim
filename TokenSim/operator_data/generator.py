"""Offline generation of operator packages from shape manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import DeviceSpec
from TokenSim.operator_data.analytical import Calibration, analytical_model_for
from TokenSim.operator_data.manifest import ShapeManifest
from TokenSim.operator_data.package import OperatorDataPackage, PackageMeta, SourceRecord
from TokenSim.operator_data.schema import TABLE_SPECS


@dataclass(frozen=True)
class GenerationReport:
    package: OperatorDataPackage
    rows_per_table: Mapping[str, int]
    skipped: Mapping[str, int]
    warnings: tuple[str, ...]


def generate_analytical_package(
    device: DeviceSpec,
    manifest: ShapeManifest,
    *,
    calibrations: Mapping[str, Calibration] | None = None,
    dataset_version: str | None = None,
    backend: str = "analytical",
    gated: bool = True,
    notes: str = "",
) -> GenerationReport:
    """Evaluate the device family's analytical model for every manifest key."""
    model = analytical_model_for(device.family)
    calibrations = dict(calibrations or {})
    tables: dict[str, list[dict[str, Any]]] = {}
    analysis: dict[str, list[dict[str, Any]]] = {}
    skipped: dict[str, int] = {}
    warnings: list[str] = []
    sources: dict[str, SourceRecord] = {}

    for table, keys in manifest.keys.items():
        if table in ("collective", "ep_all2all"):
            # Communication is produced by the hierarchical communication model at
            # runtime; only measured tables are stored for it.
            skipped[table] = len(keys)
            continue
        calibration = calibrations.get(table) or calibrations.get("default") or Calibration()
        for key in keys:
            try:
                estimate = model.estimate(table, key, device, calibration, gated=gated)
            except (ConfigurationError, ValueError) as exc:
                skipped[table] = skipped.get(table, 0) + 1
                if len(warnings) < 20:
                    warnings.append(f"{table} {key}: {exc}")
                continue
            source_id = estimate.source_id
            if source_id not in sources:
                sources[source_id] = SourceRecord(
                    source_id=source_id,
                    grade="C" if calibration.calibration_id != "uncalibrated" else "D",
                    method="analytical",
                    reference="TokenSim/operator_data/analytical",
                    notes=(
                        f"{type(model).__name__} roofline estimate on {device.device_id}; "
                        f"calibration={calibration.calibration_id}"
                    ),
                    device=device.device_id,
                    backend=backend,
                )
            row = {**key, "latency_us": estimate.latency_us, "source_id": source_id}
            tables.setdefault(table, []).append(row)
            analysis_row = {**key, **estimate.analysis_row(), "source_id": source_id}
            analysis_row.update({f"detail_{k}": v for k, v in estimate.details.items() if isinstance(v, (int, float, str, bool))})
            analysis.setdefault(table, []).append(analysis_row)
            if estimate.details.get("fits_on_chip") is False and len(warnings) < 40:
                warnings.append(
                    f"{table} {key}: rank-local weights ({estimate.work.weight_bytes:.3e} B) exceed on-chip memory"
                )

    if not tables:
        raise ConfigurationError("manifest produced no analytical rows")
    meta = PackageMeta(
        dataset_version=dataset_version or f"{device.device_id}-{backend}-{manifest.experiment_id}",
        device_id=device.device_id,
        backend=backend,
        framework_version="",
        sources=sources,
        calibration={table: cal.to_dict() for table, cal in calibrations.items()},
        notes=notes or f"Analytical package generated from manifest {manifest.experiment_id!r}.",
        extra={"manifest": manifest.experiment_id, "device_family": device.family},
    )
    package = OperatorDataPackage.from_rows(meta, tables, analysis)
    return GenerationReport(
        package=package,
        rows_per_table={t: len(r) for t, r in tables.items()},
        skipped=skipped,
        warnings=tuple(warnings),
    )


def write_package_with_manifest(package: OperatorDataPackage, manifest: ShapeManifest, root: str | Path) -> Path:
    root_path = package.write(root)
    manifest.write(root_path / "shape_manifests" / f"{manifest.experiment_id}.yaml")
    return root_path


def merge_packages(base: OperatorDataPackage, extra: OperatorDataPackage, *, prefer: str = "extra") -> OperatorDataPackage:
    """Union of two packages for the same device; on key clashes keep ``prefer``."""
    if base.device_id != extra.device_id:
        raise ConfigurationError(
            f"cannot merge packages for different devices {base.device_id!r} and {extra.device_id!r}"
        )
    tables: dict[str, dict[tuple, Mapping[str, Any]]] = {}
    for package, is_extra in ((base, False), (extra, True)):
        for table, rows in package.tables.items():
            spec = TABLE_SPECS[table]
            bucket = tables.setdefault(table, {})
            for row in rows:
                key = tuple(row[f] for f in spec.key_fields)
                if key in bucket and (prefer == "extra") != is_extra:
                    continue
                bucket[key] = row
    analysis: dict[str, list[Mapping[str, Any]]] = {}
    for package in (base, extra):
        for table, rows in package.analysis.items():
            analysis.setdefault(table, []).extend(rows)
    sources = {**base.meta.sources, **extra.meta.sources}
    meta = PackageMeta(
        dataset_version=f"{base.dataset_version}+{extra.dataset_version}",
        device_id=base.device_id,
        backend=base.backend,
        framework_version=base.meta.framework_version or extra.meta.framework_version,
        sources=sources,
        calibration={**base.meta.calibration, **extra.meta.calibration},
        notes=(base.meta.notes + "\n" + extra.meta.notes).strip(),
        extra={**base.meta.extra, **extra.meta.extra},
    )
    return OperatorDataPackage.from_rows(meta, {t: list(rows.values()) for t, rows in tables.items()}, analysis)
