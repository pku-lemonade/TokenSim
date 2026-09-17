"""Command line tools for operator data.

Examples::

    python -m TokenSim.operator_data.cli manifest --device groqchip_v1 --models llama-3-8b --tp 1,8 \
        --out data/operator_data/groqchip_v1/analytical/shape_manifests/exp1.yaml
    python -m TokenSim.operator_data.cli generate --device groqchip_v1 --models llama-3-8b,mixtral-8x7b --tp 1,8,64
    python -m TokenSim.operator_data.cli import-aiconfigurator --system-dir <aic>/systems/data/a100_sxm --device a100_sxm_80g
    python -m TokenSim.operator_data.cli import-nccl --device a100_sxm_80g --file all_reduce_8.txt --operation all_reduce --group-size 8
    python -m TokenSim.operator_data.cli validate data/operator_data/a100_sxm_80g/trtllm
    python -m TokenSim.operator_data.cli coverage --device a100_sxm_80g --backend trtllm --models llama-3-70b --tp 4
    python -m TokenSim.operator_data.cli calibrate --device a100_sxm_80g --backend trtllm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from TokenSim.errors import ConfigurationError
from TokenSim.hardware.context import DEFAULT_DATA_ROOT, HardwareContext
from TokenSim.operator_data.analytical import Calibration
from TokenSim.operator_data.calibrate import calibrate_package
from TokenSim.operator_data.coverage import coverage_report
from TokenSim.operator_data.generator import (
    generate_analytical_package,
    merge_packages,
    write_package_with_manifest,
)
from TokenSim.operator_data.importers.aiconfigurator import import_aiconfigurator_system
from TokenSim.operator_data.importers.nccl_tests import parse_nccl_tests_output
from TokenSim.operator_data.manifest import ShapeManifest, WorkloadGrid, build_manifest
from TokenSim.operator_data.package import OperatorDataPackage, PackageMeta, SourceRecord


def _ints(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if not value:
        return default
    return tuple(int(v) for v in value.split(",") if v.strip())


def _grid(args: argparse.Namespace) -> WorkloadGrid:
    base = WorkloadGrid()
    return WorkloadGrid(
        batch_sizes=_ints(args.batch, base.batch_sizes),
        prefill_tokens=_ints(args.prefill_tokens, base.prefill_tokens),
        context_lens=_ints(args.context, base.context_lens),
        tp_sizes=_ints(args.tp, base.tp_sizes),
        ep_sizes=_ints(args.ep, base.ep_sizes),
        include_collectives=not getattr(args, "no_collectives", False),
    )


def _manifest_from_args(hardware: HardwareContext, args: argparse.Namespace) -> ShapeManifest:
    if getattr(args, "manifest", None):
        return ShapeManifest.load(args.manifest)
    if not args.models:
        raise ConfigurationError("--models or --manifest is required")
    models = [hardware.model(name.strip()) for name in args.models.split(",") if name.strip()]
    experiment_id = args.experiment_id or "-".join(m.model_id for m in models)
    return build_manifest(experiment_id, models, _grid(args))


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_manifest(args: argparse.Namespace) -> int:
    hardware = HardwareContext.load(args.data_root)
    manifest = _manifest_from_args(hardware, args)
    out = Path(args.out) if args.out else Path(args.data_root) / "operator_data" / "manifests" / f"{manifest.experiment_id}.yaml"
    manifest.write(out)
    _print({"manifest": str(out), "counts": manifest.count(), "total": manifest.total()})
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    hardware = HardwareContext.load(args.data_root)
    device = hardware.device(args.device)
    manifest = _manifest_from_args(hardware, args)
    calibrations: dict[str, Calibration] = {}
    if args.calibration:
        import yaml

        raw = yaml.safe_load(Path(args.calibration).read_text(encoding="utf-8")) or {}
        for table, values in (raw.get("calibration") or raw).items():
            if isinstance(values, dict) and ("multiplier" in values or "offset_us" in values):
                calibrations[table] = Calibration.from_mapping(values)
        analytical_overrides = raw.get("analytical") or {}
        if analytical_overrides:
            from dataclasses import replace

            device = replace(device, analytical={**device.analytical, **{k: float(v) for k, v in analytical_overrides.items()}})
    report = generate_analytical_package(
        device,
        manifest,
        calibrations=calibrations,
        dataset_version=args.dataset_version,
        backend=args.backend,
    )
    out = Path(args.out) if args.out else Path(args.data_root) / "operator_data" / device.device_id / args.backend
    package = report.package
    if args.merge and (out / "generation_meta.yaml").is_file():
        package = merge_packages(OperatorDataPackage.load(out), package, prefer="base" if args.keep_existing else "extra")
    write_package_with_manifest(package, manifest, out)
    _print(
        {
            "package": str(out),
            "rows": report.rows_per_table,
            "skipped": report.skipped,
            "warnings": list(report.warnings[:10]),
            "summary": package.summary(),
        }
    )
    return 0


def cmd_import_aiconfigurator(args: argparse.Namespace) -> int:
    package = import_aiconfigurator_system(
        args.system_dir,
        device_id=args.device,
        backend=args.backend,
        version=args.version,
        nccl_version=args.nccl_version,
        dataset_version=args.dataset_version,
        include_custom_allreduce=not args.no_custom_allreduce,
        upstream_commit=args.upstream_commit,
    )
    out = Path(args.out) if args.out else Path(args.data_root) / "operator_data" / args.device / args.backend
    package.write(out)
    _print({"package": str(out), **package.summary(), "resolved_versions": package.meta.extra.get("resolved_versions"), "dropped_rows": package.meta.extra.get("dropped_rows")})
    return 0


def cmd_import_nccl(args: argparse.Namespace) -> int:
    source_id = args.source_id or f"nccl-tests:{args.device}:{Path(args.file).stem}"
    rows = parse_nccl_tests_output(
        Path(args.file),
        operation=args.operation,
        group_size=args.group_size,
        nodes=args.nodes,
        source_id=source_id,
    )
    out = Path(args.out) if args.out else Path(args.data_root) / "operator_data" / args.device / args.backend
    source = SourceRecord(
        source_id=source_id,
        grade="A",
        method="measured",
        reference=args.reference or str(Path(args.file)),
        notes=args.notes or f"nccl-tests {args.operation} over {args.group_size} ranks on {args.nodes} node(s)",
        device=args.device,
        backend=args.backend,
    )
    meta = PackageMeta(
        dataset_version=args.dataset_version or f"{args.device}-{args.backend}-nccl-tests",
        device_id=args.device,
        backend=args.backend,
        sources={source_id: source},
        notes="Collective latencies parsed from nccl-tests output.",
    )
    package = OperatorDataPackage.from_rows(meta, {"collective": rows})
    if (out / "generation_meta.yaml").is_file():
        package = merge_packages(OperatorDataPackage.load(out), package, prefer="extra")
    package.write(out)
    _print({"package": str(out), "rows_added": len(rows), **package.summary()})
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    package = OperatorDataPackage.load(args.package)
    _print({"valid": True, **package.summary(), "notes": package.meta.notes})
    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    hardware = HardwareContext.load(args.data_root)
    manifest = _manifest_from_args(hardware, args)
    package = hardware.operator_package(hardware.device(args.device).device_id, args.backend)
    if package is None:
        raise ConfigurationError(f"device {args.device!r} has no operator data under {args.data_root}/operator_data")
    report = coverage_report(package, manifest)
    summary = report.summary()
    if args.out:
        import yaml

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            yaml.safe_dump({"summary": summary, "missing": report.missing}, sort_keys=False), encoding="utf-8"
        )
        summary["missing_written_to"] = args.out
    else:
        summary["missing_examples"] = report.missing[:10]
    _print(summary)
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    hardware = HardwareContext.load(args.data_root)
    device = hardware.device(args.device)
    package = hardware.operator_package(device.device_id, args.backend)
    if package is None:
        raise ConfigurationError(f"device {args.device!r} has no operator data to calibrate against")
    results = calibrate_package(device, package, holdout_fraction=args.holdout, seed=args.seed, max_rows=args.max_rows)
    output = {
        "device_id": device.device_id,
        "backend": package.backend,
        "dataset_version": package.dataset_version,
        "analytical": {},
        "tables": {},
    }
    for table, result in results.items():
        output["tables"][table] = result.to_dict()
        # Parameters are shared across tables; the last table wins for shared
        # knobs, so report per-table values and a merged suggestion.
        output["analytical"].update(result.parameters)
    if args.out:
        import yaml

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(yaml.safe_dump(output, sort_keys=False, default_flow_style=False), encoding="utf-8")
        print(f"wrote {args.out}")
    compact = {
        table: {
            "parameters": r.parameters,
            "baseline_mean_rel_err": r.baseline_fit.mean_relative_error,
            "fit_mean_rel_err": r.fit.mean_relative_error,
            "holdout_mean_rel_err": r.holdout.mean_relative_error,
            "holdout_p90_rel_err": r.holdout.p90_relative_error,
            "holdout_count": r.holdout.count,
        }
        for table, r in results.items()
    }
    _print(compact)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate, import, validate and calibrate operator latency tables")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), dest="data_root")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_shape_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--models", help="comma-separated model ids from data/models")
        p.add_argument("--manifest", help="existing shape manifest YAML (overrides --models)")
        p.add_argument("--experiment-id", dest="experiment_id")
        p.add_argument("--tp", help="comma-separated TP sizes (default 1)")
        p.add_argument("--ep", help="comma-separated EP sizes (default 1)")
        p.add_argument("--batch", help="comma-separated decode batch sizes")
        p.add_argument("--prefill-tokens", dest="prefill_tokens", help="comma-separated prefill token counts")
        p.add_argument("--context", help="comma-separated decode context lengths")
        p.add_argument("--no-collectives", dest="no_collectives", action="store_true")

    p = sub.add_parser("manifest", help="write a shape manifest for models x parallel layouts x workload grid")
    add_shape_args(p)
    p.add_argument("--out")
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("generate", help="generate an analytical package for a device")
    p.add_argument("--device", required=True)
    p.add_argument("--backend", default="analytical")
    p.add_argument("--calibration", help="YAML with per-table calibration and/or analytical parameter overrides")
    p.add_argument("--dataset-version", dest="dataset_version")
    p.add_argument("--out")
    p.add_argument("--merge", action="store_true", help="merge into an existing package at --out")
    p.add_argument("--keep-existing", dest="keep_existing", action="store_true", help="on key clash keep existing rows")
    add_shape_args(p)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("import-aiconfigurator", help="convert an AIConfigurator system directory")
    p.add_argument("--system-dir", dest="system_dir", required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--backend", default="trtllm")
    p.add_argument("--version")
    p.add_argument("--nccl-version", dest="nccl_version")
    p.add_argument("--dataset-version", dest="dataset_version")
    p.add_argument("--upstream-commit", dest="upstream_commit", default="main")
    p.add_argument("--no-custom-allreduce", dest="no_custom_allreduce", action="store_true")
    p.add_argument("--out")
    p.set_defaults(func=cmd_import_aiconfigurator)

    p = sub.add_parser("import-nccl", help="add nccl-tests output to a package's collective table")
    p.add_argument("--device", required=True)
    p.add_argument("--backend", default="nccl")
    p.add_argument("--file", required=True)
    p.add_argument("--operation", required=True, choices=["all_reduce", "all_gather", "reduce_scatter", "all_to_all", "send_recv", "broadcast"])
    p.add_argument("--group-size", dest="group_size", type=int, required=True)
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--source-id", dest="source_id")
    p.add_argument("--reference")
    p.add_argument("--notes")
    p.add_argument("--dataset-version", dest="dataset_version")
    p.add_argument("--out")
    p.set_defaults(func=cmd_import_nccl)

    p = sub.add_parser("validate", help="validate a package directory")
    p.add_argument("package")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("coverage", help="report exact/interpolated/extrapolated/missing per manifest query")
    p.add_argument("--device", required=True)
    p.add_argument("--backend")
    p.add_argument("--out", help="write summary and missing shapes to YAML")
    add_shape_args(p)
    p.set_defaults(func=cmd_coverage)

    p = sub.add_parser("calibrate", help="fit analytical parameters to a measured package")
    p.add_argument("--device", required=True)
    p.add_argument("--backend")
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-rows", dest="max_rows", type=int, default=3000)
    p.add_argument("--out")
    p.set_defaults(func=cmd_calibrate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigurationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
